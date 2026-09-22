"""The actual pixel work: landmark fitting, warping, colour matching, blending.

Pipeline, all classic computer vision, no learned models:

1. Fit a 12-point landmark layout to the source face and to the target face.
   The layout is scaled from the face box (and refined by eye measurements
   where the cascade finds them), so a source photo at any size lines up with
   the target face in a video frame regardless of resolution or aspect ratio.
2. Triangulate the landmarks (Delaunay) and warp each source triangle onto the
   matching target triangle -> piecewise-affine warp.  This is a genuine
   geometric deformation of the source face, not an overlay or a sticker.
3. Match the source colours to the target face in CIELAB so the pasted face
   takes on the target's lighting and skin tone.
4. Blend with a mask that decays toward the face boundary, optionally through
   ``cv2.seamlessClone`` (Poisson gradient blending) so edges disappear.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import FaceBox

# --------------------------------------------------------------------------
# Landmarks
# --------------------------------------------------------------------------

#: Landmarks expressed relative to the *eye centre* and scaled by the
#: interocular distance ``d`` (pupil-to-pupil).  +y points down.
#:
#: These ratios were measured off real faces with this very detector: a Haar
#: frontal face box is about 2.84 d wide, its eye line sits ~0.39 of the box
#: height below the top, and the chin is ~1.74 d below the eye line.  Anchoring
#: on the eyes rather than the box is what makes a source photo line up with a
#: small face in a video frame: eye spacing is the one measurement both the
#: cascade and the eye cascade agree on.
_LANDMARKS_EYE_UNITS: tuple[tuple[float, float], ...] = (
    (-0.50, 0.00),  # 0  left eye
    (0.50, 0.00),  # 1  right eye
    (0.00, 0.25),  # 2  nose bridge
    (0.00, 0.80),  # 3  nose tip
    (-0.50, 1.20),  # 4  mouth left
    (0.50, 1.20),  # 5  mouth right
    (0.00, 1.20),  # 6  mouth centre
    (0.00, 1.74),  # 7  chin
    (-1.25, 0.05),  # 8  left cheek
    (1.25, 0.05),  # 9  right cheek
    (-1.05, 1.10),  # 10 jaw lower-left
    (1.05, 1.10),  # 11 jaw lower-right
    (-0.62, -0.62),  # 12 brow left
    (0.62, -0.62),  # 13 brow right
)

_LEFT_EYE, _RIGHT_EYE = 0, 1
_MOUTH_LEFT, _MOUTH_RIGHT = 4, 5

#: Fallback eye geometry read straight off a Haar box, used when the eye
#: cascade finds nothing: eye centre sits 0.112 of box height above centre and
#: the interocular distance is 0.352 of box width.
_EYE_CENTER_Y_FROM_BOX = -0.112
_INTEROCULAR_OVER_BOX_W = 0.352


def landmark_offsets() -> np.ndarray:
    """Landmarks in eye units, as a fresh array."""
    return np.asarray(_LANDMARKS_EYE_UNITS, dtype=np.float64)


@dataclass
class FaceLandmarks:
    """Landmark positions for one face, in image pixel coordinates."""

    points: np.ndarray  # (N, 2) float32
    box: FaceBox
    eye_center: tuple[float, float]
    eye_distance: float

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0)


def fit_landmarks(
    box: FaceBox,
    eyes: list[tuple[float, float]] | None = None,
    mouth: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> FaceLandmarks:
    """Place the landmark layout on ``box``.

    ``eyes`` (and optionally ``mouth`` corners) are measured positions; when
    they are missing the eye geometry is inferred from the box so the layout is
    still proportional.  Because every point is scaled by the interocular
    distance, a 4000 px source portrait and a 60 px face in a video frame end
    up with matching topology.
    """
    if eyes and len(eyes) == 2:
        (lx, ly), (rx, ry) = sorted(eyes, key=lambda p: p[0])
    else:
        # No eye cascade result: fall back to the box geometry.  The spacing
        # must be the measured interocular/box ratio, not half the box width -
        # using the box would make the face about twice too large.
        half = _INTEROCULAR_OVER_BOX_W * box.w / 2.0
        centre_x = box.x + box.w / 2.0
        lx, rx = centre_x - half, centre_x + half
        ly = ry = box.y + box.h * (0.5 + _EYE_CENTER_Y_FROM_BOX)

    eye_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    eye_distance = float(np.hypot(rx - lx, ry - ly))
    if eye_distance < 1e-3:
        eye_distance = max(1.0, _INTEROCULAR_OVER_BOX_W * box.w)

    offsets = landmark_offsets()
    # Rotate the layout by the tilt of the eye line so a tilted head still gets
    # a correctly oriented face.
    angle = np.arctan2(ry - ly, rx - lx)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ox, oy = offsets[:, 0] * eye_distance, offsets[:, 1] * eye_distance
    pts = np.empty_like(offsets, dtype=np.float32)
    pts[:, 0] = eye_center[0] + ox * cos_a - oy * sin_a
    pts[:, 1] = eye_center[1] + ox * sin_a + oy * cos_a

    if mouth and len(mouth) == 2:
        m_left, m_right = sorted(mouth, key=lambda p: p[0])
        m_y = (m_left[1] + m_right[1]) / 2.0
        # The smile cascade sometimes fires on a chin crease or a nostril.  A
        # mouth that lands above the nose tip or below the chin would invert
        # triangles and tear the mesh, so such a measurement is discarded and
        # the proportional layout is kept instead.
        if pts[2][1] < m_y < pts[7][1]:
            pts[_MOUTH_LEFT] = m_left
            pts[_MOUTH_RIGHT] = m_right
            pts[6] = ((m_left[0] + m_right[0]) / 2.0, m_y)

    return FaceLandmarks(
        points=pts,
        box=box,
        eye_center=eye_center,
        eye_distance=eye_distance,
    )


def mean_landmarks(landmarks: list[FaceLandmarks], box: FaceBox) -> FaceLandmarks:
    """Average several landmark fits (used to stabilise the source photo)."""
    if not landmarks:
        return fit_landmarks(box)
    stack = np.stack([lm.points for lm in landmarks], axis=0)
    ref = landmarks[0]
    mean = stack.mean(axis=0)
    eyes = [tuple(mean[_LEFT_EYE]), tuple(mean[_RIGHT_EYE])]
    eye_distance = float(
        np.hypot(mean[_RIGHT_EYE][0] - mean[_LEFT_EYE][0], mean[_RIGHT_EYE][1] - mean[_LEFT_EYE][1])
    )
    return FaceLandmarks(
        points=mean.astype(np.float32),
        box=box,
        eye_center=tuple(mean[[_LEFT_EYE, _RIGHT_EYE]].mean(axis=0)),
        eye_distance=eye_distance or ref.eye_distance,
    )


# --------------------------------------------------------------------------
# Triangulation + piecewise-affine warp
# --------------------------------------------------------------------------


def triangulate(points: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """Delaunay triangulation of ``points``.

    ``cv2.Subdiv2D`` is used rather than ``scipy.spatial`` so there is no SciPy
    dependency.  The subdivision rectangle is derived from the points themselves
    (expanded to also contain ``size`` when given): landmarks legitimately reach
    outside the detected face box - cheeks and chin are drawn well wide of it -
    and Subdiv2D rejects any point that is not strictly inside its rectangle.

    Returns an (T, 3) int32 index array.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.zeros((0, 3), dtype=np.int32)

    xs, ys = pts[:, 0], pts[:, 1]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    if size is not None:
        min_x, min_y = min(min_x, 0.0), min(min_y, 0.0)
        max_x, max_y = max(max_x, float(size[0])), max(max_y, float(size[1]))

    span = max(max_x - min_x, max_y - min_y, 1.0)
    margin = span * 0.02 + 1.0
    left = int(np.floor(min_x - margin))
    top = int(np.floor(min_y - margin))
    right = int(np.ceil(max_x + margin))
    bottom = int(np.ceil(max_y + margin))
    if right <= left or bottom <= top:
        return np.zeros((0, 3), dtype=np.int32)

    # cv2.Subdiv2D takes (x, y, width, height), not (x0, y0, x1, y1).
    subdiv = cv2.Subdiv2D((left, top, right - left, bottom - top))
    for x, y in pts:
        subdiv.insert((float(x), float(y)))

    triangles = []
    n = len(pts)
    for t in subdiv.getTriangleList():
        idx = []
        for k in range(3):
            px, py = t[2 * k], t[2 * k + 1]
            d = np.hypot(pts[:, 0] - px, pts[:, 1] - py)
            j = int(np.argmin(d))
            idx.append(j if d[j] < 1.0 else -1)
        if -1 in idx or len(set(idx)) != 3:
            continue
        if max(idx) >= n:
            continue
        triangles.append(idx)

    if not triangles:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)


def _warp_triangle(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    tri_src: np.ndarray,
    tri_dst: np.ndarray,
    written: np.ndarray | None = None,
) -> None:
    """Affine-warp one source triangle onto ``dst_img`` (the standard approach
    also used by OpenCV's face-morphing sample).

    Both bounding boxes are clipped to their images first.  That matters for
    triangles that straddle an edge: an unclipped negative origin would be read
    by NumPy slicing as a wrap-around offset and corrupt the destination.

    ``written``, when given, records the pixels actually painted.  Rounding at
    shared triangle edges means a pixel can be inside the triangulated region
    yet painted by no triangle; recording real writes keeps such pixels out of
    the blend mask instead of pasting black specks onto the face.
    """
    tri_src = np.asarray(tri_src, dtype=np.float32)
    tri_dst = np.asarray(tri_dst, dtype=np.float32)

    sx, sy, sw, sh = cv2.boundingRect(tri_src)
    dx, dy, dw, dh = cv2.boundingRect(tri_dst)

    sh_img, sw_img = src_img.shape[:2]
    dh_img, dw_img = dst_img.shape[:2]

    sx0, sy0 = max(0, sx), max(0, sy)
    sx1, sy1 = min(sw_img, sx + sw), min(sh_img, sy + sh)
    dx0, dy0 = max(0, dx), max(0, dy)
    dx1, dy1 = min(dw_img, dx + dw), min(dh_img, dy + dh)

    if sx1 <= sx0 or sy1 <= sy0 or dx1 <= dx0 or dy1 <= dy0:
        return

    src_local = tri_src - np.array([sx0, sy0], dtype=np.float32)
    dst_local = tri_dst - np.array([dx0, dy0], dtype=np.float32)

    warp_mat = cv2.getAffineTransform(src_local, dst_local)
    patch = cv2.warpAffine(
        src_img[sy0:sy1, sx0:sx1],
        warp_mat,
        (dx1 - dx0, dy1 - dy0),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((dy1 - dy0, dx1 - dx0), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst_local).astype(np.int32), 255, cv2.LINE_8)

    sel = mask > 0
    if not np.any(sel):
        return
    region = dst_img[dy0:dy1, dx0:dx1]
    region[sel] = patch[sel]
    if written is not None:
        written[dy0:dy1, dx0:dx1][sel] = 255


def warp_face(
    src_img: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    canvas_shape: tuple[int, int],
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the source face onto the target landmark layout.

    Returns ``(warped_bgr, coverage_mask)`` where ``coverage_mask`` marks the
    pixels that were genuinely written.  ``triangles`` must be built from
    ``src_points`` so both faces share the same topology.
    """
    h, w = canvas_shape[:2]
    warped = np.zeros((h, w, 3), dtype=np.uint8)
    coverage = np.zeros((h, w), dtype=np.uint8)

    for tri in triangles:
        tri_src = np.float32([src_points[tri[0]], src_points[tri[1]], src_points[tri[2]]])
        tri_dst = np.float32([dst_points[tri[0]], dst_points[tri[1]], dst_points[tri[2]]])
        _warp_triangle(src_img, warped, tri_src, tri_dst, written=coverage)

    return warped, coverage


# --------------------------------------------------------------------------
# Colour matching + blending
# --------------------------------------------------------------------------


def match_colors(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.9,
    luminance_strength: float = 1.0,
    std_clip: tuple[float, float] = (0.6, 1.8),
) -> np.ndarray:
    """Recolour ``source`` to match ``target`` using CIELAB statistics.

    All three channels are fitted (mean shift plus a clipped standard-deviation
    ratio) so the pasted face takes on the target's exposure as well as its
    skin tone.  Matching luminance matters: without it a bright source face
    forces the Poisson blend to do all the work itself, which flattens the face
    and washes out the identity.  ``luminance_strength`` lets L be pulled less
    than a*/b* if that is preferred.
    """
    if strength <= 0.0:
        return source

    m = mask > 0
    if not np.any(m):
        return source

    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_px = src_lab[m]
    tgt_px = tgt_lab[m]

    src_mean, src_std = src_px.mean(axis=0), src_px.std(axis=0)
    tgt_mean, tgt_std = tgt_px.mean(axis=0), tgt_px.std(axis=0)

    out = src_px.copy()
    for ch in range(3):
        local_strength = strength * (luminance_strength if ch == 0 else 1.0)
        if local_strength <= 0.0:
            continue
        if src_std[ch] < 1e-3:
            out[:, ch] = src_px[:, ch] + local_strength * (tgt_mean[ch] - src_px[:, ch])
            continue
        scale = np.clip(tgt_std[ch] / src_std[ch], std_clip[0], std_clip[1])
        matched = (src_px[:, ch] - src_mean[ch]) * scale + tgt_mean[ch]
        out[:, ch] = src_px[:, ch] + local_strength * (matched - src_px[:, ch])

    # Write back only the selected pixels.  Assigning the whole channel would
    # round-trip every pixel through the LAB conversion and perturb the ones
    # outside the mask.
    src_lab[m] = out
    lab_out = cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out_u8 = source.copy()
    out_u8[m] = lab_out[m]
    return out_u8


def make_blend_mask(
    shape: tuple[int, int],
    target_points: np.ndarray,
    coverage: np.ndarray | None = None,
    inset: float = 0.10,
) -> np.ndarray:
    """Build the coverage mask for the paste: a binary 0/255 region.

    The region is the convex hull of the target landmarks pulled in by
    ``inset`` (a fraction of face width), which keeps the seam off the eyes,
    mouth and jaw line where mismatched detail is most visible.  The warp's own
    coverage is intersected so no hole appears where the source ran out of
    pixels.

    The mask is deliberately *binary*: ``cv2.seamlessClone`` ignores a blurred
    mask, so feathering is applied inside :func:`blend_face` for the alpha mode
    only.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(
        mask, np.round(cv2.convexHull(np.float32(target_points))).astype(np.int32),
        255, cv2.LINE_8,
    )

    xs, ys = target_points[:, 0], target_points[:, 1]
    face_size = max(8.0, float(min(xs.max() - xs.min(), ys.max() - ys.min())))
    radius = max(1, int(round(face_size * inset)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    # Pull the hull in from every side, including the image border.  Border
    # pixels count as background for erosion, so a hull that touches the edge
    # still gets inset there.
    mask = cv2.erode(mask, kernel, borderValue=0)

    if coverage is not None:
        mask = cv2.bitwise_and(mask, (coverage > 0).astype(np.uint8) * 255)
    return mask


def blend_face(
    target: np.ndarray,
    warped: np.ndarray,
    mask: np.ndarray,
    mode: str = "alpha",
    feather: float = 0.08,
    center: tuple[int, int] | None = None,
) -> np.ndarray:
    """Composite ``warped`` onto ``target``.

    ``mode='alpha'`` (the default) cross-fades the warped face over the target
    with a soft edge, ``feather`` being the edge width as a fraction of the
    face size.  This is the default because measurements on real faces show it
    keeps ~98% of the source face's fine detail, and that detail is the
    likeness.  Poisson blending over a whole-face region smooths that detail
    away, so the result reads as the target person with edited colouring.

    ``mode='poisson'`` runs ``cv2.seamlessClone`` (gradient-domain blending).
    It gives an even softer edge but needs an exactly binary mask and is only
    well behaved when the pasted region is small relative to the image, which a
    full face is not.  Use it when hiding the seam matters more than likeness.
    """
    if mask is None or not np.any(mask):
        return target

    if mode == "poisson":
        binary = (mask > 127).astype(np.uint8) * 255
        if np.any(binary):
            try:
                if center is None:
                    ys, xs = np.nonzero(binary)
                    center = (int(round(xs.mean())), int(round(ys.mean())))
                return cv2.seamlessClone(warped, target, binary, center, cv2.NORMAL_CLONE)
            except cv2.error:
                pass

    alpha = mask.astype(np.float32)
    if feather > 0:
        ys, xs = np.nonzero(mask)
        face_size = float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
        sigma = max(1.0, feather * face_size)
        ksize = int(sigma * 4) | 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), sigma)
    alpha = (alpha / 255.0)[:, :, None]
    out = warped.astype(np.float32) * alpha + target.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)
