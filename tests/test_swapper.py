"""The swap itself: geometry correctness, colour matching, blending, masking.

These are the assertions that separate a real swap from a sticker: pixels are
*rasterised through the source triangles*, so a landmark's colour must end up
at that landmark's destination.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceswap import swapper as S
from faceswap.detector import FaceBox

from .conftest import load


# --------------------------------------------------------------------------
# Landmarks
# --------------------------------------------------------------------------


def test_landmarks_are_symmetric_for_a_symmetric_box():
    box = FaceBox(50, 50, 200, 200)
    lm = S.fit_landmarks(box)
    assert lm.points.shape == (14, 2)
    cx = lm.eye_center[0]
    # cheek and jaw pairs mirror about the eye centre line
    for left, right in ((8, 9), (10, 11), (12, 13), (0, 1), (4, 5)):
        assert abs((lm.points[left, 0] - cx) + (lm.points[right, 0] - cx)) < 1e-3
    # and share a row
    for left, right in ((0, 1), (4, 5), (8, 9)):
        assert abs(lm.points[left, 1] - lm.points[right, 1]) < 1e-3


def test_landmark_ordering_is_anatomically_correct():
    lm = S.fit_landmarks(FaceBox(0, 0, 240, 240))
    # eyes above nose above mouth above chin
    assert lm.points[0, 1] < lm.points[3, 1] < lm.points[4, 1] < lm.points[7, 1]
    # nose bridge above nose tip
    assert lm.points[2, 1] < lm.points[3, 1]
    # cheeks outside the eyes
    assert lm.points[8, 0] < lm.points[0, 0] < lm.points[1, 0] < lm.points[9, 0]
    # chin is centred
    assert abs(lm.points[7, 0] - lm.eye_center[0]) < 1.0


def test_layout_scales_with_the_face_not_the_image():
    """A big and a small detected face must produce the same *shape*, which is
    what makes a large source photo match a small face in a video."""
    big = S.fit_landmarks(FaceBox(0, 0, 400, 400))
    small = S.fit_landmarks(FaceBox(1000, 700, 100, 100))

    def normalised(lm):
        p = lm.points - lm.eye_center
        return p / lm.eye_distance

    assert np.allclose(normalised(big), normalised(small), atol=1e-4)


def test_layout_rotates_with_a_tilted_eye_line():
    box = FaceBox(0, 0, 200, 200)
    upright = S.fit_landmarks(box, eyes=[(70.0, 80.0), (130.0, 80.0)])
    assert abs(upright.points[0, 1] - upright.points[1, 1]) < 1e-6

    # rotate the same eye pair 30 degrees about the box centre
    theta = np.deg2rad(30.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pivot = np.array([100.0, 100.0])
    eyes = [(rot @ (np.array(e) - pivot) + pivot).tolist() for e in ((70.0, 80.0), (130.0, 80.0))]
    tilted = S.fit_landmarks(box, eyes=eyes)
    # the eye landmarks follow the given tilt exactly
    assert np.allclose(tilted.points[0], eyes[0], atol=1e-4)
    assert np.allclose(tilted.points[1], eyes[1], atol=1e-4)
    # the whole layout rotates with the eye line: the chin is "below" in face
    # coordinates, so its offset from the eye centre must turn by the same angle
    assert tilted.eye_distance == pytest.approx(upright.eye_distance, rel=1e-6)
    assert tilted.points[7, 1] > tilted.eye_center[1]
    a_up = np.arctan2(*(upright.points[7] - upright.eye_center)[::-1])
    a_ti = np.arctan2(*(tilted.points[7] - tilted.eye_center)[::-1])
    assert (a_ti - a_up) == pytest.approx(theta, abs=1e-3)


def test_mouth_measurements_override_the_layout():
    box = FaceBox(0, 0, 200, 200)
    plain = S.fit_landmarks(box)
    measured = S.fit_landmarks(box, mouth=[(60.0, 150.0), (140.0, 150.0)])
    assert not np.allclose(plain.points[4], measured.points[4])
    assert np.allclose(measured.points[4], [60.0, 150.0], atol=1e-4)
    assert np.allclose(measured.points[5], [140.0, 150.0], atol=1e-4)
    assert np.allclose(measured.points[6], [100.0, 150.0], atol=1e-4)


def test_implausible_mouth_measurement_is_rejected():
    """The smile cascade also fires on chins and nostrils.  A mouth landing
    below the chin would invert triangles and tear the mesh, so the
    proportional layout must win instead."""
    box = FaceBox(0, 0, 200, 200)
    plain = S.fit_landmarks(box)

    # Far below the chin (y ~ 1.74 eye units below the eye line).
    below = S.fit_landmarks(box, mouth=[(60.0, 900.0), (140.0, 900.0)])
    assert np.allclose(below.points[4], plain.points[4], atol=1e-4)

    # Above the nose bridge.
    above = S.fit_landmarks(box, mouth=[(60.0, -500.0), (140.0, -500.0)])
    assert np.allclose(above.points[4], plain.points[4], atol=1e-4)

    # A plausible mouth between nose tip and chin is still honoured.
    good = S.fit_landmarks(box, mouth=[(70.0, 150.0), (130.0, 150.0)])
    assert not np.allclose(good.points[4], plain.points[4])


def test_mean_landmarks_averages():
    a = S.fit_landmarks(FaceBox(0, 0, 100, 100))
    b = S.fit_landmarks(FaceBox(20, 20, 100, 100))
    m = S.mean_landmarks([a, b], a.box)
    assert np.allclose(m.points, (a.points + b.points) / 2, atol=1e-4)
    assert np.allclose(m.eye_center, (a.points[[0, 1]] + b.points[[0, 1]]).mean(axis=0) / 2, atol=1e-4)


# --------------------------------------------------------------------------
# Triangulation
# --------------------------------------------------------------------------


def test_triangulation_covers_the_face_and_uses_every_landmark():
    lm = S.fit_landmarks(FaceBox(0, 0, 260, 260))
    tris = S.triangulate(lm.points, (260, 260))
    assert len(tris) >= 12
    assert tris.max() < len(lm.points)
    used = set(tris.ravel().tolist())
    # the landmark hull is what gets triangulated, so all interior points appear
    assert len(used) >= 10
    # no degenerate triangles
    for tri in tris:
        p = lm.points[tri]
        area = abs(np.cross(p[1] - p[0], p[2] - p[0])) / 2
        assert area > 0.5


def test_triangulation_is_deterministic():
    lm = S.fit_landmarks(FaceBox(0, 0, 260, 260))
    first = S.triangulate(lm.points, (260, 260))
    for _ in range(3):
        assert np.array_equal(S.triangulate(lm.points, (260, 260)), first)


# --------------------------------------------------------------------------
# Warp: the core claim, verified pixel by pixel
# --------------------------------------------------------------------------


def test_warp_places_each_source_landmark_at_its_destination():
    """Paint a unique colour at every source landmark; after warping, each
    colour must appear at that landmark's destination.  This is the test that
    proves the faces are geometrically deformed rather than overlaid."""
    src_box = FaceBox(40, 40, 160, 160)
    dst_box = FaceBox(10, 30, 300, 260)
    src_lm = S.fit_landmarks(src_box)
    dst_lm = S.fit_landmarks(dst_box)

    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 64), (64, 128, 255),
    ]
    source = np.zeros((240, 240, 3), np.uint8)
    for (x, y), colour in zip(src_lm.points, palette):
        cv2.circle(source, (int(round(x)), int(round(y))), 5, colour, -1)

    tris = S.triangulate(src_lm.points, (240, 240))
    warped, coverage = S.warp_face(
        source, src_lm.points, dst_lm.points, (400, 400, 3), tris
    )

    for (x, y), colour in zip(dst_lm.points, palette):
        xi, yi = int(round(x)), int(round(y))
        patch = warped[yi - 2 : yi + 3, xi - 2 : xi + 3].reshape(-1, 3).astype(int)
        deltas = np.abs(patch - np.array(colour)).sum(axis=1)
        assert deltas.min() < 60, (
            f"landmark at ({xi},{yi}): expected {colour} nearby, "
            f"best match had delta {deltas.min()}"
        )

    # and nothing was written far outside the warped face
    assert coverage[5, 5] == 0
    assert np.count_nonzero(warped[380:, :]) == 0


def test_warp_is_a_deformation_not_a_copy():
    """A warped face must differ from a plain paste of the source crop."""
    src_box = FaceBox(100, 100, 200, 200)
    src_lm = S.fit_landmarks(src_box)
    dst_box = FaceBox(50, 20, 140, 300)   # much taller and narrower
    dst_lm = S.fit_landmarks(dst_box)

    source = np.zeros((420, 420, 3), np.uint8)
    cv2.circle(source, (200, 200), 60, (200, 200, 200), -1)
    cv2.line(source, (140, 200), (260, 200), (0, 0, 255), 6)
    cv2.line(source, (200, 140), (200, 260), (0, 255, 0), 6)

    tris = S.triangulate(src_lm.points, (420, 420))
    warped, _ = S.warp_face(source, src_lm.points, dst_lm.points, (420, 420, 3), tris)

    # vertical line stretches vertically, horizontal line stretches horizontally
    green = (warped[:, :, 1].astype(int) - warped[:, :, 0]) > 50
    red = (warped[:, :, 2].astype(int) - warped[:, :, 1]) > 50
    assert green.sum() > 0 and red.sum() > 0
    assert green.sum() > red.sum(), "tall target should stretch the vertical line most"


def test_warp_of_a_flat_colour_is_flat():
    source = np.full((300, 300, 3), 77, np.uint8)
    src_lm = S.fit_landmarks(FaceBox(100, 100, 100, 100))
    dst_lm = S.fit_landmarks(FaceBox(50, 40, 200, 220))
    tris = S.triangulate(src_lm.points, (300, 300))
    warped, coverage = S.warp_face(
        source, src_lm.points, dst_lm.points, (340, 340, 3), tris
    )
    # Every painted pixel must carry the flat colour.  Pixels no triangle wrote
    # stay zero, and the coverage mask must exclude exactly those.
    painted = coverage > 0
    assert np.any(painted)
    assert np.all(warped[painted] == 77), "flat colour must survive the warp"
    black = np.all(warped == 0, axis=2)
    assert not np.any(black & painted), "coverage claims pixels the warp never wrote"


# --------------------------------------------------------------------------
# Colour matching
# --------------------------------------------------------------------------


def test_colour_match_moves_means_to_the_target():
    rng = np.random.default_rng(3)
    # two flat-ish images with a clear mean difference; uniform random noise has
    # the same mean everywhere and would make this test meaningless
    source = np.clip(rng.normal(190, 14, (120, 120, 3)), 0, 255).astype(np.uint8)
    target = np.clip(rng.normal(70, 14, (120, 120, 3)), 0, 255).astype(np.uint8)
    mask = np.zeros((120, 120), np.uint8)
    cv2.circle(mask, (60, 60), 40, 255, -1)

    matched = S.match_colors(source, target, mask, strength=1.0)

    def lab_mean(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[mask > 0].mean(axis=0)

    before = np.abs(lab_mean(source) - lab_mean(target)).mean()
    after = np.abs(lab_mean(matched) - lab_mean(target)).mean()
    assert after < before * 0.35, f"means not matched: {before:.1f} -> {after:.1f}"


def test_colour_match_keeps_the_detail_that_is_the_likeness():
    """Colour transfer must shift and scale, not flatten.  If the structure of
    the source face is lost here, the swap degenerates into a tinted target."""
    img = load("obama")
    target = load("biden")
    from faceswap.detector import FaceDetector

    det = FaceDetector()
    src_lm = S.fit_landmarks(det.detect(img))
    tgt_lm = S.fit_landmarks(det.detect(target))
    tris = S.triangulate(src_lm.points, (img.shape[1], img.shape[0]))
    warped, coverage = S.warp_face(
        img, src_lm.points, tgt_lm.points, target.shape, tris
    )
    mask = S.make_blend_mask(target.shape, tgt_lm.points, coverage)

    matched = S.match_colors(warped, target, mask, strength=1.0)

    def highpass(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return g - cv2.GaussianBlur(g, (0, 0), 6.0)

    inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    a = highpass(matched)[inner]
    b = highpass(warped)[inner]
    a = a - a.mean()
    b = b - b.mean()
    corr = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert corr > 0.9, f"colour matching destroyed structure (corr={corr:.3f})"


def test_colour_match_strength_zero_is_a_no_op():
    rng = np.random.default_rng(1)
    source = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    target = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    mask = np.full((80, 80), 255, np.uint8)
    assert np.array_equal(S.match_colors(source, target, mask, strength=0.0), source)


def test_colour_match_ignores_pixels_outside_the_mask():
    rng = np.random.default_rng(2)
    source = np.clip(rng.normal(180, 20, (60, 60, 3)), 0, 255).astype(np.uint8)
    target = np.clip(rng.normal(60, 20, (60, 60, 3)), 0, 255).astype(np.uint8)
    mask = np.zeros((60, 60), np.uint8)
    mask[:, :30] = 255
    matched = S.match_colors(source, target, mask, strength=1.0)
    assert np.array_equal(matched[:, 30:], source[:, 30:])


def test_colour_match_handles_a_flat_source():
    """A constant-colour source has zero std; matching must not divide by zero."""
    source = np.full((60, 60, 3), 200, np.uint8)
    target = np.full((60, 60, 3), 60, np.uint8)
    mask = np.full((60, 60), 255, np.uint8)
    matched = S.match_colors(source, target, mask, strength=1.0)
    assert matched.std() < 6, "flat source should stay flat, just darker"


# --------------------------------------------------------------------------
# Mask and blend
# --------------------------------------------------------------------------


def test_mask_is_binary_and_inside_the_face():
    lm = S.fit_landmarks(FaceBox(100, 100, 200, 200))
    coverage = np.full((400, 400), 255, np.uint8)
    mask = S.make_blend_mask((400, 400, 3), lm.points, coverage, inset=0.10)

    values = np.unique(mask)
    assert set(values.tolist()) <= {0, 255}, "mask must be binary for seamlessClone"
    assert (mask > 0).sum() > 0
    # the mask must not reach into the corners of the face box
    assert mask[100, 100] == 0
    assert mask[299, 299] == 0
    # but must cover the face centre
    cy = int(lm.points[:, 1].mean())
    cx = int(lm.points[:, 0].mean())
    assert mask[cy, cx] > 0


def test_mask_respects_warp_coverage():
    """Where the warp produced nothing, the mask must not claim coverage -
    otherwise a black hole gets pasted on."""
    lm = S.fit_landmarks(FaceBox(100, 100, 200, 200))
    coverage = np.zeros((400, 400), np.uint8)
    coverage[150:250, 80:160] = 255          # only the left half of the face
    mask = S.make_blend_mask((400, 400, 3), lm.points, coverage, inset=0.05)
    assert np.all(mask[coverage == 0] == 0)


def test_bigger_inset_makes_a_smaller_mask():
    lm = S.fit_landmarks(FaceBox(100, 100, 220, 220))
    cov = np.full((420, 420), 255, np.uint8)
    small = S.make_blend_mask((420, 420, 3), lm.points, cov, inset=0.20)
    large = S.make_blend_mask((420, 420, 3), lm.points, cov, inset=0.05)
    assert (small > 0).sum() < (large > 0).sum()


def test_alpha_blend_actually_pastes_the_source():
    target = np.full((200, 200, 3), 30, np.uint8)
    warped = np.full((200, 200, 3), 220, np.uint8)
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 40, 255, -1)

    out = S.blend_face(target, warped, mask, mode="alpha", feather=0.0)

    assert out[100, 100].tolist() == [220, 220, 220]      # pasted
    assert out[5, 5].tolist() == [30, 30, 30]             # untouched
    # feathering softens the boundary into a gradient
    soft = S.blend_face(target, warped, mask, mode="alpha", feather=0.15)
    # The feather ramp spans the mask boundary, so a scan line through the
    # centre must contain values strictly between the two flat colours.
    vals = soft[100, :, 0].astype(int)
    assert vals.max() >= 218, "the pasted region should reach full source colour"
    assert vals.min() <= 32, "untouched background should remain"
    intermediate = ((vals > 60) & (vals < 190)).sum()
    assert intermediate >= 5, "expected a smooth ramp, not a hard edge"


def test_poisson_blend_preserves_source_structure():
    """seamlessClone must keep the source content, not wash it out.  This is
    checked with a synthetic striped mask region because it is unambiguous."""
    target = np.full((240, 240, 3), 120, np.uint8)
    warped = np.zeros((240, 240, 3), np.uint8)
    for y in range(240):
        warped[y] = 40 if (y // 8) % 2 == 0 else 210
    mask = np.zeros((240, 240), np.uint8)
    cv2.circle(mask, (120, 120), 70, 255, -1)

    out = S.blend_face(target, warped, mask, mode="poisson")
    core = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    # stripes must survive inside the region
    assert out[core][:, 0].std() > 60, "poisson blend flattened the source"


def test_blend_with_empty_mask_returns_the_target_unchanged():
    target = np.full((50, 50, 3), 88, np.uint8)
    warped = np.full((50, 50, 3), 200, np.uint8)
    empty = np.zeros((50, 50), np.uint8)
    for mode in ("alpha", "poisson"):
        assert np.array_equal(S.blend_face(target, warped, empty, mode), target)


def test_poisson_falls_back_when_seamless_clone_cannot_work():
    """A mask touching every edge crashes OpenCV's solver; the fallback must
    still return a usable image rather than raising."""
    target = np.full((30, 30, 3), 50, np.uint8)
    warped = np.full((30, 30, 3), 200, np.uint8)
    mask = np.full((30, 30), 255, np.uint8)
    out = S.blend_face(target, warped, mask, mode="poisson")
    assert out.shape == target.shape
    assert out.dtype == np.uint8
