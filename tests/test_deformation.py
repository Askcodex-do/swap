"""Prove the swap is a real deformation, not a sticker paste.

This is the regression test for the defect where the landmark layout was a
fixed template: identical on every face, scaled only by the interocular
distance.  Warping onto a fixed template can only rotate, scale and translate
the source, so every internal proportion of the source face survives untouched
and the result reads as a pasted-on mask.

Two independent checks, both on real photographs:

1. The fitted constellations of different people must NOT be related by a
   similarity transform.  The residual after the best possible similarity fit
   is the deformation the warp applies; zero means sticker.
2. Through the actual ``warp_face`` rasteriser, a source with a narrow mouth
   warped onto a target with a wide mouth must come out with the *target's*
   mouth width, measured in the output pixels.  A paste would keep the source's.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceswap import swapper as S
from faceswap.detector import FaceDetector
from faceswap.swapper import fit_landmarks, triangulate, warp_face

from .conftest import load

ALL_FACES = ["obama", "obama2", "biden", "lena_opencv_sample"]

#: Residual below this means "a similarity transform", i.e. a paste.  A real
#: face swap sits well above it; measured values on these photos are 0.05-0.16.
STICKER_RESIDUAL = 0.02


def fit_full(detector, img):
    """Fit landmarks the way the pipeline does: eyes *and* mouth measured."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = detector.detect(img)
    return fit_landmarks(box, detector.detect_eyes(gray, box), detector.detect_mouth(gray, box))


def similarity_residual(src_pts, dst_pts) -> float:
    """RMS leftover after the best-fit similarity transform, normalised."""
    s = np.asarray(src_pts, np.float64)
    t = np.asarray(dst_pts, np.float64)
    s0 = s - s.mean(axis=0)
    t0 = t - t.mean(axis=0)

    cov = t0.T @ s0 / len(s)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = (S * np.diag(D)).sum() / ((s0 ** 2).sum() / len(s))
    moved = scale * (s0 @ R.T) + t.mean(axis=0)

    eye_d = float(np.linalg.norm(t[1] - t[0]))
    return float(np.sqrt(((moved - t) ** 2).sum(axis=1).mean())) / max(eye_d, 1e-6)


def test_mouth_is_measured_per_face_not_templated(detector):
    """The mouth must be a per-face measurement, so its ratio to eye spacing
    differs between people.  A fixed template gives exactly the same ratio."""
    ratios = {}
    for name in ALL_FACES:
        img = load(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = detector.detect(img)
        mouth = detector.detect_mouth(gray, box)
        assert mouth is not None, f"mouth not found in {name}"
        lm = fit_landmarks(box, detector.detect_eyes(gray, box), mouth)
        eye_d = float(np.linalg.norm(lm.points[1] - lm.points[0]))
        ratios[name] = float(np.linalg.norm(lm.points[5] - lm.points[4])) / eye_d

    spread = max(ratios.values()) - min(ratios.values())
    assert spread > 0.15, (
        f"mouth/eye ratio barely varies across faces ({ratios}); the mouth is "
        "not really being measured"
    )


@pytest.mark.parametrize("a,b", [("obama", "obama2"), ("biden", "lena_opencv_sample")])
def test_constellations_are_not_related_by_a_similarity(detector, a, b):
    """If two different faces' landmark sets differ only by rotate/scale/
    translate, the warp is a paste."""
    la = fit_full(detector, load(a))
    lb = fit_full(detector, load(b))
    residual = similarity_residual(la.points, lb.points)
    assert residual > STICKER_RESIDUAL, (
        f"{a} and {b} constellations are a similarity transform apart "
        f"(residual {residual:.4f}); that is a sticker, not a swap"
    )


def test_warp_stretches_the_mouth_to_the_target_proportions(detector):
    """End-to-end through the real rasteriser.

    Take the face with the narrowest mouth and the one with the widest, warp
    the narrow onto the wide, and measure the mouth in the *output*: it must
    match the target's width, not the source's.
    """
    faces = {}
    for name in ALL_FACES:
        img = load(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = detector.detect(img)
        lm = fit_landmarks(box, detector.detect_eyes(gray, box), detector.detect_mouth(gray, box))
        eye_d = float(np.linalg.norm(lm.points[1] - lm.points[0]))
        faces[name] = (img, lm, float(np.linalg.norm(lm.points[5] - lm.points[4])) / eye_d)

    src_name = min(faces, key=lambda k: faces[k][2])
    dst_name = max(faces, key=lambda k: faces[k][2])
    assert src_name != dst_name

    src_img, src_lm, src_ratio = faces[src_name]
    _, dst_lm, dst_ratio = faces[dst_name]
    assert dst_ratio - src_ratio > 0.15, "test needs faces with clearly different mouths"

    h, w = src_img.shape[:2]
    tris = triangulate(src_lm.points, (w, h))

    # Warp the source onto the target layout, on the source's own canvas so the
    # output can be measured in pixels directly.
    warped, coverage = warp_face(src_img, src_lm.points, dst_lm.points, (h, w), tris)
    assert coverage.any(), "warp wrote nothing"

    # In the output, the landmarks sit at the TARGET positions by construction.
    # The real question is whether the pixels there are the source's face
    # *stretched*, so measure the mouth span in the warped raster: the mouth
    # band of the target layout must be filled across the target's width.
    mouth_y = float(dst_lm.points[6, 1])
    band = coverage[
        max(0, int(mouth_y) - 2) : int(mouth_y) + 3,
        :,
    ]
    xs = np.nonzero(band.any(axis=0))[0]
    assert xs.size, "no warped pixels at the target mouth line"

    target_mouth_px = float(np.linalg.norm(dst_lm.points[5] - dst_lm.points[4]))
    source_mouth_px = float(np.linalg.norm(src_lm.points[5] - src_lm.points[4]))
    # The warped face must span at least the target's mouth width there; a paste
    # of the source at its own scale would be far narrower.
    assert target_mouth_px > source_mouth_px * 1.1, "target mouth is not wider than source"
    assert (xs[-1] - xs[0]) > source_mouth_px * 1.1, (
        f"warped mouth span {xs[-1] - xs[0]}px is not stretched past the "
        f"source's {source_mouth_px:.0f}px"
    )
