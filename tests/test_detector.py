"""Detection and landmark placement, checked against real photographs."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceswap.detector import FaceBox, FaceDetector, load_cascade
from faceswap.swapper import fit_landmarks

from .conftest import load

ALL_FACES = ["obama", "obama2", "biden", "lena_opencv_sample"]

#: A Haar frontal face box is this fraction of the interocular distance wide,
#: and its eye line sits this fraction of the box height below the top.
#: Measured from these photos; see tests/test_landmarks.py for the tolerance.
EXPECTED_INTEROCULAR_RATIO = 0.352
EXPECTED_EYE_LINE = 0.388


def test_bundled_cascades_load_without_download():
    """The cascades must come from the installed package, not the network."""
    for name in (
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_eye.xml",
        "haarcascade_profileface.xml",
        "haarcascade_smile.xml",
    ):
        cascade = load_cascade(name)
        assert not cascade.empty()


@pytest.mark.parametrize("name", ALL_FACES)
def test_finds_exactly_the_expected_face(name, detector):
    img = load(name)
    box = detector.detect(img)
    assert box is not None, f"no face found in {name}"
    # a real face box is a sane size relative to the image
    assert 0.05 < box.w / img.shape[1] < 0.95
    assert 0.05 < box.h / img.shape[0] < 1.0
    assert box.x >= 0 and box.y >= 0
    assert box.x + box.w <= img.shape[1]
    assert box.y + box.h <= img.shape[0]


@pytest.mark.parametrize("name", ALL_FACES)
def test_eye_spacing_is_consistent_across_photos(name, detector):
    """Eye spacing must scale with the box, or a source photo will not line up
    with a face at a different distance/size."""
    img = load(name)
    box = detector.detect(img)
    eyes = detector.detect_eyes(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), box)
    assert len(eyes) == 2, f"two eyes not found in {name}"
    (lx, ly), (rx, ry) = eyes
    ratio = float(np.hypot(rx - lx, ry - ly)) / box.w
    assert 0.30 < ratio < 0.40, f"{name}: interocular/box_w = {ratio:.3f}"
    eye_line = ((ly + ry) / 2 - box.y) / box.h
    assert 0.30 < eye_line < 0.46, f"{name}: eye line at {eye_line:.3f} of box height"


@pytest.mark.parametrize("name", ALL_FACES)
def test_landmark_layout_lands_on_the_real_face(name, detector):
    """The fitted chin and cheeks must fall inside the actual face, and the
    eye landmarks must sit on the measured eyes."""
    img = load(name)
    box = detector.detect(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = detector.detect_eyes(gray, box)
    lm = fit_landmarks(box, eyes)

    # eye landmarks coincide with the measured eye centres
    for idx, (ex, ey) in zip((0, 1), sorted(eyes, key=lambda p: p[0])):
        assert np.hypot(*(lm.points[idx] - np.array([ex, ey]))) < 1.0

    # chin sits at roughly the bottom of the detected box
    chin_y = float(lm.points[7, 1])
    assert abs(chin_y - (box.y + box.h)) < 0.12 * box.h, f"{name}: chin off box bottom"

    # the cheek landmarks straddle the face centre
    assert lm.points[8, 0] < lm.points[0, 0] < lm.points[9, 0]
    # mouth sits between the nose and the chin
    assert lm.points[3, 1] < lm.points[6, 1] < lm.points[7, 1]
    # face landmarks are roughly symmetric about the eye centre
    left = lm.points[8, 0] - lm.eye_center[0]
    right = lm.points[9, 0] - lm.eye_center[0]
    assert abs(left + right) < 0.12 * box.w, f"{name}: layout is not symmetric"


def test_no_face_returns_none(detector):
    blank = np.full((200, 200, 3), 120, np.uint8)
    assert detector.detect(blank) is None
    assert detector.detect(np.zeros((0, 0, 3), np.uint8)) is None


def test_negative_images_still_return_no_face(detector):
    """The rotated retry must not invent faces on images that have none.

    It only runs when the upright cascade fails, which is exactly the situation
    on a featureless image, so this is the case where a sloppy fallback would
    start producing false positives.
    """
    rng = np.random.default_rng(0)
    ramp = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (240, 1))
    negatives = {
        "flat grey": np.full((240, 320), 120, np.uint8),
        "black": np.zeros((240, 320), np.uint8),
        "uniform noise": rng.integers(0, 256, (240, 320), dtype=np.uint8),
        "checkerboard": (np.indices((240, 320)).sum(axis=0) % 2 * 255).astype(np.uint8),
        "gradient": ramp,
    }
    for name, gray in negatives.items():
        assert detector.detect(gray) is None, f"false face found in {name}"


@pytest.mark.parametrize("angle", [-25.0, 25.0])
def test_tilted_head_is_recovered_when_upright_detection_fails(detector, angle):
    """A head roll of about 25 degrees defeats the frontal cascade.

    Without a retry those frames get no swap at all - a clip that opens with the
    head turned keeps the original face for its first seconds.  The detector must
    recover the face; the assertion that the cascade alone finds nothing is what
    keeps this test honest about why the fallback exists.
    """
    from faceswap.detector import load_cascade

    img = load("obama")
    box = detector.detect(img)
    pad = int(box.w * 0.5)
    y0, x0 = max(0, box.y - pad), max(0, box.x - pad)
    crop = img[y0 : box.y + box.h + pad, x0 : box.x + box.w + pad]
    side = min(crop.shape[:2])
    face = cv2.resize(crop[:side, :side], (90, 90), interpolation=cv2.INTER_AREA)

    h, w = 240, 320
    ramp = np.linspace(0, 200, w, dtype=np.float32)
    frame = cv2.cvtColor(np.tile(ramp, (h, 1)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    matrix = cv2.getRotationMatrix2D((45.0, 45.0), angle, 1.0)
    rolled = cv2.warpAffine(face, matrix, (90, 90), borderMode=cv2.BORDER_REPLICATE)
    px, py = 170, 85
    frame[py : py + 90, px : px + 90] = rolled

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    upright = detector._frontal.detectMultiScale(gray, 1.1, 5, minSize=(16, 16))
    assert len(upright) == 0, (
        "this test only means something if the upright cascade misses the "
        "rolled head; the cascade got better, so pick a larger angle"
    )

    found = detector.detect(frame)
    assert found is not None, f"tilted face at {angle:+.0f} degrees was not recovered"

    # It must be the planted face, not something else in the frame.
    ax0, ay0, ax1, ay1 = px, py, px + 90, py + 90
    bx0, by0 = found.x, found.y
    bx1, by1 = found.x + found.w, found.y + found.h
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    assert inter > 0, f"recovered box {found} does not overlap the planted face"


def test_largest_face_wins(detector, face_obama, obama_box):
    """With the same face pasted twice at different sizes, the big one wins."""
    pad = int(obama_box.w * 0.5)
    y0 = max(0, obama_box.y - pad)
    x0 = max(0, obama_box.x - pad)
    crop = face_obama[
        y0 : obama_box.y + obama_box.h + pad,
        x0 : obama_box.x + obama_box.w + pad,
    ]
    small = cv2.resize(crop, (180, 180), interpolation=cv2.INTER_AREA)
    big = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_AREA)

    canvas = np.full((1000, 1400, 3), 100, np.uint8)
    canvas[60 : 60 + small.shape[0], 60 : 60 + small.shape[1]] = small
    canvas[400 : 400 + big.shape[0], 800 : 800 + big.shape[1]] = big

    box = detector.detect(canvas)
    assert box is not None
    # the detected box must be centred on the large paste, not the small one
    assert box.w > small.shape[0], "picked the small face"
    cx, cy = box.center
    assert abs(cx - (800 + big.shape[1] / 2)) < 0.25 * big.shape[1], "not on the large face"
    assert abs(cy - (400 + big.shape[0] / 2)) < 0.25 * big.shape[0], "not on the large face"


def test_region_search_agrees_with_full_frame(detector, face_biden, biden_box):
    """Searching near a known box must find the same face as a full scan."""
    full = detector.detect(face_biden)
    near = detector.detect(face_biden, region=biden_box, pad=0.4)
    assert near is not None
    assert abs(near.w - full.w) < 0.1 * full.w
    assert abs(near.x - full.x) < 0.1 * full.w


def test_region_search_rejects_a_region_with_no_face(detector, face_biden):
    """If the search window excludes the face, nothing is reported."""
    corner = FaceBox(0, 0, 120, 120)
    assert detector.detect(face_biden, region=corner, pad=0.0) is None


def test_detection_is_deterministic(detector, face_lena):
    first = detector.detect(face_lena)
    for _ in range(3):
        assert detector.detect(face_lena) == first
