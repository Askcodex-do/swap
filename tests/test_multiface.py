"""Single-face enforcement.

The tool swaps exactly one face.  These tests check that an ambiguous source or
target is refused rather than guessed at.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceswap.detector import FaceDetector
from faceswap.pipeline import FaceSwapVideo

from .test_pipeline import make_video


def two_face_image(face_img, box):
    """Compose a 2x2 grid of face crops so several faces are clearly present."""
    pad = int(box.w * 0.45)
    y0, x0 = max(0, box.y - pad), max(0, box.x - pad)
    crop = face_img[y0 : box.y + box.h + pad, x0 : box.x + box.w + pad]
    side = min(crop.shape[:2])
    tile = cv2.resize(crop[:side, :side], (160, 160), interpolation=cv2.INTER_AREA)

    canvas = np.full((360, 360, 3), 110, np.uint8)
    for r in range(2):
        for c in range(2):
            canvas[r * 180 + 10 : r * 180 + 170, c * 180 + 10 : c * 180 + 170] = tile
    return canvas


def test_count_faces_finds_one_on_a_single_face_image(face_lena):
    assert FaceDetector().count_faces(face_lena) == 1


def test_count_faces_finds_several_on_a_grid(face_lena):
    box = FaceDetector().detect(face_lena)
    grid = two_face_image(face_lena, box)
    assert FaceDetector().count_faces(grid) >= 2


def test_count_faces_ignores_a_far_smaller_background_face(face_lena, face_obama):
    """A big face plus a tiny one should count as one subject, not two."""
    big_box = FaceDetector().detect(face_lena)
    pad = int(big_box.w * 0.45)
    y0, x0 = max(0, big_box.y - pad), max(0, big_box.x - pad)
    crop = face_lena[y0 : big_box.y + big_box.h + pad, x0 : big_box.x + big_box.w + pad]
    side = min(crop.shape[:2])
    big = cv2.resize(crop[:side, :side], (260, 260), interpolation=cv2.INTER_AREA)

    small_box = FaceDetector().detect(face_obama)
    spad = int(small_box.w * 0.45)
    sy0, sx0 = max(0, small_box.y - spad), max(0, small_box.x - spad)
    scrop = face_obama[sy0 : small_box.y + small_box.h + spad, sx0 : small_box.x + small_box.w + spad]
    sside = min(scrop.shape[:2])
    small = cv2.resize(scrop[:sside, :sside], (50, 50), interpolation=cv2.INTER_AREA)

    canvas = np.full((400, 400, 3), 110, np.uint8)
    canvas[40:300, 40:300] = big
    canvas[330:380, 330:380] = small
    assert FaceDetector().count_faces(canvas) == 1


def test_multi_face_source_is_refused(tmp_path, face_lena):
    box = FaceDetector().detect(face_lena)
    src = tmp_path / "grid.png"
    cv2.imwrite(str(src), two_face_image(face_lena, box))
    with pytest.raises(ValueError, match="faces; crop it"):
        FaceSwapVideo(src)


def test_single_face_source_is_accepted(tmp_path, face_lena):
    src = tmp_path / "one.png"
    cv2.imwrite(str(src), face_lena)
    FaceSwapVideo(src)  # must not raise


def test_multi_face_target_is_refused_before_output_is_written(
    tmp_path, face_obama, face_lena
):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)

    # a target clip showing a grid of faces
    from faceswap.detector import FaceDetector as FD

    box = FD().detect(face_lena)
    grid = two_face_image(face_lena, box)
    target = tmp_path / "many.mp4"
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (grid.shape[1], grid.shape[0])
    )
    for _ in range(4):
        writer.write(grid)
    writer.release()

    engine = FaceSwapVideo(src)
    out = tmp_path / "out.mp4"
    with pytest.raises(ValueError, match="faces are visible in the target"):
        engine.process(target, out)
    # the rejected run must not leave a partial video behind
    assert not out.exists()
