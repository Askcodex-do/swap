"""The swap must follow a face that actually moves, shrinks and turns.

The other pipeline tests use a face that only wobbles laterally at a fixed
size in an otherwise flat gradient.  That does not exercise the tracking path,
the detection cadence or the landmark scaling, so this file drives a clip where
the face changes position, size and roll, and asserts the swapped region follows
it - which is what separates a per-frame swap from a one-off paste.
"""

from __future__ import annotations

import cv2
import numpy as np

from faceswap.detector import FaceDetector
from faceswap.pipeline import FaceSwapVideo, SwapOptions

from .test_pipeline import read_video

W, H, N = 320, 240, 40


def make_moving_clip(path, face_img):
    """A clip where the face drifts, shrinks and rotates over 40 frames."""
    detector = FaceDetector()
    box = detector.detect(face_img)
    assert box is not None
    pad = int(box.w * 0.5)
    y0, x0 = max(0, box.y - pad), max(0, box.x - pad)
    crop = face_img[y0 : box.y + box.h + pad, x0 : box.x + box.w + pad]
    side = min(crop.shape[:2])
    face = crop[:side, :side]

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
    assert writer.isOpened()
    for i in range(N):
        t = i / (N - 1)
        frame = np.full((H, W, 3), 45, np.uint8)
        scale = (0.85 - 0.35 * t) * (H / face.shape[0])
        angle = -12.0 + 24.0 * t
        cx = W * (0.30 + 0.40 * t)
        cy = H * (0.50 + 0.06 * np.sin(t * np.pi))

        matrix = cv2.getRotationMatrix2D(
            (face.shape[1] / 2.0, face.shape[0] / 2.0), angle, 1.0
        )
        rolled = cv2.warpAffine(
            face, matrix, (face.shape[1], face.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        )
        small = cv2.resize(rolled, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        fh, fw = small.shape[:2]
        px = int(np.clip(cx - fw / 2, 0, W - fw))
        py = int(np.clip(cy - fh / 2, 0, H - fh))
        frame[py : py + fh, px : px + fw] = small
        writer.write(frame)
    writer.release()
    return path


def swap_regions(original, swapped):
    """Bounding info of the pixels that actually changed, per frame."""
    out = []
    for a, b in zip(original, swapped):
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=2) > 30
        ys, xs = np.nonzero(diff)
        out.append((int(diff.sum()), float(xs.mean()), float(ys.mean())) if xs.size else (0, 0.0, 0.0))
    return out


def test_swap_follows_a_moving_shrinking_turning_face(tmp_path, face_obama, face_lena):
    source_in = make_moving_clip(tmp_path / "moving.mp4", face_lena)
    output = tmp_path / "swapped.mp4"
    source_photo = tmp_path / "source.jpg"
    assert cv2.imwrite(str(source_photo), face_obama)

    pipeline = FaceSwapVideo(str(source_photo), SwapOptions())
    stats = pipeline.process(str(source_in), str(output))

    original = read_video(source_in)
    swapped = read_video(output)
    assert len(swapped) == len(original) == N

    # Every frame including the first carries a swapped region.  The first frame
    # is the one the upright cascade used to miss, because the head starts rolled.
    regions = swap_regions(original, swapped)
    changed = [i for i, (n, _, _) in enumerate(regions) if n > 0]
    assert len(changed) >= N - 1, (
        f"swap missing on {N - len(changed)} frames; per-frame stats: {stats}"
    )
    assert 0 in changed, "the first frame must be swapped too"

    # The swapped region must travel with the face rather than sit at one spot.
    xs = [cx for n, cx, _ in regions if n > 0]
    assert max(xs) - min(xs) > 60, (
        f"swapped region barely moved ({min(xs):.0f}..{max(xs):.0f}); it is not "
        "tracking the face"
    )

    # And it must shrink as the face shrinks.
    first_px = regions[changed[0]][0]
    last_px = regions[changed[-1]][0]
    assert last_px < first_px, (
        f"swapped area did not shrink with the face ({first_px} -> {last_px})"
    )
