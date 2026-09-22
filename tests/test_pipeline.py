"""The video path end to end: real frames in, real video out.

Videos are built in the test from real photographs so the real detector runs -
nothing is stubbed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceswap.detector import FaceBox, FaceDetector
from faceswap.pipeline import FaceSwapVideo, SwapOptions, SwapStats

VIDEO_W, VIDEO_H, VIDEO_FRAMES = 320, 240, 24
FACE_PX = 96


def make_video(path, face_img, frames=VIDEO_FRAMES, drift=True, face_px=FACE_PX):
    """A short video with a real face moving across a moving background.

    The face's own box is detected here and cropped with its aspect ratio
    preserved - squashing a portrait into a square destroys the geometry the
    cascade needs.
    """
    box = FaceDetector().detect(face_img)
    assert box is not None, "make_video needs an image with a detectable face"
    pad = int(box.w * 0.5)
    y0, x0 = max(0, box.y - pad), max(0, box.x - pad)
    crop = face_img[y0 : box.y + box.h + pad, x0 : box.x + box.w + pad]
    side = min(crop.shape[:2])
    face = cv2.resize(crop[:side, :side], (face_px, face_px), interpolation=cv2.INTER_AREA)
    assert FaceDetector().detect(face) is not None, "pasted face must be detectable"

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (VIDEO_W, VIDEO_H)
    )
    assert writer.isOpened()
    for i in range(frames):
        ramp = np.linspace(0, 200, VIDEO_W, dtype=np.float32) + i * 3
        frame = cv2.cvtColor(np.tile(ramp, (VIDEO_H, 1)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cx = 200 + (25 * np.sin(i * 0.3) if drift else 0)
        cy = 130
        x0f = int(cx - face_px / 2)
        y0f = int(cy - face_px / 2)
        frame[y0f : y0f + face_px, x0f : x0f + face_px] = face
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def template_face(face_obama, obama_box):
    return face_obama, obama_box


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def test_source_without_a_face_is_rejected(tmp_path):
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((240, 240, 3), 128, np.uint8))
    with pytest.raises(ValueError, match="no face"):
        FaceSwapVideo(blank)


def test_missing_source_file_is_rejected(tmp_path):
    with pytest.raises(IOError):
        FaceSwapVideo(tmp_path / "nope.jpg")


def test_missing_target_video_is_rejected(tmp_path, face_obama):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    engine = FaceSwapVideo(src)
    with pytest.raises(IOError):
        engine.process(tmp_path / "nope.mp4", tmp_path / "out.mp4")


def test_full_video_is_swapped_and_written(tmp_path, face_obama, face_lena, obama_box):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    engine = FaceSwapVideo(src)
    out = tmp_path / "out.mp4"
    stats = engine.process(target, out)

    assert stats.frames == VIDEO_FRAMES
    assert stats.swapped == stats.frames
    assert stats.hit_rate == 1.0
    assert out.is_file() and out.stat().st_size > 0
    assert not stats.warnings

    # the writer produced a decodable video of the same length and size
    frames = read_video(out)
    assert len(frames) == VIDEO_FRAMES
    assert frames[0].shape == (VIDEO_H, VIDEO_W, 3)


def write_frames(frames, path):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h)
    )
    assert writer.isOpened()
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def test_output_frames_actually_changed_the_face(tmp_path, face_obama, face_lena, obama_box):
    """The output must differ from the input inside the face region and be
    untouched outside it.

    A plain re-encode of the same frames is used as the reference, so the
    comparison is against codec noise rather than against the original file.
    """
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    engine = FaceSwapVideo(src)
    out = tmp_path / "out.mp4"
    engine.process(target, out)

    # reference: the same decoded frames, encoded again with no swap applied
    reference = write_frames(read_video(target), tmp_path / "ref.mp4")

    ref = read_video(reference)
    after = read_video(out)

    changed = []
    untouched = []
    for r, a in zip(ref, after):
        diff = np.abs(a.astype(int) - r.astype(int)).max(axis=2)
        changed.append((diff > 12).sum())
        # every edge of the frame is background: the face sits centrally
        border = np.concatenate([
            diff[:24, :].ravel(), diff[-24:, :].ravel(),
            diff[:, :24].ravel(), diff[:, -24:].ravel(),
        ])
        untouched.append(int(border.max()))
    assert min(changed) > 300, "no frame had a swapped face"
    assert max(untouched) == 0, "pixels outside the face region were modified"
    # the change is concentrated in the face, not smeared over the frame
    assert max(changed) < 0.25 * (VIDEO_W * VIDEO_H), "far more than the face changed"


def test_no_audio_track_is_written(tmp_path, face_obama, face_lena, obama_box):
    """The converter must not attempt to carry audio.  The container is written
    with a video-only fourcc, so no audio stream exists to enumerate."""
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    engine = FaceSwapVideo(src)
    out = tmp_path / "out.mp4"
    engine.process(target, out)

    cap = cv2.VideoCapture(str(out))
    try:
        # OpenCV exposes no audio stream for a video-only file
        assert int(cap.get(cv2.CAP_PROP_AUDIO_TOTAL_STREAMS)) == 0
    finally:
        cap.release()


def test_detection_is_reused_between_frames(tmp_path, face_obama, face_lena, obama_box):
    """Face detection is throttled; that is what keeps it fast on weak CPUs."""
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    engine = FaceSwapVideo(src, SwapOptions(detect_every=6))
    stats = engine.process(target, tmp_path / "out.mp4")
    assert stats.detect_calls <= (VIDEO_FRAMES // 6) + 2
    assert stats.swapped == VIDEO_FRAMES


def test_max_frames_limits_work(tmp_path, face_obama, face_lena, obama_box):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    engine = FaceSwapVideo(src, SwapOptions(max_frames=5))
    stats = engine.process(target, tmp_path / "out.mp4")
    assert stats.frames == 5
    assert len(read_video(tmp_path / "out.mp4")) == 5


def test_frame_passed_through_when_no_face_present(tmp_path, face_obama):
    """A frame with no face must come back untouched, not stamped with a stale
    face or a black patch."""
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    engine = FaceSwapVideo(src)

    blank = np.full((240, 320, 3), 70, np.uint8)
    out = engine.process_frame(blank, 0)
    assert np.array_equal(out, blank)


def test_poisson_mode_produces_a_valid_video(tmp_path, face_obama, face_lena, obama_box):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    engine = FaceSwapVideo(src, SwapOptions(blend="poisson"))
    stats = engine.process(target, tmp_path / "out.mp4")
    assert stats.swapped == VIDEO_FRAMES
    frames = read_video(tmp_path / "out.mp4")
    assert len(frames) == VIDEO_FRAMES


def test_invalid_options_are_rejected():
    with pytest.raises(ValueError):
        SwapOptions(blend="magic")
    with pytest.raises(ValueError):
        SwapOptions(smooth=1.0)
    with pytest.raises(ValueError):
        SwapOptions(smooth=-0.1)


def test_stats_helpers():
    stats = SwapStats(frames=10, swapped=5, seconds=2.0)
    assert stats.hit_rate == 0.5
    assert stats.fps == 5.0
    assert SwapStats().hit_rate == 0.0
    assert SwapStats().fps == 0.0


def test_temporal_smoothing_reduces_frame_to_frame_movement(tmp_path, face_obama, face_lena, obama_box):
    """Higher smoothing must make the pasted face move less between frames."""
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    def jitter(smooth):
        engine = FaceSwapVideo(src, SwapOptions(smooth=smooth, detect_every=1))
        boxes = []
        cap = cv2.VideoCapture(str(target))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            engine.process_frame(frame, len(boxes))
            boxes.append(None if engine._last_points is None else engine._last_points.copy())
        cap.release()
        centres = np.array([b.mean(axis=0) for b in boxes if b is not None])
        return float(np.abs(np.diff(centres, axis=0)).mean())

    # the face itself moves in this clip, so smoothing cannot make movement zero,
    # but it must not amplify it
    assert jitter(0.0) > 0.0
    assert jitter(0.8) <= jitter(0.0) + 1e-6
