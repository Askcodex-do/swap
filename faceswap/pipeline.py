"""Frame-by-frame video processing.

Deliberately single-pass and allocation-light: frames are read, swapped in
place and written straight out, so peak memory is a handful of frames rather
than the whole video.  That is what lets it run on 2 GB of RAM.

No audio is copied: the output video carries video frames only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detector import FaceBox, FaceDetector
from .swapper import (
    blend_face,
    fit_landmarks,
    make_blend_mask,
    match_colors,
    triangulate,
    warp_face,
)


@dataclass
class SwapOptions:
    """Everything the pipeline needs, with the tuned defaults baked in."""

    blend: str = "alpha"
    feather: float = 0.08
    colour_strength: float = 0.9
    colour_luminance: float = 1.0
    mask_inset: float = 0.10
    detect_every: int = 5
    """Re-run face detection every N frames; in between the previous box is
    reused.  Detection is the most expensive step, and a face rarely moves far
    in 5 frames at normal frame rates."""

    smooth: float = 0.6
    """Temporal smoothing of landmark positions, 0 = none, 1 = frozen.
    Takes the jitter out of the frame-to-frame cascade output."""

    max_frames: int = 0
    """0 means process the whole video (used by tests to keep runs short)."""

    def __post_init__(self) -> None:
        if self.blend not in ("alpha", "poisson"):
            raise ValueError(f"unknown blend mode {self.blend!r}")
        if not 0.0 <= self.smooth < 1.0:
            raise ValueError("smooth must be in [0, 1)")


@dataclass
class SwapStats:
    frames: int = 0
    swapped: int = 0
    undetected: int = 0
    detect_calls: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.swapped / self.frames if self.frames else 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def _open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {path}")
    return cap


class FaceSwapVideo:
    """Applies one source face to every frame of a video.

    Detection uses a local search window around the previous frame's face, so
    the cost per frame is low and steady.  If the face is lost the search
    widens back to the full frame; if it stays lost the frame is passed through
    unchanged rather than left with a stale face pasted on.
    """

    def __init__(
        self,
        source_face_path: str | Path,
        options: SwapOptions | None = None,
        detector: FaceDetector | None = None,
        source_box: FaceBox | None = None,
    ):
        """``detector`` and ``source_box`` exist for tests and for callers that
        already know where the face is; normal use passes only a path."""
        self.options = options or SwapOptions()
        self.detector = detector or FaceDetector()
        self.source_image = cv2.imread(str(source_face_path), cv2.IMREAD_COLOR)
        if self.source_image is None:
            raise IOError(f"could not read source image: {source_face_path}")

        box = source_box or self.detector.detect(self.source_image)
        if box is None:
            raise ValueError(
                "no face found in the source image - use a clear, front-facing "
                "photo with the face reasonably large in frame"
            )

        # This tool maps one face onto one face.  Refuse an ambiguous source
        # rather than silently picking whichever face the cascade ranked first.
        if source_box is None:
            others = self.detector.count_faces(self.source_image)
            if others > 1:
                raise ValueError(
                    f"the source image contains {others} faces; crop it to a "
                    "single face"
                )

        gray = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2GRAY)
        self.source_landmarks = fit_landmarks(
            box,
            self.detector.detect_eyes(gray, box),
            self.detector.detect_mouth(gray, box),
        )
        # Triangulate once, in source space: both faces share this topology.
        self.triangles = triangulate(
            self.source_landmarks.points,
            (self.source_image.shape[1], self.source_image.shape[0]),
        )
        if len(self.triangles) < 4:
            raise ValueError(
                "could not build a stable triangulation for the source face"
            )

        self._last_box: FaceBox | None = None
        self._last_points: np.ndarray | None = None
        self._detect_calls = 0
        self._multi_face_checked = False
        self._last_mouth: tuple[tuple[float, float], tuple[float, float]] | None = None

    def _detect_frame(self, frame: np.ndarray, index: int) -> FaceBox | None:
        opts = self.options
        if self._last_box is not None and index % opts.detect_every != 0:
            return self._last_box

        self._detect_calls += 1
        # Local search first when we already have a position, otherwise full frame.
        box = None
        if self._last_box is not None:
            box = self.detector.detect(frame, region=self._last_box, pad=0.30)
        if box is None:
            box = self.detector.detect(frame)
        self._last_box = box
        return box

    def _measure_mouth(self, gray: np.ndarray, box: FaceBox, index: int):
        """Mouth corners for this frame, re-measured on the detection cadence.

        Mouth detection costs about as much as eye detection, so it rides the
        same every-N-frames schedule as the face box and is reused in between.
        The mouth barely moves over a few frames, and the landmark smoothing
        below removes what little jitter remains.
        """
        opts = self.options
        if self._last_mouth is not None and index % opts.detect_every != 0:
            return self._last_mouth
        mouth = self.detector.detect_mouth(gray, box)
        self._last_mouth = mouth
        return mouth

    def process(self, input_path: str | Path, output_path: str | Path) -> SwapStats:
        opts = self.options
        cap = _open_video(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not width or not height:
            cap.release()
            raise IOError("video reports zero frame size")
        if not fps or fps != fps or fps <= 0:
            fps = 25.0

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        stats = SwapStats()
        started = time.time()
        index = 0

        # Read and validate the first frame *before* opening the output, so a
        # rejected clip leaves no file behind for the writer to have created.
        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            raise IOError(f"no frames could be read from {input_path}")
        try:
            self._check_frame_is_usable(first_frame)
        except Exception:
            cap.release()
            raise

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise IOError(f"could not open video writer for {output_path}")

        try:
            writer.write(self.process_frame(first_frame, 0, stats))
            stats.frames += 1
            index = 1

            while True:
                if opts.max_frames and index >= opts.max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                writer.write(self.process_frame(frame, index, stats))
                stats.frames += 1
                index += 1
        finally:
            cap.release()
            writer.release()

        stats.seconds = time.time() - started
        if stats.hit_rate < 0.5:
            stats.warnings.append(
                f"face found in only {stats.hit_rate:.0%} of frames; "
                "output keeps the original face elsewhere"
            )
        return stats

    def _check_frame_is_usable(self, frame: np.ndarray) -> None:
        """Fail on a target clip this tool is not meant to handle."""
        if self._multi_face_checked:
            return
        box = self.detector.detect(frame)
        if box is None:
            return
        count = self.detector.count_faces(frame, region=box, pad=1.5)
        self._multi_face_checked = True
        if count > 1:
            raise ValueError(
                f"{count} faces are visible in the target video; this tool swaps "
                "exactly one face, so use a clip that shows a single face"
            )

    def process_frame(
        self, frame: np.ndarray, index: int = 0, stats: SwapStats | None = None
    ) -> np.ndarray:
        """Swap one frame.  Public so tests and scripts can drive it directly."""
        opts = self.options
        box = self._detect_frame(frame, index)
        if stats is not None:
            stats.detect_calls = self._detect_calls
        if box is None:
            if stats is not None:
                stats.undetected += 1
            self._last_points = None
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        target = fit_landmarks(
            box,
            self.detector.detect_eyes(gray, box),
            self._measure_mouth(gray, box, index),
        )

        templates = self.source_landmarks.points
        # Temporal smoothing: blend the new fit with the previous one so the
        # paste does not twitch with per-frame detection noise.
        if self._last_points is not None and opts.smooth > 0.0:
            points = (
                opts.smooth * self._last_points + (1.0 - opts.smooth) * target.points
            )
        else:
            points = target.points
        self._last_points = points

        warped, coverage = warp_face(
            self.source_image, templates, points, frame.shape, self.triangles
        )
        if not np.any(coverage):
            if stats is not None:
                stats.undetected += 1
            return frame

        mask = make_blend_mask(frame.shape, points, coverage, inset=opts.mask_inset)
        if not np.any(mask):
            if stats is not None:
                stats.undetected += 1
            return frame

        matched = match_colors(
            warped,
            frame,
            mask,
            strength=opts.colour_strength,
            luminance_strength=opts.colour_luminance,
        )
        out = blend_face(frame, matched, mask, opts.blend, feather=opts.feather)

        if stats is not None:
            stats.swapped += 1
        return out
