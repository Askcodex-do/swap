"""Single-face detection with OpenCV's bundled Haar cascades.

The cascade XML files ship inside the opencv-python wheel, so nothing is
downloaded at runtime and no neural-network weights are involved.

Only the *largest* face is ever returned, which is what this tool wants:
one person, one face, no multi-face handling.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

_FRONTAL_CASCADE = "haarcascade_frontalface_alt2.xml"
_PROFILE_CASCADE = "haarcascade_profileface.xml"
_EYE_CASCADE = "haarcascade_eye.xml"


def _cascade_search_paths(name: str) -> list[str]:
    """Every place a bundled cascade may live, including inside a PyInstaller exe."""
    paths: list[str] = []

    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_dir:
        paths.append(os.path.join(data_dir, name))

    pkg_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    paths.append(os.path.join(pkg_dir, "data", name))
    paths.append(os.path.join(pkg_dir, name))

    # PyInstaller one-file builds unpack into sys._MEIPASS.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        paths.append(os.path.join(meipass, "cv2", "data", name))
        paths.append(os.path.join(meipass, name))

    return paths


def load_cascade(name: str) -> cv2.CascadeClassifier:
    for path in _cascade_search_paths(name):
        if os.path.isfile(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
    raise FileNotFoundError(
        f"Could not locate bundled cascade {name!r}. "
        "Reinstall opencv-python (the cascade XMLs ship inside the wheel)."
    )


@dataclass(frozen=True)
class FaceBox:
    """A detected face rectangle in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_bounds(cls, left: float, top: float, right: float, bottom: float) -> "FaceBox":
        """Build a box from float edges, rounding so the box still contains them."""
        x = int(np.floor(left))
        y = int(np.floor(top))
        return cls(x, y, int(np.ceil(right)) - x, int(np.ceil(bottom)) - y)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def expand(self, fx: float, fy: float) -> "FaceBox":
        """Grow the box by a fraction of its size on each axis, clamped at 0."""
        dx = int(round(self.w * fx))
        dy = int(round(self.h * fy))
        x = max(0, self.x - dx)
        y = max(0, self.y - dy)
        return FaceBox(x, y, self.w + 2 * dx, self.h + 2 * dy)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def area(self) -> int:
        return self.w * self.h


class FaceDetector:
    """Detects the single largest frontal face in an image or video frame."""

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_ratio: float = 0.06,
        max_detect_width: int = 640,
        profile_fallback: bool = True,
    ) -> None:
        self._frontal = load_cascade(_FRONTAL_CASCADE)
        self._profile = load_cascade(_PROFILE_CASCADE) if profile_fallback else None
        self._eyes = load_cascade(_EYE_CASCADE)
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_face_ratio = min_face_ratio
        self._max_detect_width = max_detect_width

    def _detect_in(
        self, gray: np.ndarray, offset: tuple[int, int], min_size: int
    ) -> list[FaceBox]:
        boxes: list[FaceBox] = []

        for cascade in (self._frontal, self._profile):
            if cascade is None:
                continue
            found = cascade.detectMultiScale(
                gray,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=(min_size, min_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for (x, y, w, h) in found:
                boxes.append(FaceBox(x + offset[0], y + offset[1], w, h))
            if boxes:
                break

        return boxes

    def detect(
        self,
        image_bgr: np.ndarray,
        region: FaceBox | None = None,
        pad: float = 0.25,
    ) -> FaceBox | None:
        """Return the largest face, or ``None``.

        Detection runs on a downscaled grayscale copy so the cost stays flat
        even on 1080p input, then the box is scaled back to source pixels.

        If ``region`` is given, only that area (grown by ``pad``) is searched,
        which both stabilises tracking across frames and avoids full-frame
        scans during video processing.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None

        if image_bgr.ndim == 2:
            full_gray = image_bgr
        else:
            full_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        height, width = full_gray.shape[:2]

        offset = (0, 0)
        search = full_gray
        if region is not None:
            grown = region.expand(pad, pad)
            x0 = max(0, grown.x)
            y0 = max(0, grown.y)
            x1 = min(width, grown.x + grown.w)
            y1 = min(height, grown.y + grown.h)
            if x1 - x0 < 8 or y1 - y0 < 8:
                return None
            search = full_gray[y0:y1, x0:x1]
            offset = (x0, y0)

        search_h, search_w = search.shape[:2]
        downscale = 1.0
        if search_w > self._max_detect_width:
            downscale = self._max_detect_width / float(search_w)
            search_small = cv2.resize(
                search,
                (self._max_detect_width, max(1, int(round(search_h * downscale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            search_small = search

        min_size = max(
            16, int(round(min(search_small.shape[:2]) * self._min_face_ratio))
        )
        boxes = self._detect_in(search_small, offset, min_size)

        if not boxes:
            return None

        inv = 1.0 / downscale
        scaled = [
            FaceBox(
                int(round(b.x * inv)),
                int(round(b.y * inv)),
                int(round(b.w * inv)),
                int(round(b.h * inv)),
            )
            for b in boxes
        ]

        best = max(scaled, key=FaceBox.area)
        return self._clamp(best, width, height)

    def count_faces(
        self,
        image_bgr: np.ndarray,
        region: FaceBox | None = None,
        pad: float = 0.25,
        min_ratio: float = 0.35,
    ) -> int:
        """How many faces are present, ignoring those much smaller than the
        largest.

        The tool swaps exactly one face.  Anything under ``min_ratio`` of the
        biggest box is treated as scenery (a face on a poster, a second person
        in the far background) rather than a second subject, so a stray small
        detection does not change the outcome.
        """
        if image_bgr is None or image_bgr.size == 0:
            return 0

        full_gray = (
            image_bgr if image_bgr.ndim == 2
            else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        )
        height, width = full_gray.shape[:2]

        offset = (0, 0)
        search = full_gray
        if region is not None:
            grown = region.expand(pad, pad)
            x0, y0 = max(0, grown.x), max(0, grown.y)
            x1 = min(width, grown.x + grown.w)
            y1 = min(height, grown.y + grown.h)
            if x1 - x0 < 8 or y1 - y0 < 8:
                return 0
            search = full_gray[y0:y1, x0:x1]
            offset = (x0, y0)

        search_h, search_w = search.shape[:2]
        downscale = 1.0
        if search_w > self._max_detect_width:
            downscale = self._max_detect_width / float(search_w)
            search_small = cv2.resize(
                search,
                (self._max_detect_width, max(1, int(round(search_h * downscale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            search_small = search

        min_size = max(16, int(round(min(search_small.shape[:2]) * self._min_face_ratio)))
        boxes = self._detect_in(search_small, offset, min_size)
        if not boxes:
            return 0

        inv = 1.0 / downscale
        areas = [b.area() * inv * inv for b in boxes]
        biggest = max(areas)
        return sum(1 for a in areas if a >= biggest * min_ratio)

    def detect_eyes(self, gray: np.ndarray, face: FaceBox) -> list[tuple[float, float]]:
        """Eye centres inside the upper part of ``face``, left to right."""
        x0 = max(0, face.x)
        y0 = max(0, face.y)
        x1 = min(gray.shape[1], face.x + face.w)
        y1 = min(gray.shape[0], face.y + face.h)
        if x1 - x0 < 12 or y1 - y0 < 12:
            return []

        # Eyes sit in the top ~60% of a Haar face box.
        roi = gray[y0 : y0 + int(round((y1 - y0) * 0.6)), x0:x1]
        found = self._eyes.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(max(6, roi.shape[1] // 12), max(6, roi.shape[0] // 12)),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(found) < 2:
            return []

        centres = sorted(
            (((x + w / 2.0) + x0, (y + h / 2.0) + y0) for (x, y, w, h) in found),
            key=lambda p: p[0],
        )

        # Split into left/right halves and take the widest-apart pair.
        mid = x0 + (x1 - x0) / 2.0
        left = [c for c in centres if c[0] < mid]
        right = [c for c in centres if c[0] >= mid]
        if not left or not right:
            return []
        return [max(left, key=lambda c: c[0]), min(right, key=lambda c: c[0])]

    @staticmethod
    def _clamp(box: FaceBox, width: int, height: int) -> FaceBox:
        x = max(0, min(box.x, width - 1))
        y = max(0, min(box.y, height - 1))
        w = max(1, min(box.w, width - x))
        h = max(1, min(box.h, height - y))
        return FaceBox(x, y, w, h)
