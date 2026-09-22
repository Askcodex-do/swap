"""Shared fixtures.  Real faces only - the point of these tests is that the
actual detector and the actual swap pipeline run on real photographs."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceswap.detector import FaceBox, FaceDetector

DATA = __import__("pathlib").Path(__file__).parent / "data"


def load(name: str) -> np.ndarray:
    img = cv2.imread(str(DATA / f"{name}.jpg"), cv2.IMREAD_COLOR)
    assert img is not None, f"missing test image {name}"
    return img


@pytest.fixture(scope="session")
def detector() -> FaceDetector:
    return FaceDetector()


@pytest.fixture(scope="session")
def face_obama() -> np.ndarray:
    return load("obama")


@pytest.fixture(scope="session")
def face_biden() -> np.ndarray:
    return load("biden")


@pytest.fixture(scope="session")
def face_obama2() -> np.ndarray:
    return load("obama2")


@pytest.fixture(scope="session")
def face_lena() -> np.ndarray:
    return load("lena_opencv_sample")


@pytest.fixture(scope="session")
def obama_box(detector, face_obama) -> FaceBox:
    box = detector.detect(face_obama)
    assert box is not None
    return box


@pytest.fixture(scope="session")
def biden_box(detector, face_biden) -> FaceBox:
    box = detector.detect(face_biden)
    assert box is not None
    return box
