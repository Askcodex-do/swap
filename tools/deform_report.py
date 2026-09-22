"""Print the deformation evidence for a build.

Reports, on the real test photographs, the two numbers that separate a real
swap from a sticker:

* the residual after the best-fit similarity transform between two people's
  landmark sets - exactly 0.0 means the warp is only rotate/scale/translate
* the mouth/eye ratio per face, which is 1.000 for every face if the mouth is
  not actually being measured

Run from the repository root:

    python tools/deform_report.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# Allow running as `python tools/deform_report.py` from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faceswap.detector import FaceDetector  # noqa: E402
from faceswap.swapper import fit_landmarks  # noqa: E402

FACES = ("obama", "obama2", "biden", "lena_opencv_sample")


def similarity_residual(src, dst) -> float:
    s = np.asarray(src, np.float64)
    t = np.asarray(dst, np.float64)
    s0, t0 = s - s.mean(axis=0), t - t.mean(axis=0)
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


def main() -> None:
    detector = FaceDetector()
    fits = {}
    ratios = {}

    print("face            box        mouth      mouth/eye")
    for name in FACES:
        img = cv2.imread(f"tests/data/{name}.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = detector.detect(img)
        mouth = detector.detect_mouth(gray, box)
        lm = fit_landmarks(box, detector.detect_eyes(gray, box), mouth)
        fits[name] = lm.points
        eye_d = float(np.linalg.norm(lm.points[1] - lm.points[0]))
        ratios[name] = float(np.linalg.norm(lm.points[5] - lm.points[4])) / eye_d
        found = "found" if mouth else "MISSING"
        print(f"{name:15s} {box.w:4d}px     {found:9s}  {ratios[name]:.3f}")

    spread = max(ratios.values()) - min(ratios.values())
    print()
    print(f"mouth/eye spread across faces : {spread:.3f}   (0.000 means templated)")

    names = list(fits)
    worst = min(
        similarity_residual(fits[a], fits[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    )
    print(f"worst similarity residual     : {worst:.4f}   (0.0000 means sticker)")
    print()

    if spread > 0.15 and worst > 0.02:
        print("VERDICT: the warp reshapes internal geometry - a real deformation.")
        return 0
    print("VERDICT: FAILED - this build pastes rather than deforms.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
