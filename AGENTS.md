# AGENTS.md

Repository-specific guidance for working on `faceswap`.

## What this project is

An offline face-swap *video* converter. A single source photo is warped onto the
single face in a video. Classic CV only - Haar cascades, Delaunay triangulation,
piecewise-affine warping, CIELAB colour transfer, alpha blending.

## Hard constraints (do not violate)

These are the point of the project. Breaking any of them is a regression:

* **Two dependencies only**: `numpy` and `opencv-python-headless`. Never add
  torch, tensorflow, onnxruntime, insightface, dlib, scipy, mediapipe, or any
  learned-model runtime.
* **No downloads at run time.** No `urllib`/`requests`/`socket` in `faceswap/`.
  Only the Haar XMLs bundled inside the OpenCV wheel are used.
* **No CUDA, no GPU.** CPU only.
* **No audio.** The output is written with the `mp4v` fourcc and carries no
  audio stream. Do not add audio muxing.
* **One face only.** Multi-face sources and multi-face targets are *refused*
  with a clear error (see `tests/test_multiface.py`). Do not add multi-face
  handling.
* **Video in, video out.** Image targets are refused. There is no image-to-image
  mode.
* **Real deformation, not a paste.** Pixels are rasterised through source
  triangles. Never replace the warp with a rectangular crop or an overlay.

## Compatibility floor

Windows 8.1 on 2 GB of RAM is a target platform. That implies:

* Python 3.8+ syntax. All files use `from __future__ import annotations`, so
  PEP 604 unions (`str | Path`) are fine in annotations, but do not use runtime
  3.9+ features such as `str.removeprefix` or `dict | dict`.
* Avoid 3.8+ only additions in code that must run on 3.7 (see `WINDOWS.md`).
  `Path.unlink(missing_ok=True)` is 3.8+, so it is avoided in `faceswap/`.
* Keep the frozen exe's memory low by processing one frame at a time. Never
  accumulate all frames in a list.

## Layout

* `faceswap/detector.py` - `FaceBox`, `FaceDetector.detect/detect_eyes/count_faces`.
  `load_cascade` searches `cv2.data`, the cv2 package dir, and `sys._MEIPASS`.
* `faceswap/swapper.py` - `fit_landmarks`, `triangulate`, `warp_face`,
  `match_colors`, `make_blend_mask`, `blend_face`.
* `faceswap/pipeline.py` - `SwapOptions`, `SwapStats`, `FaceSwapVideo`.
* `faceswap/cli.py` - argument parsing, validation, exit codes (2 = bad input,
  1 = runtime failure).
* `run_faceswap.py` - PyInstaller entry point. A frozen program has no parent
  package, so `faceswap/__main__.py`'s relative import cannot be the bundle
  target.
* `faceswap.spec`, `make_exe.py` - the build.

## Testing

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

* Tests use **real photographs** in `tests/data` (public-domain US government
  portraits plus the standard OpenCV `lena`). Do not stub the detector; do not
  add synthetic faces - a drawn face is not Haar-detectable, which was tried and
  rejected.
* `FaceSwapVideo` accepts `detector=` and `source_box=` for injection, but tests
  pass only a path so the real detector runs.
* When building a test video, **detect the face's own box first**. Squashing a
  portrait into a square destroys the geometry the cascade needs, and detection
  silently returns nothing.

## Gotchas already hit (do not re-introduce)

* `cv2.Subdiv2D` takes `(x, y, width, height)`, not `(x0, y0, x1, y1)`. It also
  rejects points outside its rectangle, so the rect is derived from the landmark
  bounds - cheeks and chin reach outside the detected box.
* `_warp_triangle` must clip both bounding boxes before slicing. A negative
  origin becomes a wrap-around index in NumPy and corrupts the destination.
* The coverage mask records *actual* writes, not the triangulated polygon,
  because edge rounding can leave a pixel inside the hull but painted by no
  triangle - which would paste black specks onto the face.
* Blend masks must be strictly binary for `seamlessClone`. Use `cv2.LINE_8`
  fill, never the antialiased default.
* Colour matching writes back only masked pixels; round-tripping the whole image
  through LAB perturbs pixels outside the mask.
* Landmark proportions are anchored to the **measured** interocular/box-width
  ratio (`_INTEROCULAR_OVER_BOX_W`), not "half the box width".
* **A rolled head is retried on rotated copies** (`FaceDetector._detect_tilted`,
  angles in `_TILT_ANGLES`). The upright frontal cascade tolerates only about
  +-10 degrees of roll, so a clip that opens with the head turned produced *no
  face at all* on those frames and kept the original face. The retry is a
  fallback only - never the first attempt - because it costs ~124 ms on a
  320x240 frame with no face in it, against ~14 ms for a tracked local search.
  A box found in a rotated frame is mapped back through the inverse rotation, so
  callers always see upright coordinates. Validated for recall (35/40 -> 40/40 on
  a moving-face clip) *and* precision (flat grey, black, noise, checkerboard and
  gradient images still return `None`). See `tests/test_detector.py` and
  `tests/test_motion.py`.
* The **cheek/jaw/chin outline is still a proportional template**, and that is
  deliberate. Measuring it from the image was tried and rejected: a
  Cr/Cb skin mask plus a per-row scan for the outline leaked into a beige
  background (obama: measured "face" width 2.68 eye-units, i.e. the whole frame)
  and found no skin at all on the grayscale `lena` sample. Without a learned
  segmentation model, background and face are not separable in that region, so a
  bad measurement is worse than a good template. Do not re-add a skin-tone jaw
  scan. What *is* measured per face is the eyes and the mouth (smile cascade);
  see `tests/test_deformation.py`.
* PyInstaller 6 does not auto-collect `secrets`, which strands numpy through
  `numpy.random`. See `hiddenimports` in `faceswap.spec`.

## Tuning defaults

`SwapOptions` defaults were chosen from measurement, not taste:

* `blend="alpha"`, `feather=0.08` - alpha preserves 0.95+ of source detail where
  Poisson preserves ~0.2.
* `colour_strength=0.9`, `colour_luminance=1.0` - lands output L* within ~5
  units of the target while keeping identity correlation 0.74-0.90.
* `mask_inset=0.10`, `detect_every=5`, `smooth=0.6`.

Re-run `python -m pytest tests/ -q` before committing, and keep the change map
tight around the face (the `test_output_frames_actually_changed_the_face` test
enforces this).
