# faceswap

An offline face-swap video converter. One photo of a face is mapped onto the
single face in a video, frame by frame, and written out as a new video with no
audio track.

It uses only classic computer-vision techniques: Haar-cascade detection,
landmark-based Delaunay triangulation, piecewise-affine warping, CIELAB colour
transfer and alpha blending. There are no neural networks, no ONNX runtime, no
CUDA, no insightface, and nothing is downloaded at run time. The only two
dependencies are NumPy and headless OpenCV, and the cascade files used for
detection ship inside the OpenCV wheel.

## What it does and does not do

| | |
|---|---|
| Source | a single photo containing one clear, front-facing face |
| Target | a video containing one face |
| Output | a video of the same size and length, video only, **no audio** |
| Refuses | images or photos as the target (this is not image-to-image) |
| Refuses | sources or clips showing more than one face |
| Never | downloads models, needs a GPU, or phones home |

The swap is a real deformation: the source face is rasterised triangle by
triangle onto the target's features, so expressions, head angle and mouth
position follow the target. It is not a sticker or a cut-out pasted on top.

## Requirements

* Python 3.8 or newer (tested on 3.10 and 3.13)
* NumPy and `opencv-python-headless` - see `requirements.txt`
* Windows, Linux or macOS. **Windows 8.1** needs Python 3.10.11 or older and the
  pinned wheels described in `WINDOWS.md`.

No CUDA, no extra model files, no internet connection once installed.

## Install

```bash
python -m pip install -r requirements.txt
```

## Use

```bash
python -m faceswap --source face.jpg --target clip.mp4 --output out.mp4
```

Short form:

```bash
python -m faceswap -s face.jpg -t clip.mp4 -o out.mp4
```

Typical output:

```
source face loaded: box=(357, 104, 274, 274) eye distance=102.0px, 19 triangles
processing clip.mp4 -> out.mp4 ...
done: 571/571 frames swapped (100%), 115 face detections, 9.1s (62.7 fps)
```

If the source has no detectable face, or either input shows more than one
face, the tool stops with an explanation instead of guessing.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--blend` | `alpha` | `alpha` keeps the most likeness; `poisson` is smoother but flatter |
| `--feather` | `0.08` | edge softness for alpha, as a fraction of face size |
| `--colour` | `0.9` | colour-match strength, 0-1 |
| `--mask-inset` | `0.10` | how far inside the face outline the paste stops |
| `--detect-every` | `5` | re-detect the face every N frames; higher is faster |
| `--smooth` | `0.6` | temporal smoothing, 0-0.95; raise it if the paste jitters |
| `--max-frames` | `0` | stop after N frames (0 means the whole video) |
| `--quiet` | off | suppress progress output |

Start with the defaults. If the pasted face jitters, raise `--smooth`. If the
edges look hard, raise `--feather`. If the skin tone looks wrong, raise
`--colour`.

## Performance

Measured on a single CPU core, using the same code path the exe runs:

| Input | Result |
|---|---|
| 320x240, 96 px face | 140 fps |
| 1280x720, 150 px face | 15.5 fps |

Peak memory stays around 180-230 MB, because the pipeline reads, swaps and
writes one frame at a time rather than loading the video. That leaves plenty of
room inside a 2 GB machine.

The two things that keep it fast are doing detection on a downscaled grey copy,
and reusing the previous frame's detection for `--detect-every` frames
(landmarks are still refitted every frame, so tracking stays accurate).

## Build a standalone exe

```bash
python -m pip install -r requirements-dev.txt
python make_exe.py
```

This produces `dist/faceswap.exe` (Windows) or `dist/faceswap` (Linux/macOS),
around 65 MB, with the OpenCV cascades and the Python runtime inside it. It
runs on a machine with no Python installed. For the Windows 8.1 runbook, see
`WINDOWS.md`.

## How it works

1. **Detect.** The target's largest frontal face is found with a Haar cascade
   on a downscaled grey frame. Eyes are found inside the upper 60% of that box.
2. **Fit landmarks.** A fourteen-point layout - eyes, brows, nose bridge and
   tip, mouth corners and centre, chin, jaw and cheeks - is fitted to the box,
   scaled by the measured eye distance and rotated to the measured eye line.
   Because every landmark is expressed as a multiple of interocular distance, a
   4000 px portrait and a 60 px video face land on identical topology.
3. **Deform.** The source landmarks are triangulated with Delaunay once. Each
   triangle is affine-warped to its destination, so the source pixels are
   stretched onto the target's geometry rather than overlaid.
4. **Match colour.** Source and target are compared in CIELAB inside the blend
   mask; the mean is shifted and the standard deviation scaled so the face takes
   on the target's exposure and skin tone without losing its detail.
5. **Blend.** The warped face is composited through a feathered mask kept inside
   the face outline, leaving the target's ears, neck and hair from the original.

## Tests

```bash
python -m pip install pytest
python -m pytest tests/ -q
```

The suite runs the real detector and the real pipeline on real photographs -
nothing is stubbed. It checks geometric correctness pixel by pixel (a colour
painted at a source landmark must appear at its destination), that colour
matching moves the mean without destroying structure, that a plain re-encode is
the only difference outside the face, that multi-face inputs are refused, and
that no audio stream is written.

## Licence

MIT. The sample photographs in `tests/data` are public-domain US government
portraits and the standard OpenCV `lena` test image.
