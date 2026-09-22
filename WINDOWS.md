# Building and running on Windows 8.1 (2 GB RAM)

This is the complete sequence, from a bare machine to a working exe. Follow it
in order. Every step has been chosen so it works on Windows 8.1 with 2 GB of
RAM.

## Why the versions matter

Windows 8.1 reached the end of extended support in January 2023, and Python
follows Microsoft's lifecycle. The consequences are specific:

* **Python 3.12 and newer do not install on Windows 8.1** - the installers
  themselves refuse. The last Python with a Windows installer that runs on 8.1
  is **3.11.9**, and the last with a *3.10* installer is **3.10.11**. Either
  works here; 3.10.11 is the safest choice and is what this project is tested
  against.
* One practical wrinkle: `python.org` download pages carry a "not supported on
  Windows 7 or earlier" note on 3.10.11, so the page can look like Windows 8.1
  is excluded. It is not - Windows 8.1 is supported on 3.10.11 and 3.11.9.
* **PyInstaller 6 dropped Python 3.7**, and **NumPy 2 dropped it too**, so on
  3.7 you must pin both (see the fallback section at the end).
* PyInstaller officially supports Windows 8 and newer, so 8.1 is fine.
* On any Python, use the **32-bit (x86)** installer only if your machine is
  32-bit. Install the matching bitness for everything, or the wheels will not
  load.

## Step 1 - Install Python 3.10.11

1. Download `python-3.10.11-amd64.exe` (64-bit) from the official release page:
   <https://www.python.org/downloads/release/python-31011/>
   For a 32-bit machine use `python-3.10.11.exe` instead.
2. Run the installer.
3. **Tick "Add Python 3.10 to PATH"** on the first screen.
4. Choose "Install Now".

Verify with a new Command Prompt:

```bat
python --version
```

Expect `Python 3.10.11`.

## Step 2 - Get the project

Copy the project folder onto the machine (a USB stick is fine - nothing here
needs an internet connection at run time). Then:

```bat
cd C:\faceswap
```

## Step 3 - Install the dependencies

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

That installs exactly two things:

* `opencv-python-headless` - the headless build, so there is no Qt/GUI baggage
* `numpy`

Both are ordinary wheels with no CUDA, ONNX or model downloads. The Haar
cascade XML files face detection uses are inside the OpenCV wheel.

Check it works before going further:

```bat
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

## Step 4 - Try it on a video

```bat
python -m faceswap -s face.jpg -t clip.mp4 -o out.mp4
```

`face.jpg` must show one clear, front-facing face. `clip.mp4` must show one
face. The tool will tell you if either has none, or more than one.

To confirm it has not degraded on the slow machine, cap the work first:

```bat
python -m faceswap -s face.jpg -t clip.mp4 -o out.mp4 --max-frames 60
```

## Step 5 - Build the exe

```bat
python -m pip install pyinstaller==5.13.2
python make_exe.py
```

The result is `dist\faceswap.exe`, about 65 MB, self-contained. Copy that one
file anywhere - including to a machine with no Python at all - and run it from
a Command Prompt:

```bat
dist\faceswap.exe -s face.jpg -t clip.mp4 -o out.mp4
```

The console window stays open on purpose: this is a command-line tool and you
want to see its progress and any error message.

### If the exe fails to start

The most common cause on Windows 8.1 is the **Universal C Runtime**. Python 3.5
and newer need it, and on Vista through 8.1 it arrives through Windows Update,
which may not be installed on a machine that has been offline. Symptom: the exe
exits immediately with a missing-DLL error. Fix it by installing one of:

* KB2999226 (Update for Universal C Runtime), or
* the Visual C++ 2015 Redistributable

Both are available from Microsoft's Download Center. A machine that has run
Windows Update recently already has them.

To find out exactly which DLL is missing, run `dist\faceswap.exe` from a
Command Prompt rather than double-clicking it, so the error stays on screen.

### If antivirus flags the exe

PyInstaller bootloaders are a common false positive, because malware has abused
the same packing technique. Add an exclusion for the folder, or build with
`--onedir` (edit `faceswap.spec`: replace the single `EXE(...)` with an
`EXE`+`COLLECT` pair) so the payload is not packed into one file.

## Step 6 - Keep it fast on 2 GB

Defaults are already tuned for a weak machine. If you need more speed:

* Raise `--detect-every` from 5 to 8 or 10. Detection is the expensive step and
  is already throttled; landmarks are still refitted on every frame, so the
  face keeps following the video.
* Keep clips at 480p or below. On a 720p clip this runs at roughly 15 fps on
  one core; at 320x240 it runs at 140 fps.
* Process in chunks with `--max-frames` if a single pass takes too long.

Memory is not a concern: peak use is about 180-230 MB because one frame is read,
swapped and written at a time. Nothing loads the whole video. If the machine is
truly tight, close the browser first - that is usually the biggest consumer, not
this tool.

## Fallback: Windows 8.1 with 32-bit Python or an older Python

If you are pinned to **Python 3.7**, pin both packages, because the newer ones
dropped 3.7:

```bat
python -m pip install "numpy==1.21.6" "opencv-python-headless==4.11.0.86" "pyinstaller==5.13.2"
```

`opencv_python_headless-4.11.0.86-cp37-abi3-win_amd64.whl` is an `abi3` wheel,
so the same file serves Python 3.7 through 3.12 - it will install on 3.7.

On **32-bit** Python, use the `win32` (not `win_amd64`) wheel files and expect
a somewhat slower run; everything else is identical. Sample pinned URLs:

* `https://files.pythonhosted.org/packages/.../opencv_python_headless-4.11.0.86-cp37-abi3-win_amd64.whl`
* `https://files.pythonhosted.org/packages/.../numpy-1.26.4-cp310-cp310-win_amd64.whl`
* `https://files.pythonhosted.org/packages/.../pyinstaller-5.13.2-py3-none-win_amd64.whl`

The code itself stays compatible: it imports only `cv2`, `numpy`, `argparse`,
`dataclasses`, `pathlib`, `os`, `sys` and `time`, uses `from __future__ import
annotations` for modern type hints, and downloads nothing.

## Quick reference

```bat
:: install
python -m pip install -r requirements.txt

:: run from source
python -m faceswap -s face.jpg -t clip.mp4 -o out.mp4

:: build
python -m pip install pyinstaller==5.13.2
python make_exe.py

:: run the exe
dist\faceswap.exe -s face.jpg -t clip.mp4 -o out.mp4
```
