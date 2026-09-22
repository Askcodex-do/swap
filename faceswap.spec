# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the face-swap CLI.

Build with:  pyinstaller --noconfirm --clean faceswap.spec
(``python make_exe.py`` does this for you and cleans up first.)

Two things this spec gets right for an OpenCV app:

* OpenCV's Haar cascade XML files live in ``cv2/data``.  PyInstaller does not
  always collect them, and without them face detection fails at runtime on a
  machine that has no separate OpenCV install - so they are added explicitly.
* cv2's own binary and numpy are pulled in automatically as normal imports;
  nothing else is needed because the project has no other dependencies.
"""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("cv2", includes=["data/*.xml"])

# numpy's typing stubs and tests are dead weight in a frozen build.
excludes = [
    "numpy.testing",
    "numpy.f2py",
    "setuptools",
    "pkg_resources",
    "pytest",
    "tkinter",
    "matplotlib",
    "PIL",
    "scipy",
]

# PyInstaller 6 stopped pulling in pathlib/tokenize and their dependency chains
# automatically, which strands numpy on a missing `secrets`.
hiddenimports = [
    "faceswap",
    "faceswap.cli",
    "faceswap.detector",
    "faceswap.swapper",
    "faceswap.pipeline",
    "secrets",
    "random",
    "hashlib",
    "hmac",
    "base64",
    "binascii",
    "tokenize",
    "pathlib",
    "urllib",
]


a = Analysis(
    ["run_faceswap.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="faceswap",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
