"""Build the standalone exe.

Run from the project root:

    python make_exe.py

Produces ``dist/faceswap.exe`` (one file, no console window shown when
double-clicked is *not* what we want here - this is a CLI, so the console
stays).  PyInstaller must be installed: ``pip install pyinstaller``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SPEC = ROOT / "faceswap.spec"


def main() -> int:
    if shutil.which("pyinstaller") is None:
        print("PyInstaller is not installed. Run: pip install pyinstaller", file=sys.stderr)
        return 1

    for directory in (DIST, BUILD):
        if directory.exists():
            shutil.rmtree(directory)

    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        str(SPEC),
    ]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        return result.returncode

    exe = DIST / ("faceswap.exe" if sys.platform == "win32" else "faceswap")
    if not exe.exists():
        print(f"build finished but {exe} is missing", file=sys.stderr)
        return 1
    print(f"built {exe} ({exe.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
