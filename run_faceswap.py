"""Launcher used as the PyInstaller entry point.

A frozen program is executed as top-level ``__main__`` with no parent package,
so ``faceswap/__main__.py``'s relative import cannot be used as the bundling
target.  This absolute-import shim is the entry point instead.
"""

from faceswap.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
