"""Entry point for both ``python -m faceswap`` and the frozen exe."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
