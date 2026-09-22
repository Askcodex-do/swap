"""Command-line entry point.

    python -m faceswap --source face.jpg --target clip.mp4 --output out.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .pipeline import FaceSwapVideo, SwapOptions

VIDEO_SUFFIXES = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faceswap",
        description=(
            "Offline face-swap video converter. One source photo is mapped onto "
            "the single face in a video. No audio, no neural networks, no model "
            "downloads, no GPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m faceswap -s face.jpg -t clip.mp4 -o out.mp4\n"
            "  python -m faceswap -s face.jpg -t clip.mp4 -o out.mp4 --blend poisson\n"
            "  python -m faceswap -s face.jpg -t clip.mp4 -o out.mp4 --max-frames 60\n"
        ),
    )
    parser.add_argument(
        "-s", "--source", required=True,
        help="source photo containing exactly one clear, front-facing face",
    )
    parser.add_argument(
        "-t", "--target", required=True,
        help="input video to convert",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="output video path (mp4)",
    )
    parser.add_argument(
        "--blend", choices=("alpha", "poisson"), default="alpha",
        help="alpha (default, keeps the most likeness) or poisson",
    )
    parser.add_argument(
        "--feather", type=float, default=0.08,
        help="edge softness as a fraction of face size for alpha blending (default 0.08)",
    )
    parser.add_argument(
        "--colour", type=float, default=0.9,
        help="colour matching strength 0-1 (default 0.9)",
    )
    parser.add_argument(
        "--mask-inset", type=float, default=0.10,
        help="how far inside the face outline the blend stops, 0-0.3 (default 0.10)",
    )
    parser.add_argument(
        "--detect-every", type=int, default=5,
        help="re-detect the face every N frames (default 5, higher is faster)",
    )
    parser.add_argument(
        "--smooth", type=float, default=0.6,
        help="temporal smoothing 0-0.95 (default 0.6); raise if the paste jitters",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="stop after N frames (0 = whole video)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress output",
    )
    parser.add_argument("--version", action="version", version=f"faceswap {__version__}")
    return parser


def _check_inputs(source: str, target: str, output: str) -> list[str]:
    problems = []
    if not Path(source).is_file():
        problems.append(f"source image not found: {source}")
    if not Path(target).is_file():
        problems.append(f"target video not found: {target}")
    elif Path(target).suffix.lower() not in VIDEO_SUFFIXES:
        problems.append(
            f"target {target!r} is not a video by extension; this tool swaps "
            "into video only, not image-to-image"
        )
    if Path(output).suffix.lower() not in VIDEO_SUFFIXES:
        problems.append(
            f"output {output!r} should end in a video extension such as .mp4"
        )
    if Path(source).resolve() == Path(output).resolve():
        problems.append("output would overwrite the source image")
    if Path(target).resolve() == Path(output).resolve():
        problems.append("output would overwrite the input video")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    problems = _check_inputs(args.source, args.target, args.output)
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 2

    try:
        options = SwapOptions(
            blend=args.blend,
            feather=args.feather,
            colour_strength=args.colour,
            mask_inset=args.mask_inset,
            detect_every=max(1, args.detect_every),
            smooth=args.smooth,
            max_frames=max(0, args.max_frames),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        engine = FaceSwapVideo(args.source, options)
    except (IOError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        lm = engine.source_landmarks
        print(f"source face loaded: box={lm.box.as_tuple()} "
              f"eye distance={lm.eye_distance:.1f}px, {len(engine.triangles)} triangles")
        print(f"processing {args.target} -> {args.output} ...")

    try:
        stats = engine.process(args.target, args.output)
    except (IOError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"done: {stats.swapped}/{stats.frames} frames swapped "
            f"({stats.hit_rate:.0%}), {stats.detect_calls} face detections, "
            f"{stats.seconds:.1f}s ({stats.fps:.1f} fps)"
        )
        for warning in stats.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
