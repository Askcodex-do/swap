"""The command-line interface: argument handling, validation, and exit codes."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceswap.cli import build_parser, main

from .test_pipeline import make_video, read_video


def test_help_lists_the_tool(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "face-swap" in text
    assert "--source" in text and "--target" in text


def test_requires_source_and_target():
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["-s", "a.jpg"])
    with pytest.raises(SystemExit):
        main(["-s", "a.jpg", "-t", "b.mp4"])


def test_missing_files_report_errors(tmp_path, capsys):
    code = main([
        "-s", str(tmp_path / "nope.jpg"),
        "-t", str(tmp_path / "nope.mp4"),
        "-o", str(tmp_path / "out.mp4"),
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "source image not found" in err
    assert "target video not found" in err


def test_image_target_is_refused(tmp_path, face_obama, face_lena, capsys):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    still = tmp_path / "still.png"
    cv2.imwrite(str(still), face_lena)

    code = main(["-s", str(src), "-t", str(still), "-o", str(tmp_path / "out.mp4")])
    assert code == 2
    assert "not a video" in capsys.readouterr().err


def test_non_video_output_is_refused(tmp_path, face_obama, capsys):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    code = main([
        "-s", str(src), "-t", str(tmp_path / "x.mp4"), "-o", str(tmp_path / "out.txt"),
    ])
    assert code == 2
    assert "video extension" in capsys.readouterr().err


def test_source_without_a_face_exits_with_one(tmp_path, capsys):
    src = tmp_path / "src.png"
    cv2.imwrite(str(src), np.full((200, 200, 3), 127, np.uint8))
    target = tmp_path / "in.mp4"
    open(target, "wb").close()

    code = main(["-s", str(src), "-t", str(target), "-o", str(tmp_path / "out.mp4")])
    assert code == 1
    assert "no face found in the source" in capsys.readouterr().err


def test_full_run_writes_a_video_and_reports(tmp_path, face_obama, face_lena, capsys):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)
    out = tmp_path / "out.mp4"

    code = main(["-s", str(src), "-t", str(target), "-o", str(out)])
    assert code == 0, capsys.readouterr().err
    text = capsys.readouterr().out
    assert "source face loaded" in text
    assert "done:" in text and "100%" in text
    assert out.is_file()
    assert len(read_video(out)) == 24


def test_quiet_suppresses_progress(tmp_path, face_obama, face_lena, capsys):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    code = main([
        "-s", str(src), "-t", str(target),
        "-o", str(tmp_path / "out.mp4"), "--quiet",
    ])
    assert code == 0
    assert capsys.readouterr().out == ""


def test_options_are_plumbed_through(tmp_path, face_obama, face_lena):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    code = main([
        "-s", str(src), "-t", str(target), "-o", str(tmp_path / "out.mp4"),
        "--blend", "poisson", "--max-frames", "4", "--detect-every", "2",
        "--quiet",
    ])
    assert code == 0
    assert len(read_video(tmp_path / "out.mp4")) == 4


def test_invalid_option_value_exits_with_two(tmp_path, face_obama, face_lena, capsys):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    code = main([
        "-s", str(src), "-t", str(target),
        "-o", str(tmp_path / "o.mp4"), "--smooth", "1.0",
    ])
    assert code == 2
    assert "smooth" in capsys.readouterr().err


def test_refuses_to_overwrite_the_source(tmp_path, face_obama, face_lena, capsys):
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), face_obama)
    target = make_video(tmp_path / "in.mp4", face_lena)

    code = main(["-s", str(src), "-t", str(target), "-o", str(src)])
    assert code == 2
    assert "overwrite" in capsys.readouterr().err


def test_module_entry_point_is_wired_up():
    import faceswap.__main__ as entry

    assert callable(entry.main)
