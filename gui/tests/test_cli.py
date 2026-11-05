from __future__ import annotations

from pathlib import Path

import pytest

from polytaxo_gui.cli import main, validate_dataset_path


def test_main_requires_dataset(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "usage: polytaxo-gui" in captured.err
    assert "one of the arguments --dataset is required" in captured.err


def test_validate_dataset_path_rejects_missing_directory(tmp_path: Path):
    missing = tmp_path / "missing"

    with pytest.raises(ValueError, match="does not exist"):
        validate_dataset_path(missing)


def test_validate_dataset_path_rejects_file(tmp_path: Path):
    file_path = tmp_path / "dataset.txt"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        validate_dataset_path(file_path)
