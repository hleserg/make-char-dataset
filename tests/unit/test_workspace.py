"""Tests for make_char_dataset.workspace."""

from __future__ import annotations

from pathlib import Path

from make_char_dataset.workspace import Workspace


def test_stage_paths_derive_from_root() -> None:
    ws = Workspace(Path("/tmp/ws"))
    assert ws.passport_import == Path("/tmp/ws/00_passport_import")
    assert ws.generated == Path("/tmp/ws/01_generated")
    assert ws.clean == Path("/tmp/ws/02_clean")
    assert ws.dataset == Path("/tmp/ws/03_dataset")
    assert ws.manual_review == Path("/tmp/ws/manual_review")


def test_training_dir_is_kohya_named() -> None:
    ws = Workspace(Path("/tmp/ws"))
    assert ws.training_dir(10, "conan") == Path("/tmp/ws/03_dataset/10_conan")


def test_ensure_base_creates_input_and_review(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.ensure_base()
    assert ws.root.is_dir()
    assert ws.passport_import.is_dir()
    assert ws.manual_review.is_dir()
