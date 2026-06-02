"""Tests for make_char_dataset.doctor (training-env validation).

Pure checks + report formatting only; the ``--probe`` torch/CUDA subprocess is
excluded from coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from make_char_dataset.config import Settings, get_settings
from make_char_dataset.doctor import (
    Check,
    _base_is_local_path,
    format_report,
    run_doctor,
    training_checks,
)


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def _fake_aitoolkit(root: Path) -> Path:
    """Create a stand-in ai-toolkit clone (dir + run.py + venv python)."""
    aitk = root / "ai-toolkit"
    (aitk / "venv" / "bin").mkdir(parents=True)
    (aitk / "run.py").write_text("# run\n", encoding="utf-8")
    (aitk / "venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    return aitk


def _by_name(checks: list[Check], name: str) -> Check:
    return next(c for c in checks if c.name == name)


# --- _base_is_local_path ---------------------------------------------------


def test_base_is_local_path() -> None:
    assert _base_is_local_path("") is False
    assert _base_is_local_path("black-forest-labs/FLUX.1-dev") is False
    assert _base_is_local_path("/models/flux1-dev.safetensors") is True  # .safetensors suffix


def test_base_is_local_path_existing(tmp_path: Path) -> None:
    target = tmp_path / "flux"
    target.mkdir()
    assert _base_is_local_path(str(target)) is True


# --- training_checks -------------------------------------------------------


def test_all_green_with_local_base_and_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    aitk = _fake_aitoolkit(tmp_path)
    base = tmp_path / "flux1-dev.safetensors"
    base.write_bytes(b"x")
    ws = tmp_path / "ws"
    dataset = ws / "03_dataset" / "10_kael"
    dataset.mkdir(parents=True)
    (dataset / "img.png").write_bytes(b"x")

    settings = _settings(
        monkeypatch,
        APP_AITOOLKIT_DIR=str(aitk),
        APP_TRAIN_BASE_MODEL=str(base),
        APP_WORKSPACE=str(ws),
        APP_TRIGGER_TOKEN="kael",
        APP_DATASET_REPEATS="10",
    )
    checks = training_checks(settings)
    assert _by_name(checks, "ai-toolkit clone").ok
    assert _by_name(checks, "ai-toolkit run.py").ok
    assert _by_name(checks, "trainer interpreter").ok
    assert _by_name(checks, "Flux base model").ok
    assert _by_name(checks, "kohya dataset").ok
    _, code = format_report(checks)
    assert code == 0


def test_missing_aitoolkit_is_critical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, APP_AITOOLKIT_DIR=str(tmp_path / "absent"))
    checks = training_checks(settings)
    clone = _by_name(checks, "ai-toolkit clone")
    assert clone.ok is False
    assert clone.critical is True
    _, code = format_report(checks)
    assert code == 1


def test_hf_id_without_token_warns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    aitk = _fake_aitoolkit(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    settings = _settings(
        monkeypatch,
        APP_AITOOLKIT_DIR=str(aitk),
        APP_TRAIN_BASE_MODEL="black-forest-labs/FLUX.1-dev",
        APP_HUGGINGFACE_TOKEN="",
        APP_WORKSPACE=str(tmp_path / "ws"),
    )
    token_check = _by_name(training_checks(settings), "HF token for gated base")
    assert token_check.ok is False
    assert token_check.critical is False  # a warning, not a hard failure


def test_hf_id_with_token_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    aitk = _fake_aitoolkit(tmp_path)
    settings = _settings(
        monkeypatch,
        APP_AITOOLKIT_DIR=str(aitk),
        APP_TRAIN_BASE_MODEL="black-forest-labs/FLUX.1-dev",
        HF_TOKEN="hf_secret",
        APP_WORKSPACE=str(tmp_path / "ws"),
    )
    assert _by_name(training_checks(settings), "HF token for gated base").ok is True


def test_missing_dataset_warns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    aitk = _fake_aitoolkit(tmp_path)
    settings = _settings(
        monkeypatch,
        APP_AITOOLKIT_DIR=str(aitk),
        APP_WORKSPACE=str(tmp_path / "ws"),
        APP_TRIGGER_TOKEN="kael",
    )
    dataset_check = _by_name(training_checks(settings), "kohya dataset")
    assert dataset_check.ok is False
    assert dataset_check.critical is False


# --- format_report + run_doctor --------------------------------------------


def test_format_report_marks() -> None:
    report, code = format_report(
        [
            Check("ok one", True, "fine"),
            Check("warn one", False, "soft", critical=False),
            Check("fail one", False, "hard", critical=True),
        ]
    )
    assert "✓ PASS  ok one: fine" in report
    assert "! WARN  warn one: soft" in report
    assert "✗ FAIL  fail one: hard" in report
    assert "RED (1 critical failure(s))" in report
    assert code == 1


def test_format_report_all_green() -> None:
    report, code = format_report([Check("ok", True, "fine")])
    assert "VERDICT: GREEN" in report
    assert code == 0


def test_run_doctor_prints_and_returns_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _settings(monkeypatch, APP_AITOOLKIT_DIR=str(tmp_path / "absent"))
    code = run_doctor(probe=False)
    out = capsys.readouterr().out
    assert "make-char-dataset doctor" in out
    assert code == 1  # missing ai-toolkit clone is critical


def test_run_doctor_probe_appends_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    aitk = _fake_aitoolkit(tmp_path)
    monkeypatch.setattr(
        "make_char_dataset.doctor._probe_torch",
        lambda settings: Check("torch / CUDA probe", True, "2.7.0 True NVIDIA RTX"),
    )
    _settings(monkeypatch, APP_AITOOLKIT_DIR=str(aitk), APP_WORKSPACE=str(tmp_path / "ws"))
    run_doctor(probe=True)
    assert "torch / CUDA probe: 2.7.0 True NVIDIA RTX" in capsys.readouterr().out
