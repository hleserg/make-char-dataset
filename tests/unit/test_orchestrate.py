"""Tests for make_char_dataset.orchestrate (run-all dispatch + resume + flags).

Hermetic: a synthetic passport export run through the real stages on the stub
backend (APP_BACKEND=stub) — no GPU, no onnxruntime, no ComfyUI, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from make_char_dataset.assembly import StubGenerator
from make_char_dataset.config import get_settings
from make_char_dataset.ingest import PASSPORT_STEPS
from make_char_dataset.orchestrate import run_all, run_stage
from make_char_dataset.workspace import Workspace


def _make_export(root: Path) -> Path:
    export = root / "passport"
    refs = export / "refs"
    refs.mkdir(parents=True)
    generator = StubGenerator(size=64)
    steps: dict[str, dict[str, str]] = {}
    for index, step in enumerate(PASSPORT_STEPS):
        generator.generate(500 + index, refs / f"{step}.png")
        steps[step] = {"approved_path": f"refs/{step}.png"}
    state = {
        "character_id": "conan",
        "current_step": None,
        "steps": steps,
        "character_table": {"archetype": "barbarian"},
        "prompt_layers": {"face": "angular face"},
        "base_outfit": {"prompt": "fur cloak"},
    }
    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return export


def _configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **extra: str) -> Workspace:
    monkeypatch.setenv("APP_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("APP_BACKEND", "stub")
    monkeypatch.setenv("APP_VARIANTS_PER_CHARACTER", "6")
    monkeypatch.setenv("APP_MIN_SIDE_PX", "16")
    monkeypatch.setenv("APP_TRIGGER_TOKEN", "conan")
    monkeypatch.setenv("APP_DATASET_REPEATS", "10")
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return Workspace(tmp_path / "ws")


def test_run_all_runs_every_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export = _make_export(tmp_path)
    ws = _configure(monkeypatch, tmp_path)

    results = run_all(export)

    assert set(results) == {"import", "generate", "clean", "caption"}
    for stage_dir in (ws.passport_import, ws.generated, ws.clean):
        assert (stage_dir / ".stage_complete").is_file()
    training = ws.dataset / "10_conan"
    images = sorted(training.glob("*.png"))
    assert images and all(i.with_suffix(".txt").is_file() for i in images)


def test_run_all_resumable_and_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export = _make_export(tmp_path)
    ws = _configure(monkeypatch, tmp_path)
    run_all(export)
    training = ws.dataset / "10_conan"
    sample = sorted(training.glob("*.png"))[0]
    mtime = sample.stat().st_mtime_ns

    run_all(export)  # resumable: every stage cached, nothing rewritten
    assert sample.stat().st_mtime_ns == mtime

    run_all(export, force=True)  # force rebuilds
    assert sorted(training.glob("*.png"))


def test_run_all_respects_stage_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export = _make_export(tmp_path)
    ws = _configure(monkeypatch, tmp_path, APP_RUN_CAPTION="false")

    results = run_all(export)

    assert "caption" not in results and "clean" in results
    assert not (ws.dataset / "10_conan").exists()


def test_run_stage_dispatches_each_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export = _make_export(tmp_path)
    ws = _configure(monkeypatch, tmp_path)

    run_stage("import", export_dir=export)
    assert (ws.passport_import / ".stage_complete").is_file()
    run_stage("generate")
    assert (ws.generated / ".stage_complete").is_file()
    run_stage("clean")
    assert (ws.clean / ".stage_complete").is_file()
    result = run_stage("caption")
    assert result.training_dir.name == "10_conan"


def test_run_all_runs_train_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export = _make_export(tmp_path)
    _configure(monkeypatch, tmp_path, APP_RUN_TRAIN="true")
    # Stub the (heavy) trainer entry so run-all's train branch runs without a GPU.
    monkeypatch.setattr("make_char_dataset.orchestrate.run_train", lambda *, force=False: "LORA")

    results = run_all(export)

    assert set(results) == {"import", "generate", "clean", "caption", "train"}
    assert results["train"] == "LORA"


def test_run_stage_train_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}

    def fake_run_train(*, force: bool = False) -> str:
        seen["force"] = force
        return "LORA_PATH"

    monkeypatch.setattr("make_char_dataset.orchestrate.run_train", fake_run_train)
    assert run_stage("train", force=True) == "LORA_PATH"
    assert seen["force"] is True


def test_run_stage_import_requires_export() -> None:
    with pytest.raises(ValueError, match="requires an export_dir"):
        run_stage("import")


def test_run_stage_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        run_stage("bogus")
