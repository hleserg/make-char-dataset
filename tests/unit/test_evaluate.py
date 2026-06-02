"""Tests for make_char_dataset.evaluate (in-stack eval grid).

Pure builders (prompt, LoRA cell plan, Flux graph, contact sheet) + the
orchestration with a GPU-free StubStackSampler. The real ComfyUI sampler is
excluded from coverage.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from make_char_dataset.config import get_settings
from make_char_dataset.evaluate import (
    DEFAULT_SCENES,
    GRID_FILENAME,
    MANIFEST_FILENAME,
    MODES,
    EvalError,
    StubStackSampler,
    assemble_grid,
    build_flux_workflow,
    build_manifest,
    build_prompt,
    load_scenes,
    loras_for_mode,
    make_cells,
    resolve_char_lora,
    run_eval,
)
from make_char_dataset.workspace import Workspace


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


# --- prompt + LoRA cell plan -------------------------------------------------


def test_build_prompt_triggers_first() -> None:
    assert (
        build_prompt("standing in a forest", style_trigger="cmcstyle", char_trigger="kael_char")
        == "cmcstyle, kael_char, standing in a forest"
    )


def test_loras_for_mode() -> None:
    kw = {
        "style_lora": "cmcstyle.safetensors",
        "style_weight": 0.8,
        "char_lora": "kael.safetensors",
        "char_weight": 0.9,
    }
    assert loras_for_mode("base", **kw) == []
    assert loras_for_mode("style", **kw) == [("cmcstyle.safetensors", 0.8)]
    assert loras_for_mode("char", **kw) == [("kael.safetensors", 0.9)]
    assert loras_for_mode("stack", **kw) == [
        ("cmcstyle.safetensors", 0.8),
        ("kael.safetensors", 0.9),
    ]


def test_loras_for_mode_skips_absent_style() -> None:
    # no style LoRA configured -> stack falls back to just the char LoRA
    assert loras_for_mode(
        "stack", style_lora="", style_weight=0.8, char_lora="kael.safetensors", char_weight=0.9
    ) == [("kael.safetensors", 0.9)]


def test_loras_for_mode_unknown_raises() -> None:
    with pytest.raises(EvalError, match="unknown eval mode"):
        loras_for_mode("bogus", style_lora="s", style_weight=1.0, char_lora="c", char_weight=1.0)


def test_make_cells_crosses_prompts_and_modes() -> None:
    cells = make_cells(["p0", "p1"])
    assert len(cells) == 2 * len(MODES)
    assert cells[0].row == 0 and cells[0].mode == "base"
    assert {c.mode for c in cells} == set(MODES)


# --- Flux graph --------------------------------------------------------------


def test_build_flux_workflow_base_has_no_lora() -> None:
    graph = build_flux_workflow(
        prompt="cmcstyle, kael_char, x",
        loras=[],
        seed=42,
        width=512,
        height=512,
        steps=20,
        guidance=3.5,
    )
    assert not any(node["class_type"] == "LoraLoaderModelOnly" for node in graph.values())
    assert graph["40"]["inputs"]["model"] == ["10", 0]  # KSampler reads the bare UNET
    assert graph["40"]["inputs"]["cfg"] == 1.0  # Flux uses FluxGuidance, not cfg
    assert graph["21"]["class_type"] == "FluxGuidance"


def test_build_flux_workflow_stacks_two_loras() -> None:
    graph = build_flux_workflow(
        prompt="p",
        loras=[("cmcstyle.safetensors", 0.8), ("kael.safetensors", 0.9)],
        seed=1,
        width=512,
        height=512,
        steps=20,
        guidance=3.5,
    )
    assert graph["130"]["inputs"]["lora_name"] == "cmcstyle.safetensors"
    assert graph["130"]["inputs"]["model"] == ["10", 0]  # first LoRA off the base
    assert graph["131"]["inputs"]["lora_name"] == "kael.safetensors"
    assert graph["131"]["inputs"]["model"] == ["130", 0]  # second chained onto the first
    assert graph["40"]["inputs"]["model"] == ["131", 0]  # KSampler reads the end of the chain


def test_build_flux_workflow_single_lora() -> None:
    graph = build_flux_workflow(
        prompt="p",
        loras=[("kael.safetensors", 0.9)],
        seed=1,
        width=512,
        height=512,
        steps=20,
        guidance=3.5,
    )
    assert graph["130"]["inputs"]["model"] == ["10", 0]
    assert graph["40"]["inputs"]["model"] == ["130", 0]
    assert "131" not in graph


# --- contact sheet + stub sampler -------------------------------------------


def test_assemble_grid_dimensions_and_missing_cell() -> None:
    scenes = ["scene one", ""]  # empty label exercises the word-wrap empty branch
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    images: dict[tuple[int, str], Image.Image | None] = {(0, m): img for m in MODES}
    # row 1 cells missing -> rendered as a "(no image)" placeholder, not a crash
    sheet = assemble_grid(images, scenes, "kael_char", thumb=64, label_w=80, header_h=20)
    assert sheet.size == (80 + len(MODES) * 64, 20 + len(scenes) * 64)


def test_stub_sampler_is_deterministic_png() -> None:
    sampler = StubStackSampler(size=16)
    g1 = build_flux_workflow(
        prompt="a", loras=[], seed=1, width=64, height=64, steps=1, guidance=1.0
    )
    g2 = build_flux_workflow(
        prompt="b", loras=[], seed=1, width=64, height=64, steps=1, guidance=1.0
    )
    a, b = sampler.sample(g1), sampler.sample(g2)
    assert a == sampler.sample(g1)  # deterministic per graph
    assert a != b  # different prompts -> different tiles
    with Image.open(io.BytesIO(a)) as im:
        assert im.size == (16, 16) and im.format == "PNG"


# --- resolve_char_lora / load_scenes / manifest -----------------------------


def test_resolve_char_lora_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_EVAL_CHAR_LORA_PATH="/loras/kael_v2.safetensors")
    assert resolve_char_lora(settings) == "/loras/kael_v2.safetensors"


def test_resolve_char_lora_derived(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRIGGER_TOKEN="kael", APP_TRAIN_OUTPUT_NAME="")
    assert resolve_char_lora(settings) == "kael.safetensors"


def test_load_scenes_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_scenes(_settings(monkeypatch)) == list(DEFAULT_SCENES)


def test_load_scenes_from_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "prompts.txt"
    f.write_text("scene a\n\n  scene b  \n", encoding="utf-8")
    settings = _settings(monkeypatch, APP_EVAL_PROMPTS_FILE=str(f))
    assert load_scenes(settings) == ["scene a", "scene b"]


def test_load_scenes_empty_file_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("  \n\n", encoding="utf-8")
    settings = _settings(monkeypatch, APP_EVAL_PROMPTS_FILE=str(f))
    with pytest.raises(EvalError, match="empty"):
        load_scenes(settings)


def test_build_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch, APP_TRIGGER_TOKEN="kael_char", APP_STYLE_LORA_PATH="/l/cmcstyle.safetensors"
    )
    manifest = build_manifest(settings, scenes=["s"], char_lora="kael_char.safetensors")
    assert manifest["char_trigger"] == "kael_char"
    assert manifest["style_trigger"] == "cmcstyle"
    assert manifest["style_lora"] == "cmcstyle.safetensors"  # basename only
    assert manifest["modes"] == list(MODES)


# --- run_eval orchestration --------------------------------------------------


def _configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Workspace:
    monkeypatch.setenv("APP_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("APP_TRIGGER_TOKEN", "kael")
    monkeypatch.setenv("APP_STYLE_LORA_PATH", "/loras/cmcstyle.safetensors")
    monkeypatch.setenv("APP_EVAL_CHAR_LORA_PATH", "kael.safetensors")
    get_settings.cache_clear()
    return Workspace(tmp_path / "ws")


def test_run_eval_writes_grid_and_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = _configure(monkeypatch, tmp_path)
    grid = run_eval(sampler=StubStackSampler(size=16))
    assert grid == ws.eval_dir / GRID_FILENAME
    assert grid.is_file()
    with Image.open(grid) as im:
        assert im.size[0] > 0 and im.size[1] > 0
    manifest = json.loads((ws.eval_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["stack"] == "Flux + cmcstyle + <char>_char"
    assert manifest["char_lora"] == "kael.safetensors"


def test_run_eval_dry_run_writes_graphs_no_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ws = _configure(monkeypatch, tmp_path)

    class BoomSampler:
        def sample(self, graph: dict) -> bytes:
            raise AssertionError("sampler must not be called in dry-run")

    grid = run_eval(sampler=BoomSampler(), dry_run=True)
    assert grid.is_file()
    # one graph json per cell = scenes * modes
    graphs = sorted(ws.eval_dir.glob("graph_*.json"))
    assert len(graphs) == len(load_scenes(get_settings())) * len(MODES)
    # the dry-run graphs are valid JSON ComfyUI graphs
    assert json.loads(graphs[0].read_text(encoding="utf-8"))["40"]["class_type"] == "KSampler"
