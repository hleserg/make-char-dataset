"""Tests for make_char_dataset.caption (clean dedup + captioner layout).

Hermetic: synthetic images + the StubCaptioner; no onnxruntime, no GPU, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from make_char_dataset.assembly import StubGenerator
from make_char_dataset.caption import (
    CaptionError,
    StubCaptioner,
    Wd14Captioner,
    _make_captioner,
    _resolve_hf_token,
    caption_dataset,
    character_locker_caption,
    clean_variants,
    layout_regularization,
    run_caption,
    run_clean,
    style_tokens,
)
from make_char_dataset.config import get_settings
from make_char_dataset.tagging import StubTagger
from make_char_dataset.workspace import Workspace


def _seed_generated(ws: Workspace, *, size: int = 64) -> None:
    """Write 4 distinct variants + 1 exact duplicate + 1 too-small into 01_generated."""
    gen = ws.generated
    gen.mkdir(parents=True, exist_ok=True)
    stub = StubGenerator(size=size)
    for i in range(4):
        stub.generate(i, gen / f"var_{i:03d}.png")
    stub.generate(0, gen / "var_004.png")  # exact duplicate of var_000
    Image.new("RGB", (8, 8), (5, 5, 5)).save(gen / "var_005.png")  # too small


def _seed_clean(ws: Workspace, n: int = 3, *, size: int = 64) -> None:
    clean = ws.clean
    clean.mkdir(parents=True, exist_ok=True)
    stub = StubGenerator(size=size)
    for i in range(n):
        stub.generate(100 + i, clean / f"var_{i:03d}.png")


# --- Character-Locker caption policy -----------------------------------------


def test_character_locker_caption_strips_geometry_and_style() -> None:
    tags = ["standing", "green coat", "forest background", "brown hair", "1boy", "comic style"]
    caption = character_locker_caption("kael", tags, style=["comic style"])
    assert caption.split(",")[0].strip() == "kael"  # trigger first
    assert "brown hair" not in caption and "1boy" not in caption  # geometry stripped
    assert "comic style" not in caption  # style stripped
    assert "standing" in caption and "green coat" in caption  # content kept


def test_style_tokens_splits_on_commas() -> None:
    assert style_tokens("comic style, ink shading ,  ") == ("comic style", "ink shading")


# --- clean stage -------------------------------------------------------------


def test_clean_dedups_and_size_filters(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_generated(ws)

    result = clean_variants(ws, dedup_distance=6, min_side_px=16)

    assert len(result.kept) == 4  # 4 distinct survive
    assert set(result.rejected) == {"var_004.png", "var_005.png"}  # duplicate + too-small
    assert len(list(ws.clean.glob("var_*.png"))) == 4
    assert (ws.manual_review / "var_004.png").is_file()
    assert (ws.manual_review / "var_005.png").is_file()
    assert len(list(ws.generated.glob("var_*.png"))) == 6  # 01_generated left intact
    assert (ws.clean / ".stage_complete").is_file()


def test_clean_requires_generated(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(CaptionError, match="generate stage first"):
        clean_variants(ws, dedup_distance=6, min_side_px=16)


def test_clean_idempotent_and_force(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_generated(ws)
    first = clean_variants(ws, dedup_distance=6, min_side_px=16)
    sample = ws.clean / first.kept[0]
    mtime = sample.stat().st_mtime_ns

    second = clean_variants(ws, dedup_distance=6, min_side_px=16)  # skip
    assert second.kept == first.kept
    assert sample.stat().st_mtime_ns == mtime  # skip does not rewrite

    forced = clean_variants(ws, dedup_distance=6, min_side_px=16, force=True)
    assert len(forced.kept) == 4


def test_clean_self_heals_on_corrupt_manifest(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_generated(ws)
    clean_variants(ws, dedup_distance=6, min_side_px=16)
    (ws.clean / "clean.json").write_text("{ broken", encoding="utf-8")

    healed = clean_variants(ws, dedup_distance=6, min_side_px=16)  # rebuild
    assert len(healed.kept) == 4


# --- caption stage -----------------------------------------------------------


def test_caption_lays_out_kohya_pairs(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_clean(ws, 3)

    result = caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10)

    assert result.training_dir == ws.dataset / "10_kael"
    images = sorted(result.training_dir.glob("*.png"))
    assert len(images) == 3
    for image in images:
        sidecar = image.with_suffix(".txt")
        assert sidecar.is_file()
        caption = sidecar.read_text(encoding="utf-8")
        assert caption.split(",")[0].strip() == "kael"  # trigger first
        assert "brown hair" not in caption and "comic style" not in caption  # no id/style
    manifest = json.loads((result.training_dir / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["trigger"] == "kael" and manifest["keep_tokens"] == 1


def test_caption_requires_clean(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(CaptionError, match="clean stage first"):
        caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10)


def test_caption_idempotent_and_force(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_clean(ws, 3)
    first = caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10)
    sample = sorted(first.training_dir.glob("*.png"))[0]
    mtime = sample.stat().st_mtime_ns

    caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10)  # skip
    assert sample.stat().st_mtime_ns == mtime

    forced = caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10, force=True)
    assert len(forced.kept) == 3


# --- regularization track ----------------------------------------------------


def test_layout_regularization_has_no_trigger(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    stub = StubGenerator()
    paths = [class_dir / f"c{i}.png" for i in range(3)]
    for i, path in enumerate(paths):
        stub.generate(200 + i, path)

    reg_dir = layout_regularization(ws, paths, class_descriptor="1man ranger", repeats=1)

    assert reg_dir == ws.dataset / "1_1man_ranger"
    txts = sorted(reg_dir.glob("*.txt"))
    assert len(txts) == 3
    for txt in txts:
        assert txt.read_text(encoding="utf-8") == "1man ranger"  # class descriptor, no trigger


# --- settings-driven entry points + purity -----------------------------------


def test_run_clean_and_run_caption_use_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("APP_TRIGGER_TOKEN", "kael")
    monkeypatch.setenv("APP_DATASET_REPEATS", "12")
    monkeypatch.setenv("APP_MIN_SIDE_PX", "16")
    monkeypatch.setenv("APP_CAPTIONER", "stub")  # GPU/network-free captioner for CI
    get_settings.cache_clear()
    ws = Workspace(tmp_path / "ws")
    _seed_generated(ws)

    clean = run_clean()
    assert len(clean.kept) == 4
    result = run_caption()  # settings-selected captioner (stub here)
    assert result.training_dir == ws.dataset / "12_kael"
    assert len(result.kept) == 4


@pytest.mark.parametrize(
    "tag",
    ["hair ornament", "hairband", "hair bow", "closed eyes", "skintight", "green coat"],
)
def test_geometry_filter_keeps_variable_tags(tag: str) -> None:
    # accessory / expression / outfit tags must survive (they should VARY)
    assert tag in character_locker_caption("kael", [tag])


@pytest.mark.parametrize(
    "tag",
    ["brown hair", "long hair", "green eyes", "1boy", "muscular", "scar", "pointy ears"],
)
def test_geometry_filter_strips_identity_tags(tag: str) -> None:
    # immutable identity geometry must be stripped (it binds to the trigger)
    assert tag not in character_locker_caption("kael", [tag, "standing"])


def test_style_filter_strips_curated_and_prompt_tokens() -> None:
    tags = ["standing", "comic style", "monochrome", "sketch", "ink shading"]
    # curated style tags + " style" suffix stripped even with NO style prompt
    default = character_locker_caption("kael", tags)
    assert "comic style" not in default and "monochrome" not in default and "sketch" not in default
    assert "standing" in default
    assert "ink shading" in default  # not a known style tag without a prompt...
    # ...until the user names it in the style prompt
    with_prompt = character_locker_caption("kael", tags, style=["ink shading"])
    assert "ink shading" not in with_prompt


def test_caption_does_not_mutate_clean(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    ws.clean.mkdir(parents=True)
    stub = StubGenerator()
    stub.generate(7, ws.clean / "var_000.png")
    stub.generate(7, ws.clean / "var_001.png")  # byte-identical -> within dedup distance

    result = caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10)

    # caption never re-dedups, so neither file is moved out of the read-only 02_clean
    assert (ws.clean / "var_000.png").is_file() and (ws.clean / "var_001.png").is_file()
    assert len(result.kept) == 2


def test_caption_self_heals_missing_pair(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_clean(ws, 3)
    first = caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10)
    sorted(first.training_dir.glob("*.png"))[0].unlink()  # delete a png, keep the marker

    healed = caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10)  # rebuild
    images = sorted(healed.training_dir.glob("*.png"))
    assert len(images) == 3 and all(image.with_suffix(".txt").is_file() for image in images)


def test_clean_self_heals_missing_kept_file(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_generated(ws)
    first = clean_variants(ws, dedup_distance=6, min_side_px=16)
    (ws.clean / first.kept[0]).unlink()  # delete a survivor; manifest + marker stay

    healed = clean_variants(ws, dedup_distance=6, min_side_px=16)  # rebuild, not stale cache
    assert (ws.clean / healed.kept[0]).is_file()
    assert len(healed.kept) == 4


def test_caption_self_heals_on_corrupt_dataset_manifest(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_clean(ws, 3)
    first = caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10)
    (first.training_dir / "dataset.json").write_text("{ broken", encoding="utf-8")

    healed = caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10)  # rebuild
    assert len(sorted(healed.training_dir.glob("*.png"))) == 3


def test_caption_force_preserves_subdirectories(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_clean(ws, 2)
    result = caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10)
    keep = result.training_dir / "keep"
    keep.mkdir()
    (keep / "x.txt").write_text("x", encoding="utf-8")

    caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10, force=True)
    assert keep.is_dir()  # clear is files-only


def test_keep_tokens_recorded(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_clean(ws, 2)
    result = caption_dataset(ws, StubCaptioner("kael"), trigger="kael", repeats=10, keep_tokens=2)
    manifest = json.loads((result.training_dir / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["keep_tokens"] == 2


def test_pure_core_no_onnxruntime_import() -> None:
    # caption + tagging are imported at module top; importing them must not pull
    # the heavy inference deps onto the CPU/CI path.
    import sys

    for heavy in ("torch", "onnxruntime"):
        assert heavy not in sys.modules


# --- captioner seam: stub / WD14 / VLM selection -----------------------------


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def test_stub_captioner_trigger_first_and_clean(tmp_path: Path) -> None:
    img = tmp_path / "var_007.png"
    StubGenerator().generate(7, img)
    caption = StubCaptioner("kael_char").caption(img)
    assert caption.startswith("kael_char,")  # trigger first
    assert "var_007" in caption  # varies per image
    for banned in ("comic", "painterly", "brown hair", "green eyes"):
        assert banned not in caption  # no style / identity geometry


def test_wd14_captioner_uses_character_locker(tmp_path: Path) -> None:
    img = tmp_path / "var_000.png"
    StubGenerator().generate(0, img)
    caption = Wd14Captioner(StubTagger(), "kael", style=("comic style",)).caption(img)
    assert caption.split(",")[0].strip() == "kael"
    assert "brown hair" not in caption and "comic style" not in caption  # geometry + style out
    assert "standing" in caption  # content kept


def test_make_captioner_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_CAPTIONER="stub", APP_TRIGGER_TOKEN="kael")
    assert isinstance(_make_captioner(settings), StubCaptioner)


def test_make_captioner_wd14(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_CAPTIONER="wd14")
    assert isinstance(_make_captioner(settings), Wd14Captioner)


def test_make_captioner_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    from make_char_dataset.vlm_caption import VlmCaptioner

    settings = _settings(monkeypatch, APP_CAPTIONER="vlm", HF_TOKEN="hf_read_tok")
    assert isinstance(_make_captioner(settings), VlmCaptioner)


def test_make_captioner_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_CAPTIONER="bogus")
    with pytest.raises(CaptionError, match="unknown captioner"):
        _make_captioner(settings)


def test_resolve_hf_token_prefers_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _resolve_hf_token(_settings(monkeypatch, HF_TOKEN="from_env")) == "from_env"


def test_resolve_hf_token_falls_back_to_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    home = tmp_path / "home"
    (home / ".cache" / "huggingface").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "token").write_text("cached_tok\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    settings = _settings(monkeypatch, APP_HUGGINGFACE_TOKEN="")
    assert _resolve_hf_token(settings) == "cached_tok"


def test_resolve_hf_token_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "empty_home")
    settings = _settings(monkeypatch, APP_HUGGINGFACE_TOKEN="")
    with pytest.raises(CaptionError, match="no HF token"):
        _resolve_hf_token(settings)


def test_caption_dataset_propagates_captioner_error(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_clean(ws, 2)

    class BoomCaptioner:
        def caption(self, image: Path) -> str:
            raise RuntimeError("vlm down")

    with pytest.raises(RuntimeError, match="vlm down"):
        caption_dataset(ws, BoomCaptioner(), trigger="kael", repeats=10)
