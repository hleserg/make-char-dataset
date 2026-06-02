"""Tests for make_char_dataset.ingest (the passport-import adapter).

Everything runs on a synthetic create-char-passport export built in ``tmp_path``,
so the suite is hermetic — no GPU, no network, no real fixture required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from make_char_dataset.config import get_settings
from make_char_dataset.ingest import (
    PASSPORT_STEPS,
    IngestError,
    _resolve,
    aspect_bucket,
    import_passport,
    read_state,
    run_import,
)
from make_char_dataset.workspace import Workspace


def _png(
    path: Path,
    size: tuple[int, int] = (1024, 1024),
    color: tuple[int, int, int] = (120, 90, 60),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="PNG")


def _make_export(
    root: Path,
    *,
    current_step: str | None = None,
    drop: tuple[str, ...] = (),
    sizes: dict[str, tuple[int, int]] | None = None,
    style: bool = False,
    outfit: bool = False,
    props: bool = False,
    emotion_refs: bool = False,
    unreferenced: bool = False,
    face_ref_override: str | None = None,
) -> Path:
    """Build a synthetic passport export and return its folder."""
    export = root / "kael-thornwood"
    refs = export / "refs"
    refs.mkdir(parents=True)
    sizes = sizes or {}
    steps: dict[str, Any] = {}
    for step in PASSPORT_STEPS:
        if step in drop:
            continue
        _png(refs / f"{step}.png", size=sizes.get(step, (1024, 1024)))
        rel = f"refs/{step}.png"
        steps[step] = {"approved_path": rel, "last_path": rel}
    if face_ref_override is not None:
        steps["passport_face"] = {
            "approved_path": face_ref_override,
            "last_path": face_ref_override,
        }

    state: dict[str, Any] = {
        "character_id": "kael-thornwood",
        "current_step": current_step,
        "steps": steps,
        "character_table": {"name": "Kael Thornwood", "archetype": "ranger"},
        "prompt_layers": {"face": "angular face", "body": "lean build", "outfit": "", "style": ""},
        "base_outfit": {"prompt": "green ranger coat", "ref": "refs/passport_body.png"},
        "style_ref": None,
        "outfits": [],
        "props": [],
        "emotions": {"base_emotion": {"ref": None}, "items": [{"ref": None, "value": "neutral"}]},
    }
    if style:
        _png(refs / "style.png")
        state["style_ref"] = "refs/style.png"
    if outfit:
        _png(refs / "outfit_rogue_front.png")
        state["outfits"] = [{"id": "rogue", "refs": {"front": "refs/outfit_rogue_front.png"}}]
    if props:
        _png(refs / "prop_sword_0.png")
        state["props"] = [{"id": "sword", "shots": [{"ref": "refs/prop_sword_0.png"}]}]
    if emotion_refs:
        _png(refs / "emotion_base.png")
        _png(refs / "emotion_0.png")
        state["emotions"] = {
            "base_emotion": {"ref": "refs/emotion_base.png"},
            "items": [{"ref": "refs/emotion_0.png", "value": "smiling"}],
        }
    if unreferenced:
        _png(refs / "junk.png")  # present on disk but NOT referenced by state.json

    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return export


def test_happy_path_imports_five_anchors(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    ws = Workspace(tmp_path / "ws")

    result = import_passport(export, ws)

    assert result.character_id == "kael-thornwood"
    assert len(result.anchors) == 5
    assert len(list(ws.passport_import.glob("*.png"))) == 5
    assert (ws.passport_import / ".stage_complete").is_file()

    manifest = json.loads((ws.passport_import / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["doctrine"] == "conditioning-only"
    assert manifest["character_id"] == "kael-thornwood"
    assert len(manifest["anchors"]) == 5
    # 00_passport_import/ is self-contained for the generate stage: textual identity
    # travels in the manifest, so [3] never reaches back to the ccp export.
    assert manifest["identity"]["character_table"]["name"] == "Kael Thornwood"
    assert manifest["identity"]["base_outfit"]["prompt"] == "green ranger coat"
    assert result.identity["prompt_layers"]["face"] == "angular face"


def test_passport_role_mapping(tmp_path: Path) -> None:
    result = import_passport(_make_export(tmp_path), Workspace(tmp_path / "ws"))
    assert {a.key: a.role for a in result.anchors} == {
        "passport_face": "face",
        "passport_profile": "face",
        "passport_body": "body",
        "passport_back": "body",
        "passport_3q": "body",
    }


def test_not_ready_raises(tmp_path: Path) -> None:
    export = _make_export(tmp_path, current_step="passport_3q")
    with pytest.raises(IngestError, match="not ready"):
        import_passport(export, Workspace(tmp_path / "ws"))


def test_missing_frame_raises(tmp_path: Path) -> None:
    export = _make_export(tmp_path, drop=("passport_back",))
    with pytest.raises(IngestError, match="passport_back"):
        import_passport(export, Workspace(tmp_path / "ws"))


def test_unresolved_pointer_raises(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    (export / "refs" / "passport_body.png").unlink()  # pointer stays, file gone
    with pytest.raises(IngestError, match="passport_body"):
        import_passport(export, Workspace(tmp_path / "ws"))


def test_path_traversal_rejected(tmp_path: Path) -> None:
    _png(tmp_path / "secret.png")
    export = _make_export(tmp_path, face_ref_override="../secret.png")
    with pytest.raises(IngestError, match="escapes"):
        import_passport(export, Workspace(tmp_path / "ws"))


def test_optional_style_and_outfit_imported(tmp_path: Path) -> None:
    export = _make_export(tmp_path, style=True, outfit=True)
    result = import_passport(export, Workspace(tmp_path / "ws"))
    roles = [a.role for a in result.anchors]
    assert len(result.anchors) == 7
    assert roles.count("style") == 1
    assert roles.count("outfit") == 1
    assert {"style", "outfit_rogue_front"} <= {a.key for a in result.anchors}


def test_props_and_emotions_imported(tmp_path: Path) -> None:
    export = _make_export(tmp_path, props=True, emotion_refs=True)
    result = import_passport(export, Workspace(tmp_path / "ws"))
    roles = [a.role for a in result.anchors]
    assert roles.count("prop") == 1
    assert roles.count("emotion") == 2  # base + one item with a ref
    assert {"prop_sword_shot_1", "emotion_base", "emotion_0"} <= {a.key for a in result.anchors}


def test_optional_missing_file_raises(tmp_path: Path) -> None:
    export = _make_export(tmp_path, style=True)
    (export / "refs" / "style.png").unlink()
    with pytest.raises(IngestError, match="style reference"):
        import_passport(export, Workspace(tmp_path / "ws"))


def test_ignores_unreferenced_files(tmp_path: Path) -> None:
    result = import_passport(_make_export(tmp_path, unreferenced=True), Workspace(tmp_path / "ws"))
    assert len(result.anchors) == 5
    assert "refs/junk.png" not in {a.source for a in result.anchors}


def test_minimal_state_optional_keys_absent(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    state = json.loads((export / "state.json").read_text(encoding="utf-8"))
    for key in ("outfits", "props", "emotions", "style_ref"):
        state.pop(key, None)
    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")

    result = import_passport(export, Workspace(tmp_path / "ws"))
    assert len(result.anchors) == 5


def test_idempotent_skip_and_force(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    ws = Workspace(tmp_path / "ws")

    first = import_passport(export, ws)
    sample = ws.passport_import / first.anchors[0].dest
    mtime = sample.stat().st_mtime_ns

    second = import_passport(export, ws)  # marker present -> skip, returns manifest
    assert [a.dest for a in second.anchors] == [a.dest for a in first.anchors]
    assert second.identity == first.identity  # identity round-trips through the manifest
    assert sample.stat().st_mtime_ns == mtime  # a skip must not rewrite anchor files

    stray = ws.passport_import / "stray.png"
    stray.write_bytes(b"x")
    forced = import_passport(export, ws, force=True)
    assert not stray.exists()  # force clears the stage folder first
    assert len(forced.anchors) == 5


def test_dest_name_collision_disambiguated(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    (export / "refs" / "a").mkdir()
    (export / "refs" / "b").mkdir()
    _png(export / "refs" / "a" / "front.png")
    _png(export / "refs" / "b" / "front.png")
    state = json.loads((export / "state.json").read_text(encoding="utf-8"))
    state["outfits"] = [
        {"id": "x", "refs": {"front": "refs/a/front.png"}},
        {"id": "y", "refs": {"front": "refs/b/front.png"}},
    ]
    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")

    result = import_passport(export, Workspace(tmp_path / "ws"))
    outfit_dests = sorted(a.dest for a in result.anchors if a.role == "outfit")
    assert outfit_dests == ["outfit_front.png", "outfit_front_1.png"]


@pytest.mark.parametrize(
    ("width", "height", "label", "dims"),
    [
        (1024, 1024, "1:1", (1024, 1024)),
        (1200, 900, "4:3", (1024, 768)),
        (900, 1200, "3:4", (768, 1024)),
        (256, 256, "1:1", (256, 256)),  # no upscale
        (400, 300, "4:3", (400, 300)),  # 4:3 but small -> shrink bucket, no upscale
    ],
)
def test_aspect_bucket_classification(
    width: int, height: int, label: str, dims: tuple[int, int]
) -> None:
    assert aspect_bucket(width, height, 1024) == (label, dims)


def test_aspect_bucket_invalid_size_raises() -> None:
    with pytest.raises(IngestError):
        aspect_bucket(0, 100, 1024)


def test_normalized_output_dims(tmp_path: Path) -> None:
    export = _make_export(tmp_path, sizes={"passport_body": (1200, 900)})
    ws = Workspace(tmp_path / "ws")
    result = import_passport(export, ws)

    body = next(a for a in result.anchors if a.key == "passport_body")
    assert body.bucket == "1024x768"
    with Image.open(ws.passport_import / body.dest) as image:
        assert image.size == (1024, 768)


def test_run_import_uses_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export = _make_export(tmp_path)
    monkeypatch.setenv("APP_WORKSPACE", str(tmp_path / "cfgws"))
    monkeypatch.setenv("APP_TARGET_SIDE", "512")
    get_settings.cache_clear()

    result = run_import(export)

    assert result.import_dir == Workspace(tmp_path / "cfgws").passport_import
    assert result.import_dir.is_dir()
    assert all(a.bucket == "512x512" for a in result.anchors)


def test_read_state_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="cannot read"):
        read_state(tmp_path)


def test_read_state_invalid_json_raises(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(IngestError, match="invalid JSON"):
        read_state(tmp_path)


def test_read_state_non_object_raises(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("[]", encoding="utf-8")
    with pytest.raises(IngestError, match="must contain a JSON object"):
        read_state(tmp_path)


def test_no_steps_table_raises(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(json.dumps({"current_step": None}), encoding="utf-8")
    with pytest.raises(IngestError, match="no 'steps'"):
        import_passport(tmp_path, Workspace(tmp_path / "ws"))


def test_malformed_optional_groups_are_skipped(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    state = json.loads((export / "state.json").read_text(encoding="utf-8"))
    state["outfits"] = ["not-a-dict", {"id": "x"}, {"id": "y", "refs": {"front": ""}}]
    state["props"] = ["nope", {"id": "p", "shots": ["bad", {"ref": ""}]}]
    state["emotions"] = {"base_emotion": "oops", "items": ["bad", {"ref": ""}]}
    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")

    result = import_passport(export, Workspace(tmp_path / "ws"))
    assert len(result.anchors) == 5  # every malformed / empty optional pointer is skipped


def test_force_preserves_subdirectories(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    ws = Workspace(tmp_path / "ws")
    import_passport(export, ws)

    keep = ws.passport_import / "keep_dir"
    keep.mkdir()
    (keep / "inner.txt").write_text("x", encoding="utf-8")

    import_passport(export, ws, force=True)
    assert keep.is_dir()  # _clear_owned removes files only, never subdirectories


def test_resolve_rejects_empty_pointer(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="empty reference path"):
        _resolve(tmp_path, "")


def test_import_does_not_write_outside_its_stage_folder(tmp_path: Path) -> None:
    """Conditioning-only at this stage: anchors land in 00_passport_import/ only."""
    ws = Workspace(tmp_path / "ws")
    import_passport(_make_export(tmp_path), ws)
    assert ws.passport_import.is_dir()
    assert not ws.generated.exists()
    assert not ws.clean.exists()
    assert not ws.dataset.exists()  # never copied into the training set


def test_pointer_walk_follows_renamed_target(tmp_path: Path) -> None:
    """Proves we follow state.json pointers, not the conventional filename."""
    export = _make_export(tmp_path)
    _png(export / "refs" / "custom_front.png")
    state = json.loads((export / "state.json").read_text(encoding="utf-8"))
    state["steps"]["passport_face"]["approved_path"] = "refs/custom_front.png"
    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")

    result = import_passport(export, Workspace(tmp_path / "ws"))
    face = next(a for a in result.anchors if a.key == "passport_face")
    assert face.source == "refs/custom_front.png"  # pointer wins over the stem


def test_outfit_details_refs_are_imported(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    _png(export / "refs" / "outfit_rogue_detail.png")
    state = json.loads((export / "state.json").read_text(encoding="utf-8"))
    state["outfits"] = [{"id": "rogue", "details": [{"ref": "refs/outfit_rogue_detail.png"}]}]
    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")

    result = import_passport(export, Workspace(tmp_path / "ws"))
    assert "outfit_rogue_detail_0" in {a.key for a in result.anchors}
    assert sum(a.role == "outfit" for a in result.anchors) == 1


def test_duplicate_optional_ids_get_unique_keys(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    _png(export / "refs" / "a.png")
    _png(export / "refs" / "b.png")
    state = json.loads((export / "state.json").read_text(encoding="utf-8"))
    state["outfits"] = [
        {"id": "dup", "refs": {"front": "refs/a.png"}},
        {"id": "dup", "refs": {"front": "refs/b.png"}},
    ]
    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")

    result = import_passport(export, Workspace(tmp_path / "ws"))
    outfit_keys = [a.key for a in result.anchors if a.role == "outfit"]
    assert sorted(outfit_keys) == ["outfit_dup_front", "outfit_dup_front_1"]  # no collision


def test_character_id_falls_back_to_export_dir_name(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    state = json.loads((export / "state.json").read_text(encoding="utf-8"))
    state.pop("character_id", None)
    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")

    result = import_passport(export, Workspace(tmp_path / "ws"))
    assert result.character_id == "kael-thornwood"  # the export folder name


def test_non_rgb_source_is_normalized_to_rgb(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    Image.new("P", (1024, 1024)).save(export / "refs" / "passport_face.png")  # palette mode

    result = import_passport(export, Workspace(tmp_path / "ws"))
    face = next(a for a in result.anchors if a.key == "passport_face")
    with Image.open(result.import_dir / face.dest) as image:
        assert image.mode == "RGB"


def test_corrupt_image_raises_ingest_error(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    (export / "refs" / "passport_body.png").write_bytes(b"not a real image")

    with pytest.raises(IngestError, match="not a readable image"):
        import_passport(export, Workspace(tmp_path / "ws"))


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    _png(outside)
    export = _make_export(tmp_path)
    link = export / "refs" / "link.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")
    state = json.loads((export / "state.json").read_text(encoding="utf-8"))
    state["steps"]["passport_face"]["approved_path"] = "refs/link.png"
    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(IngestError, match="escapes"):
        import_passport(export, Workspace(tmp_path / "ws"))


def test_skip_self_heals_when_anchor_file_deleted(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    ws = Workspace(tmp_path / "ws")
    first = import_passport(export, ws)

    (ws.passport_import / first.anchors[0].dest).unlink()  # marker stays, file gone
    healed = import_passport(export, ws)  # not force -> must rebuild, not lie

    assert all((ws.passport_import / a.dest).is_file() for a in healed.anchors)
    assert len(healed.anchors) == 5


def test_skip_self_heals_on_corrupt_manifest(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    ws = Workspace(tmp_path / "ws")
    import_passport(export, ws)
    (ws.passport_import / "manifest.json").write_text("{ broken", encoding="utf-8")

    healed = import_passport(export, ws)  # not force -> rebuild over corrupt manifest
    assert len(healed.anchors) == 5
    assert json.loads((ws.passport_import / "manifest.json").read_text(encoding="utf-8"))


def test_pure_core_no_heavy_imports() -> None:
    import sys

    import make_char_dataset.ingest  # noqa: F401

    for heavy in ("torch", "diffusers", "onnxruntime", "create_char_passport"):
        assert heavy not in sys.modules
