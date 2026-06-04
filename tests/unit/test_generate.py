"""Tests for make_char_dataset.generate (planner + stub backend + stage) and the
ComfyUI workflow builder. All hermetic: no GPU, no ComfyUI server, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from make_char_dataset.backends.comfy import (
    ComfyBackend,
    FluxImg2ImgBackend,
    _combine_prompt,
    _extract_error,
)
from make_char_dataset.config import get_settings
from make_char_dataset.generate import (
    SHOT_CLASSES,
    GenerationError,
    StubBackend,
    VariantSpec,
    class_counts,
    generate_dataset,
    plan_variants,
    run_generate,
)
from make_char_dataset.workspace import Workspace


def _seed_import(ws: Workspace) -> dict:
    """Write a minimal 00_passport_import/ manifest + anchor images (as ingest would)."""
    imp = ws.passport_import
    imp.mkdir(parents=True, exist_ok=True)
    anchors = []
    groups = {
        "face": ["face_passport_face", "face_passport_profile"],
        "body": ["body_passport_body", "body_passport_back", "body_passport_3q"],
    }
    for role, names in groups.items():
        for name in names:
            dest = f"{name}.png"
            Image.new("RGB", (16, 16), (10, 20, 30)).save(imp / dest)
            anchors.append(
                {
                    "role": role,
                    "key": name,
                    "source": f"refs/{name}.png",
                    "dest": dest,
                    "bucket": "1024x1024",
                    "width": 1024,
                    "height": 1024,
                }
            )
    manifest = {
        "character_id": "kael",
        "doctrine": "conditioning-only",
        "identity": {
            "character_table": {"archetype": "wandering ranger"},
            "base_outfit": {"prompt": "green hooded coat"},
            "prompt_layers": {
                "face": "angular face with a pale scar",
                "body": "lean athletic build",
                "style": "gritty ink shading",
            },
        },
        "anchors": anchors,
    }
    (imp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_class_counts_exact_recipe_split() -> None:
    # Pin the exact 40/30/20/10 split, not just monotonicity (an equal split is monotone).
    assert class_counts(40) == {"full_body": 16, "mid_shot": 12, "portrait": 8, "angle": 4}
    # Non-divisible total locks the largest-remainder tie-break.
    assert class_counts(36) == {"full_body": 14, "mid_shot": 11, "portrait": 7, "angle": 4}
    assert sum(class_counts(37).values()) == 37
    assert class_counts(0) == {cls: 0 for cls in SHOT_CLASSES}


def test_class_counts_negative_raises() -> None:
    with pytest.raises(GenerationError):
        class_counts(-1)


def test_plan_count_and_class_distribution() -> None:
    manifest = {
        "anchors": [{"role": "body", "dest": "b.png"}, {"role": "face", "dest": "f.png"}],
        "identity": {},
    }
    specs = plan_variants(manifest, 10)
    assert len(specs) == 10
    counts = {cls: sum(s.shot_class == cls for s in specs) for cls in SHOT_CLASSES}
    assert counts == class_counts(10)
    assert [s.index for s in specs] == list(range(10))


def test_plan_role_fanout_portraits_use_face_anchors() -> None:
    manifest = {
        "anchors": [
            {"role": "face", "dest": "face_a.png"},
            {"role": "body", "dest": "body_a.png"},
        ],
        "identity": {},
    }
    specs = plan_variants(manifest, 20)
    for spec in specs:
        if spec.shot_class == "portrait":
            assert spec.anchor_dest == "face_a.png"
        else:
            assert spec.anchor_dest == "body_a.png"


def test_plan_is_deterministic() -> None:
    manifest = {"anchors": [{"role": "body", "dest": "b.png"}], "identity": {}}
    assert plan_variants(manifest, 12, seed=7) == plan_variants(manifest, 12, seed=7)


def test_plan_prompt_carries_identity_and_is_style_free() -> None:
    manifest = {
        "anchors": [{"role": "body", "dest": "b.png"}],
        "identity": {
            "character_table": {"archetype": "wandering ranger"},
            "base_outfit": {"prompt": "green hooded coat"},
            "prompt_layers": {
                "face": "angular face with a pale scar",
                "body": "lean athletic build",
                "style": "gritty ink shading",  # must never leak into the prompt
            },
        },
    }
    specs = plan_variants(manifest, 8)
    for spec in specs:
        assert "wandering ranger" in spec.prompt  # archetype
        assert "angular face with a pale scar" in spec.prompt  # prompt_layers.face folded
        assert "lean athletic build" in spec.prompt  # prompt_layers.body folded
        assert "green hooded coat" in spec.prompt  # outfit
        assert "ink shading" not in spec.prompt  # prompt_layers.style excluded (style-free)
        assert "style" not in spec.prompt.lower()  # style is applied at the backend, not here


def test_plan_no_anchors_raises() -> None:
    with pytest.raises(GenerationError, match="no anchors"):
        plan_variants({"anchors": []}, 5)


def test_stub_backend_distinct_and_deterministic(tmp_path: Path) -> None:
    backend = StubBackend()
    spec_a = VariantSpec(0, "portrait", "f.png", "p", 1, 0.75, "openpose", "64x64")
    spec_b = VariantSpec(1, "portrait", "f.png", "p", 2, 0.75, "openpose", "64x64")
    a1, a2, b = tmp_path / "a1.png", tmp_path / "a2.png", tmp_path / "b.png"
    backend.render(spec_a, tmp_path / "anchor.png", a1)
    backend.render(spec_a, tmp_path / "anchor.png", a2)
    backend.render(spec_b, tmp_path / "anchor.png", b)
    assert a1.read_bytes() == a2.read_bytes()  # same seed -> identical
    assert a1.read_bytes() != b.read_bytes()  # different seed -> distinct


def test_generate_dataset_end_to_end(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_import(ws)

    result = generate_dataset(ws, StubBackend(), count=12)

    assert len(result.variants) == 12
    images = sorted(p.name for p in ws.generated.glob("var_*.png"))
    assert len(images) == 12
    assert (ws.generated / ".stage_complete").is_file()

    plan = json.loads((ws.generated / "plan.json").read_text(encoding="utf-8"))
    assert plan["doctrine"] == "conditioning-only"
    assert len(plan["variants"]) == 12
    assert "style" not in json.dumps(plan["specs"]).lower()  # no style tokens persisted


def test_generate_idempotent_and_force(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_import(ws)
    first = generate_dataset(ws, StubBackend(), count=8)
    sample = ws.generated / first.variants[0].image
    mtime = sample.stat().st_mtime_ns

    second = generate_dataset(ws, StubBackend(), count=8)  # skip
    assert [v.image for v in second.variants] == [v.image for v in first.variants]
    assert sample.stat().st_mtime_ns == mtime  # skip does not rewrite

    forced = generate_dataset(ws, StubBackend(), count=8, force=True)
    assert len(forced.variants) == 8


def test_generate_self_heals_missing_variant(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_import(ws)
    first = generate_dataset(ws, StubBackend(), count=6)
    (ws.generated / first.variants[0].image).unlink()  # marker stays, image gone

    healed = generate_dataset(ws, StubBackend(), count=6)  # not force -> rebuild
    assert all((ws.generated / v.image).is_file() for v in healed.variants)
    assert len(healed.variants) == 6


def test_generate_requires_import(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(GenerationError, match="import stage first"):
        generate_dataset(ws, StubBackend(), count=4)


def test_run_generate_uses_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("APP_VARIANTS_PER_CHARACTER", "9")
    monkeypatch.setenv("APP_BACKEND", "stub")
    get_settings.cache_clear()
    _seed_import(Workspace(tmp_path / "ws"))

    result = run_generate()
    assert len(result.variants) == 9


def test_each_variant_renders_from_its_own_anchor(tmp_path: Path) -> None:
    """Identity-constant: every render is an img2img from the character's own anchor."""
    ws = Workspace(tmp_path / "ws")
    _seed_import(ws)
    calls: list[tuple[str, str, Path]] = []

    class Recording:
        def render(self, spec: VariantSpec, anchor_path: Path, out_path: Path) -> None:
            calls.append((spec.shot_class, spec.anchor_dest, anchor_path))
            StubBackend().render(spec, anchor_path, out_path)

    generate_dataset(ws, Recording(), count=20)
    for shot_class, anchor_dest, anchor_path in calls:
        assert anchor_path == ws.passport_import / anchor_dest  # from its own anchor
        assert anchor_path.is_file()
        assert anchor_dest.startswith("face_" if shot_class == "portrait" else "body_")


def test_select_backend_forwards_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import make_char_dataset.backends.comfy as comfy_mod
    from make_char_dataset.generate import _select_backend

    captured: dict[str, object] = {}

    class FakeComfy:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(comfy_mod, "ComfyBackend", FakeComfy)

    class Settings:
        backend = "comfyui"
        comfy_url = "http://comfy:8188"
        base_model = "sdxl"
        style_lora_path = "/models/lora/comic.safetensors"
        style_lora_weight = 0.8
        style_prompt = "comic style"
        controlnet_model = "control_openpose.safetensors"

    _select_backend(Settings())
    assert captured["controlnet_model"] == "control_openpose.safetensors"  # the wiring regression
    assert captured["style_lora_path"] == "/models/lora/comic.safetensors"
    assert captured["base_url"] == "http://comfy:8188"


def test_plan_anchor_missing_dest_raises() -> None:
    with pytest.raises(GenerationError, match="missing its 'dest'"):
        plan_variants({"anchors": [{"role": "body"}], "identity": {}}, 4)


def test_plan_falls_back_when_role_pool_empty() -> None:
    # Only an outfit anchor: face/body pools are empty -> fall back to any anchor.
    manifest = {"anchors": [{"role": "outfit", "dest": "o.png"}], "identity": {}}
    specs = plan_variants(manifest, 8)
    assert all(s.anchor_dest == "o.png" for s in specs)


def test_generate_malformed_manifest_raises(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_import(ws)
    (ws.passport_import / "manifest.json").write_text("{ broken", encoding="utf-8")
    with pytest.raises(GenerationError, match="unreadable"):
        generate_dataset(ws, StubBackend(), count=4)
    (ws.passport_import / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(GenerationError, match="malformed"):
        generate_dataset(ws, StubBackend(), count=4)


def test_generate_self_heals_on_corrupt_plan(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_import(ws)
    generate_dataset(ws, StubBackend(), count=6)
    (ws.generated / "plan.json").write_text("{ broken", encoding="utf-8")

    healed = generate_dataset(ws, StubBackend(), count=6)  # not force -> rebuild
    assert len(healed.variants) == 6


def test_force_preserves_subdirectories(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_import(ws)
    generate_dataset(ws, StubBackend(), count=6)
    keep = ws.generated / "keep"
    keep.mkdir()
    (keep / "x.txt").write_text("x", encoding="utf-8")

    generate_dataset(ws, StubBackend(), count=6, force=True)
    assert keep.is_dir()  # clear is files-only


def test_rebuild_clears_orphans_from_interrupted_run(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    _seed_import(ws)
    ws.generated.mkdir(parents=True)
    orphan = ws.generated / "var_999.png"
    orphan.write_bytes(b"x")  # leftover from an interrupted, markerless run

    result = generate_dataset(ws, StubBackend(), count=6)
    assert not orphan.exists()  # cleared on every (re)build path
    assert len(result.variants) == 6


def test_pure_core_lazy_backend() -> None:
    import sys

    import make_char_dataset.generate  # noqa: F401

    for heavy in ("torch", "diffusers", "requests", "websocket"):
        assert heavy not in sys.modules


def test_comfy_build_workflow_wires_img2img_lora_controlnet() -> None:
    backend = ComfyBackend(
        base_model="sdxl",
        style_lora_path="/models/lora/comic.safetensors",
        style_prompt="comic book style",
        controlnet_model="control_openpose.safetensors",
    )
    spec = VariantSpec(3, "full_body", "a.png", "a ranger", 11, 0.75, "openpose", "768x1024")
    graph = backend._build_workflow(spec, "a.png")

    class_types = {node["class_type"] for node in graph.values()}
    assert {"ImageScale", "VAEEncode", "LoraLoader", "ControlNetApply"} <= class_types
    assert "OpenposePreprocessor" in class_types  # mode-specific structural hint
    # license-safe: never InsightFace-based nodes
    assert not class_types & {"PuLID", "IPAdapter", "IPAdapterApply", "InstantID"}
    # edges, not just membership — an orphaned node must not pass
    assert graph["scale"]["inputs"]["width"] == 768  # bucket lands on the scaler...
    assert "width" not in graph["sampler"]["inputs"]  # ...not on KSampler (img2img)
    assert graph["vae_encode"]["inputs"]["pixels"] == ["scale", 0]
    assert graph["sampler"]["inputs"]["latent_image"] == ["vae_encode", 0]
    assert graph["sampler"]["inputs"]["model"] == ["style_lora", 0]  # LoRA rewires the model
    assert graph["positive"]["inputs"]["clip"] == ["style_lora", 1]
    assert graph["sampler"]["inputs"]["positive"] == ["controlnet_apply", 0]
    assert graph["controlnet_apply"]["inputs"]["image"] == ["controlnet_pre", 0]
    assert graph["sampler"]["inputs"]["denoise"] == 0.75
    assert "comic book style" in graph["positive"]["inputs"]["text"]


def test_comfy_build_workflow_degrades_without_optional_models() -> None:
    backend = ComfyBackend(base_model="flux")  # no LoRA, no ControlNet
    spec = VariantSpec(0, "portrait", "a.png", "a ranger", 1, 0.7, "none", "1024x1024")
    graph = backend._build_workflow(spec, "a.png")

    class_types = {node["class_type"] for node in graph.values()}
    assert "LoraLoader" not in class_types
    assert "ControlNetApply" not in class_types
    assert graph["checkpoint"]["inputs"]["ckpt_name"] == "flux.safetensors"
    assert graph["sampler"]["inputs"]["positive"] == ["positive", 0]  # not rerouted via controlnet
    assert "width" not in graph["sampler"]["inputs"]
    assert graph["vae_encode"]["inputs"]["pixels"] == ["scale", 0]  # img2img still wired


def test_comfy_controlnet_requires_both_model_and_mode() -> None:
    pose = VariantSpec(0, "full_body", "a.png", "p", 1, 0.75, "openpose", "768x1024")
    none_mode = VariantSpec(0, "full_body", "a.png", "p", 1, 0.75, "none", "768x1024")
    # model set but mode 'none' -> no ControlNet
    with_model = ComfyBackend(controlnet_model="cn.safetensors")
    assert "ControlNetApply" not in {
        n["class_type"] for n in with_model._build_workflow(none_mode, "a.png").values()
    }
    # mode set but no model -> no ControlNet (the production-wiring regression)
    no_model = ComfyBackend()
    assert "ControlNetApply" not in {
        n["class_type"] for n in no_model._build_workflow(pose, "a.png").values()
    }


# --- Flux restylization backend (HLE-804) -----------------------------------


def test_flux_build_workflow_img2img_wires_style_lora() -> None:
    backend = FluxImg2ImgBackend(
        style_lora_path="/models/loras/cmcstyle_1400.safetensors",
        style_lora_weight=0.85,
        style_prompt="cmcstyle",
        guidance=3.5,
        steps=20,
    )
    spec = VariantSpec(7, "full_body", "serg0.png", "a man standing", 1234, 0.5, "none", "768x1152")
    graph = backend._build_workflow(spec, "serg0.png")

    class_types = {node["class_type"] for node in graph.values()}
    # split Flux loaders, not a single SDXL checkpoint
    assert {"UNETLoader", "DualCLIPLoader", "VAELoader"} <= class_types
    assert "CheckpointLoaderSimple" not in class_types
    # img2img (anchor -> latent), NOT txt2img from an empty latent
    assert {"LoadImage", "ImageScale", "VAEEncode"} <= class_types
    assert "EmptySD3LatentImage" not in class_types
    # Flux conditioning + style LoRA
    assert {"FluxGuidance", "LoraLoaderModelOnly"} <= class_types
    # license-safe: never InsightFace-based nodes
    assert not class_types & {"PuLID", "IPAdapter", "IPAdapterApply", "InstantID"}

    # edges, not just membership
    assert graph["16"]["inputs"]["width"] == 768  # bucket lands on the scaler...
    assert graph["16"]["inputs"]["height"] == 1152
    assert "width" not in graph["40"]["inputs"]  # ...not on KSampler (img2img)
    assert graph["17"]["inputs"]["pixels"] == ["16", 0]  # encode the scaled anchor
    assert graph["17"]["inputs"]["vae"] == ["12", 0]
    assert graph["40"]["inputs"]["latent_image"] == ["17", 0]
    assert graph["13"]["inputs"]["model"] == ["10", 0]  # style LoRA chains off the UNet
    assert graph["13"]["inputs"]["lora_name"] == "cmcstyle_1400.safetensors"  # basename only
    assert graph["40"]["inputs"]["model"] == ["13", 0]  # ...and rewires the sampler's model
    assert graph["40"]["inputs"]["positive"] == ["21", 0]  # via FluxGuidance
    assert graph["21"]["inputs"]["conditioning"] == ["20", 0]
    assert graph["21"]["inputs"]["guidance"] == 3.5
    assert graph["40"]["inputs"]["negative"] == ["22", 0]
    assert graph["22"]["inputs"]["text"] == ""  # empty negative (cfg 1.0)
    assert graph["40"]["inputs"]["cfg"] == 1.0  # Flux steers via FluxGuidance
    assert graph["40"]["inputs"]["denoise"] == 0.5  # the calibrated identity/style knob
    assert graph["40"]["inputs"]["steps"] == 20
    assert "cmcstyle" in graph["20"]["inputs"]["text"]  # style trigger folded in
    assert "a man standing" in graph["20"]["inputs"]["text"]


def test_flux_build_workflow_degrades_without_style_lora() -> None:
    backend = FluxImg2ImgBackend()  # no style LoRA configured
    spec = VariantSpec(0, "portrait", "a.png", "a man", 1, 0.6, "none", "768x768")
    graph = backend._build_workflow(spec, "a.png")

    assert "LoraLoaderModelOnly" not in {node["class_type"] for node in graph.values()}
    assert graph["40"]["inputs"]["model"] == ["10", 0]  # sampler reads the UNet directly
    assert graph["17"]["inputs"]["pixels"] == ["16", 0]  # img2img still wired
    assert graph["20"]["inputs"]["text"] == "a man"  # no style prompt appended


def test_flux_backend_uses_configured_model_filenames() -> None:
    backend = FluxImg2ImgBackend(
        unet="flux1-dev-fp8.safetensors",
        clip_l="clip_l.safetensors",
        t5xxl="t5xxl_fp8_e4m3fn.safetensors",
        vae="ae.safetensors",
    )
    spec = VariantSpec(0, "full_body", "a.png", "a man", 1, 0.5, "none", "768x1152")
    graph = backend._build_workflow(spec, "a.png")
    assert graph["10"]["inputs"]["unet_name"] == "flux1-dev-fp8.safetensors"
    assert graph["11"]["inputs"]["clip_name1"] == "clip_l.safetensors"
    assert graph["11"]["inputs"]["clip_name2"] == "t5xxl_fp8_e4m3fn.safetensors"
    assert graph["11"]["inputs"]["type"] == "flux"
    assert graph["12"]["inputs"]["vae_name"] == "ae.safetensors"


def test_select_backend_flux_routes_to_flux_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import make_char_dataset.backends.comfy as comfy_mod
    from make_char_dataset.generate import _select_backend

    captured: dict[str, object] = {}

    class FakeFlux:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(comfy_mod, "FluxImg2ImgBackend", FakeFlux)

    class Settings:
        backend = "comfyui"
        comfy_url = "http://comfy:8188"
        base_model = "flux"  # <- routes to the restylization backend
        style_lora_path = "/models/loras/cmcstyle_1400.safetensors"
        style_lora_weight = 0.85
        style_prompt = "cmcstyle"
        gen_unet = "flux1-dev-fp8.safetensors"
        gen_clip_l = "clip_l.safetensors"
        gen_t5xxl = "t5xxl_fp8_e4m3fn.safetensors"
        gen_vae = "ae.safetensors"
        gen_guidance = 3.5
        gen_steps = 24

    backend = _select_backend(Settings())
    assert isinstance(backend, FakeFlux)
    assert captured["t5xxl"] == "t5xxl_fp8_e4m3fn.safetensors"  # the Flux-specific wiring
    assert captured["style_lora_path"] == "/models/loras/cmcstyle_1400.safetensors"
    assert captured["guidance"] == 3.5
    assert captured["base_url"] == "http://comfy:8188"


def test_combine_prompt_appends_style_only_when_present() -> None:
    assert _combine_prompt("a man standing", "cmcstyle") == "a man standing, cmcstyle"
    assert _combine_prompt("a man standing", "") == "a man standing"


def test_extract_error_reads_comfy_history_status() -> None:
    # a failed prompt: execution_error message with an exception detail
    errored = {
        "status": {
            "status_str": "error",
            "messages": [
                ["execution_start", {}],
                [
                    "execution_error",
                    {"node_type": "KSampler", "exception_message": "CUDA out of memory"},
                ],
            ],
        }
    }
    msg = _extract_error(errored)
    assert msg is not None
    assert "CUDA out of memory" in msg
    assert "KSampler" in msg
    # a bare error status with no detailed message still reports an error
    assert _extract_error({"status": {"status_str": "error", "messages": []}}) == "execution error"
    # a successful prompt has no error
    assert _extract_error({"status": {"status_str": "success", "messages": []}}) is None
    # a malformed/absent status is tolerated
    assert _extract_error({}) is None


def test_plan_fans_across_outfit_and_emotion_anchors() -> None:
    """The whole passport set is multiplied: outfit + emotion anchors are used too."""
    anchors = (
        [{"role": "face", "dest": f"face_{i}.png"} for i in range(2)]
        + [{"role": "body", "dest": f"body_{i}.png"} for i in range(3)]
        + [{"role": "outfit", "dest": f"outfit_{i}.png"} for i in range(5)]
        + [{"role": "emotion", "dest": f"emotion_{i}.png"} for i in range(3)]
    )
    specs = plan_variants({"anchors": anchors, "identity": {}}, 40)

    used = {spec.anchor_dest for spec in specs}
    assert len(used) == 13  # every anchor conditions at least one variant (not just body/face)
    for spec in specs:
        role = spec.anchor_dest.split("_", 1)[0]
        if spec.shot_class == "portrait":
            assert role in ("face", "emotion")  # portraits carry expression variety
        else:
            assert role in ("body", "outfit")  # fuller shots carry clothing variety


def test_comfy_controlnet_unknown_mode_falls_back_to_scaled_anchor() -> None:
    # model + a mode with no preprocessor -> ControlNet wired off the raw scaled anchor
    backend = ComfyBackend(controlnet_model="cn.safetensors")
    spec = VariantSpec(0, "full_body", "a.png", "p", 1, 0.75, "canny", "768x1024")
    graph = backend._build_workflow(spec, "a.png")
    class_types = {node["class_type"] for node in graph.values()}
    assert "ControlNetApply" in class_types
    assert "OpenposePreprocessor" not in class_types
    assert "Zoe-DepthMapPreprocessor" not in class_types
    assert graph["controlnet_apply"]["inputs"]["image"] == ["scale", 0]
