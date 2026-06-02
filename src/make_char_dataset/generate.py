"""Generate stage: multiply golden anchors into ~30-40 diverse, in-style variants.

Reads the ``00_passport_import/`` manifest (role-tagged anchors + textual
identity), **plans** a class-balanced, role-fanned-out set of variant specs, and
**renders** each through an injected backend into ``01_generated/``. Identity is
held constant — every variant is an img2img from the character's own anchors —
while pose / plan / background / lighting vary (the project doctrine: consistency
in the character, diversity in everything else).

Style is external: the style LoRA + style prompt are applied only inside the
backend's ``render`` call; the planned/persisted prompt is **style-free**, and the
per-variant provenance sidecar carries no style tokens (captioning in [4] tags the
generated images with WD14, so it needs provenance — seed/anchor/class — not style).

Layering (pure-core-lazy-backend): :class:`GenerationBackend` is the spec-aware
[3] seam. The heavy ComfyUI adapter lives in ``backends/comfy.py`` and is
lazy-imported, so this module stays torch/network-free. :class:`StubBackend`
reuses :class:`make_char_dataset.assembly.StubGenerator` as its render primitive,
so there is a single GPU-free stub shared with the dataset core and the verifier.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from make_char_dataset.assembly import StubGenerator
from make_char_dataset.config import get_settings
from make_char_dataset.workspace import Workspace

MANIFEST_FILENAME = "manifest.json"
PLAN_FILENAME = "plan.json"
STAGE_MARKER = ".stage_complete"

# Shot classes and the recipe mix (epic recipe): ~40% full body, 30% mid shot,
# 20% portrait/expression, 10% dynamic angle.
SHOT_CLASSES: tuple[str, ...] = ("full_body", "mid_shot", "portrait", "angle")
RECIPE_WEIGHTS: dict[str, float] = {
    "full_body": 0.40,
    "mid_shot": 0.30,
    "portrait": 0.20,
    "angle": 0.10,
}

# Which anchor role best conditions each shot class (identity stays constant by
# img2img-ing the character's own anchor).
_CLASS_ROLE: dict[str, str] = {
    "full_body": "body",
    "mid_shot": "body",
    "portrait": "face",
    "angle": "body",
}

_SHOT_FRAMING: dict[str, str] = {
    "full_body": "full body shot, head to toe",
    "mid_shot": "medium shot, waist up",
    "portrait": "close-up portrait, head and shoulders",
    "angle": "dynamic camera angle, full body",
}

# Deterministic scene variation per class — only pose / setting / lighting vary,
# never identity. Cycled by per-class index so a plan is reproducible.
_SCENES: dict[str, tuple[str, ...]] = {
    "full_body": (
        "standing in a sunlit forest clearing",
        "walking down a rain-slicked cobbled street",
        "mid-stride across a windswept moor",
        "seated on a mossy boulder",
        "crouched behind tall grass at dusk",
    ),
    "mid_shot": (
        "leaning against a stone wall, soft side light",
        "turning to look over one shoulder, overcast light",
        "arms crossed in a torchlit hall",
        "reaching for a satchel, golden-hour light",
    ),
    "portrait": (
        "calm expression, soft window light",
        "wry half-smile, warm rim light",
        "determined look, cool blue light",
    ),
    "angle": (
        "low-angle hero shot against the sky",
        "high-angle overhead view on a forest path",
    ),
}


class GenerationError(Exception):
    """Raised when the generate stage cannot run (e.g. nothing imported yet)."""


@dataclass(frozen=True)
class VariantSpec:
    """A single planned variant — what to render and how (style-free)."""

    index: int
    shot_class: str
    anchor_dest: str  # filename in 00_passport_import/ to img2img from
    prompt: str  # positive prompt: identity + framing + scene (no style tokens)
    seed: int
    denoise: float
    controlnet: str
    bucket: str  # target dimensions, e.g. "768x1024"


@dataclass(frozen=True)
class GeneratedVariant:
    """Provenance of one rendered variant (consumed by clean/caption in [4])."""

    index: int
    shot_class: str
    image: str  # filename in 01_generated/
    anchor_dest: str
    seed: int


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of :func:`generate_dataset`."""

    out_dir: Path
    variants: list[GeneratedVariant]


@runtime_checkable
class GenerationBackend(Protocol):
    """Renders one :class:`VariantSpec` (img2img from ``anchor_path``) to ``out_path``.

    Production backends drive a GPU (ComfyUI img2img + ControlNet, loaded with the
    external style LoRA). They are lazy-imported inside their adapter and injected
    here, so this pure module never pulls torch/diffusers/network clients.
    """

    def render(self, spec: VariantSpec, anchor_path: Path, out_path: Path) -> None:
        """Render ``spec`` (conditioned on ``anchor_path``) and write ``out_path``."""
        ...


@dataclass
class StubBackend:
    """GPU-free backend for tests and the verifier smoke.

    Reuses :class:`make_char_dataset.assembly.StubGenerator` (the single render
    primitive) seeded by ``spec.seed``, so distinct variants get distinct
    perceptual hashes — no ComfyUI, model download, or network.
    """

    size: int = 64

    def render(self, spec: VariantSpec, anchor_path: Path, out_path: Path) -> None:
        """Write a deterministic, per-seed image (anchor is ignored by the stub)."""
        StubGenerator(size=self.size).generate(spec.seed, out_path)


def class_counts(total: int) -> dict[str, int]:
    """Split ``total`` variants across shot classes by the recipe weights.

    Uses the largest-remainder method so the counts always sum to ``total`` and
    the split is deterministic.
    """
    if total < 0:
        raise GenerationError(f"variant count must be non-negative, got {total}")
    exact = {cls: total * RECIPE_WEIGHTS[cls] for cls in SHOT_CLASSES}
    counts = {cls: int(value) for cls, value in exact.items()}
    remainder = total - sum(counts.values())
    by_frac = sorted(SHOT_CLASSES, key=lambda cls: (exact[cls] - counts[cls], cls), reverse=True)
    for cls in by_frac[:remainder]:
        counts[cls] += 1
    return counts


def _anchors_by_role(anchors: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_role: dict[str, list[dict[str, Any]]] = {}
    for anchor in anchors:
        by_role.setdefault(str(anchor.get("role", "")), []).append(dict(anchor))
    return by_role


def _pick_anchor(by_role: dict[str, list[dict[str, Any]]], role: str, nth: int) -> dict[str, Any]:
    """Round-robin an anchor of ``role``; fall back to body, then any role."""
    pool = by_role.get(role) or by_role.get("body") or by_role.get("face")
    if not pool:
        pool = [anchor for anchors in by_role.values() for anchor in anchors]
    return pool[nth % len(pool)]


def _build_prompt(identity: Mapping[str, Any], shot_class: str, scene: str) -> str:
    """Build a style-free positive prompt from identity + framing + scene.

    Folds the textual identity the import stage captured (``character_table``
    archetype, ``prompt_layers`` face/body anatomy, ``base_outfit`` prompt) to
    reinforce identity during generation. Deliberately omits ``prompt_layers``
    ``style`` (would break the style-free invariant — style is the external LoRA's
    job) and ``composition`` (fights the shot framing / scene variation here).
    """
    table = identity.get("character_table")
    archetype = str(table.get("archetype", "")) if isinstance(table, Mapping) else ""
    layers = identity.get("prompt_layers")
    face = str(layers.get("face", "")) if isinstance(layers, Mapping) else ""
    body = str(layers.get("body", "")) if isinstance(layers, Mapping) else ""
    base_outfit = identity.get("base_outfit")
    outfit = str(base_outfit.get("prompt", "")) if isinstance(base_outfit, Mapping) else ""
    parts = [archetype, face, body, outfit, _SHOT_FRAMING[shot_class], scene]
    return ", ".join(part for part in parts if part)


def _class_bucket(shot_class: str, target_side: int) -> str:
    """Aspect bucket per class: portrait orientation for full-length, else square."""
    short = round(target_side * 3 / 4)
    if shot_class in ("full_body", "angle"):
        return f"{short}x{target_side}"
    return f"{target_side}x{target_side}"


def plan_variants(
    manifest: Mapping[str, Any],
    count: int,
    *,
    seed: int = 0,
    denoise: float = 0.75,
    controlnet: str = "openpose",
    target_side: int = 1024,
) -> list[VariantSpec]:
    """Plan ``count`` class-balanced, role-fanned-out variant specs from a manifest.

    Deterministic given its inputs. Identity comes from the manifest's anchors +
    identity block; only framing / scene / seed vary across the plan.
    """
    anchors = manifest.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        raise GenerationError("import manifest has no anchors; run the import stage first")
    identity = manifest.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}

    by_role = _anchors_by_role(anchors)
    counts = class_counts(count)

    specs: list[VariantSpec] = []
    index = 0
    for shot_class in SHOT_CLASSES:
        scenes = _SCENES[shot_class]
        for nth in range(counts[shot_class]):
            anchor = _pick_anchor(by_role, _CLASS_ROLE[shot_class], nth)
            dest = anchor.get("dest")
            if not dest:
                raise GenerationError("manifest anchor is missing its 'dest' filename")
            scene = scenes[nth % len(scenes)]
            specs.append(
                VariantSpec(
                    index=index,
                    shot_class=shot_class,
                    anchor_dest=str(dest),
                    prompt=_build_prompt(identity, shot_class, scene),
                    seed=seed + index,
                    denoise=denoise,
                    controlnet=controlnet,
                    bucket=_class_bucket(shot_class, target_side),
                )
            )
            index += 1
    return specs


def _read_import_manifest(workspace: Workspace) -> dict[str, Any]:
    """Load the import stage's manifest, or fail telling the user to import first."""
    path = workspace.passport_import / MANIFEST_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GenerationError(f"no import manifest at {path}; run the import stage first") from exc
    except json.JSONDecodeError as exc:
        raise GenerationError(f"import manifest at {path} is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise GenerationError(f"import manifest at {path} is malformed")
    return data


def _clear_files(out: Path) -> None:
    """Remove the flat files this stage owns before a rebuild (files only)."""
    for child in out.iterdir():
        if child.is_file():
            child.unlink()


def _write_plan(
    out: Path, character_id: str, specs: Sequence[VariantSpec], variants: Sequence[GeneratedVariant]
) -> None:
    """Persist the style-free generation plan + per-variant provenance for [4]."""
    payload = {
        "character_id": character_id,
        "doctrine": "conditioning-only",
        "specs": [asdict(spec) for spec in specs],
        "variants": [asdict(variant) for variant in variants],
    }
    (out / PLAN_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_plan(out: Path) -> GenerationResult | None:
    """Return a prior plan's result iff it loads and every variant image exists."""
    try:
        data = json.loads((out / PLAN_FILENAME).read_text(encoding="utf-8"))
        variants = [GeneratedVariant(**entry) for entry in data.get("variants", [])]
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not variants or not all((out / variant.image).is_file() for variant in variants):
        return None
    return GenerationResult(out_dir=out, variants=variants)


def generate_dataset(
    workspace: Workspace,
    backend: GenerationBackend,
    *,
    count: int,
    seed: int = 0,
    denoise: float = 0.75,
    controlnet: str = "openpose",
    target_side: int = 1024,
    force: bool = False,
) -> GenerationResult:
    """Render ``count`` variants from the imported anchors into ``01_generated/``.

    Reads the import manifest, plans a class-balanced/role-fanned-out set, renders
    each spec through ``backend``, and writes a style-free ``plan.json`` provenance
    sidecar. Idempotent: a complete prior run is returned as-is unless ``force``;
    a stale marker over a missing image self-heals by rebuilding.
    """
    out = workspace.generated
    marker = out / STAGE_MARKER
    # Serve a complete prior run without re-reading the upstream manifest or
    # re-planning — so a finished generate survives the export being cleaned up.
    if marker.exists() and not force:
        cached = _load_plan(out)
        if cached is not None:
            return cached

    manifest = _read_import_manifest(workspace)
    specs = plan_variants(
        manifest, count, seed=seed, denoise=denoise, controlnet=controlnet, target_side=target_side
    )

    out.mkdir(parents=True, exist_ok=True)
    # Always clear owned files before a (re)build, so an interrupted markerless run
    # leaves no orphaned variants behind. Files-only, so subdirectories survive.
    _clear_files(out)

    variants: list[GeneratedVariant] = []
    for spec in specs:
        anchor_path = workspace.passport_import / spec.anchor_dest
        image_name = f"var_{spec.index:03d}.png"
        backend.render(spec, anchor_path, out / image_name)
        variants.append(
            GeneratedVariant(
                index=spec.index,
                shot_class=spec.shot_class,
                image=image_name,
                anchor_dest=spec.anchor_dest,
                seed=spec.seed,
            )
        )

    _write_plan(out, str(manifest.get("character_id", "")), specs, variants)
    marker.write_text("ok\n", encoding="utf-8")
    return GenerationResult(out_dir=out, variants=variants)


def _select_backend(settings: Any) -> GenerationBackend:
    """Pick the configured backend; the heavy ComfyUI adapter is imported lazily."""
    if settings.backend == "comfyui":
        from make_char_dataset.backends.comfy import ComfyBackend

        return ComfyBackend(
            base_url=settings.comfy_url,
            base_model=settings.base_model,
            style_lora_path=settings.style_lora_path,
            style_lora_weight=settings.style_lora_weight,
            style_prompt=settings.style_prompt,
            controlnet_model=settings.controlnet_model,
        )
    return StubBackend()


def run_generate(*, seed: int = 0, force: bool = False) -> GenerationResult:
    """Settings-driven entry point: generate variants into the configured workspace."""
    settings = get_settings()
    workspace = Workspace(settings.workspace)
    return generate_dataset(
        workspace,
        _select_backend(settings),
        count=settings.variants_per_character,
        seed=seed,
        denoise=settings.img2img_denoise,
        controlnet=settings.controlnet,
        target_side=settings.target_side,
        force=force,
    )
