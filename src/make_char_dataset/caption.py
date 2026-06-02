"""Clean + caption stages: dedup the generated variants and lay out a kohya dataset.

clean: ``01_generated/`` -> ``02_clean/`` — min-size filter + perceptual-hash dedup
of the **generated variants only** (golden anchors are conditioning-only and never
entered the pipeline here). Too-small / near-duplicate frames are copied to
``manual_review/`` (never deleted; ``01_generated/`` is left intact).

caption: ``02_clean/`` -> ``03_dataset/<repeats>_<trigger>/`` — WD14-tag each image,
build a **Character-Locker** caption (trigger first; identity geometry stripped so it
binds to the trigger, style tokens stripped because style lives in the external LoRA),
and lay out kohya ``image`` + ``.txt`` pairs via :func:`assembly.assemble_dataset`.
An opt-in regularization track lays out trigger-free class images.

Builds on the pure ``assembly`` core; the heavy WD14 tagger is injected behind the
:class:`make_char_dataset.tagging.Tagger` Protocol and lazy-imported, so this stage
and its tests stay torch/onnxruntime-free.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from make_char_dataset.assembly import (
    AssemblyResult,
    Variant,
    assemble_dataset,
    build_character_caption,
    dedup_by_phash,
)
from make_char_dataset.config import get_settings
from make_char_dataset.tagging import StubTagger, Tagger
from make_char_dataset.workspace import Workspace

CLEAN_MANIFEST = "clean.json"
DATASET_MANIFEST = "dataset.json"
STAGE_MARKER = ".stage_complete"

# Hair/eye descriptors that, as the *leading phrase* of a "<desc> hair" / "<desc> eyes"
# tag, mark it as immutable identity — so accessory compounds ("hair ornament",
# "hairband") and expressions ("closed eyes") are NOT matched.
_HAIR_DESCRIPTORS: frozenset[str] = frozenset(
    {
        "black",
        "brown",
        "blonde",
        "blond",
        "red",
        "white",
        "grey",
        "gray",
        "silver",
        "pink",
        "blue",
        "green",
        "purple",
        "orange",
        "auburn",
        "ginger",
        "dark",
        "light",
        "long",
        "short",
        "very long",
        "medium",
        "shoulder-length",
        "spiky",
        "wavy",
        "curly",
        "straight",
    }
)
_EYE_COLORS: frozenset[str] = frozenset(
    {
        "black",
        "brown",
        "blue",
        "green",
        "red",
        "grey",
        "gray",
        "purple",
        "yellow",
        "amber",
        "hazel",
        "pink",
        "orange",
        "olive",
        "violet",
    }
)
# Whole-tag identity markers (body type, distinguishing marks, species traits, counts).
_IDENTITY_TAGS: frozenset[str] = frozenset(
    {
        "1boy",
        "1girl",
        "solo",
        "muscular",
        "abs",
        "dark skin",
        "pale skin",
        "tan",
        "scar",
        "scar on face",
        "mole",
        "freckles",
        "beard",
        "mustache",
        "goatee",
        "facial hair",
        "pointy ears",
        "animal ears",
        "tail",
        "horns",
        "wings",
    }
)
# Art-style / medium tags WD14 emits that must never enter a character caption
# (style lives in the external LoRA) — stripped independent of the user style prompt.
_STYLE_TAGS: frozenset[str] = frozenset(
    {
        "monochrome",
        "greyscale",
        "grayscale",
        "sketch",
        "lineart",
        "line art",
        "watercolor",
        "oil painting",
        "flat color",
        "cel shading",
        "comic",
        "manga",
        "traditional media",
        "pixel art",
        "chibi",
    }
)

# caption never re-dedups: the clean stage is the single dedup owner, so layout must
# not shutil.move an image out of the read-only 02_clean input.
_NO_DEDUP = -1


class CaptionError(Exception):
    """Raised when a clean/caption stage cannot run (e.g. nothing upstream)."""


@dataclass(frozen=True)
class CleanResult:
    """Outcome of :func:`clean_variants`."""

    out_dir: Path
    kept: list[str]  # filenames in 02_clean/
    rejected: list[str]  # filenames copied to manual_review/


def _is_geometry(tag: str) -> bool:
    """True when ``tag`` is immutable identity geometry (filtered from captions).

    Whole-tag / leading-phrase matching, so variable accessory and expression tags
    survive: ``hair ornament`` / ``hairband`` / ``closed eyes`` / ``skintight`` are
    kept, while ``brown hair`` / ``long hair`` / ``green eyes`` / ``scar`` are stripped.
    """
    lowered = tag.lower().strip()
    if lowered in _IDENTITY_TAGS:
        return True
    if lowered.endswith(" hair") and lowered[: -len(" hair")] in _HAIR_DESCRIPTORS:
        return True
    return lowered.endswith(" eyes") and lowered[: -len(" eyes")] in _EYE_COLORS


def _is_style(tag: str, style_prompt_tokens: frozenset[str]) -> bool:
    """True when ``tag`` is an art-style/medium token (kept out of character captions)."""
    lowered = tag.lower().strip()
    return lowered in _STYLE_TAGS or lowered.endswith(" style") or lowered in style_prompt_tokens


def style_tokens(style_prompt: str) -> tuple[str, ...]:
    """Split a style prompt into comma-separated tokens to strip from captions."""
    return tuple(token.strip() for token in style_prompt.split(",") if token.strip())


def character_locker_caption(
    trigger: str, tags: Sequence[str], *, style: Sequence[str] = ()
) -> str:
    """Build a Character-Locker caption: trigger first, no geometry, no style tokens.

    Style is stripped via both a curated art-style/medium denylist (so booru style
    tags never leak even when no style prompt is configured) and the user's
    ``style`` tokens; identity geometry is stripped so it binds to the trigger.
    """
    style_set = frozenset(token.lower().strip() for token in style)
    content = [tag for tag in tags if not _is_geometry(tag) and not _is_style(tag, style_set)]
    return build_character_caption(trigger, content)


def _clear_files(out: Path) -> None:
    """Remove the flat files this stage owns before a rebuild (files only)."""
    for child in out.iterdir():
        if child.is_file():
            child.unlink()


def _too_small(path: Path, min_side_px: int) -> bool:
    with Image.open(path) as image:
        return min(image.size) < min_side_px


def _dataset_image_count(training_dir: Path) -> int | None:
    """Read the expected image count from a prior layout's dataset.json (or None)."""
    try:
        data = json.loads((training_dir / DATASET_MANIFEST).read_text(encoding="utf-8"))
        return int(data["images"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _load_clean(out: Path) -> CleanResult | None:
    """Return a prior clean result iff its manifest loads and kept files exist."""
    try:
        data = json.loads((out / CLEAN_MANIFEST).read_text(encoding="utf-8"))
        kept = [str(name) for name in data["kept"]]
        rejected = [str(name) for name in data["rejected"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if not all((out / name).is_file() for name in kept):
        return None
    return CleanResult(out_dir=out, kept=kept, rejected=rejected)


def clean_variants(
    workspace: Workspace, *, dedup_distance: int, min_side_px: int, force: bool = False
) -> CleanResult:
    """Dedup + size-filter the generated variants from ``01_generated`` into ``02_clean``.

    Reads ``01_generated`` read-only; copies survivors to ``02_clean`` and rejects
    (too-small or near-duplicate) to ``manual_review`` — never deletes upstream.
    Idempotent: a complete prior clean is returned as-is unless ``force``.
    """
    out = workspace.clean
    marker = out / STAGE_MARKER
    if marker.exists() and not force:
        cached = _load_clean(out)
        if cached is not None:
            return cached

    variants = sorted(workspace.generated.glob("var_*.png"))
    if not variants:
        raise CaptionError("no generated variants in 01_generated; run the generate stage first")

    sized = [path for path in variants if not _too_small(path, min_side_px)]
    too_small = [path for path in variants if _too_small(path, min_side_px)]
    kept, duplicates = dedup_by_phash(sized, dedup_distance)
    rejected = too_small + duplicates

    out.mkdir(parents=True, exist_ok=True)
    workspace.manual_review.mkdir(parents=True, exist_ok=True)
    _clear_files(out)
    for path in kept:
        shutil.copyfile(path, out / path.name)
    for path in rejected:
        shutil.copyfile(path, workspace.manual_review / path.name)

    result = CleanResult(
        out_dir=out, kept=[path.name for path in kept], rejected=[path.name for path in rejected]
    )
    (out / CLEAN_MANIFEST).write_text(
        json.dumps({"kept": result.kept, "rejected": result.rejected}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    marker.write_text("ok\n", encoding="utf-8")
    return result


def caption_dataset(
    workspace: Workspace,
    tagger: Tagger,
    *,
    trigger: str,
    repeats: int,
    style_prompt: str = "",
    keep_tokens: int = 1,
    force: bool = False,
) -> AssemblyResult:
    """Tag the cleaned variants and lay out a kohya ``03_dataset/<repeats>_<trigger>/``.

    Each ``02_clean`` image is WD14-tagged, captioned Character-Locker style (trigger
    first, geometry + style stripped), and written with a ``.txt`` sidecar via
    :func:`assembly.assemble_dataset`. Dedup is **not** re-run here (the clean stage
    owns it), so the read-only ``02_clean`` input is never mutated. Idempotent: a
    complete prior layout is returned as-is; ``kept`` is the laid-out image paths.
    """
    images = sorted(workspace.clean.glob("var_*.png"))
    if not images:
        raise CaptionError("no cleaned variants in 02_clean; run the clean stage first")

    training_dir = workspace.training_dir(repeats, trigger)
    marker = training_dir / STAGE_MARKER
    if marker.exists() and not force:
        existing = sorted(training_dir.glob("*.png"))
        expected = _dataset_image_count(training_dir)
        complete = (
            bool(existing)
            and len(existing) == expected
            and all(image.with_suffix(".txt").is_file() for image in existing)
        )
        if complete:
            return AssemblyResult(
                training_dir=training_dir, kept=existing, deduped=[], anchors_used=0
            )

    style = style_tokens(style_prompt)
    variants = [
        Variant(image, character_locker_caption(trigger, tagger.tag(image), style=style))
        for image in images
    ]

    training_dir.mkdir(parents=True, exist_ok=True)
    _clear_files(training_dir)
    assemble_dataset(
        workspace,
        variants=variants,
        trigger=trigger,
        repeats=repeats,
        dedup_distance=_NO_DEDUP,
    )
    laid_out = sorted(training_dir.glob("*.png"))
    (training_dir / DATASET_MANIFEST).write_text(
        json.dumps(
            {
                "trigger": trigger,
                "repeats": repeats,
                "keep_tokens": keep_tokens,
                "images": len(laid_out),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    marker.write_text("ok\n", encoding="utf-8")
    return AssemblyResult(training_dir=training_dir, kept=laid_out, deduped=[], anchors_used=0)


def layout_regularization(
    workspace: Workspace,
    class_images: Iterable[Path],
    *,
    class_descriptor: str,
    repeats: int,
) -> Path:
    """Opt-in regularization track: lay out trigger-free class images for kohya.

    Each class image gets a ``.txt`` holding only the class descriptor (no trigger),
    in ``03_dataset/<repeats>_<class>/`` — the kohya regularization-folder convention.
    """
    folder = class_descriptor.replace(" ", "_")
    reg_dir = workspace.dataset / f"{repeats}_{folder}"
    reg_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(class_images):
        stem = f"reg_{index:03d}"
        shutil.copyfile(image, reg_dir / f"{stem}.png")
        (reg_dir / f"{stem}.txt").write_text(class_descriptor, encoding="utf-8")
    return reg_dir


def run_clean(*, force: bool = False) -> CleanResult:
    """Settings-driven clean stage."""
    settings = get_settings()
    return clean_variants(
        Workspace(settings.workspace),
        dedup_distance=settings.dedup_hamming_distance,
        min_side_px=settings.min_side_px,
        force=force,
    )


def run_caption(*, tagger: Tagger | None = None, force: bool = False) -> AssemblyResult:
    """Settings-driven caption stage (defaults to the GPU-free StubTagger)."""
    settings = get_settings()
    return caption_dataset(
        Workspace(settings.workspace),
        tagger or StubTagger(),
        trigger=settings.trigger_token,
        repeats=settings.dataset_repeats,
        keep_tokens=settings.keep_tokens,
        style_prompt=settings.style_prompt,
        force=force,
    )
