"""Dataset assembly: deduplicate generated character variants and lay them out
in a kohya-ready training folder.

The heavy image-generation backend (ComfyUI / diffusers, landed in a later
subtask of HLE-757) is injected behind the :class:`Generator` Protocol, so this
module — together with its unit tests and the verifier smoke — never imports
torch/diffusers and stays fast on the CPU/CI path.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from make_char_dataset.workspace import Workspace


@runtime_checkable
class Generator(Protocol):
    """Renders one character variant image to ``out_path``.

    Real implementations drive a GPU backend (ComfyUI over HTTP+websocket, or
    local diffusers) loaded with the external style LoRA. They are imported
    lazily inside their adapter and injected here; this pure module depends only
    on the Protocol.

    # PLAYBOOK-START
    # id: pure-core-lazy-backend
    # title: Pure core, heavy backend behind an injected Protocol
    # status: refined
    # category: pipeline
    # tags: [protocol, di, testing, lazy-import]
    # Orchestration/assembly depends only on a Protocol; the heavy, optional
    # backend (GPU model, network client) is imported lazily inside its adapter
    # and injected. Unit tests and the verifier inject a stub, so the fast path
    # never pays the heavy import and coverage stays high without a GPU.
    # Substitution test passes: applies to any pipeline with a costly backend.
    # PLAYBOOK-END
    """

    def generate(self, index: int, out_path: Path) -> None:
        """Render variant ``index`` and write it to ``out_path``."""
        ...


@dataclass
class StubGenerator:
    """Deterministic, dependency-light :class:`Generator` for tests and the
    verifier smoke.

    Writes per-index pseudo-random PNGs (so distinct indices get distinct
    perceptual hashes); ``duplicate_of`` maps one index onto another's seed so
    the dedup path is exercised — all without a GPU, a model download, or a
    running ComfyUI server.
    """

    size: int = 64
    duplicate_of: dict[int, int] = field(default_factory=dict)

    def generate(self, index: int, out_path: Path) -> None:
        """Render a deterministic noise image for ``index`` (or its duplicate)."""
        seed = self.duplicate_of.get(index, index)
        rng = np.random.default_rng(seed)
        pixels = rng.integers(0, 256, size=(self.size, self.size, 3), dtype=np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(out_path)


def generate_variants(
    generator: Generator, count: int, out_dir: Path, *, prefix: str = "gen"
) -> list[Path]:
    """Render ``count`` variants into ``out_dir`` via the injected generator.

    Returns the written paths in generation order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        path = out_dir / f"{prefix}_{index:03d}.png"
        generator.generate(index, path)
        paths.append(path)
    return paths


def _phash(path: Path) -> Any:
    """Perceptual hash of an image. imagehash is imported lazily so importing
    this module stays cheap (it pulls in scipy)."""
    import imagehash

    with Image.open(path) as image:
        return imagehash.phash(image)


def dedup_by_phash(paths: Sequence[Path], max_distance: int) -> tuple[list[Path], list[Path]]:
    """Split ``paths`` into ``(kept, duplicates)`` by perceptual hash.

    A path is a duplicate when its phash is within ``max_distance`` Hamming
    distance of an already-kept path. Input order is preserved, so the first
    occurrence in each near-duplicate group is the one kept.
    """
    kept: list[Path] = []
    kept_hashes: list[Any] = []
    duplicates: list[Path] = []
    for path in paths:
        digest = _phash(path)
        if any((digest - other) <= max_distance for other in kept_hashes):
            duplicates.append(path)
        else:
            kept.append(path)
            kept_hashes.append(digest)
    return kept, duplicates


def build_character_caption(
    trigger: str, tags: Sequence[str], *, style_tokens: Sequence[str] = ()
) -> str:
    """Build a Character-Locker caption: the character ``trigger`` first, then
    only *variable* context (pose / outfit / background / lighting).

    Inverse of the style-locker strategy used for the style dataset: we describe
    what should VARY so it does not bind to identity, deliberately omit face/body
    geometry (it binds to the trigger), and strip any art-style tokens — those
    live in the external style LoRA, never in the character captions.

    # PLAYBOOK-START
    # id: character-locker-caption
    # title: Character-Locker captioning (trigger-first, geometry- and style-free)
    # status: draft
    # category: dataset
    # tags: [lora, captioning, disentanglement]
    # When a separate style LoRA carries the art style, the character dataset's
    # captions must put the identity trigger first, describe only what should
    # VARY (pose/outfit/background), omit the geometry that should bind to the
    # trigger, and strip style tokens entirely. This is the mirror image of the
    # style dataset's "trigger=STYLE, describe content" rule.
    # PLAYBOOK-END
    """
    banned = {token.strip().lower() for token in style_tokens if token.strip()}
    kept = [tag.strip() for tag in tags if tag.strip() and tag.strip().lower() not in banned]
    return ", ".join([trigger, *kept])


@dataclass(frozen=True)
class Variant:
    """An image plus the Character-Locker caption to train it with."""

    image: Path
    caption: str


@dataclass(frozen=True)
class AssemblyResult:
    """Outcome of :func:`assemble_dataset`."""

    training_dir: Path
    kept: list[Path]
    deduped: list[Path]
    anchors_used: int


def assemble_dataset(
    workspace: Workspace,
    *,
    variants: Sequence[Variant],
    trigger: str,
    repeats: int,
    dedup_distance: int,
    anchors: Sequence[Variant] = (),
) -> AssemblyResult:
    """Lay out a kohya-ready training folder from generated ``variants``.

    Near-duplicate ``variants`` (by perceptual hash) are moved to
    ``manual_review/`` — never deleted.

    ``anchors`` is an **opt-in primitive**, not the default route. Under the
    project's *conditioning-only* doctrine the golden passport set feeds the
    generate stage as img2img/ControlNet references and is kept OUT of the
    training set (its shared grey-studio background + neutral expression would
    bind those to identity). When supplied, ``anchors`` are written through
    unconditionally and exempt from dedup; the default pipeline supplies none.

    Writes ``<repeats>_<trigger>/`` with a ``.png`` per kept image and a matching
    ``.txt`` caption sidecar, using deterministic names so re-runs overwrite
    rather than duplicate.
    """
    workspace.manual_review.mkdir(parents=True, exist_ok=True)
    training_dir = workspace.training_dir(repeats, trigger)
    training_dir.mkdir(parents=True, exist_ok=True)

    by_path = {variant.image: variant for variant in variants}
    kept_paths, dup_paths = dedup_by_phash(list(by_path), dedup_distance)

    for dup in dup_paths:
        shutil.move(str(dup), str(workspace.manual_review / dup.name))

    def _emit(items: Sequence[Variant], prefix: str) -> None:
        for index, variant in enumerate(items):
            stem = f"{prefix}_{index:03d}"
            shutil.copyfile(variant.image, training_dir / f"{stem}.png")
            (training_dir / f"{stem}.txt").write_text(variant.caption, encoding="utf-8")

    _emit(list(anchors), "anchor")
    _emit([by_path[path] for path in kept_paths], "var")

    return AssemblyResult(
        training_dir=training_dir,
        kept=kept_paths,
        deduped=dup_paths,
        anchors_used=len(anchors),
    )
