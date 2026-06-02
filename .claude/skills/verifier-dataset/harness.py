"""Reusable plumbing for the dataset verifier (imported by smoke.py and scenarios).

Mirrors the create-char-passport ``verifier-gradio`` harness shape — helpers +
a ``Check`` accumulator + a ``report()`` with an exit-code contract — but for the
make-char-dataset pipeline. The honest cheap surface is the **end-to-end stage
contract** under the *conditioning-only* doctrine: a synthetic passport export is
run through the REAL stages ``import -> generate -> clean -> caption`` against
stub backends (no GPU, no model download, no ComfyUI server), and the on-disk
result is asserted — crucially, that the golden anchors stay conditioning-only
(in ``00_passport_import`` only) and never leak into the trained ``03_dataset``.
The real backend is checked in the heavy, human-eyeball tier (see SKILL.md).
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from make_char_dataset import caption as caption_mod
from make_char_dataset import generate as generate_mod
from make_char_dataset import ingest as ingest_mod
from make_char_dataset.assembly import StubGenerator
from make_char_dataset.tagging import StubTagger
from make_char_dataset.workspace import Workspace

# Captions/labels use — and ✓; reconfigure so a cp1251 Windows console won't choke.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8")


def make_workspace(prefix: str = "char-verify-") -> tuple[Workspace, Path]:
    """Create an isolated temp workspace so the verifier never touches APP_WORKSPACE."""
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    workspace = Workspace(tmp / "workspace")
    workspace.ensure_base()
    return workspace, tmp


def _make_passport_export(root: Path, *, character_id: str = "conan") -> Path:
    """Write a synthetic, ready create-char-passport export (state.json + 5 refs)."""
    export = root / "passport"
    refs = export / "refs"
    refs.mkdir(parents=True)
    generator = StubGenerator(size=64)
    steps: dict[str, dict[str, str]] = {}
    for index, step in enumerate(ingest_mod.PASSPORT_STEPS):
        generator.generate(500 + index, refs / f"{step}.png")
        steps[step] = {"approved_path": f"refs/{step}.png"}
    state = {
        "character_id": character_id,
        "current_step": None,
        "steps": steps,
        "character_table": {"archetype": "wandering barbarian"},
        "prompt_layers": {"face": "angular face", "body": "broad shoulders", "style": "ink"},
        "base_outfit": {"prompt": "fur cloak"},
    }
    (export / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return export


@dataclass
class SamplePipeline:
    """What build_sample_pipeline produced, so scenarios can assert against it."""

    workspace: Workspace
    trigger: str
    repeats: int
    count: int
    clean: object  # caption.CleanResult
    dataset: object  # assembly.AssemblyResult
    anchor_hashes: set[bytes]  # bytes of every imported 00_passport_import anchor


def build_sample_pipeline(
    workspace: Workspace,
    *,
    trigger: str = "conan",
    repeats: int = 10,
    count: int = 8,
    style_prompt: str = "ink, halftone",
) -> SamplePipeline:
    """Run the REAL stages import -> generate -> clean -> caption on a synthetic export.

    Plants one byte-identical duplicate into ``01_generated`` so the clean stage's
    dedup -> ``manual_review`` path is exercised. Returns the stage results plus the
    content hashes of the golden anchors, for the conditioning-only anti-regression.
    """
    export = _make_passport_export(workspace.root.parent, character_id=trigger)

    # 1) import: golden anchors land as conditioning in 00_passport_import only.
    ingest_mod.import_passport(export, workspace, target_side=64)
    anchor_hashes = {path.read_bytes() for path in workspace.passport_import.glob("*.png")}

    # 2) generate: multiply into variants in 01_generated (stub backend, no GPU).
    generate_mod.generate_dataset(
        workspace, generate_mod.StubBackend(), count=count, target_side=64
    )

    # 3) plant a near-duplicate variant to exercise dedup -> manual_review.
    first = sorted(workspace.generated.glob("var_*.png"))[0]
    shutil.copyfile(first, workspace.generated / "var_planted_dup.png")

    # 4) clean: dedup + size filter -> 02_clean (small min side so 64px stubs pass).
    clean = caption_mod.clean_variants(workspace, dedup_distance=6, min_side_px=16)

    # 5) caption: Character-Locker captions + kohya layout -> 03_dataset.
    dataset = caption_mod.caption_dataset(
        workspace, StubTagger(), trigger=trigger, repeats=repeats, style_prompt=style_prompt
    )

    return SamplePipeline(
        workspace=workspace,
        trigger=trigger,
        repeats=repeats,
        count=count,
        clean=clean,
        dataset=dataset,
        anchor_hashes=anchor_hashes,
    )


@dataclass
class Check:
    """Accumulates pass/fail checks; ``report`` prints them and returns an exit code."""

    items: list[tuple[bool, str, bool]] = field(default_factory=list)

    def expect(self, ok: bool, msg: str, *, critical: bool = True) -> None:
        """Record a check outcome."""
        self.items.append((bool(ok), msg, critical))

    def note(self, msg: str) -> None:
        """Print an informational note (not a pass/fail)."""
        print(f"  note: {msg}")


def report(check: Check) -> int:
    """Print results and return 1 if any critical check failed, else 0."""
    failed_critical = 0
    for ok, msg, critical in check.items:
        if ok:
            mark = "✓ PASS"
        elif critical:
            mark = "✗ FAIL"
            failed_critical += 1
        else:
            mark = "! WARN"
        print(f"  {mark}  {msg}")
    verdict = "GREEN" if failed_critical == 0 else f"RED ({failed_critical} critical failure(s))"
    print(f"VERDICT: {verdict}")
    return 1 if failed_critical else 0
