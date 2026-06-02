"""Reusable plumbing for the dataset verifier (imported by smoke.py and future
scenarios).

Mirrors the create-char-passport ``verifier-gradio`` harness shape — helpers +
a ``Check`` accumulator + a ``report()`` with an exit-code contract — but for the
make-char-dataset pipeline. The honest cheap surface here is the dataset-assembly
CONTRACT (dedup -> Character-Locker caption -> kohya layout), exercised against a
``StubGenerator`` so no GPU, model download, or ComfyUI server is needed. The real
generation backend is checked in the heavy, human-eyeball tier (see SKILL.md).
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from make_char_dataset.assembly import (
    StubGenerator,
    Variant,
    assemble_dataset,
    build_character_caption,
    generate_variants,
)
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


@dataclass
class SampleResult:
    """What build_sample_dataset produced, so scenarios can assert against it."""

    result: object  # assembly.AssemblyResult
    trigger: str
    repeats: int
    n_variants: int
    duplicates: int
    n_anchors: int
    style_tokens: list[str]


def build_sample_dataset(
    workspace: Workspace,
    *,
    trigger: str = "conan",
    repeats: int = 10,
    n_variants: int = 6,
    duplicate_of: dict[int, int] | None = None,
    n_anchors: int = 2,
    style_tokens: list[str] | None = None,
) -> SampleResult:
    """Run the REAL assembly path against a StubGenerator, planting one near-duplicate
    variant and ``n_anchors`` dedup-exempt golden anchors."""
    duplicate_of = duplicate_of if duplicate_of is not None else {n_variants - 1: 0}
    style_tokens = style_tokens if style_tokens is not None else ["painterly", "inked"]

    generator = StubGenerator(duplicate_of=duplicate_of)
    paths = generate_variants(generator, n_variants, workspace.generated)
    # Raw tags include a style token, which Character-Locker captioning strips out.
    variants = [
        Variant(
            path,
            build_character_caption(
                trigger, ["standing", style_tokens[0], "outdoor"], style_tokens=style_tokens
            ),
        )
        for path in paths
    ]

    anchor_generator = StubGenerator()
    anchors: list[Variant] = []
    for i in range(n_anchors):
        anchor = workspace.passport_import / f"passport_{i}.png"
        anchor_generator.generate(1000 + i, anchor)
        anchors.append(Variant(anchor, build_character_caption(trigger, ["neutral", "studio"])))

    result = assemble_dataset(
        workspace,
        variants=variants,
        anchors=anchors,
        trigger=trigger,
        repeats=repeats,
        dedup_distance=6,
    )
    return SampleResult(
        result=result,
        trigger=trigger,
        repeats=repeats,
        n_variants=n_variants,
        duplicates=len(duplicate_of),
        n_anchors=n_anchors,
        style_tokens=style_tokens,
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
