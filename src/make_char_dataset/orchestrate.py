"""``run-all`` orchestrator: drive import -> generate -> clean -> caption end-to-end.

Each stage owns its idempotency (a ``.stage_complete`` marker + self-heal), so
:func:`run_all` is **resumable**: a crash/restart re-invokes the stages and the
already-complete ones return their cached result instead of recomputing. Stage
enable flags (``APP_RUN_IMPORT/GENERATE/CLEAN/CAPTION``) gate which stages run;
``force`` re-runs every enabled stage. Side effects stay inside each stage's folder.

This is a thin coordinator over the per-stage entry points — the stage logic and its
tests live in :mod:`ingest`, :mod:`generate`, and :mod:`caption`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from make_char_dataset.caption import run_caption, run_clean
from make_char_dataset.config import get_settings
from make_char_dataset.generate import run_generate
from make_char_dataset.ingest import run_import
from make_char_dataset.train import run_train

# Order matters: each stage feeds the next. ``train`` is last and opt-in — it is
# heavy (a GPU + the Flux base) and gated off by default (``run_train=False``), so
# ``run-all`` runs the dataset stages only unless training is explicitly enabled.
STAGES: tuple[str, ...] = ("import", "generate", "clean", "caption", "train")


def run_stage(stage: str, *, export_dir: Path | str | None = None, force: bool = False) -> Any:
    """Run a single named pipeline stage and return its result."""
    if stage == "import":
        if export_dir is None:
            raise ValueError("the 'import' stage requires an export_dir")
        return run_import(export_dir, force=force)
    if stage == "generate":
        return run_generate(force=force)
    if stage == "clean":
        return run_clean(force=force)
    if stage == "caption":
        return run_caption(force=force)
    if stage == "train":
        return run_train(force=force)
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")


def run_all(export_dir: Path | str, *, force: bool = False) -> dict[str, Any]:
    """Run every enabled stage in order; return ``{stage: result}`` for those that ran.

    Resumable and idempotent via each stage's marker; ``APP_RUN_*`` flags gate which
    stages execute (a disabled upstream stage with an enabled downstream one will
    surface the downstream stage's "run the previous stage first" error).
    """
    settings = get_settings()
    enabled = {
        "import": settings.run_import,
        "generate": settings.run_generate,
        "clean": settings.run_clean,
        "caption": settings.run_caption,
        "train": settings.run_train,
    }
    results: dict[str, Any] = {}
    for stage in STAGES:
        if enabled[stage]:
            results[stage] = run_stage(stage, export_dir=export_dir, force=force)
    return results
