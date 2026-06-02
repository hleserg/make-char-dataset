"""Workspace layout contract for the character-dataset pipeline.

A single, typed description of where each stage reads and writes. Every path is
derived from one ``root`` so the whole pipeline can be relocated or sandboxed by
changing ``APP_WORKSPACE``. See ``docs/architecture/WORKSPACE.md`` for the prose
contract.

Layout::

    <root>/
      00_passport_import/      role-tagged golden anchors imported from a passport export
      01_generated/            raw variants produced by the generation backend
      02_clean/                deduplicated, size-filtered variants
      03_dataset/<N>_<trig>/   kohya-ready images + caption sidecars
      06_lora/<name>/          trained character LoRA (ai-toolkit output; opt-in)
      manual_review/           anything kicked out for a human to inspect
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    """Resolved, typed view of the pipeline's on-disk layout.

    # PLAYBOOK-START
    # id: derived-workspace-layout
    # title: Single-root derived directory contract
    # status: refined
    # category: pipeline
    # tags: [filesystem, pipeline, config]
    # Derive every stage directory from one configurable root behind named
    # properties instead of scattering string joins across the codebase.
    # Relocating or sandboxing the whole pipeline becomes a one-line change,
    # and stages depend on an interface, not on literal path strings.
    # PLAYBOOK-END
    """

    root: Path

    @property
    def passport_import(self) -> Path:
        """Stage 0 input: role-tagged golden anchors from a passport export."""
        return self.root / "00_passport_import"

    @property
    def generated(self) -> Path:
        """Raw variants produced by the generation backend."""
        return self.root / "01_generated"

    @property
    def clean(self) -> Path:
        """Deduplicated, size-filtered variants."""
        return self.root / "02_clean"

    @property
    def dataset(self) -> Path:
        """Root of the kohya-ready dataset output."""
        return self.root / "03_dataset"

    @property
    def manual_review(self) -> Path:
        """Artifacts kicked out for a human to inspect."""
        return self.root / "manual_review"

    @property
    def lora(self) -> Path:
        """Trained character LoRA output (``06_lora/``).

        The ``train`` stage shells out to ai-toolkit with this as its
        ``training_folder``; the LoRA lands at ``06_lora/<name>/<name>.safetensors``.
        """
        return self.root / "06_lora"

    def training_dir(self, repeats: int, trigger: str) -> Path:
        """Return the kohya training subfolder named ``<repeats>_<trigger>``.

        kohya_ss encodes the per-image repeat count in the folder name, so a
        dataset trained 10x under the trigger ``conan`` lives in ``10_conan``.
        """
        return self.dataset / f"{repeats}_{trigger}"

    def ensure_base(self) -> None:
        """Create the input and review directories that no single stage owns."""
        for path in (self.root, self.passport_import, self.manual_review):
            path.mkdir(parents=True, exist_ok=True)
