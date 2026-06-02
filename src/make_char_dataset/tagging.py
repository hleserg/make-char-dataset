"""Auto-tagging behind a Protocol — WD14 (onnxruntime) in production, a Stub in CI.

The caption stage ([4]) needs content tags (pose / outfit / background / lighting)
for each generated image. WD14 (an ONNX tagger) is the production source, run via
**onnxruntime** so the path stays torch-free. It is heavy (model download +
inference) and exercised only in the local heavy tier, so it is lazy-imported
behind :class:`Tagger` and its inference is omitted from coverage; the pure
score-thresholding + label-parsing helpers are unit-tested. :class:`StubTagger` is
the GPU-free CI double — deterministic tags that exercise the Character-Locker
filters.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

# Default WD14 confidence threshold (SmilingWolf's recommended general-tag cutoff).
DEFAULT_TAG_THRESHOLD = 0.35


@runtime_checkable
class Tagger(Protocol):
    """Returns content tags for one image (most-confident first)."""

    def tag(self, image_path: Path) -> list[str]:
        """Tag ``image_path`` and return tag strings, highest confidence first."""
        ...


@dataclass
class StubTagger:
    """Deterministic, GPU-free :class:`Tagger` for tests and the verifier smoke.

    Returns a fixed, representative tag set (content + geometry + a style token) so
    the Character-Locker geometry/style filtering is exercised without onnxruntime.
    """

    tags: list[str] = field(
        default_factory=lambda: [
            "standing",
            "green coat",
            "forest background",
            "soft lighting",
            "brown hair",  # geometry — must be filtered out
            "1boy",  # geometry — must be filtered out
            "comic style",  # style — must be filtered out
        ]
    )

    def tag(self, image_path: Path) -> list[str]:
        """Return the configured tag list (independent of the image)."""
        return list(self.tags)


def threshold_tags(
    scores: list[float], labels: list[str], threshold: float = DEFAULT_TAG_THRESHOLD
) -> list[str]:
    """Select labels whose score is >= ``threshold``, highest score first (pure)."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    chosen = [
        (score, label) for score, label in zip(scores, labels, strict=True) if score >= threshold
    ]
    chosen.sort(key=lambda pair: pair[0], reverse=True)
    return [label.replace("_", " ") for _, label in chosen]


def parse_wd14_labels(csv_text: str, count: int) -> list[str]:
    """Extract the first ``count`` WD14 tag names from a ``selected_tags`` CSV.

    The canonical SmilingWolf CSV starts with a ``tag_id,name,category,count``
    header; it is skipped so labels stay aligned with the model's score vector
    (a header left in shifts every label by one — silently mislabelling tags).
    """
    rows = list(csv.reader(io.StringIO(csv_text)))
    if rows and not (rows[0] and rows[0][0].isdigit()):
        rows = rows[1:]  # drop the header row
    return [row[1] for row in rows[:count] if len(row) > 1]


@dataclass
class WD14Tagger:  # pragma: no cover
    """Production :class:`Tagger` driving a WD14 ONNX model via onnxruntime.

    Heavy and network/model-bound (the ONNX model + ``selected_tags.csv`` are
    human-provided inputs, HLE-759), so it is lazy-imported and coverage-omitted;
    only :func:`threshold_tags` and :func:`parse_wd14_labels` are unit-tested.
    """

    model_path: str = ""
    labels_path: str = ""
    threshold: float = DEFAULT_TAG_THRESHOLD

    def tag(self, image_path: Path) -> list[str]:
        """Run WD14 inference and return thresholded tags (heavy; lazy imports)."""
        import numpy as np
        import onnxruntime as ort  # pyright: ignore[reportMissingImports]  # gpu extra (HLE-759)
        from PIL import Image

        session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        _, height, width, _ = session.get_inputs()[0].shape
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            # Square-pad before resize (WD14 reference preprocessing) so non-square
            # inputs are not distorted.
            side = max(rgb.size)
            square = Image.new("RGB", (side, side), (255, 255, 255))
            square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
            resized = square.resize((int(width), int(height)))
            array = np.asarray(resized, dtype=np.float32)[:, :, ::-1][None, ...]
        scores = session.run(None, {session.get_inputs()[0].name: array})[0][0].tolist()
        labels = parse_wd14_labels(Path(self.labels_path).read_text(encoding="utf-8"), len(scores))
        return threshold_tags(scores, labels, self.threshold)
