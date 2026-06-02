"""Tests for make_char_dataset.tagging (Stub tagger + pure threshold helper)."""

from __future__ import annotations

from pathlib import Path

import pytest

from make_char_dataset.tagging import (
    DEFAULT_TAG_THRESHOLD,
    StubTagger,
    parse_wd14_labels,
    threshold_tags,
)


def test_threshold_tags_selects_and_orders_by_score() -> None:
    scores = [0.9, 0.2, 0.5]
    labels = ["standing", "low_conf", "green_coat"]
    # underscores become spaces; below-threshold dropped; ordered by score
    assert threshold_tags(scores, labels, 0.35) == ["standing", "green coat"]


def test_threshold_tags_default_threshold() -> None:
    assert threshold_tags([DEFAULT_TAG_THRESHOLD, 0.1], ["a", "b"]) == ["a"]


def test_threshold_tags_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        threshold_tags([0.9], ["a", "b"], 0.3)


def test_parse_wd14_labels_skips_header() -> None:
    # canonical SmilingWolf CSV starts with a tag_id,name,... header that MUST be
    # skipped, or every label shifts by one relative to its score (silent mislabel).
    csv_text = "tag_id,name,category,count\n1,standing,0,100\n2,green_coat,0,50\n3,extra,0,10\n"
    assert parse_wd14_labels(csv_text, 2) == ["standing", "green_coat"]


def test_parse_wd14_labels_handles_headerless() -> None:
    assert parse_wd14_labels("1,a,0,5\n2,b,0,3\n", 2) == ["a", "b"]


def test_stub_tagger_is_deterministic(tmp_path: Path) -> None:
    tagger = StubTagger()
    tags = tagger.tag(tmp_path / "anything.png")
    assert tags == tagger.tag(tmp_path / "other.png")  # independent of image
    assert "brown hair" in tags and "comic style" in tags  # exercises the filters
