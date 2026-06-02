"""Tests for make_char_dataset.assembly (the pure dataset-assembly core)."""

from __future__ import annotations

from pathlib import Path

from make_char_dataset.assembly import (
    StubGenerator,
    Variant,
    assemble_dataset,
    build_character_caption,
    dedup_by_phash,
    generate_variants,
)
from make_char_dataset.workspace import Workspace


def test_stub_generator_is_deterministic_and_can_duplicate(tmp_path: Path) -> None:
    gen = StubGenerator(duplicate_of={1: 0})
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    dup = tmp_path / "dup.png"
    gen.generate(0, a)
    gen.generate(0, b)
    gen.generate(1, dup)  # index 1 borrows index 0's seed -> identical image
    assert a.read_bytes() == b.read_bytes()
    assert dup.read_bytes() == a.read_bytes()


def test_generate_variants_writes_numbered_files(tmp_path: Path) -> None:
    paths = generate_variants(StubGenerator(), 3, tmp_path / "gen")
    assert [p.name for p in paths] == ["gen_000.png", "gen_001.png", "gen_002.png"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)


def test_dedup_keeps_first_and_flags_near_duplicates(tmp_path: Path) -> None:
    paths = generate_variants(StubGenerator(duplicate_of={3: 0}), 4, tmp_path / "g")
    kept, dups = dedup_by_phash(paths, max_distance=6)
    assert len(kept) == 3
    assert dups == [paths[3]]


def test_build_character_caption_trigger_first_and_style_stripped() -> None:
    caption = build_character_caption(
        "conan", ["standing", "painterly", "outdoor"], style_tokens=["painterly"]
    )
    assert caption == "conan, standing, outdoor"
    assert build_character_caption("conan", []) == "conan"
    assert build_character_caption("conan", ["  ", "sitting"]) == "conan, sitting"


def test_assemble_dataset_lays_out_kohya_folder_with_anchors_exempt(tmp_path: Path) -> None:
    """Exercise the opt-in ``anchors=`` primitive (dedup-exempt pass-through).

    This is NOT the default route: under the conditioning-only doctrine the
    pipeline passes no anchors, so the golden passport set stays out of the
    training set. The primitive is verified here so generators can opt in later.
    """
    ws = Workspace(tmp_path)
    gen = StubGenerator(duplicate_of={3: 0})
    variant_paths = generate_variants(gen, 4, ws.generated)
    variants = [
        Variant(p, build_character_caption("conan", ["standing", "outdoor"])) for p in variant_paths
    ]

    ws.passport_import.mkdir(parents=True, exist_ok=True)
    anchor_gen = StubGenerator()
    a0, a1 = ws.passport_import / "passport_face.png", ws.passport_import / "passport_body.png"
    anchor_gen.generate(100, a0)
    anchor_gen.generate(101, a1)
    anchors = [Variant(a0, "conan, front portrait"), Variant(a1, "conan, full body")]

    result = assemble_dataset(
        ws,
        variants=variants,
        anchors=anchors,
        trigger="conan",
        repeats=10,
        dedup_distance=6,
    )

    # kohya folder named exactly <repeats>_<trigger>
    assert result.training_dir == ws.dataset / "10_conan"
    assert result.training_dir.name == "10_conan"

    # one near-duplicate variant was routed to manual_review, never deleted
    assert result.deduped == [variant_paths[3]]
    assert (ws.manual_review / "gen_003.png").is_file()
    assert not variant_paths[3].exists()  # moved out of 01_generated

    # 2 anchors (exempt from dedup) + 3 kept variants = 5 image/caption pairs
    images = sorted(p.name for p in result.training_dir.glob("*.png"))
    captions = sorted(p.name for p in result.training_dir.glob("*.txt"))
    assert len(images) == 5
    assert len(captions) == 5
    assert result.anchors_used == 2
    assert len(result.kept) == 3

    # every image has a sibling .txt and every caption starts with the trigger
    for image in result.training_dir.glob("*.png"):
        sidecar = image.with_suffix(".txt")
        assert sidecar.is_file()
        assert sidecar.read_text(encoding="utf-8").split(",")[0].strip() == "conan"
