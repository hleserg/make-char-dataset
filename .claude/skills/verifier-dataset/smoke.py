"""Free, no-GPU mechanism/regression check for the dataset-assembly pipeline.

Runs the REAL assembly path (perceptual-hash dedup -> Character-Locker caption ->
kohya layout) against a StubGenerator — no GPU, no model download, no ComfyUI
server — and asserts the on-disk dataset contract. This is the check the
PreToolUse PR gate runs on every ``gh pr create``.

Run from the project dir:
    uv run python .claude/skills/verifier-dataset/smoke.py

Exit code is non-zero if any critical contract check regressed. Python adds this
script's directory to sys.path, so ``harness`` imports without any path juggling.
"""

from __future__ import annotations

from harness import Check, build_sample_dataset, make_workspace, report


def run(check: Check) -> None:
    """Assert the full on-disk dataset contract against a stub-generated sample."""
    workspace, _tmp = make_workspace()
    sample = build_sample_dataset(workspace, trigger="conan", repeats=10, n_variants=6)
    result = sample.result
    training_dir = result.training_dir  # type: ignore[attr-defined]
    images = sorted(training_dir.glob("*.png"))
    captions = sorted(training_dir.glob("*.txt"))
    texts = [c.read_text(encoding="utf-8") for c in captions]

    # 1) kohya folder named exactly <repeats>_<trigger>
    expected_name = f"{sample.repeats}_{sample.trigger}"
    check.expect(
        training_dir.name == expected_name,
        f"kohya folder named '{expected_name}' (got '{training_dir.name}')",
    )

    # 2) every image has a sibling .txt caption
    check.expect(
        len(images) == len(captions) and all(i.with_suffix(".txt").is_file() for i in images),
        f"every image has a caption sidecar ({len(images)} images / {len(captions)} captions)",
    )

    # 3) expected count: anchors (exempt) + (variants - duplicates)
    expected_count = sample.n_anchors + (sample.n_variants - sample.duplicates)
    check.expect(
        len(images) == expected_count,
        f"image count == anchors + unique variants ({len(images)} == {expected_count})",
    )

    # 4) all golden anchors used (dedup-exempt)
    n_anchor_images = len(list(training_dir.glob("anchor_*.png")))
    check.expect(
        n_anchor_images == sample.n_anchors,
        f"all {sample.n_anchors} golden anchors present and dedup-exempt",
    )

    # 5) planted near-duplicate routed to manual_review, never deleted
    review_images = list(workspace.manual_review.glob("*.png"))
    check.expect(
        len(result.deduped) == sample.duplicates  # type: ignore[attr-defined]
        and len(review_images) == sample.duplicates,
        f"{sample.duplicates} near-duplicate variant(s) routed to manual_review (not deleted)",
    )

    # 6) every caption starts with the character trigger
    check.expect(
        all(t.split(",")[0].strip() == sample.trigger for t in texts),
        "every caption starts with the character trigger token",
    )

    # 7) NO style tokens leaked into captions (style lives in the external LoRA)
    leaked = sorted({tok for tok in sample.style_tokens for t in texts if tok in t.lower()})
    check.expect(not leaked, f"no style tokens in captions (Style Locker); leaked={leaked}")

    # 8) no zero-byte / unreadable images
    check.expect(all(i.stat().st_size > 0 for i in images), "no zero-byte images")


def main() -> int:
    """Run all contract checks and return the aggregate exit code."""
    check = Check()
    run(check)
    print(f"\ndataset verifier — {len(check.items)} checks")
    return report(check)


if __name__ == "__main__":
    raise SystemExit(main())
