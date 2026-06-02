"""Free, no-GPU mechanism/regression check for the full make-char-dataset pipeline.

Runs the REAL stages ``import -> generate -> clean -> caption`` against stub
backends — no GPU, no model download, no ComfyUI server, no onnxruntime — and
asserts the on-disk contract under the *conditioning-only* doctrine: the golden
anchors stay in ``00_passport_import`` and never leak into the trained
``03_dataset``. This is the check the PreToolUse PR gate runs on every
``gh pr create``.

Run from the project dir:
    uv run python .claude/skills/verifier-dataset/smoke.py

Exit code is non-zero if any critical contract check regressed. Python adds this
script's directory to sys.path, so ``harness`` imports without any path juggling.
"""

from __future__ import annotations

from harness import Check, build_sample_pipeline, make_workspace, report

from make_char_dataset.caption import Wd14Captioner, style_tokens
from make_char_dataset.tagging import StubTagger


def run(check: Check) -> None:
    """Assert the full conditioning-only pipeline contract against a stub sample."""
    workspace, _tmp = make_workspace()
    sample = build_sample_pipeline(workspace, trigger="conan", repeats=10, count=8)

    training_dir = sample.dataset.training_dir  # type: ignore[attr-defined]
    images = sorted(training_dir.glob("*.png"))
    captions = sorted(training_dir.glob("*.txt"))
    texts = [c.read_text(encoding="utf-8") for c in captions]

    # 1) every stage wrote its idempotency marker (00 -> 01 -> 02 -> 03)
    for stage_dir in (
        workspace.passport_import,
        workspace.generated,
        workspace.clean,
        training_dir,
    ):
        check.expect(
            (stage_dir / ".stage_complete").is_file(),
            f"stage '{stage_dir.name}' wrote a .stage_complete marker",
        )

    # 2) kohya folder named exactly <repeats>_<trigger>
    expected_name = f"{sample.repeats}_{sample.trigger}"
    check.expect(
        training_dir.name == expected_name,
        f"kohya folder named '{expected_name}' (got '{training_dir.name}')",
    )

    # 3) every image has a sibling .txt caption
    check.expect(
        len(images) == len(captions) and all(i.with_suffix(".txt").is_file() for i in images),
        f"every image has a caption sidecar ({len(images)} images / {len(captions)} captions)",
    )

    # 4) CONDITIONING-ONLY anti-regression: no golden anchor image is in the dataset
    dataset_hashes = {i.read_bytes() for i in images}
    leaked = dataset_hashes & sample.anchor_hashes
    check.expect(
        not leaked,
        "conditioning-only: no golden anchor leaked into 03_dataset (trained set is "
        "generated variants only)",
    )
    check.expect(
        not list(training_dir.glob("anchor_*.png")),
        "no anchor_* files in the dataset (anchors are conditioning, not training images)",
    )

    # 5) the 5 golden anchors live in 00_passport_import (conditioning) — and there
    check.expect(
        len(list(workspace.passport_import.glob("*.png"))) == 5,
        "5 golden anchors imported into 00_passport_import",
    )

    # 6) the dataset is exactly the cleaned generated variants
    n_kept = len(sample.clean.kept)  # type: ignore[attr-defined]
    check.expect(
        len(images) == n_kept,
        f"dataset images == cleaned variants ({len(images)} == {n_kept})",
    )

    # 7) every caption starts with the character trigger
    check.expect(
        all(t.split(",")[0].strip() == sample.trigger for t in texts),
        "every caption starts with the character trigger token",
    )

    # 8a) the laid-out captions (default StubCaptioner prose) carry no style/medium or
    #     identity-geometry words — guards the default VLM/stub captioner output.
    banned_words = [
        "comic",
        "painterly",
        "watercolor",
        "brown hair",
        "blonde",
        "green eyes",
        "muscular",
    ]
    bled = sorted({w for w in banned_words for t in texts if w in t.lower()})
    check.expect(
        not bled,
        f"default captions carry no style/identity words (Character-Locker); leaked={bled}",
    )

    # 8b) Character-Locker stripping on the WD14 fallback path: StubTagger DOES emit
    #     geometry + style tags ('brown hair', '1boy', 'comic style'); the WD14
    #     captioner must strip them and keep the trigger first. (The default VLM/stub
    #     path omits them by prompt, so this exercises the actual stripping logic.)
    sample_image = sorted(workspace.clean.glob("var_*.png"))[0]
    wd14_caption = (
        Wd14Captioner(StubTagger(), sample.trigger, style_tokens("")).caption(sample_image).lower()
    )
    check.expect(
        all(tok not in wd14_caption for tok in ("comic style", "brown hair", "1boy")),
        "Character-Locker: WD14 captioner strips style + identity-geometry tags",
    )
    check.expect(
        wd14_caption.split(",")[0].strip() == sample.trigger,
        "Character-Locker: WD14 caption begins with the trigger token",
    )

    # 9) the planted near-duplicate was routed to manual_review (never deleted)
    check.expect(
        len(list(workspace.manual_review.glob("*.png"))) >= 1,
        "planted near-duplicate routed to manual_review (not deleted)",
    )

    # 10) no zero-byte / unreadable images
    check.expect(all(i.stat().st_size > 0 for i in images), "no zero-byte images")


def main() -> int:
    """Run all contract checks and return the aggregate exit code."""
    check = Check()
    run(check)
    print(f"\ndataset verifier — {len(check.items)} checks")
    return report(check)


if __name__ == "__main__":
    raise SystemExit(main())
