#!/usr/bin/env python3
"""Character-agnostic WD14 captioning for curated round-N keepers.

Community best-practice char-LoRA captioning: WD14 (wd-eva02-large-tagger-v3) reads
the ACTUAL rendered frame (not the prompt, which drifts), then we PRUNE every tag
that describes Serg0's CONSTANT identity / build / art-style so those bind to the
trigger, keeping only the VARIABLE scene tags (pose, framing, expression, clothing,
background, lighting, objects). Final sidecar caption: `serg0, 1boy, solo, <tags>`.

The denylist IS the quality lever (per review): anything left in the caption is
something the LoRA will treat as optional — so identity/build/style MUST be pruned,
and any leaked idealization tag (muscular, abs) MUST be killed so it never binds.

Refs are NOT captioned here — they keep their hand-written captions in
build_trainset.py (incl. the tattoo close-ups). This only writes <img>.txt next to
each keeper PNG. Runs in the isolated CPU-onnxruntime venv (no torch/cuDNN clash).

Usage: wd14venv/bin/python caption_keepers.py <dir> [--trigger serg0] [--show N]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# --- PRUNE: constants that must bind to the trigger, not sit in the caption ---
# exact-match danbooru tags (underscores already normalised to spaces)
PRUNE_EXACT = {
    # gender/meta anchors we re-add ourselves
    "1boy",
    "male focus",
    "solo",
    "manly",
    # head / hair (he's bald w/ light stubble — constant)
    "bald",
    "shaved head",
    "very short hair",
    "buzz cut",
    "no hair",
    "short hair",
    "bald head",
    # facial hair (constant: light stubble, no mustache)
    "facial hair",
    "beard",
    "stubble",
    "goatee",
    "mustache",
    "sideburns",
    "soul patch",
    "full beard",
    "thick eyebrows",
    # age / face descriptors that are identity
    "mature male",
    "old man",
    "old",
    "wrinkles",
    "forehead",
    "nose",
    # skin tone (constant)
    "dark skin",
    "dark-skinned male",
    "tan",
    "tanned",
    "pale skin",
    "light skin",
    "brown skin",
    # eye COLOUR (constant) — gaze/expression eye tags are kept (variable)
    "brown eyes",
    "blue eyes",
    "green eyes",
    "black eyes",
    "grey eyes",
    "gray eyes",
    "red eyes",
    "yellow eyes",
    "hazel eyes",
    "amber eyes",
    "heterochromia",
    # piercings (constant: left-ear plug + eyebrow piercing)
    "piercing",
    "ear piercing",
    "earrings",
    "eyebrow piercing",
    "ear plug",
    "industrial piercing",
    # ANTI-IDEALIZATION — must never bind to serg0 (he's ordinary build)
    "muscular",
    "muscular male",
    "abs",
    "pectorals",
    "toned",
    "bara",
    "large pectorals",
    "biceps",
    "veiny",
    "muscular arms",
    "six pack",
    # art-style / medium (the comic/watercolor look binds to the trigger)
    "watercolor",
    "watercolor (medium)",
    "traditional media",
    "sketch",
    "painting",
    "painting (medium)",
    "lineart",
    "comic",
    "ink",
    "oil painting",
    "art",
    "marker (medium)",
    "graphite (medium)",
    "colored pencil (medium)",
    "drawing",
    "illustration",
    # tattoos (constant identity — bind to trigger)
    "tattoo",
    "arm tattoo",
    "forearm tattoo",
    "tattoos",
}
# substring rules: drop any tag containing one of these fragments
PRUNE_SUBSTR = ("tattoo", "piercing", "muscular", " abs", "pectoral", "beard", "stubble")


def keep(tag: str) -> bool:
    t = tag.replace("_", " ").strip().lower()
    if not t:
        return False
    if t in PRUNE_EXACT:
        return False
    return all(frag not in t for frag in PRUNE_SUBSTR)


def caption_for(tagger, path, trigger):
    from PIL import Image

    res = tagger.tag(Image.open(path).convert("RGB"))
    # wdtagger >=0.x exposes .general_tags (tuple); be defensive across versions
    raw = getattr(res, "general_tags", None)
    if raw is None:
        gts = getattr(res, "general_tags_string", "") or ""
        raw = [x for x in gts.split(",")]
    tags = []
    seen = set()
    for tg in raw:
        t = tg.replace("_", " ").strip().lower()
        if keep(t) and t not in seen:
            seen.add(t)
            tags.append(t)
    return f"{trigger}, 1boy, solo, " + ", ".join(tags)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--trigger", default="serg0")
    ap.add_argument("--show", type=int, default=0, help="print first N captions")
    args = ap.parse_args()

    from wdtagger import Tagger

    tagger = Tagger()
    imgs = sorted(glob.glob(os.path.join(args.dir, "*.png")))
    n = 0
    for i, p in enumerate(imgs):
        cap = caption_for(tagger, p, args.trigger)
        with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as fh:
            fh.write(cap + "\n")
        n += 1
        if i < args.show:
            print(f"{os.path.basename(p)} :: {cap}", flush=True)
    print(f"captioned {n} images in {args.dir}", flush=True)


if __name__ == "__main__":
    main()
