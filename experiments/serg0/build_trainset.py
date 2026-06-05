#!/usr/bin/env python3
"""Build a kohya SDXL char-LoRA trainset from a character's reference images.

Character-agnostic by design: captions are `trigger, 1boy/1girl, solo, <variable
scene tags>` — pose/view/outfit/expression/background only. Identity, body type
and art style are NOT captioned, so they bind to the trigger from the refs (works
for an athlete or an average build alike — no anti-idealization hacks).

Layout produced: <out>/<repeats>_<trigger>/<img> + <img>.txt  (kohya folder form).
Reused across rounds of the iterative loop and across future characters.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Per-file scene captions for the Serg0 golden set (descriptive names → variable tags).
# Falls back to a generic caption for any unlisted image (e.g. curated round-N gens).
SERG0_CAPTIONS = {
    "face_front_neutral": "portrait, looking at viewer, grey shirt, simple background",
    "face_profile_left": "portrait, profile, from side, grey shirt, simple background",
    "emotion_neutral_front": "portrait, looking at viewer, grey shirt, simple background",
    "emotion_angry_front": "portrait, angry, looking at viewer, grey shirt, simple background",
    "emotion_smile_front": "portrait, smile, looking at viewer, grey shirt, simple background",
    "body_front_itoutfit": "full body, standing, grey t-shirt, jeans, sneakers, simple background",
    "body_front_itoutfit_b": "full body, standing, grey t-shirt, jeans, sneakers, simple background",
    "body_back_itoutfit": "full body, from behind, standing, grey t-shirt, jeans, simple background",
    "body_front_summer": "full body, standing, white t-shirt, shorts, sandals, simple background",
    "body_front_summer_armsout": "full body, standing, arms at sides, white t-shirt, shorts, sandals, simple background",
    "body_profile_summer": "full body, profile, from side, white t-shirt, shorts, simple background",
    "body_profile_summer_b": "full body, profile, from side, white t-shirt, shorts, simple background",
    "body_back_summer": "full body, from behind, white t-shirt, shorts, simple background",
    "body_front_blazer": "full body, standing, brown jacket, shirt, pants, sneakers, simple background",
    "body_back_blazer": "full body, from behind, brown jacket, pants, simple background",
    "detail_leftforearm_dumspirospero_maze_watch": "tattoo, forearm, close-up, watch, simple background",
    "detail_rightforearm_clock_gears_wind": "tattoo, forearm, close-up, simple background",
    "detail_righthand_sun": "tattoo, hand, close-up, simple background",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="/workspace/serg0_refs", help="dir of reference PNGs")
    ap.add_argument("--out", default="/workspace/serg0_train", help="kohya train root")
    ap.add_argument("--trigger", default="serg0")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--gender", default="1boy")
    ap.add_argument(
        "--extra-dir", default="", help="optional dir of curated round-N gens to fold in"
    )
    args = ap.parse_args()

    dst = Path(args.out) / f"{args.repeats}_{args.trigger}"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(Path(args.refs).glob("*.png")):
        tags = SERG0_CAPTIONS.get(img.stem, "simple background")
        shutil.copy(img, dst / img.name)
        (dst / (img.stem + ".txt")).write_text(
            f"{args.trigger}, {args.gender}, solo, {tags}\n", encoding="utf-8"
        )
        n += 1
    # fold in curated generations (expect a sidecar .txt next to each, else generic)
    if args.extra_dir and Path(args.extra_dir).is_dir():
        for img in sorted(Path(args.extra_dir).glob("*.png")):
            shutil.copy(img, dst / img.name)
            cap = img.with_suffix(".txt")
            text = (
                cap.read_text(encoding="utf-8").strip()
                if cap.exists()
                else f"{args.trigger}, {args.gender}, solo, simple background"
            )
            (dst / (img.stem + ".txt")).write_text(text + "\n", encoding="utf-8")
            n += 1
    print(f"built {n} image+caption pairs in {dst} (trigger '{args.trigger}')")


if __name__ == "__main__":
    main()
