#!/usr/bin/env python3
"""Publish the curated Serg0 char-LoRA dataset to a PRIVATE HF dataset repo, then
verify the push by listing remote files and matching counts.

Uploads ONLY: curated keepers (png + WD14 .txt) + original refs (png + build_trainset
captions) + README. NEVER the proprietary style model or any key/.env. Repo is private
by default — it is a real person's likeness.

Usage: wd14venv/bin/python hf_upload.py [--repo hleserg/serg0-char-dataset] [--public]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys

WS = "/workspace"
KEEPERS = f"{WS}/serg0_dataset"
REFS = f"{WS}/serg0_refs"
STAGE = f"{WS}/hf_stage"
GENDER = "1boy"
TRIGGER = "serg0"


def load_ref_captions():
    """Reuse build_trainset.py's hand-written ref captions if present."""
    for cand in (f"{WS}/build_trainset.py", f"{WS}/runpod_serg0/build_trainset.py"):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("bt", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "SERG0_CAPTIONS", {})
    return {}


def stage(keep_counts):
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(f"{STAGE}/keepers")
    os.makedirs(f"{STAGE}/refs")
    # keepers: png + existing WD14 .txt
    nk = 0
    for f in sorted(os.listdir(KEEPERS)):
        if f.endswith((".png", ".txt")):
            shutil.copy(os.path.join(KEEPERS, f), f"{STAGE}/keepers/{f}")
            if f.endswith(".png"):
                nk += 1
    # refs: png + generated caption
    refcaps = load_ref_captions()
    nr = 0
    for f in sorted(os.listdir(REFS)):
        if not f.endswith(".png"):
            continue
        shutil.copy(os.path.join(REFS, f), f"{STAGE}/refs/{f}")
        stem = os.path.splitext(f)[0]
        tags = refcaps.get(stem, "simple background")
        with open(f"{STAGE}/refs/{stem}.txt", "w", encoding="utf-8") as fh:
            fh.write(f"{TRIGGER}, {GENDER}, solo, {tags}\n")
        nr += 1
    readme = f"""---
license: other
language: [en]
tags: [character-lora, dataset, sdxl, illustrious, personal]
pretty_name: Serg0 character LoRA dataset
---

# Serg0 — character LoRA dataset (curated, improved, captioned)

Private dataset for training a **character LoRA** of a single real person
(non-idealized, hand-drawn comic/watercolor style). Trigger word: **`{TRIGGER}`**.

## Contents
- `refs/` — {nr} original passport reference images (the gold ground truth: exact
  identity, build, and the forearm/hand tattoos) + character-agnostic captions.
- `keepers/` — {keep_counts["total"]} curated, quality-improved generated variants
  ({keep_counts["sdxl"]} from vanilla SDXL, {keep_counts["ill"]} from Illustrious-XL),
  each with a WD14 character-agnostic caption.

## Pipeline
18 refs → txt2img multiplication on two SDXL-family bases (with round-0 char-LoRAs) →
best-practice improver (FaceDetailer face+hands + UltimateSDUpscale ESRGAN) →
manual curation (kept the faithful frames only) → WD14 captioning.

## Captioning doctrine (character-agnostic)
Captions describe ONLY what VARIES (pose, framing, expression, clothing, background,
lighting). Identity, build, the comic/watercolor art style, piercings and tattoos are
intentionally NOT captioned, so they bind to the `{TRIGGER}` trigger.

## Usage / license
Likeness of a real person. **Personal, non-commercial use only. Do not redistribute.**
"""
    with open(f"{STAGE}/README.md", "w", encoding="utf-8") as fh:
        fh.write(readme)
    return nk, nr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="hleserg/serg0-char-dataset")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = open(f"{WS}/.hf_token").read().strip()
    api = HfApi(token=token)

    # split counts for README
    keeps = [f for f in os.listdir(KEEPERS) if f.endswith(".png")]
    kc = {
        "total": len(keeps),
        "sdxl": sum("_sdxl_" in f for f in keeps),
        "ill": sum("_illustrious_" in f for f in keeps),
    }
    nk, nr = stage(kc)
    local_files = sum(len(fs) for _, _, fs in os.walk(STAGE))
    print(f"staged: {nk} keeper png, {nr} ref png -> {local_files} files total", flush=True)

    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=STAGE,
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Add curated+improved+captioned Serg0 char-LoRA dataset",
    )
    remote = api.list_repo_files(args.repo, repo_type="dataset")
    rk = sum(f.startswith("keepers/") and f.endswith(".png") for f in remote)
    rkt = sum(f.startswith("keepers/") and f.endswith(".txt") for f in remote)
    rr = sum(f.startswith("refs/") and f.endswith(".png") for f in remote)
    print(f"REMOTE verify: keepers png={rk} txt={rkt}, refs png={rr}, total={len(remote)}")
    ok = rk == nk and rkt == nk and rr == nr
    print("UPLOAD_OK" if ok else "UPLOAD_MISMATCH")
    vis = "public" if args.public else "private"
    print(f"repo: https://huggingface.co/datasets/{args.repo} ({vis})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
