#!/usr/bin/env python3
"""Quantitative checkpoint eval over the already-rendered A/B grid (no regeneration).

Turns the visual A/B into numbers so the training PEAK and the onset of overfit are
objective, not just eyeballed. Uses CLIP (openai/clip-vit-large-patch14, already cached):

  • likeness   = mean cosine(generated, the 18 ORIGINAL refs)        ← image↔image, no captions
  • adherence  = mean cosine(generated, its PROMPT text)  (CLIPScore) ← does the scene obey the prompt
  • diversity  = 1 - mean pairwise cosine across the 20 prompt images ← low ⇒ collapsed/overfit

Peak = likeness high while adherence + diversity have NOT yet dropped. Overfit shows as
likeness ↑ but adherence/diversity ↓. The base (no-LoRA) rows anchor the tradeoff: ~max
adherence/diversity, ~min likeness. CLIP measures overall look (identity+style), not exact
facial ID — a strong guide, with the human eye final on fine details.

Run:  /root/comfy-venv/bin/python eval_checkpoints.py
Out:  prints tables + writes /home/serg/serg0_ab_out/eval.json and eval_summary.md
"""
# ruff: noqa: E501 - docstring tables / a few long metric-format lines

from __future__ import annotations

import json
import os
import re
import sys

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_ab import PROMPTS

AB = "/home/serg/serg0_ab_out"
REFS = "/home/serg/serg0_train_local/10_serg0"
MODEL = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip():
    model = CLIPModel.from_pretrained(MODEL).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    return model, proc


@torch.no_grad()
def embed_images(model, proc, paths, batch=16):
    # explicit projection (vision_model -> visual_projection) = the CLIP shared space,
    # robust across transformers versions where get_image_features wraps its output.
    embs = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch]]
        px = proc(images=imgs, return_tensors="pt").to(DEVICE)
        out = model.vision_model(pixel_values=px["pixel_values"])
        e = model.visual_projection(out.pooler_output)
        embs.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.cat(embs) if embs else torch.empty(0, device=DEVICE)


@torch.no_grad()
def embed_texts(model, proc, texts):
    tok = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = model.text_model(input_ids=tok["input_ids"], attention_mask=tok.get("attention_mask"))
    e = model.text_projection(out.pooler_output)
    return torch.nn.functional.normalize(e, dim=-1)


def variant_dirs():
    out = {"sdxl": {}, "ill": {}}
    for d in sorted(os.listdir(AB)):
        full = os.path.join(AB, d)
        if not os.path.isdir(full):
            continue
        if d in ("base_sdxl", "base_ill"):
            out[d.split("_")[1]][-1] = full  # -1 = base (no LoRA)
        else:
            m = re.match(r"(sdxl|ill)_s(\d+)$", d)
            if m:
                out[m.group(1)][int(m.group(2))] = full
    return out


def score_variant(model, proc, path, ref_embs, txt_embs):
    paths = [os.path.join(path, f"{i:03d}.png") for i in range(len(PROMPTS))]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None
    img = embed_images(model, proc, paths)
    likeness = (img @ ref_embs.T).mean().item()  # mean cos to all refs, over all images
    # adherence: each image vs its own prompt text (index-aligned)
    idx = [int(os.path.basename(p)[:3]) for p in paths]
    adh = torch.stack([img[k] @ txt_embs[idx[k]] for k in range(len(paths))]).mean().item()
    # diversity: 1 - mean pairwise cosine among the variant's images
    sim = img @ img.T
    n = sim.shape[0]
    off = (sim.sum() - n) / (n * (n - 1)) if n > 1 else torch.tensor(1.0)
    diversity = (1 - off).item()
    return {"n": len(paths), "likeness": likeness, "adherence": adh, "diversity": diversity}


def main():
    print(f"loading CLIP {MODEL} on {DEVICE} ...", flush=True)
    model, proc = load_clip()
    refs = [os.path.join(REFS, f) for f in sorted(os.listdir(REFS)) if f.endswith(".png")]
    print(f"refs: {len(refs)} | prompts: {len(PROMPTS)}", flush=True)
    ref_embs = embed_images(model, proc, refs)
    txt_embs = embed_texts(model, proc, list(PROMPTS))

    dirs = variant_dirs()
    report = {}
    for base in ("sdxl", "ill"):
        rows = []
        for step in sorted(dirs[base]):
            s = score_variant(model, proc, dirs[base][step], ref_embs, txt_embs)
            if s:
                s["step"] = step
                rows.append(s)
        report[base] = rows

    # composite + peak pick (over LoRA steps only): balance likeness × adherence,
    # with a soft diversity guard (penalise collapse below 60% of the base diversity).
    summary_lines = []
    for base in ("sdxl", "ill"):
        rows = report[base]
        base_row = next((r for r in rows if r["step"] == -1), None)
        base_div = base_row["diversity"] if base_row else 1.0
        steps = [r for r in rows if r["step"] >= 0]
        for r in steps:
            guard = 1.0 if r["diversity"] >= 0.6 * base_div else (r["diversity"] / (0.6 * base_div))
            r["composite"] = r["likeness"] * r["adherence"] * guard
        peak = max(steps, key=lambda r: r["composite"]) if steps else None
        for r in rows:
            r["peak"] = bool(peak and r["step"] == peak["step"])

        title = f"=== {base.upper()} (base no-LoRA + checkpoints) ==="
        print("\n" + title, flush=True)
        hdr = f"{'step':>6} {'likeness':>9} {'adherence':>10} {'diversity':>10} {'composite':>10}  peak"
        print(hdr, flush=True)
        summary_lines += [title, hdr]
        for r in rows:
            lbl = "base" if r["step"] == -1 else str(r["step"])
            comp = f"{r.get('composite', 0):.4f}" if r["step"] >= 0 else "   —"
            line = (
                f"{lbl:>6} {r['likeness']:>9.4f} {r['adherence']:>10.4f} "
                f"{r['diversity']:>10.4f} {comp:>10}  {'<= PEAK' if r['peak'] else ''}"
            )
            print(line, flush=True)
            summary_lines.append(line)
        if peak:
            msg = f"-> {base.upper()} recommended peak: step {peak['step']} (composite {peak['composite']:.4f})"
            print(msg, flush=True)
            summary_lines.append(msg)

    with open(os.path.join(AB, "eval.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(AB, "eval_summary.md"), "w") as fh:
        fh.write(
            "# Serg0 R2 checkpoint eval (CLIP)\n\n```\n" + "\n".join(summary_lines) + "\n```\n"
        )
    print("\nwrote eval.json + eval_summary.md", flush=True)


if __name__ == "__main__":
    main()
