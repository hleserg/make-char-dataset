---
name: verifier-dataset
description: >-
  Runtime verification for the make-char-dataset pipeline. Use when verifying a change to
  the import / generation / dedup / caption / kohya-layout stages (the /verify skill
  auto-discovers this for the pipeline surface). The free check runs the REAL stages
  (import -> generate -> clean -> caption) against stub backends, so the on-disk
  conditioning-only contract is exercised with no GPU, no model download, no onnxruntime,
  and no running ComfyUI server.
---

# verifier-dataset

The repo's evidence-capture protocol for the character-dataset pipeline. The
honest cheap surface is not image *quality* (a human must eye that) — it is the
**end-to-end stage contract** under the *conditioning-only* doctrine: a synthetic
passport export is run through the REAL stages `import → generate → clean →
caption`, and the on-disk result is asserted — crucially, that the golden anchors
stay conditioning-only (in `00_passport_import` only) and **never leak into the
trained `03_dataset`**. The heavy backends sit behind Protocols, so we verify this
without a GPU.

This is the pipeline analog of create-char-passport's `verifier-gradio` skill. We
took only the *pattern* — a **free, dependency-light, every-PR regression check**
plus an optional **heavy human-eyeball check** — not its Gradio/Playwright
specifics. Here the "free" axis is **no-GPU / no-model-download** instead of
no-API-key/browser.

## Why a stub generator, not the real model

The heavy generation backend (ComfyUI over HTTP+websocket, or local diffusers,
loaded with the external style LoRA) and the captioner (VLM prose via the Gemini
proxy by default, or the legacy WD14 tagger) sit behind Protocols and are **lazily
imported** (the `pure-core-lazy-backend` PLAYBOOK marker in
`src/make_char_dataset/assembly.py`). That lazy-Protocol design is exactly what
lets this check run at all: the smoke uses a `StubCaptioner` and the ≥90 % unit path
never imports torch / diffusers / onnxruntime / the network, so they are fast and need no GPU. The same
reason the heavy deps live in a PEP-735 `[dependency-groups] gpu` group, not an
extra.

## Free, no-GPU mechanism / regression check — one command

Run **from the project dir** (so `uv` resolves the env and the package imports):

```bash
uv run python .claude/skills/verifier-dataset/smoke.py
```

It seeds a temp workspace, writes a synthetic passport export, and runs the
**real** stages `import → generate → clean → caption` (stub generation + stub
captioner, one planted near-duplicate variant), then asserts the on-disk contract:

- every stage wrote its `.stage_complete` marker (`00 → 01 → 02 → 03`),
- the dataset folder is named exactly `<repeats>_<trigger>` (the kohya convention),
- every image has a sibling `.txt` caption,
- **conditioning-only:** no golden anchor image leaks into `03_dataset` (by content
  hash) and there are no `anchor_*` files — the trained set is the generated
  variants only; the 5 anchors live in `00_passport_import` (conditioning) and there,
- the dataset is exactly the cleaned generated variants,
- the planted near-duplicate is routed to `manual_review/` (never deleted),
- every caption starts with the **character** trigger token,
- **no style or identity-geometry tokens** leak into captions (Character-Locker:
  style lives in the external LoRA; geometry binds to the trigger), and
- no zero-byte images.

Exit code is non-zero if any **critical** mechanism check regressed. This is the
check the `PreToolUse` PR gate in `.claude/settings.json` runs on every
`gh pr create`, for free, with no API key and no GPU.

## Heavy, real-generation check (GPU + models + human eye)

The generation backend (HLE-766) and the VLM/WD14 captioners now exist; use this
tier when you want to judge **output quality / character-identity consistency** —
which a human must eye. Provide:

- a running ComfyUI (`APP_COMFY_URL`) or local diffusers weights, on a CUDA GPU,
- the pre-trained **style LoRA** (`APP_STYLE_LORA_PATH`) + `APP_STYLE_PROMPT`,
- license-safe **ControlNet** models (OpenPose/depth) — never InsightFace-based
  PuLID/IP-Adapter/InstantID for the commercial dataset (HLE-668),
- a real **passport export** (`state.json` + `refs/`) to multiply.

Point the real `ComfyBackend` / `WD14Tagger` at the live models, run the same
`import → generate → clean → caption` flow on the real export (the
`kael-thornwood` fixture is handy), and write a contact sheet for human review.
Reuse `harness.py` helpers; do not re-invent the workspace/stub plumbing.

## Extending for new stages

Import `harness.py` and write a small scenario (see `smoke.py`). Helpers:
`make_workspace`, `build_sample_pipeline` (runs the real stages on a synthetic
export), `Check.expect` / `Check.note`, and `report` (the exit-code contract).

## Gotchas already handled (don't re-discover)

- **Run `uv` from the project dir** — `cd $TMP` then `uv run` loses the project
  env and `make_char_dataset` won't import.
- **UTF-8 stdout** — `harness.py` reconfigures it; captions/labels use `—` / `✓`
  and the Windows console is cp1251.
- **Isolated workspace** — `make_workspace()` seeds a fresh temp dir, so the
  verifier never touches the real `APP_WORKSPACE`.
- **Keep torch/diffusers/onnxruntime OUT of the smoke import path** — inject the
  `StubGenerator`; the real backend stays behind its lazily-imported Protocol.
- **Solid-color stub images collapse under phash** (a flat image hashes to a
  constant). `StubGenerator` writes per-index pseudo-random noise so distinct
  indices get distinct hashes and `duplicate_of` produces a real near-duplicate.
