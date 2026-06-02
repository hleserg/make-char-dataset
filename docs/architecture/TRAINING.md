# Character-LoRA training (`train` stage)

The dataset pipeline (`import → generate → clean → caption`) ends at a kohya-ready
`03_dataset/<repeats>_<trigger>/`. The optional **`train`** stage turns that into a
trained character LoRA at `06_lora/<name>/<name>.safetensors`.

This stage exists to satisfy the **style-coherent stack** (HLE-802): the goal is a
char-LoRA that, loaded together with the comic **style** LoRA, renders a consistent
character *in the comic style* —

```
Flux.1-dev  +  cmcstyle (style LoRA)  +  <char>_char (this LoRA)
```

— where the style does not distort the face/body and the character does not break
the style.

## Why ai-toolkit, not kohya sd-scripts

The char-LoRA **must** train on the same base as the style LoRA — **FLUX.1-dev** —
or the two LoRAs will not stack. Training Flux on a ~16 GB GPU needs the base
quantized. kohya sd-scripts loads the whole Flux DiT to the GPU *before* its
block-swap offload and OOMs at the load peak; it has no sub-fp8 **training**
quantization. [ostris **ai-toolkit**](https://github.com/ostris/ai-toolkit) can
quantize the transformer **and** the text encoder to `qfloat8` with `low_vram`,
which fits. So this stage shells out to ai-toolkit (the sibling *style* pipeline uses
kohya; the char side does not).

The trainer runs in **ai-toolkit's own venv** (its cu128/sm_120 torch). We only
*spawn* it — never import it — so its torch/diffusers pins stay isolated from this
package's numpy/Pillow (the same pure-core/lazy-backend split as `generate.py`).

## The recipe

`build_aitoolkit_config` emits an ai-toolkit job config (JSON; ai-toolkit reads
`.json`/`.yaml` alike) mirroring the recipe that has actually trained a Flux LoRA
on this box. The load-bearing parts:

| Config | Value | Why |
|--------|-------|-----|
| `model.name_or_path` | `black-forest-labs/FLUX.1-dev` | same base as the style LoRA — mandatory to stack |
| `model.is_flux` | `true` | Flux training path |
| `model.quantize` + `qtype`/`qtype_te` | `true` / **`qfloat8`** | quantize transformer **and** text encoder — the bit kohya can't do. **Not `qint4`** (see gotchas) |
| `model.low_vram` | `true` | quantize on CPU + stream blocks; keeps the GPU load peak low (the GPU also drives the display) |
| `train.dtype` | `bf16` | Flux is bf16-native; fp16 risks an empty/NaN LoRA |
| `train.optimizer` | `adafactor` | near-zero optimizer state → fits ~16 GB (adamw8bit needs more) |
| `network` | `lora`, `linear=32`, `linear_alpha=16` | LoRA rank/alpha (16–32 fine; tune by eval) |
| `datasets[0].folder_path` | `03_dataset/<N>_<trigger>` | our kohya dataset |
| `datasets[0].caption_ext` + `shuffle_tokens=false` | `txt` | `.txt` sidecars; the leading `<char>_char` trigger stays pinned |
| `train.train_text_encoder` | `false` | not supported for Flux |
| `train.noise_scheduler` | `flowmatch` | Flux flow-matching |
| `train.disable_sampling` | `true` | in-training sampling moves the whole transformer to the GPU at once and OOMs ~16 GB; eval **post-hoc in ComfyUI** instead |

**Captions** stay content + identity only (the trigger token + only what *varies*:
pose, outfit, expression, background, camera shot). Identity features (face shape,
hair colour, body) are **left undescribed** so they bind to the trigger — the
inverse of a style LoRA. Style words are never written; the cmcstyle LoRA carries
the style at inference. Every knob is overridable via `APP_TRAIN_*` (see `.env.example`).

### Box gotchas (RTX 5070 Ti, 16 GB, drives the display)

- **Never `qint4` with `low_vram`.** qint4's `int4pack` kernel is CUDA-only, but
  `low_vram` quantizes on the CPU → `NotImplementedError`; ai-toolkit then writes a
  config but **no `.safetensors`** (a silent failure). `build_aitoolkit_config`
  raises on this combo. Use `qfloat8` (fp8 casts fine on CPU).
- **Run online.** Do not set `HF_HUB_OFFLINE=1` — ai-toolkit's `model_info` call
  raises instead of using the cache; the cached rev equals `main`, so online does
  not re-download. The HF token may live in the `huggingface-cli` cache rather than
  `.env` (the `doctor` token check then WARNs, not FAILs — both are honored).
- **Pick the best checkpoint by eval**, not the last (`max_step_saves_to_keep=8`):
  characters want likeness *and* prompt flexibility, which peak before the final step.
- **De-risk first:** before a 4–5 h run (~7–9 s/it at 512 under `low_vram`), do a
  fit-check (`APP_TRAIN_STEPS=5 APP_TRAIN_SAVE_EVERY=5`) and confirm a `.safetensors`
  is written.

## Running it

Training is **opt-in** and excluded from `run-all` by default (`APP_RUN_TRAIN=false`)
because it needs a GPU and the Flux base:

```bash
# 1) check the heavy-tier environment first (fast, no GPU):
make-char-dataset doctor          # add --probe to also report torch + CUDA

# 2) train the char-LoRA from 03_dataset into 06_lora:
make-char-dataset train --trigger kael
#   -> 06_lora/kael/kael.safetensors  (+ aitoolkit_config.json, train.log)
```

`doctor` validates the prerequisites (ai-toolkit clone + venv, an `HF_TOKEN` for the
gated FLUX.1-dev base, a built dataset) so a missing piece is caught before a long
run starts. `train` is idempotent: an already-trained LoRA is returned as-is unless
`--force`.

## Evaluating (DoD)

Acceptance is judged **strictly in the stack** `Flux + cmcstyle + <char>_char`:
identity holds, style holds, neither overrides the other, and the triggers stay
isolated. That side-by-side eval grid is produced by a sibling stage (HLE-804
"stack eval"); restylizing the identity assets into the style space (the denoise
bridge) is the sibling "restylization generation" stage. This stage is only the
training step between them.

## Running on a cloud GPU (opt-in)

The local box trains Flux at the edge of 16 GB. To run the heavy training on a
cheap cloud GPU instead, set `APP_TRAIN_BACKEND=ssh` (+ `APP_TRAIN_SSH_*`): the
stage rsyncs the dataset to a rented GPU box, runs ai-toolkit there, and rsyncs the
LoRA back — the local default is unchanged. Inference (the eval grid) goes to a
remote ComfyUI just by pointing `APP_COMFY_URL` at it. See
[Cloud GPU](CLOUD.md).

## Verification boundary

Like the rest of the pipeline, the **logic** is built and unit-tested GPU-free: the
config/launch/progress builders are pure and pinned to the proven recipe with a
`FakeTrainer`; only the ai-toolkit subprocess launch is uncovered. A real Flux run
needs the heavy tier (a GPU, the FLUX.1-dev base, the cmcstyle LoRA, ai-toolkit's
venv) — provided locally, not in CI.
