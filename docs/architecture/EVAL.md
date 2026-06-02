# In-stack eval grid (`eval` stage)

The acceptance test for HLE-804: judge the trained char-LoRA **in the stack** it
ships in — `Flux + cmcstyle + <char>_char` — not in isolation. `make-char-dataset
eval` renders a fixed-seed contact sheet into `07_eval/` so a human can sign off the
DoD: the style does not distort the face/build, the character does not break the
style, and neither overrides the other.

## The grid: trigger isolation

Rows are prompts; the four columns are **trigger-isolation cells** — the same
prompt (carrying both triggers) rendered with different LoRAs loaded:

| Column | LoRAs loaded | What it proves |
|--------|--------------|----------------|
| `base` | none | the triggers alone do nothing → the LoRAs are what *add* identity + style |
| `style` | cmcstyle | the comic style, on a generic subject (no specific character) |
| `char` | `<char>_char` | the character's identity, **without** the style |
| `stack` | cmcstyle + `<char>_char` | the target: the character rendered in the comic style |

Reading a row left-to-right answers the DoD at a glance: identity holds (compare
`char` vs `stack`), style holds (compare `style` vs `stack`), neither overrides (the
`stack` cell keeps both), and the triggers stay isolated (each only fires when its
LoRA is loaded).

## Running it

Needs a running ComfyUI (Flux.1-dev) plus both LoRAs in `ComfyUI/models/loras`:

```bash
# GPU-free harness check — writes the per-cell ComfyUI graphs + an empty grid:
make-char-dataset eval --trigger kael --dry-run

# the real grid (ComfyUI must be up):
make-char-dataset eval --trigger kael      # -> 07_eval/stack_eval_grid.png + stack_eval.json
```

The style LoRA comes from `APP_STYLE_LORA_PATH`; the char LoRA from
`APP_EVAL_CHAR_LORA_PATH` (or derived as `<train_output_name|trigger>.safetensors`
from the trained output). Both are taken as basenames relative to `models/loras`. Sample at guidance ~3.5 (inference), a fixed seed so cells are comparable.
All knobs are `APP_EVAL_*` (see `.env.example`).

## Design

Pure-core / lazy-backend, like the other stages: the Flux txt2img **graph builder**
(`build_flux_workflow` — chains a `LoraLoaderModelOnly` per LoRA, so 0/1/2 compose),
the cell plan (`loras_for_mode` / `make_cells`) and the Pillow **grid assembler**
(`assemble_grid`) are pure and unit-tested. Only `ComfyStackSampler` (the ComfyUI
HTTP sampler) is uncovered; a `StubStackSampler` renders deterministic tiles so the
grid and `--dry-run` work with no GPU and no server.

## Verification boundary

A real grid needs the heavy tier (ComfyUI + the cmcstyle LoRA + a trained char-LoRA
— human-provided / from the train stage). The harness logic is built and verified
GPU-free; the eyeball acceptance is the human step the grid exists to support.
