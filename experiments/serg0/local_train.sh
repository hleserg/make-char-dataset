#!/usr/bin/env bash
# Local char-LoRA training on the RTX 5070 Ti (16GB, cu128) — extend to 4000 steps from
# scratch to find the quality PEAK (clean cosine curve, save every 400 -> 10 checkpoints).
# Same proven recipe as the pod R1 (refs x10 + keepers x2, dim32/alpha16, AdamW8bit, bf16),
# one base at a time (16GB can't fit two SDXL trainings). Outputs serg0_<base>_r2.
#
# Usage: bash local_train.sh sdxl   |   bash local_train.sh ill
set -uo pipefail
BASE=${1:?sdxl|ill}
KOHYA=/home/serg/sd-scripts
PY=$KOHYA/venv/bin/python
CKPTDIR=/home/serg/ComfyUI/models/checkpoints
ROOT=/home/serg/serg0_train_local
OUTBASE=/home/serg/serg0_lora_out
STEPS=${STEPS:-4000}
if [ "$BASE" = "sdxl" ]; then
  CKPT=$CKPTDIR/sd_xl_base_1.0.safetensors; NAME=serg0_sdxl_r2; OUT=$OUTBASE/sdxl_r2
elif [ "$BASE" = "ill" ]; then
  CKPT=$CKPTDIR/Illustrious-XL-v1.0.safetensors; NAME=serg0_ill_r2; OUT=$OUTBASE/ill_r2
else
  echo "base must be sdxl|ill"; exit 1
fi
[ -f "$CKPT" ] || { echo "MISSING ckpt $CKPT"; exit 1; }
mkdir -p "$OUT"

cd "$KOHYA"
exec "$PY" -m accelerate.commands.launch --num_processes 1 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no sdxl_train_network.py \
  --pretrained_model_name_or_path="$CKPT" \
  --train_data_dir="$ROOT" --output_dir="$OUT" --output_name="$NAME" \
  --resolution=1024,1024 --network_module=networks.lora \
  --network_dim=32 --network_alpha=16 --train_batch_size=1 --max_train_steps="$STEPS" \
  --learning_rate=1e-4 --optimizer_type=AdamW8bit --lr_scheduler=cosine \
  --mixed_precision=bf16 --save_precision=fp16 --save_every_n_steps=400 \
  --save_model_as=safetensors --cache_latents --cache_latents_to_disk \
  --gradient_checkpointing --sdpa --caption_extension=.txt --shuffle_caption \
  --keep_tokens=1 --seed=42 --max_data_loader_n_workers=2 --persistent_data_loader_workers
