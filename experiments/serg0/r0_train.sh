#!/usr/bin/env bash
# Round-0 of the iterative char-LoRA loop, ON THE POD.
# Trains TWO SDXL char-LoRAs in parallel on one 48GB GPU: vanilla SDXL vs Illustrious.
# Compare which base holds the character's drawn style + non-idealized identity.
# Run:  bash r0_train.sh   (detached launch is done by the caller)
set -uo pipefail
WS=/workspace
KOHYA=$WS/sd-scripts
CKPT=$WS/ComfyUI/models/checkpoints
log(){ echo -e "\n=== $* ===" > /proc/1/fd/1 2>/dev/null; echo -e "\n=== $* ==="; }

# 1) kohya sd-scripts + venv (its own cu124 torch, matches the pod)
log "kohya setup"
[ -d "$KOHYA/.git" ] || git clone -q https://github.com/kohya-ss/sd-scripts "$KOHYA"
cd "$KOHYA"
[ -x venv/bin/python ] || python -m venv venv
venv/bin/pip install -q -U pip
venv/bin/pip install -q torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu124
venv/bin/pip install -q -r requirements.txt
venv/bin/pip install -q bitsandbytes

# 2) base checkpoints (Illustrious already present; fetch vanilla SDXL)
log "base checkpoints"
mkdir -p "$CKPT"
[ -f "$CKPT/Illustrious-XL-v1.0.safetensors" ] || wget -c -q -O "$CKPT/Illustrious-XL-v1.0.safetensors" "https://huggingface.co/OnomaAIResearch/Illustrious-XL-v1.0/resolve/main/Illustrious-XL-v1.0.safetensors?download=true"
[ -f "$CKPT/sd_xl_base_1.0.safetensors" ] || wget -c -q -O "$CKPT/sd_xl_base_1.0.safetensors" "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"

# 3) trainset (character-agnostic captions)
log "trainset"
python "$WS/runpod_serg0/build_trainset.py" --refs "$WS/serg0_refs" --out "$WS/serg0_train" --trigger serg0 --repeats 10

# 4) two parallel trainings (≈13GB each, fits 48GB)
log "launch trainings"
COMMON="--train_data_dir=$WS/serg0_train --resolution=1024,1024 --network_module=networks.lora \
 --network_dim=32 --network_alpha=16 --train_batch_size=1 --max_train_steps=1600 \
 --learning_rate=1e-4 --optimizer_type=AdamW8bit --lr_scheduler=cosine --mixed_precision=bf16 \
 --save_precision=fp16 --save_every_n_steps=400 --save_model_as=safetensors --cache_latents \
 --gradient_checkpointing --sdpa --caption_extension=.txt --shuffle_caption --keep_tokens=1 --seed=42"
launch(){ # launch <base_ckpt> <out_dir> <name> <logfile>
  cd "$KOHYA"
  setsid nohup venv/bin/python -m accelerate.commands.launch --num_processes 1 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no sdxl_train_network.py \
    --pretrained_model_name_or_path="$1" --output_dir="$2" --output_name="$3" $COMMON \
    > "$4" 2>&1 </dev/null &
  echo "launched $3 -> pid $!" > /proc/1/fd/1 2>/dev/null
}
launch "$CKPT/sd_xl_base_1.0.safetensors"        "$WS/out/lora_sdxl"        serg0_sdxl "$WS/train_sdxl.log"
launch "$CKPT/Illustrious-XL-v1.0.safetensors"   "$WS/out/lora_illustrious" serg0_ill  "$WS/train_ill.log"
log "R0 launched (serg0_sdxl + serg0_ill); tail /workspace/train_*.log"
