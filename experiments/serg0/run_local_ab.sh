#!/usr/bin/env bash
# Local A/B grid runner: needs ComfyUI up on :8188. The python script self-symlinks the
# R2 checkpoints into ComfyUI/models/loras, discovers every checkpoint, and renders raw
# txt2img (base + each ckpt) x 20 prompts x 2 bases. Stdlib-only -> plain python3.
set -uo pipefail
curl -s -m5 -o /dev/null http://127.0.0.1:8188/system_stats || { echo "ComfyUI NOT up on :8188"; exit 1; }
exec python3 /home/serg/make-char-dataset/experiments/serg0/local_ab.py
