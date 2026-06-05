#!/usr/bin/env bash
# Install the IP-Adapter + ControlNet reference backends for the single-gen UI, the
# cu128-SAFE way: ZERO pip installs. We only (a) git-clone the IP-Adapter custom node
# (it has no hard deps beyond torch/PIL, already present) and (b) download model weights.
# ControlNet uses the CORE ComfyUI Canny node + a Canny SDXL model — so NO controlnet_aux,
# NO opencv/mediapipe/timm churn, NO chance of swapping the cu128 nightly torch.
#
# Guarded: records torch.__version__ before and after; if it moved, that's a RED ALERT
# (it must NOT). Idempotent: skips clones/downloads that already exist.
#
# Run AFTER training (GPU free helps the node load test). Then restart ComfyUI and flip
# BACKEND_REF in ui_single.py once the graph wiring (HLE-923/#21) is verified live.
set -uo pipefail
PY=/root/comfy-venv/bin/python
CN=/home/serg/ComfyUI/custom_nodes
M=/home/serg/ComfyUI/models

echo "=== torch BEFORE (must be unchanged at the end) ==="
BEFORE=$("$PY" -c "import torch;print(torch.__version__)")
echo "$BEFORE"

echo "=== (1) IP-Adapter custom node (git clone, no pip) ==="
if [ ! -d "$CN/ComfyUI_IPAdapter_plus" ]; then
  git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus "$CN/ComfyUI_IPAdapter_plus" \
    && echo "cloned IPAdapter_plus" || echo "CLONE FAILED"
else
  echo "IPAdapter_plus already present"
fi

echo "=== (2) model downloads (HF, default token) ==="
"$PY" - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download

M = "/home/serg/ComfyUI/models"
# (repo_id, filename_in_repo, dest_dir, save_as)
JOBS = [
    # IP-Adapter SDXL (image-prompt; ViT-H variant) + its CLIP vision encoder
    ("h94/IP-Adapter", "sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
     f"{M}/ipadapter", "ip-adapter_sdxl_vit-h.safetensors"),
    ("h94/IP-Adapter", "models/image_encoder/model.safetensors",
     f"{M}/clip_vision", "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"),
    # ControlNet Canny SDXL (used with the CORE Canny node — no controlnet_aux needed)
    ("xinsir/controlnet-canny-sdxl-1.0", "diffusion_pytorch_model.safetensors",
     f"{M}/controlnet", "controlnet-canny-sdxl-1.0.safetensors"),
]
for repo, fn, dest, save_as in JOBS:
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, save_as)
    if os.path.exists(target):
        print(f"skip (exists): {target}")
        continue
    print(f"downloading {repo}/{fn} -> {target}", flush=True)
    p = hf_hub_download(repo_id=repo, filename=fn)
    os.symlink(p, target)  # symlink from HF cache; keeps one copy
    print(f"  linked {target}")
PYEOF

echo "=== (3) verify models in place ==="
ls -la "$M/ipadapter/" "$M/clip_vision/" "$M/controlnet/" 2>/dev/null | grep -iE 'serg0|ip-adapter|CLIP-ViT|canny' || true

echo "=== torch AFTER (RED ALERT if != before) ==="
AFTER=$("$PY" -c "import torch;print(torch.__version__)")
echo "$AFTER"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "!!! cu128 torch CHANGED ($BEFORE -> $AFTER) — ROLL BACK with the cu128 wheel before using ComfyUI !!!"
  exit 3
fi
echo "=== OK: torch unchanged. Restart ComfyUI, then verify IPAdapter/ControlNet nodes via /object_info. ==="
