#!/usr/bin/env bash
# Serg0 comic panel-factory. Stdlib-only -> plain python3. Needs ComfyUI up on :8188.
# Serves http://localhost:8770. Reuses ui_single plumbing + gen_improve graphs.
set -uo pipefail
curl -s -m5 -o /dev/null http://127.0.0.1:8188/system_stats || { echo "ComfyUI NOT up on :8188"; exit 1; }
exec python3 /home/serg/make-char-dataset/experiments/serg0/ui_comic.py
