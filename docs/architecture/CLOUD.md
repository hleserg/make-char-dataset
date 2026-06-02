# Running the heavy steps on a cloud GPU

The local box (16 GB RTX 5070 Ti) trains Flux at the edge of its VRAM. The two
heavy steps can run on a cheap cloud GPU instead — **without changing the local
default**; both are opt-in.

| Heavy step | How it goes to the cloud | Code change to enable |
|------------|--------------------------|------------------------|
| **Inference** — `eval` grid, (future) generation/restylization | point `APP_COMFY_URL` at a remote ComfyUI | none — already supported |
| **Training** — `train` (ai-toolkit Flux) | the opt-in **SSH backend** (`APP_TRAIN_BACKEND=ssh`) | rsync up → run on the remote → rsync the LoRA back |

> Which provider, current pricing, registration and a $/run estimate are covered by
> a separate research report (RunPod / Vast.ai / Modal / Lambda …). The mechanics
> below are **provider-agnostic** — they work with any box that exposes SSH.

## Inference on the cloud (free — no code)

Rent any GPU box, run ComfyUI on it (Flux.1-dev + your LoRAs in `models/loras`),
expose its port, and:

```bash
APP_COMFY_URL=http://<cloud-host>:8188 make-char-dataset eval --trigger kael
```

The `eval` stage (and the generation backend) already talk to ComfyUI over HTTP, so
nothing else changes. Use an SSH tunnel (`ssh -L 8188:localhost:8188 <host>`) so the
port is not public.

## Training on the cloud (the SSH backend)

`APP_TRAIN_BACKEND=ssh` makes `make-char-dataset train`:

1. `rsync` the kohya dataset (`03_dataset/<N>_<trigger>`) up to the remote workdir;
2. upload a config whose paths are rewritten for the remote (`rewrite_config_for_remote`);
3. run ai-toolkit on the remote over SSH (clearing the run dir first, so no stale
   resume), streaming its progress;
4. `rsync` the resulting LoRA back into local `06_lora/<name>/`.

```bash
# .env (or env):
APP_TRAIN_BACKEND=ssh
APP_TRAIN_SSH_HOST=root@1.2.3.4        # or an ~/.ssh/config alias
APP_TRAIN_SSH_PORT=22
APP_TRAIN_SSH_WORKDIR=~/make-char-train
APP_TRAIN_SSH_AITOOLKIT_DIR=~/ai-toolkit

make-char-dataset train --trigger kael   # -> local 06_lora/kael/kael.safetensors
```

The local default (`APP_TRAIN_BACKEND=local`) is unchanged; flip the flag back to
train on this box. `APP_TRAIN_SSH_WORKDIR` may be `~/…` (the `~` is resolved to the
remote `$HOME` so the path baked into ai-toolkit's config is absolute) or an absolute
path (e.g. RunPod's persistent `/workspace/…`). Keep the SSH paths and the trigger
token shell-safe (no spaces/metacharacters).

### Provision the remote box once

The SSH backend assumes the remote already has the toolchain. On a fresh rented box
(adjust paths to `APP_TRAIN_SSH_*`):

```bash
# 1) ai-toolkit + its venv (a 24 GB GPU trains Flux comfortably — no qint4 issues)
git clone https://github.com/ostris/ai-toolkit ~/ai-toolkit
cd ~/ai-toolkit && python -m venv venv && venv/bin/pip install -r requirements.txt
# 2) HF auth for the gated FLUX.1-dev base (cached; not passed over the wire)
venv/bin/huggingface-cli login        # paste a read token
# 3) the base downloads on the first run (or pre-warm it)
```

The HF token is **never** sent over the SSH command line (it would show in remote
`ps`); the remote uses its own `huggingface-cli` login cache. SSH auth is your key.

## Cost & recipe notes

- A 24 GB cloud GPU removes the local VRAM tightness, so `APP_TRAIN_QTYPE` can stay
  `qfloat8` (or go `qfloat8`/bf16 with EMA on) and you can raise `APP_TRAIN_RESOLUTION`.
- Cheap **spot/community** tiers can be interrupted — for a 2–5 h run keep
  `APP_TRAIN_SAVE_EVERY` modest so a restart loses little; the per-run cost estimate
  and the spot-vs-on-demand trade-off are in the provider research report.
- Occasional bursty runs favour either a cheap SSH spot box or a serverless option
  (Modal, which ai-toolkit supports via `run_modal.py`) — see the report for the
  price/effort call.
