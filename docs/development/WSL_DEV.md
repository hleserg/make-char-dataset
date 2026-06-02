# Developing on WSL + CUDA

This project **runs image generation**, so the recommended development environment
is **WSL2 (Ubuntu) with an NVIDIA CUDA driver**, where models can be cached locally
and a generation backend (ComfyUI / diffusers) is available to produce and validate
variants. Linux also removes Windows-specific friction (non-ASCII paths breaking
image I/O, missing `jq`/`make`, non-portable shell hooks).

## Which stages need the GPU

| Stage | GPU | Why |
|-------|-----|-----|
| `import` | no | walk `state.json`, copy + normalize anchors (Pillow) |
| `generate` | **yes** | diffusion via ComfyUI/diffusers + style LoRA + ControlNet |
| `clean` | no | perceptual-hash dedup (`imagehash`) |
| `caption` | **yes** | WD14 tagger (ONNX) for Character-Locker captions |
| downstream | **yes** | kohya LoRA training + validating output in a generation UI |

`import` and `clean` run fine CPU-only; `generate`/`caption` and the end goal want
CUDA. The heavy backends sit behind Protocols and are imported lazily, so unit
tests, `make check`, and the `verifier-dataset` smoke run with **no GPU**.

## One-time setup

```bash
# In WSL (Ubuntu), with the NVIDIA driver installed on Windows and `nvidia-smi` working:
git clone https://github.com/hleserg/make-char-dataset.git
cd make-char-dataset

curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if absent
uv sync --all-extras                              # create .venv, install the light/CPU deps
uv run pre-commit install                         # repo hooks
make check                                         # DoD gate — must be green
```

`make` and `jq` exist on Linux, so the `Makefile` targets and any shell hooks work
directly (unlike the Windows host).

### GPU dependencies

The light/CPU deps (`numpy`, `pillow`, `imagehash`) are in `[project.dependencies]`
and install with `uv sync --all-extras`. The heavy generation/captioning stack
(`torch`, `torchvision`, `diffusers`, `transformers`, `onnxruntime`, …) is a
PEP-735 dependency group — install it only on the GPU box:

```bash
uv sync --group gpu     # pulls CUDA torch from the cu128 index (see pyproject.toml)
nvidia-smi              # confirm the GPU is visible from WSL
```

`uv sync --all-extras` (CI, fresh clone) deliberately does **not** install the
`gpu` group, so the CPU path and `make check` stay fast and `pip-audit` never scans
the heavy tree. The group is still resolved and pinned in `uv.lock` on every
platform (macOS falls back to CPU torch + `onnxruntime`).

### Generation backend + models (provided by the human)

The `generate` stage needs a running **ComfyUI** (`APP_COMFY_URL`) or local
diffusers weights, plus the inputs tracked in the Linear "human inputs" task:

- the pre-trained **style LoRA** (`APP_STYLE_LORA_PATH`) + `APP_STYLE_PROMPT`,
- license-safe **ControlNet** models (OpenPose/depth) — never InsightFace-based
  PuLID / IP-Adapter / InstantID for the commercial dataset (HLE-668),
- a real **passport export** to multiply.

## Exposing a local UI / service to the LAN (WSL networking)

> **THE RULE: a WSL service must listen on `0.0.0.0` inside WSL, never `127.0.0.1`.**

Windows reaches a WSL service through the WSL distro's **eth0 IP** (e.g.
`172.22.x.x`) via a `netsh portproxy`; a service bound to WSL **loopback**
(`127.0.0.1`) is invisible to that proxy and unreachable from the host or LAN. Bind
any server (a generation UI, a verifier) to `0.0.0.0`.

**Reaching it from the Windows host / another LAN machine (Windows 10):**

> WSL **mirrored networking** (`networkingMode=mirrored`, no portproxy needed)
> requires **Windows 11 22H2+**; on Windows 10 the `.wslconfig` line is ignored and
> WSL falls back to NAT — so you need a host-side portproxy + firewall rule.

```powershell
# Elevated PowerShell. Re-derive the WSL IP (it changes when WSL restarts):
$ip = (wsl hostname -I).Trim().Split(' ')[0]; "WSL IP = $ip"
$port = 7860
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$port 2>$null
netsh interface portproxy add    v4tov4 listenaddress=0.0.0.0 listenport=$port connectaddress=$ip connectport=$port
netsh advfirewall firewall delete rule name=WSL_LAN_$port 2>$null
netsh advfirewall firewall add    rule name=WSL_LAN_$port dir=in action=allow protocol=TCP localport=$port remoteip=LocalSubnet
```

Then open `http://<windows-LAN-IP>:<port>` from any LAN machine (or `localhost:<port>`
on the host). To survive reboots/WSL restarts, drive that script from a Scheduled Task
on logon + a short interval; `schtasks /Create` is more robust than
`Register-ScheduledTask` on localized / non-domain Windows. Prefer a LAN-scoped
(`remoteip=LocalSubnet`) firewall rule over disabling the firewall.

## Keep the code OS-agnostic

The pipeline should keep running on Windows too:

- **Never call image I/O with a raw path that may be non-ASCII** — prefer Pillow
  (`Image.open`) or decode via `np.fromfile` + `cv2.imdecode` if OpenCV is used.
- Prefer pure stdlib path handling; don't hardcode shell tools in committed code.

## Continuing the pipeline (agent onboarding)

Read these first, in order: `AGENTS.md` (canonical rules), then
`docs/architecture/SYSTEM.md` and `docs/architecture/WORKSPACE.md`.

Definition of Done is a hard gate: `make check` green (ruff + ruff format + pyright
+ bandit + pip-audit + pytest with ≥90 % coverage). Conventional Commits, small
reviewable PRs, English commits/comments. On taking a Linear issue set it **In
Progress**; on finishing set **Done** and link the PR.

`src/make_char_dataset/assembly.py` is the reference for the remaining stages —
reuse its patterns:

- An **injectable backend behind a `Protocol`** (`Generator`) so the heavy/optional
  model is swappable and tests inject a stub. See the `pure-core-lazy-backend`
  PLAYBOOK marker.
- **Pure policy/data functions** (dedup, captioning, layout) with the heavy
  dependency imported lazily inside the adapter — keeps coverage high without the
  heavy dep on the unit path.
- **Deterministic, traceable output names** so re-runs overwrite instead of
  duplicating.
- A **`manual_review/` fallback** for inputs the automation should not force.

Next stages are the **passport-ingest adapter**, then the **generation engine**,
then **captioning + layout** — pick up the corresponding Linear issue (HLE-757
subtasks).

### Set up local hooks in WSL

The committed `.claude/settings.json` carries the shared command allowlist + hooks
(format-on-edit, a pre-PR `make check` + verifier gate, a SessionStart pre-commit
install). Personal/machine-specific overrides (e.g. an autonomous `acceptEdits`
mode, or push/PR permissions) go in `.claude/settings.local.json` (git-ignored) —
run `/update-config` to manage them.
