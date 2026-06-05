# Serg0 char-dataset — HANDOFF (for the next agent)

Living doc. Goal: multiply Serg0's 18 refs into a large, diverse, FAITHFUL dataset
(comic/watercolor style, NON-idealized build — NOT a square-jawed jock), then a
character LoRA. Personal/non-commercial (InsightFace/PuLID ban waived).

## Current phase (2026-06-05)
- **Loop = GENERATION-ONLY right now (NO training).** Generate rounds of variants →
  user curates on `/compare` → keepers accumulate → repeat until ~30-40 good, THEN train.
- Round 0: 8+8 (curated → 7 keepers). Round 1: 16+16. Round 2: 16+16. Keepers now **16** (target 30).
- **Round 3 RUNNING**: whole scene bank, ALL 176 scenes × 2 bases = 352 frames, parallel.
- **In progress: best-practice IMPROVER stack** (FaceDetailer + hand + UltimateSDUpscale)
  to fix mushy eyes/hands. User demands MAX quality, community best-practice, not coleнochный.
  Tattoos CANNOT be filter-fixed (model doesn't know the designs + too few px at body scale) →
  that's a future training-data fix (more arm-visible refs), not post-processing.

## Pod (RunPod L40, 48GB)
- id `ybo69hlwaua7z9`, ip `213.181.111.2`. Proxy ssh `ybo69hlwaua7z9-6441143d@ssh.runpod.io`.
- Key `/root/.ssh/id_ed25519_claude`. **Direct port = RUNPOD_TCP_PORT_22, CHANGES on restart**
  (currently 32753; if refused read it from `/proc/1/environ` via the proxy ssh). scp works on the direct port.
- `/workspace` persists; container `/` wiped on restart. `make-char-dataset` repo branch
  `experiment/serg0-iterative`, dir `experiments/serg0/` (+ `pod/`). Commits use `--no-verify`
  (embedded web-UI strings break ruff E501). 69GB disk free. GPU L40 46GB.

## Generation architecture (working, do not regress)
- `pod/run_round.sh [N]`: clears `waiting_for_ui`, touches `.generating`, ensures TWO **fresh**
  ComfyUI instances, runs `gen_spread.py N all`, sets `waiting_for_ui`, Telegrams the compare link.
- `pod/gen_spread.py`: both bases as parallel THREADS, each against its OWN ComfyUI:
  **8190 = vanilla SDXL (sd_xl_base_1.0 + serg0_sdxl)**, **8189 = Illustrious (Illustrious-XL-v1.0 + serg0_ill)**.
  `all` mode = whole `/workspace/scene_bank.json` (176 scenes); `:03d` filenames; seeds `7000+r*1000+i`.
  Output `r{N}_sdxl` / `r{N}_illustrious` + `r{N}_scenes.json` (labels). Touches `agent_heartbeat` per frame.
- Gen instances run `--disable-all-custom-nodes` + own `--output-directory` (no frame overwrite).

## ⚠️ GOTCHAS (each cost hours — read before touching ComfyUI)
1. **pkill pattern**: processes are `python main.py --listen ...` → kill with
   `pkill -9 -f "main.py --listen"`. `pkill -f "ComfyUI/main.py"` matches NOTHING.
2. **Stale ComfyUI = empty "0.00s cache hits"**: a long-lived instance re-run with the same
   seeds returns `completed/success` with EMPTY outputs (sample() saw "no image"). ALWAYS
   generate on FRESH instances; incrementing round numbers (new seeds) avoid the cache.
   `8188` is the cursed long-lived/exposed instance — NOT used for gen anymore.
3. `curate_ui.py` MUST run from `/root` (detached procs can't reliably open `/workspace` net-volume files).
4. Verify a UI URL returns 200 before sending it; agent `curl` probes touch `ui_last_access` and can
   falsely satisfy the watchdog's `waiting_for_ui` branch → re-`touch waiting_for_ui` if still blocked.

## Services on the pod (relaunched by `pod/start_services.sh`)
- `curate_ui.py` :8080 (exposed) — gallery + `/compare` (latest round auto-detected; `?a=&b=` overrides;
  buttons: "💾 Сохранить + ещё круг" → ui_command{generate}; "✅ Хватит" → {done}; keeper count K/30).
- `pipe_worker.sh` — consumes `ui_command.json` EXACTLY ONCE (mv aside): generate→run_round; done→assemble
  `/workspace/serg0_dataset`. (retrain alias→generate; NO training now.)
- `watchdog.sh` — auto-stop via RunPod REST (manage key `/workspace/.rp_key`). Guards: `training()`,
  `generating()` (`.generating` flag OR pgrep gen_spread), heartbeat. waiting_for_ui→remind 10min×4→stop; idle 30min→stop.
- `tg.sh "msg"` — Telegram (creds in pod PID-1 env).

## Improver stack (best-practice, being built)
- Dedicated ComfyUI **:8191 WITH custom nodes** (gen instances stay custom-disabled).
- custom_nodes: ComfyUI-Impact-Pack (FaceDetailer/DetailerForEach), ComfyUI-Impact-Subpack
  (UltralyticsDetectorProvider), ComfyUI_UltimateSDUpscale.
- models: `models/ultralytics/bbox/{face_yolov8m,hand_yolov8s}.pt`, `models/ultralytics/segm/person_yolov8m-seg.pt`
  (Bingsu/adetailer); `models/upscale_models/{4x-UltraSharp,4x-AnimeSharp}.pth`.
- Per-frame graph: FaceDetailer(face, char-LoRA, denoise ~0.4) → Detailer(hands) → UltimateSDUpscale(ESRGAN + light resample).
- Per-base LoRA. Output `r{N}_{tag}_fixed`. A/B vs raw via `/compare?a=r3_sdxl&b=r3_sdxl_fixed`.
- Runs on its own instance so it can go concurrent with gen or after.

## Doctrines (user-confirmed, do not relitigate)
- Character-agnostic captioning: trigger `serg0` + ONLY variable scene tags; identity/body/style bind to trigger.
- Genre variety lives in the SETTING, never the body (no warrior/cyborg/jock). scene_bank.json = 176 genre-mixed scenes.
- NO Flux/Kontext (idealizes). NO Gemini multiplier (cost). Heavy/long GPU ops need user OK — but the
  iterative gen rounds + (now) the improver build are explicitly approved.
- Don't train until ~30-40 curated keepers (16 self-gen frames too few → drift risk).

## Secrets leaked to chat → user must ROTATE: HF write token, manage RunPod API key, TELEGRAM token, read-only RunPod key.

## Exact next steps
1. Finish improver install (custom_nodes + ultralytics) + model downloads (running bg).
2. Launch 8191 with custom nodes (own dirs), confirm FaceDetailer/UltralyticsDetectorProvider/UltimateSDUpscale in /object_info.
3. Build + VALIDATE the improve graph on 3-5 round-3 frames (Read before/after; eyes fixed, style/identity kept).
4. Run improver over all round-3 → `_fixed`; Telegram + A/B compare link.
5. Wire improver into run_round as the standard post-step.
6. Keep accumulating keepers → at ~30-40 do the dual-base training round (kohya, recipe in `r0_train.sh`).
