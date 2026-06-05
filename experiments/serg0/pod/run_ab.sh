#!/usr/bin/env bash
# Build the R1 A/B comparison page (base vs +LoRA, across all checkpoints), serve it on
# 8080 (exposed) and Telegram the link. Keeps the pod alive during gen via .generating.
set -uo pipefail
WS=/workspace
TG=$WS/tg.sh
PODID=$(tr '\0' '\n' </proc/1/environ 2>/dev/null | grep -m1 '^RUNPOD_POD_ID=' | cut -d= -f2-)
LINK="https://${PODID}-8080.proxy.runpod.net/"
LORAS=$WS/ComfyUI/models/loras

# 1) symlink every R1 checkpoint LoRA into ComfyUI's loras dir (BEFORE comfy starts, so
#    fresh instances list them). final + step files.
for s in 400 800 1200 1600 2000 2400; do
  p=$(printf "%08d" "$s")
  ln -sf "$WS/out/lora_sdxl_r1/serg0_sdxl_r1-step${p}.safetensors" "$LORAS/serg0_sdxl_r1-step${p}.safetensors"
  ln -sf "$WS/out/lora_illustrious_r1/serg0_ill_r1-step${p}.safetensors" "$LORAS/serg0_ill_r1-step${p}.safetensors"
done

# 2) keep the pod alive while generating
rm -f "$WS/waiting_for_ui"
touch "$WS/.generating" "$WS/agent_heartbeat"
bash "$TG" "⏳ Строю A/B-страницу R1 (база vs +LoRA, 20 промтов × 6 чекпойнтов × 2 базы ≈ 280 кадров, без улучшайзера). ~20-25 мин, пришлю ссылку."

# 3) two fresh ComfyUI instances (raw txt2img, no custom nodes), own dirs
ensure(){ # port outdir
  curl -sf -m3 "http://127.0.0.1:$1/" >/dev/null 2>&1 && return 0
  mkdir -p "$2"
  (cd "$WS/ComfyUI" && setsid nohup python main.py --listen 127.0.0.1 --port "$1" \
    --output-directory "$2" --temp-directory "$2/tmp" --disable-all-custom-nodes \
    >"$WS/comfy_$1.log" 2>&1 </dev/null &)
}
ensure 8190 "$WS/comfy_8190_out"
ensure 8189 "$WS/comfy_8189_out"
for _ in $(seq 1 90); do
  curl -sf -m3 http://127.0.0.1:8190/ >/dev/null 2>&1 &&
    curl -sf -m3 http://127.0.0.1:8189/ >/dev/null 2>&1 && break
  sleep 2
done

# 4) generate the whole grid + build page + zip
python "$WS/ab_grid.py" >"$WS/ab_grid.log" 2>&1
rc=$?
n=$(find "$WS/ab_out" -name "*.png" 2>/dev/null | wc -l)

rm -f "$WS/.generating"
if [ "$rc" != "0" ] || [ "$n" -lt 1 ]; then
  touch "$WS/waiting_for_ui"
  bash "$TG" "⚠️ A/B: проблема (rc=$rc, кадров=$n). Лог ab_grid.log."
  exit 1
fi

# 5) serve the static page on 8080 (free it from curate_ui first)
pkill -f curate_ui.py 2>/dev/null
pkill -f "http.server 8080" 2>/dev/null
sleep 1
(cd "$WS/ab_out" && setsid nohup python -m http.server 8080 --bind 0.0.0.0 \
  >"$WS/ab_serve.log" 2>&1 </dev/null &)
ok=down
for _ in $(seq 1 20); do curl -sf -m3 http://127.0.0.1:8080/ >/dev/null 2>&1 && { ok=up; break; }; sleep 1; done

touch "$WS/waiting_for_ui"
zs=$(du -h "$WS/ab_out/ab_compare.zip" 2>/dev/null | cut -f1)
bash "$TG" "✅ A/B-страница R1 готова ($n кадров, серв=$ok). Открой и выбери лучший чекпойнт: $LINK  (кнопка «Скачать страницу (zip)» ~$zs внутри). Под уснёт по простою ~через 40 мин — zip качни для офлайна."
echo "=== run_ab done: served=$ok frames=$n link=$LINK ==="
