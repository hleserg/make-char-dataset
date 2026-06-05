#!/usr/bin/env bash
# Quality pass over a round's frames -> r{R}_{tag}_fixed (both bases), watchdog-safe.
# FaceDetailer (face/eyes) + FaceDetailer (hands) + UltimateSDUpscale, on the dedicated
# improver ComfyUI (8191, custom nodes). Sets .improving so the cost watchdog won't stop
# the pod mid-run; Telegrams the raw-vs-fixed compare link when done.
WS=/workspace
TG=$WS/tg.sh
R=${1:-3}
PODID=$(tr '\0' '\n' </proc/1/environ 2>/dev/null | grep -m1 '^RUNPOD_POD_ID=' | cut -d= -f2-)
# curation link: the IMPROVED frames of both bases (vanilla SDXL vs Illustrious) to pick from
LINK="https://${PODID}-8080.proxy.runpod.net/compare"

# ensure the improver instance (8191, WITH custom nodes) is up
if ! curl -sf -m3 http://127.0.0.1:8191/ >/dev/null 2>&1; then
  mkdir -p "$WS/comfy_8191_out" "$WS/comfy_8191_tmp"
  (cd "$WS/ComfyUI" && setsid nohup python main.py --listen 127.0.0.1 --port 8191 \
    --output-directory "$WS/comfy_8191_out" --temp-directory "$WS/comfy_8191_tmp" \
    >"$WS/comfy_8191.log" 2>&1 </dev/null &)
  for _ in $(seq 1 80); do curl -sf -m3 http://127.0.0.1:8191/ >/dev/null 2>&1 && break; sleep 3; done
fi

rm -f "$WS/waiting_for_ui"
touch "$WS/.improving" "$WS/agent_heartbeat"
bash "$TG" "✨ Улучшайзер пошёл по кругу $R (FaceDetailer лицо+руки + ESRGAN-апскейл, обе базы). Долго (~1.5-2ч). Пришлю ссылку, как готово."

python "$WS/improve.py" "$R" sdxl        >"$WS/improve_r${R}_sdxl.log" 2>&1
python "$WS/improve.py" "$R" illustrious >"$WS/improve_r${R}_ill.log"  2>&1

rm -f "$WS/.improving"
nsd=$(ls "$WS"/out/"r${R}_sdxl_fixed"/*.png 2>/dev/null | wc -l)
nil=$(ls "$WS"/out/"r${R}_illustrious_fixed"/*.png 2>/dev/null | wc -l)
touch "$WS/waiting_for_ui"
bash "$TG" "✅ Улучшенные кадры круга $R готовы: sdxl $nsd, ill $nil. Сравни raw vs fixed: $LINK"
