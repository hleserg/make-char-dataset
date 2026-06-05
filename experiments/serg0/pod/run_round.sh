#!/usr/bin/env bash
# run_round.sh [N] — one generation round of the iterative char-dataset loop, end to
# end: clear the UI-wait flag, mark .generating (so the cost watchdog won't stop the
# pod mid-gen), generate round N's spread (16 pairs, both bases, R0 LoRAs), then set
# waiting_for_ui and Telegram the compare link. Used for the manual round AND by
# pipe_worker for auto rounds, so every round exercises the exact same notify path.
WS=/workspace
TG=$WS/tg.sh
PODID=$(tr '\0' '\n' </proc/1/environ 2>/dev/null | grep -m1 '^RUNPOD_POD_ID=' | cut -d= -f2-)
LINK="https://${PODID}-8080.proxy.runpod.net/compare"

if [ -n "$1" ]; then
  N=$1
else
  N=$(ls -d "$WS"/out/r*_sdxl 2>/dev/null | sed -E 's#.*/r([0-9]+)_sdxl#\1#' | sort -n | tail -1)
  N=$(((${N:--1}) + 1))
fi

NS=$(python3 -c "import json;print(len(json.load(open('$WS/scene_bank.json'))))" 2>/dev/null || echo "?")
# Two fresh ComfyUI instances (8189 + 8190), one per base, so both generate in parallel.
# Each gets its OWN output/temp dir so they never overwrite each other's frames. We avoid
# the long-lived 8188 (its warm cache returned empty 0.00s "cache hits").
ensure_comfy(){  # <port> <outdir> <tmpdir>
  curl -sf -m3 "http://127.0.0.1:$1/" >/dev/null 2>&1 && return 0
  mkdir -p "$2" "$3"
  (cd "$WS/ComfyUI" && setsid nohup python main.py --listen 127.0.0.1 --port "$1" \
    --output-directory "$2" --temp-directory "$3" --disable-all-custom-nodes \
    >"$WS/comfy_$1.log" 2>&1 </dev/null &)
}
ensure_comfy 8189 "$WS/comfy_8189_out" "$WS/comfy_8189_tmp"
ensure_comfy 8190 "$WS/comfy_8190_out" "$WS/comfy_8190_tmp"
for _ in $(seq 1 90); do
  curl -sf -m3 http://127.0.0.1:8189/ >/dev/null 2>&1 &&
    curl -sf -m3 http://127.0.0.1:8190/ >/dev/null 2>&1 && break
  sleep 2
done
rm -f "$WS/waiting_for_ui"
touch "$WS/.generating" "$WS/agent_heartbeat"
bash "$TG" "⏳ Генерю круг $N — ВСЕ $NS сцен на обеих базах ($((NS * 2)) кадров, надолго ~40-50 мин). Пришлю ссылку, когда будет готово."

python "$WS/gen_spread.py" "$N" all >"$WS/gen_r${N}.log" 2>&1
rc=$?

rm -f "$WS/.generating"
nsd=$(ls "$WS"/out/"r${N}_sdxl"/*.png 2>/dev/null | wc -l)
nil=$(ls "$WS"/out/"r${N}_illustrious"/*.png 2>/dev/null | wc -l)
touch "$WS/waiting_for_ui"

if [ "$rc" = "0" ] && [ "$nsd" -gt 0 ]; then
  bash "$TG" "✅ Круг $N готов: $nsd+$nil кадров. Отбери верные и жми «Сохранить + ещё круг» → $LINK"
else
  bash "$TG" "⚠️ Круг $N: проблема при генерации (rc=$rc, $nsd+$nil кадров). Лог gen_r${N}.log. UI: $LINK"
fi
