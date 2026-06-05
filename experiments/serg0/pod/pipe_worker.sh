#!/usr/bin/env bash
# Acts on UI buttons (curate_ui writes /workspace/ui_command.json). The loop is now
# GENERATION-ONLY (no training):
#   action=generate -> launch the next round (run_round.sh, auto round number)
#   action=done     -> assemble all accepted frames into /workspace/serg0_dataset
# The command file is consumed EXACTLY ONCE (moved aside immediately) so a poll can
# never re-fire it into an unbounded-round cost runaway.
WS=/workspace
TG=$WS/tg.sh
CMD=$WS/ui_command.json
LAST=$WS/.ui_cmd_last
PODID=$(tr '\0' '\n' </proc/1/environ 2>/dev/null | grep -m1 '^RUNPOD_POD_ID=' | cut -d= -f2-)
LINK="https://${PODID}-8080.proxy.runpod.net/compare"
generating(){ [ -f "$WS/.generating" ] || [ "$(pgrep -fc gen_spread.py)" != "0" ]; }

while true; do
  if [ -f "$CMD" ]; then
    mv -f "$CMD" "$LAST" 2>/dev/null || { sleep 5; continue; }   # consume exactly once
    action=$(python3 -c "import json;print(json.load(open('$LAST')).get('action',''))" 2>/dev/null)
    n=$(python3 -c "import json;print(len(json.load(open('$LAST')).get('accepted',[])))" 2>/dev/null)
    if [ "$action" = "generate" ] || [ "$action" = "retrain" ]; then
      if generating; then
        bash "$TG" "↻ Уже идёт генерация круга — дождись её конца. Всего отобрано: $n."
      else
        bash "$TG" "💾 Отобрано всего: $n. Запускаю следующий круг…"
        setsid nohup bash "$WS/run_round.sh" >"$WS/run_round.log" 2>&1 </dev/null &
      fi
    elif [ "$action" = "done" ]; then
      rm -rf "$WS/serg0_dataset"; mkdir -p "$WS/serg0_dataset"
      python3 -c "
import json, os, shutil
d = json.load(open('$LAST')); root = d.get('root', '$WS/out')
for p in d.get('accepted', []):
    s = os.path.join(root, p)
    if os.path.exists(s):
        shutil.copy(s, os.path.join('$WS/serg0_dataset', p.replace('/', '_')))
"
      cnt=$(ls "$WS"/serg0_dataset/*.png 2>/dev/null | wc -l)
      rm -f "$WS/waiting_for_ui"
      bash "$TG" "✅ Датасет собран: $cnt кадров в /workspace/serg0_dataset. Готов к обучению/выгрузке. $LINK"
    fi
  fi
  sleep 15
done
