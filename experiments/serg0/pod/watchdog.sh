#!/usr/bin/env bash
# Cost-safety watchdog (uses the manage-scope RunPod key at /workspace/.rp_key).
#  - When the pipe is blocked waiting for the user (/workspace/waiting_for_ui exists):
#    Telegram a link+warning every 10 min; on the 4th un-answered reminder -> STOP the pod.
#  - Idle guard: if NOTHING is happening (no training, no agent heartbeat, no UI access)
#    for 30 min -> STOP the pod (the user is away).
# Never stops while a training is running or the agent touched its heartbeat recently.
WS=/workspace; TG=$WS/tg.sh; KEYF=$WS/.rp_key
PODID=$(tr '\0' '\n' </proc/1/environ 2>/dev/null | grep -m1 '^RUNPOD_POD_ID=' | cut -d= -f2-)
LINK="https://${PODID}-8080.proxy.runpod.net"
WAIT=$WS/waiting_for_ui; ACCESS=$WS/ui_last_access; HB=$WS/agent_heartbeat
IDLE=1800; REMIND=600; MAXR=4
mtime(){ stat -c %Y "$1" 2>/dev/null || echo 0; }
stop_pod(){ K=$(cat "$KEYF" 2>/dev/null); curl -s -m20 -X POST -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/${PODID}/stop" >/dev/null 2>&1; }
training(){ [ "$(pgrep -fc sdxl_train_network.py)" != "0" ]; }
last_act=$(date +%s); reminders=0; reminded=0
while true; do
  now=$(date +%s)
  if training; then last_act=$now; reminders=0; sleep 120; continue; fi
  # any UI access or agent heartbeat counts as activity
  a=$(cat "$ACCESS" 2>/dev/null || echo 0); h=$(mtime "$HB")
  [ "${a:-0}" -gt "$last_act" ] && last_act=$a
  [ "$h" -gt "$last_act" ] && last_act=$h
  if [ -f "$WAIT" ]; then
    flag=$(mtime "$WAIT")
    if [ "${a:-0}" -ge "$flag" ]; then reminders=0; sleep 120; continue; fi   # user opened UI since the block
    if [ $((now - reminded)) -ge $REMIND ]; then
      reminders=$((reminders + 1)); reminded=$now
      if [ "$reminders" -ge "$MAXR" ]; then
        bash "$TG" "⏹ 4 напоминания без ответа — останавливаю под (экономия). Данные на /workspace + HF целы. UI вернётся при возобновлении: $LINK"
        stop_pod; break
      fi
      bash "$TG" "⚠️ Нужно твоё действие в UI ($reminders/4): отбракуй/прими кадры и пни пайп → $LINK"
    fi
    sleep 60; continue
  fi
  if [ $((now - last_act)) -ge $IDLE ]; then
    bash "$TG" "⏹ Простой 30+ мин (нет обучения и активности) — останавливаю под для экономии. Возобнови, когда нужно: $LINK"
    stop_pod; break
  fi
  sleep 120
done
