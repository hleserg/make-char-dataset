#!/usr/bin/env bash
# /workspace/tg.sh "message" — send a Telegram message (creds from PID-1 env on the pod)
TOK=$(tr '\0' '\n' </proc/1/environ 2>/dev/null | grep -m1 '^TELEGRAM_TOKEN=' | cut -d= -f2-)
CHAT=$(tr '\0' '\n' </proc/1/environ 2>/dev/null | grep -m1 '^TELEGRAM_CHAT_ID=' | cut -d= -f2-)
[ -z "$TOK" ] && { echo "no TELEGRAM_TOKEN"; exit 1; }
curl -s -m 15 -o /dev/null -w '%{http_code}\n' "https://api.telegram.org/bot${TOK}/sendMessage" \
  --data-urlencode "chat_id=${CHAT}" --data-urlencode "text=$1"
