#!/usr/bin/env bash
# Detached watcher: waits for BOTH R1 trainings to end, then Telegrams the result.
# Survives the agent session. Does NOT auto-start any generation (user is sensitive to
# surprise GPU use) — it proposes the A/B and waits for an explicit "давай A/B".
WS=/workspace
TG=$WS/tg.sh
while [ "$(pgrep -fc sdxl_train_network.py)" != "0" ]; do
  touch "$WS/agent_heartbeat"
  sleep 120
done
rm -f "$WS/.training"
SD=$WS/out/lora_sdxl_r1/serg0_sdxl_r1.safetensors
IL=$WS/out/lora_illustrious_r1/serg0_ill_r1.safetensors
ck=$(ls "$WS"/out/lora_sdxl_r1/serg0_sdxl_r1-step*.safetensors 2>/dev/null | wc -l)
if [ -f "$SD" ] && [ -f "$IL" ]; then
  bash "$TG" "✅ R1 обучена С НУЛЯ: serg0_sdxl_r1 + serg0_ill_r1 готовы (по $ck промежуточных чекпойнтов на базу, шаги 400..2400). R0 не тронут — остаётся baseline. Дальше A/B R1 vs R0: напиши 'давай A/B' и я сгенерю сравнение для отбора. Логи train_*_r1.log."
else
  sdok=$([ -f "$SD" ] && echo OK || echo НЕТ)
  ilok=$([ -f "$IL" ] && echo OK || echo НЕТ)
  bash "$TG" "⚠️ R1 обучение завершилось, но финальный файл отсутствует: sdxl=$sdok, ill=$ilok. Глянь train_sdxl_r1.log / train_ill_r1.log."
fi
