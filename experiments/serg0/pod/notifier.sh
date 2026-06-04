#!/usr/bin/env bash
# R0 training notifier: status+ETA to Telegram every 30 min; done/error notice.
TG=/workspace/tg.sh; sleep 90
prog(){ grep -aoE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+' "$1" 2>/dev/null | tail -1; }
err(){ grep -aiE 'error|traceback|out of memory|cuda' "$1" 2>/dev/null | tail -1; }
while true; do
  alive=$(pgrep -fc sdxl_train_network.py); donen=$(ls /workspace/out/lora_*/*.safetensors 2>/dev/null | wc -l)
  if [ "${alive:-0}" = "0" ]; then
    [ "$donen" -ge 2 ] && bash "$TG" "✅ R0 готово. LoRA: $donen. Можно генерить тест-спред для курации (8080)." \
      || bash "$TG" "⚠️ R0 остановилось, LoRA лишь $donen. SDXL:$(err /workspace/train_sdxl.log) ILL:$(err /workspace/train_ill.log)"
    break; fi
  bash "$TG" "⏳ R0 (трейнов:$alive, LoRA:$donen) • SDXL: $(prog /workspace/train_sdxl.log) • ILL: $(prog /workspace/train_ill.log)"
  sleep 1800
done
