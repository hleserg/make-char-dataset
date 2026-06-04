#!/usr/bin/env bash
# Acts on UI "pipe" buttons: reads /workspace/ui_command.json (written by curate_ui)
# and runs the next pipe step. retrain = fold accepted gens into the trainset + relaunch
# kohya on Illustrious; regenerate = notify (gen script wired after a base is chosen).
WS=/workspace; TG=$WS/tg.sh; CMD=$WS/ui_command.json; SEEN=$WS/.ui_cmd_seen
CKPT=$WS/ComfyUI/models/checkpoints
touch "$SEEN"
while true; do
  if [ -f "$CMD" ] && [ "$CMD" -nt "$SEEN" ]; then
    touch "$SEEN"
    action=$(python3 -c "import json;print(json.load(open('$CMD')).get('action',''))" 2>/dev/null)
    n=$(python3 -c "import json;print(len(json.load(open('$CMD')).get('accepted',[])))" 2>/dev/null)
    bash "$TG" "🛠 UI: команда '$action' (принятых: $n) — обрабатываю..."
    if [ "$action" = "retrain" ]; then
      rm -rf "$WS/serg0_accepted"; mkdir -p "$WS/serg0_accepted"
      python3 -c "
import json, os, shutil
d = json.load(open('$CMD')); root = d.get('root', '$WS/out')
for p in d.get('accepted', []):
    s = os.path.join(root, p)
    if os.path.exists(s):
        shutil.copy(s, os.path.join('$WS/serg0_accepted', os.path.basename(p)))
"
      python "$WS/runpod_serg0/build_trainset.py" --refs "$WS/serg0_refs" --out "$WS/serg0_train_r" \
        --trigger serg0 --repeats 10 --extra-dir "$WS/serg0_accepted"
      cd "$WS/sd-scripts"
      setsid nohup venv/bin/python -m accelerate.commands.launch --num_processes 1 --mixed_precision bf16 \
        --dynamo_backend no sdxl_train_network.py \
        --pretrained_model_name_or_path="$CKPT/Illustrious-XL-v1.0.safetensors" \
        --output_dir="$WS/out/lora_ill_r" --output_name=serg0_ill_r \
        --train_data_dir="$WS/serg0_train_r" --resolution=1024,1024 --network_module=networks.lora \
        --network_dim=32 --network_alpha=16 --train_batch_size=1 --max_train_steps=1800 \
        --learning_rate=1e-4 --optimizer_type=AdamW8bit --lr_scheduler=cosine --mixed_precision=bf16 \
        --save_precision=fp16 --save_every_n_steps=400 --save_model_as=safetensors --cache_latents \
        --gradient_checkpointing --sdpa --caption_extension=.txt --shuffle_caption --keep_tokens=1 --seed=42 \
        >"$WS/train_ill_r.log" 2>&1 </dev/null &
      rm -f "$WS/waiting_for_ui"
      bash "$TG" "⇪ Дообучение запущено: Illustrious + база + $n принятых кадров. Лог train_ill_r.log."
    elif [ "$action" = "regenerate" ]; then
      bash "$TG" "↻ Запрос на догенерацию ($n принятых как референс). Ген-скрипт подключу после выбора базы по R0."
    fi
  fi
  sleep 20
done
