#!/usr/bin/env bash
# Relaunch all pod services after a RunPod restart (run on the pod).
WS=/workspace; export HF_HOME=$WS/hf PIP_CACHE_DIR=$WS/pipcache
python -c "import torchsde,safetensors,aiohttp,kornia" 2>/dev/null || pip install -q -r $WS/ComfyUI/requirements.txt
pgrep -f "ComfyUI/main.py" >/dev/null || (cd $WS/ComfyUI && setsid nohup python main.py --listen 0.0.0.0 --port 8188 --disable-all-custom-nodes >$WS/comfy.log 2>&1 </dev/null &)
# curate UI from /root: detached procs can't reliably open files on the /workspace net-volume
pgrep -f curate_ui.py >/dev/null || (cp $WS/curate_ui.py /root/curate_ui.py; CURATE_PORT=8080 CURATE_ROOT=$WS/out setsid nohup python -u /root/curate_ui.py >/root/curate.log 2>&1 </dev/null &)
pgrep -f sdxl_train_network.py >/dev/null || setsid nohup bash $WS/runpod_serg0/r0_train.sh >$WS/r0_setup.log 2>&1 </dev/null &
pgrep -f "/workspace/notifier.sh"    >/dev/null || setsid nohup bash $WS/notifier.sh    >$WS/notifier.log 2>&1 </dev/null &
pgrep -f "/workspace/pipe_worker.sh" >/dev/null || setsid nohup bash $WS/pipe_worker.sh >$WS/pipe_worker.log 2>&1 </dev/null &
pgrep -f "/workspace/watchdog.sh"    >/dev/null || setsid nohup bash $WS/watchdog.sh    >$WS/watchdog.log 2>&1 </dev/null &
echo "services launched"
