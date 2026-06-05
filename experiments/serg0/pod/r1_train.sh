#!/usr/bin/env bash
# R1 char-LoRA training — FROM SCRATCH on the curated improved dataset, ON THE POD.
# Trains TWO SDXL char-LoRAs in parallel on the L40 (vanilla SDXL + Illustrious).
#
# Data composition (refs ANCHOR identity/build/style/tattoos; keepers ADD diversity):
#   <root>/<REF_REP>_serg0/   = 18 original refs        (high repeats)
#   <root>/<KEEP_REP>_serg0/  = curated keepers (.txt)  (low repeats)
# MODE:
#   shared (default) — both bases train on ALL 113 keepers + refs (one root).
#   split            — sdxl base trains on sdxl-source keepers+refs; ill on ill-source.
#
# Outputs go to NEW names (serg0_*_r1) and dirs — the deployed R0 LoRAs
# (serg0_sdxl/serg0_ill) are left intact (live in gen+improver + A/B baseline).
# Promote R1 only after it beats R0.
set -uo pipefail
WS=/workspace
KOHYA=$WS/sd-scripts
CKPT=$WS/ComfyUI/models/checkpoints
MODE=${1:-shared}
REF_REP=${REF_REP:-10}
KEEP_REP=${KEEP_REP:-2}
STEPS=${STEPS:-2400}
log(){ echo -e "\n=== $* ===" > /proc/1/fd/1 2>/dev/null; echo -e "\n=== $* ==="; }

# ---- assemble kohya train roots (refs captions from build_trainset; keepers keep their WD14 .txt) ----
log "assemble trainset(s) mode=$MODE refs x$REF_REP keepers x$KEEP_REP"
ASM=$(python3 - "$MODE" "$REF_REP" "$KEEP_REP" <<'PY'
import importlib.util, os, shutil, sys, glob
WS="/workspace"; mode=sys.argv[1]; rr=sys.argv[2]; kr=sys.argv[3]
REFS=f"{WS}/serg0_refs"; KEEP=f"{WS}/serg0_dataset"; TRIG="serg0"; GEN="1boy"
spec=importlib.util.spec_from_file_location("bt", f"{WS}/build_trainset.py")
bt=importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
CAPS=getattr(bt,"SERG0_CAPTIONS",{})
def put_refs(dst):
    os.makedirs(dst,exist_ok=True)
    for img in sorted(glob.glob(f"{REFS}/*.png")):
        st=os.path.splitext(os.path.basename(img))[0]
        shutil.copy(img,os.path.join(dst,os.path.basename(img)))
        open(os.path.join(dst,st+".txt"),"w").write(f"{TRIG}, {GEN}, solo, {CAPS.get(st,'simple background')}\n")
def put_keep(dst,pred):
    os.makedirs(dst,exist_ok=True)
    for img in sorted(glob.glob(f"{KEEP}/*.png")):
        b=os.path.basename(img)
        if not pred(b): continue
        shutil.copy(img,os.path.join(dst,b))
        t=os.path.splitext(img)[0]+".txt"
        shutil.copy(t,os.path.join(dst,os.path.splitext(b)[0]+".txt"))
def build(root,pred):
    if os.path.exists(root): shutil.rmtree(root)
    put_refs(f"{root}/{rr}_{TRIG}"); put_keep(f"{root}/{kr}_{TRIG}",pred)
    nr=len(glob.glob(f"{root}/{rr}_{TRIG}/*.png")); nk=len(glob.glob(f"{root}/{kr}_{TRIG}/*.png"))
    return nr,nk
if mode=="split":
    a=build(f"{WS}/serg0_train_r1_sdxl", lambda b:"_sdxl_" in b)
    c=build(f"{WS}/serg0_train_r1_ill",  lambda b:"_illustrious_" in b)
    print(f"SDXLROOT={WS}/serg0_train_r1_sdxl ILLROOT={WS}/serg0_train_r1_ill SDXL_refs={a[0]} SDXL_keep={a[1]} ILL_refs={c[0]} ILL_keep={c[1]}")
else:
    a=build(f"{WS}/serg0_train_r1", lambda b:True)
    print(f"SDXLROOT={WS}/serg0_train_r1 ILLROOT={WS}/serg0_train_r1 refs={a[0]} keep={a[1]}")
PY
)
echo "$ASM"
SDXLROOT=$(echo "$ASM" | grep -oE 'SDXLROOT=[^ ]+' | cut -d= -f2)
ILLROOT=$(echo "$ASM" | grep -oE 'ILLROOT=[^ ]+' | cut -d= -f2)

# ---- base checkpoints present? ----
[ -f "$CKPT/Illustrious-XL-v1.0.safetensors" ] || { log "MISSING Illustrious ckpt"; exit 1; }
[ -f "$CKPT/sd_xl_base_1.0.safetensors" ]      || { log "MISSING SDXL ckpt"; exit 1; }

# ---- launch two parallel trainings (≈13GB each, fits 48GB) ----
log "launch R1 trainings (from scratch) steps=$STEPS"
COMMON="--resolution=1024,1024 --network_module=networks.lora \
 --network_dim=32 --network_alpha=16 --train_batch_size=1 --max_train_steps=$STEPS \
 --learning_rate=1e-4 --optimizer_type=AdamW8bit --lr_scheduler=cosine --mixed_precision=bf16 \
 --save_precision=fp16 --save_every_n_steps=400 --save_model_as=safetensors --cache_latents \
 --gradient_checkpointing --sdpa --caption_extension=.txt --shuffle_caption --keep_tokens=1 --seed=42"
launch(){ # launch <base_ckpt> <train_root> <out_dir> <name> <logfile>
  mkdir -p "$3"
  cd "$KOHYA"
  setsid nohup venv/bin/python -m accelerate.commands.launch --num_processes 1 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no sdxl_train_network.py \
    --pretrained_model_name_or_path="$1" --train_data_dir="$2" --output_dir="$3" --output_name="$4" $COMMON \
    > "$5" 2>&1 </dev/null &
  echo "launched $4 -> pid $!" > /proc/1/fd/1 2>/dev/null
  echo "launched $4 -> pid $!"
}
touch "$WS/.training" "$WS/agent_heartbeat"
launch "$CKPT/sd_xl_base_1.0.safetensors"      "$SDXLROOT" "$WS/out/lora_sdxl_r1"        serg0_sdxl_r1 "$WS/train_sdxl_r1.log"
launch "$CKPT/Illustrious-XL-v1.0.safetensors" "$ILLROOT"  "$WS/out/lora_illustrious_r1" serg0_ill_r1  "$WS/train_ill_r1.log"
log "R1 launched (serg0_sdxl_r1 + serg0_ill_r1); tail /workspace/train_*_r1.log"
