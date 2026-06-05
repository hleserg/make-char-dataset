#!/usr/bin/env python3
"""Best-practice quality pass for generated frames (community standard, on ComfyUI 8191
with custom nodes). Per frame: FaceDetailer (face/eyes, YOLO face + the base char-LoRA)
-> FaceDetailer (hands, YOLO hand) -> UltimateSDUpscale (ESRGAN + light tile re-draw).
Fixes mushy SDXL eyes/hands and sharpens, WITHOUT changing composition or identity
(re-render uses the same char-LoRA). Writes r{N}_{tag}_fixed/ next to the raw frames.

Tattoos are NOT addressed here (the model doesn't know the exact designs; that's a
training-data fix) — this is purely face/eyes/hands/sharpness.

Usage:
  python improve.py <round> <tag>            # process whole r{round}_{tag} dir
  python improve.py <round> <tag> 0,16,51    # only these indices (validation)
  tag in {sdxl, illustrious}
"""

import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request

WS = "/workspace"
OUT = f"{WS}/out"
URL = "http://127.0.0.1:8191"
HB = f"{WS}/agent_heartbeat"
# tag -> (checkpoint, char-lora, esrgan upscaler)
BASES = {
    "sdxl": ("sd_xl_base_1.0.safetensors", "serg0_sdxl.safetensors", "4x-UltraSharp.pth"),
    "illustrious": (
        "Illustrious-XL-v1.0.safetensors",
        "serg0_ill.safetensors",
        "4x-AnimeSharp.pth",
    ),
}
POS = "serg0, 1boy, solo, detailed face, detailed symmetric eyes, sharp focus, high quality"
NEG = "blurry, lowres, blank eyes, white eyes, deformed eyes, extra eyes, closed eyes, bad anatomy, bad hands, watermark, text, signature"


def upload(path):
    name = os.path.basename(path)
    with open(path, "rb") as fh:
        data = fh.read()
    boundary = "----improvb"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode()
        + data
        + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{boundary}--\r\n'.encode()
    )
    req = urllib.request.Request(
        URL + "/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return r["name"] if not r.get("subfolder") else f"{r['subfolder']}/{r['name']}"


def fd(image_in, model, clip, vae, pos, neg, bbox, seed, guide, denoise, thr):
    """A FaceDetailer node dict (works for any bbox detector — face or hands)."""
    return {
        "class_type": "FaceDetailer",
        "inputs": {
            "image": image_in,
            "model": model,
            "clip": clip,
            "vae": vae,
            "positive": pos,
            "negative": neg,
            "bbox_detector": bbox,
            "guide_size": guide,
            "guide_size_for": True,
            "max_size": 1024,
            "seed": seed,
            "steps": 20,
            "cfg": 6.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": denoise,
            "feather": 5,
            "noise_mask": True,
            "force_inpaint": True,
            "bbox_threshold": thr,
            "bbox_dilation": 10,
            "bbox_crop_factor": 3.0,
            "sam_detection_hint": "center-1",
            "sam_dilation": 0,
            "sam_threshold": 0.93,
            "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False",
            "drop_size": 10,
            "wildcard": "",
            "cycle": 1,
        },
    }


def graph(tag, frame_name, seed):
    ckpt, lora, upscaler = BASES[tag]
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": lora,
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": POS}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": NEG}},
        "5": {"class_type": "LoadImage", "inputs": {"image": frame_name}},
        "10": {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": "bbox/face_yolov8m.pt"},
        },
        "11": {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": "bbox/hand_yolov8s.pt"},
        },
        # face/eyes
        "20": fd(
            ["5", 0],
            ["2", 0],
            ["2", 1],
            ["1", 2],
            ["3", 0],
            ["4", 0],
            ["10", 0],
            seed,
            512,
            0.40,
            0.50,
        ),
        # hands
        "21": fd(
            ["20", 0],
            ["2", 0],
            ["2", 1],
            ["1", 2],
            ["3", 0],
            ["4", 0],
            ["11", 0],
            seed + 1,
            384,
            0.35,
            0.40,
        ),
        "31": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": upscaler}},
        "30": {
            "class_type": "UltimateSDUpscale",
            "inputs": {
                "image": ["21", 0],
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "vae": ["1", 2],
                "upscale_by": 1.5,
                "seed": seed + 2,
                "steps": 12,
                "cfg": 6.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 0.18,
                "upscale_model": ["31", 0],
                "mode_type": "Linear",
                "tile_width": 1024,
                "tile_height": 1024,
                "mask_blur": 8,
                "tile_padding": 32,
                "seam_fix_mode": "None",
                "seam_fix_denoise": 1.0,
                "seam_fix_width": 64,
                "seam_fix_mask_blur": 8,
                "seam_fix_padding": 16,
                "force_uniform_tiles": True,
                "tiled_decode": False,
                "batch_size": 1,
            },
        },
        "40": {
            "class_type": "SaveImage",
            "inputs": {"images": ["30", 0], "filename_prefix": "fix"},
        },
    }
    return g


def run(g, timeout=600, poll=2.0):
    req = urllib.request.Request(
        URL + "/prompt",
        data=json.dumps({"prompt": g}).encode(),
        headers={"Content-Type": "application/json"},
    )
    pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    end = time.monotonic() + timeout
    empty = 0
    while time.monotonic() < end:
        e = json.loads(urllib.request.urlopen(URL + f"/history/{pid}", timeout=15).read()).get(pid)
        if e:
            for n in e.get("outputs", {}).values():
                for im in n.get("images", []):
                    return urllib.request.urlopen(
                        URL + "/view?" + urllib.parse.urlencode(im), timeout=60
                    ).read()
            st = e.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(f"error: {json.dumps(st)[:300]}")
            if st.get("completed"):
                empty += 1
                if empty >= 8:
                    raise RuntimeError(f"completed no image: {json.dumps(st)[:300]}")
        time.sleep(poll)
    raise TimeoutError(pid)


def main():
    r = int(sys.argv[1])
    tag = sys.argv[2]
    src = f"{OUT}/r{r}_{tag}"
    dst = f"{OUT}/r{r}_{tag}_fixed"
    os.makedirs(dst, exist_ok=True)
    frames = sorted(glob.glob(f"{src}/*.png"))
    if len(sys.argv) > 3:
        want = {w.strip().zfill(3) for w in sys.argv[3].split(",") if w.strip()}
        frames = [f for f in frames if os.path.basename(f).split(".")[0] in want]
    for p in frames:
        o = os.path.join(dst, os.path.basename(p))
        if os.path.exists(o):
            continue
        idx = int(os.path.basename(p).split(".")[0])
        t = time.time()
        try:
            name = upload(p)
            png = run(graph(tag, name, 5000 + idx))
            with open(o, "wb") as fh:
                fh.write(png)
            print(f"{tag} {os.path.basename(p)}: ok {time.time() - t:.0f}s", flush=True)
        except Exception as exc:
            print(f"{tag} {os.path.basename(p)}: FAIL {str(exc)[:160]}", flush=True)
        try:
            open(HB, "w").write(str(int(time.time())))
        except OSError:
            pass
    print(f"=== improve r{r}_{tag} done ===", flush=True)


if __name__ == "__main__":
    main()
