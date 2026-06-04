#!/usr/bin/env python3
"""Round-aware test-spread generator for the iterative char-dataset loop.

Each round generates PAIRS scenes per base (vanilla SDXL + Illustrious) using the
R0 char-LoRAs (NO retraining) into r{N}_sdxl / r{N}_illustrious, so the user can
curate and accumulate faithful frames until an optimal dataset size is reached.

Scenes come from /workspace/scene_bank.json if present (a flat JSON list of strings),
else a built-in diverse list. Strictly character-agnostic: only pose / setting /
camera / lighting vary; identity + body + art-style bind to the `serg0` trigger from
the refs. Touches /workspace/agent_heartbeat per render so the cost watchdog will not
stop the pod mid-generation.

Usage: python gen_spread.py [ROUND]   # ROUND default = max existing r{N} + 1
"""

import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

WS = "/workspace"
BASE = "http://127.0.0.1:8188"
HB = f"{WS}/agent_heartbeat"
OUT = f"{WS}/out"
PAIRS = 16  # doubled: 16 scene-pairs => 16 SDXL + 16 Illustrious per round
JOBS = [
    ("sd_xl_base_1.0.safetensors", "serg0_sdxl.safetensors", "sdxl"),
    ("Illustrious-XL-v1.0.safetensors", "serg0_ill.safetensors", "illustrious"),
]
# Character-agnostic scenes (pose / setting / camera / lighting only).
INLINE = [
    "standing on a city street, full body, golden hour",
    "sitting at a cafe with a laptop, medium shot, indoor",
    "walking in a park, full body, side view, daylight",
    "close-up portrait, smiling, soft light",
    "leaning against a brick wall, arms crossed, three-quarter view",
    "sitting on a sofa holding a coffee mug, relaxed, window light",
    "cooking at a kitchen stove, medium shot, warm light",
    "on a rooftop at sunset, full body, looking at the city",
    "reading a book by a window, medium shot, overcast light",
    "waiting at a bus stop, full body, rainy evening, neon reflections",
    "browsing in a bookstore, medium shot, soft indoor light",
    "crossing a street at a crosswalk, full body, low angle",
    "sitting on stone steps outdoors, relaxed, afternoon sun",
    "standing in a subway car holding a rail, three-quarter, fluorescent light",
    "close-up profile portrait, side lighting, dark background",
    "laughing with head slightly back, close-up, bright daylight",
    "tying a shoelace, crouching, full body, sidewalk",
    "holding an umbrella in the rain, full body, wide shot",
    "leaning over a balcony railing, medium shot, city at night",
    "sitting cross-legged on the floor, top-down angle, lamp light",
    "standing in a field of tall grass, full body, windy, sunset",
    "by a lakeside at dawn, full body, misty light",
    "hiking on a forest trail, full body, dappled sunlight",
    "sitting on a park bench, medium shot, morning",
    "portrait looking over the shoulder, three-quarter back view, soft light",
    "standing under a streetlamp at night, full body, dramatic shadow",
    "working at a desk with papers, medium shot, desk lamp",
    "pouring coffee in a kitchen, medium shot, morning light",
    "standing in a doorway, full body, backlit silhouette",
    "sitting at a bar counter, medium shot, warm dim light",
    "walking down a staircase, full body, side view",
    "close-up portrait, neutral expression, even studio light",
    "looking up at the sky, low angle, bright overcast",
    "carrying grocery bags on a sidewalk, full body, daylight",
    "resting against a tree trunk, medium shot, golden hour",
    "standing on a bridge over a river, full body, wide shot",
    "close-up three-quarter portrait, thoughtful, rim light",
    "sitting in a window seat of a train, medium shot, passing scenery",
    "stretching arms outdoors in the morning, full body, sunrise",
    "browsing a market stall, medium shot, colorful daylight",
    "standing in light rain, looking up, medium shot",
    "sitting on the edge of a rooftop, full body, dusk",
    "leaning on a kitchen counter with a mug, medium shot, morning",
    "walking a dog in a park, full body, autumn light",
    "portrait in shadow with a single light source, close-up, chiaroscuro",
    "standing in a snowy street, full body, cold overcast light",
    "sitting on a staircase indoors, three-quarter, soft window light",
    "close-up portrait from a low angle, confident, daylight",
]
NEG = (
    "lowres, bad anatomy, extra fingers, extra limbs, deformed, watermark, text, signature, blurry"
)


def scenes_all():
    """Prefer the curated scene bank if present; else the built-in diverse list."""
    try:
        data = json.load(open(f"{WS}/scene_bank.json"))
        if isinstance(data, list) and len(data) >= PAIRS:
            return [str(s) for s in data]
    except (OSError, ValueError):
        pass
    return INLINE


def next_round():
    rs = [
        int(m.group(1))
        for d in glob.glob(f"{OUT}/r*_sdxl")
        if (m := re.search(r"/r(\d+)_sdxl$", d))
    ]
    return (max(rs) + 1) if rs else 0


def sample(graph_, timeout=300, poll=2.0):
    req = urllib.request.Request(
        BASE + "/prompt",
        data=json.dumps({"prompt": graph_}).encode(),
        headers={"Content-Type": "application/json"},
    )
    pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        e = json.loads(urllib.request.urlopen(BASE + f"/history/{pid}", timeout=15).read()).get(pid)
        if e:
            for n in e.get("outputs", {}).values():
                for im in n.get("images", []):
                    return urllib.request.urlopen(
                        BASE + "/view?" + urllib.parse.urlencode(im), timeout=30
                    ).read()
            raise RuntimeError(f"no image; status={e.get('status', {})}")
        time.sleep(poll)
    raise TimeoutError(pid)


def graph(ckpt, lora, prompt, seed):
    return {
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
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": prompt}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": NEG}},
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": 28,
                "cfg": 6.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "r"}},
    }


def main():
    r = int(sys.argv[1]) if len(sys.argv) > 1 else next_round()
    allsc = scenes_all()
    length = len(allsc)
    scenes = [allsc[(r * PAIRS + i) % length] for i in range(PAIRS)]
    os.makedirs(OUT, exist_ok=True)
    json.dump(scenes, open(f"{OUT}/r{r}_scenes.json", "w"))
    for ckpt, lora, tag in JOBS:
        outdir = f"{OUT}/r{r}_{tag}"
        os.makedirs(outdir, exist_ok=True)
        for i, sc in enumerate(scenes):
            try:
                open(HB, "w").write(str(int(time.time())))
            except OSError:
                pass
            t = time.time()
            try:
                prompt = f"serg0, 1boy, solo, {sc}, masterpiece, best quality"
                png = sample(graph(ckpt, lora, prompt, 7000 + r * 1000 + i))
                open(os.path.join(outdir, f"{i:02d}.png"), "wb").write(png)
                print(f"r{r}_{tag} {i}: ok {time.time() - t:.0f}s", flush=True)
            except Exception as exc:
                print(f"r{r}_{tag} {i}: FAIL {str(exc)[:120]}", flush=True)
    print(f"=== round {r} gen done ===", flush=True)


if __name__ == "__main__":
    main()
