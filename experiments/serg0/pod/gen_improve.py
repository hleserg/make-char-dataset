#!/usr/bin/env python3
"""Production generation backend for Serg0: txt2img on a chosen base + char-LoRA
(+ optional style LoRA), with the community-best-practice IMPROVER as a single
TOGGLE — FaceDetailer (face/eyes) → FaceDetailer (hands) → UltimateSDUpscale (ESRGAN).

This locks in the user's decision: the improver gave a big quality win on the training
photos, so it is part of prod generation (default ON), runnable on whichever models we
end up using (`serg0_*_r1`, falling back to `serg0_*` R0 if R1 not present yet).

One ComfyUI graph per image (gen → improve → save) so there is no intermediate I/O.
Importable: `generate(...)` returns the list of saved PNG paths and reports progress via
an `on_event(kind, data)` callback (kinds: "start", "done", "fail") — the single-gen UI
uses that plus ComfyUI's websocket for step-level progress. Runs against ANY ComfyUI URL
(local or pod) that has the improver custom nodes (Impact-Pack/Subpack/UltimateSDUpscale).

CLI (smoke test):
  python gen_improve.py --base sdxl --prompt "on a rooftop at sunset, full body" \
      --count 2 --url http://127.0.0.1:8191 --outdir /workspace/prod_test
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Callable, Optional

# base tag -> (checkpoint, char-lora basename stem, esrgan upscaler)
BASES = {
    "sdxl": ("sd_xl_base_1.0.safetensors", "serg0_sdxl", "4x-UltraSharp.pth"),
    "illustrious": ("Illustrious-XL-v1.0.safetensors", "serg0_ill", "4x-AnimeSharp.pth"),
}
TRIGGER = "serg0, 1boy, solo"
QUALITY = "masterpiece, best quality"
DEFAULT_NEG = (
    "lowres, bad anatomy, bad hands, extra fingers, extra limbs, deformed, "
    "blank eyes, white eyes, deformed eyes, watermark, text, signature, blurry"
)


def _get(url: str, path: str, timeout: int = 15):
    return json.loads(urllib.request.urlopen(url + path, timeout=timeout).read())


def list_loras(url: str) -> list[str]:
    """Available LoRA filenames from a running ComfyUI (works local or remote)."""
    try:
        info = _get(url, "/object_info/LoraLoader")
        return list(info["LoraLoader"]["input"]["required"]["lora_name"][0])
    except Exception:
        return []


def resolve_char_lora(url: str, base_tag: str) -> str:
    """Prefer the R1 char-LoRA (serg0_*_r1) if ComfyUI sees it, else R0 (serg0_*)."""
    stem = BASES[base_tag][1]
    loras = list_loras(url)
    for cand in (f"{stem}_r1.safetensors", f"{stem}.safetensors"):
        if cand in loras:
            return cand
    return f"{stem}.safetensors"


def _fd(image_in, model, clip, vae, pos, neg, bbox, seed, guide, denoise, thr):
    """A FaceDetailer node (works for any bbox detector — face or hands)."""
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


def build_graph(
    base_tag: str,
    char_lora: str,
    prompt: str,
    negative: str,
    seed: int,
    *,
    improve: bool,
    style_lora: Optional[str] = None,
    style_strength: float = 0.85,
    steps: int = 28,
    cfg: float = 6.5,
    width: int = 1024,
    height: int = 1024,
) -> dict:
    ckpt, _stem, esrgan = BASES[base_tag]
    pos = f"{TRIGGER}, {prompt}, {QUALITY}"
    # checkpoint -> char LoRA -> (optional) style LoRA
    g: dict = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": char_lora,
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        },
    }
    last = "2"
    if style_lora:
        g["2s"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": [last, 0],
                "clip": [last, 1],
                "lora_name": style_lora,
                "strength_model": style_strength,
                "strength_clip": style_strength,
            },
        }
        last = "2s"
    g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": [last, 1], "text": pos}}
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": [last, 1], "text": negative}}
    g["5"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    g["6"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": [last, 0],
            "positive": ["3", 0],
            "negative": ["4", 0],
            "latent_image": ["5", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0,
        },
    }
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}}
    if not improve:
        g["40"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["7", 0], "filename_prefix": "gen"},
        }
        return g
    # improver: face -> hands -> ESRGAN upscale
    g["10"] = {
        "class_type": "UltralyticsDetectorProvider",
        "inputs": {"model_name": "bbox/face_yolov8m.pt"},
    }
    g["11"] = {
        "class_type": "UltralyticsDetectorProvider",
        "inputs": {"model_name": "bbox/hand_yolov8s.pt"},
    }
    g["20"] = _fd(
        ["7", 0],
        [last, 0],
        [last, 1],
        ["1", 2],
        ["3", 0],
        ["4", 0],
        ["10", 0],
        seed,
        512,
        0.40,
        0.50,
    )
    g["21"] = _fd(
        ["20", 0],
        [last, 0],
        [last, 1],
        ["1", 2],
        ["3", 0],
        ["4", 0],
        ["11", 0],
        seed + 1,
        384,
        0.35,
        0.40,
    )
    g["31"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": esrgan}}
    g["30"] = {
        "class_type": "UltimateSDUpscale",
        "inputs": {
            "image": ["21", 0],
            "model": [last, 0],
            "positive": ["3", 0],
            "negative": ["4", 0],
            "vae": ["1", 2],
            "upscale_by": 1.5,
            "seed": seed + 2,
            "steps": 12,
            "cfg": cfg,
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
    }
    g["40"] = {"class_type": "SaveImage", "inputs": {"images": ["30", 0], "filename_prefix": "gen"}}
    return g


def _submit(url: str, graph: dict) -> str:
    req = urllib.request.Request(
        url + "/prompt",
        data=json.dumps({"prompt": graph}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]


def _await_image(url: str, pid: str, timeout: int = 600, poll: float = 1.5) -> bytes:
    end = time.monotonic() + timeout
    empty = 0
    while time.monotonic() < end:
        e = _get(url, f"/history/{pid}").get(pid)
        if e:
            for n in e.get("outputs", {}).values():
                for im in n.get("images", []):
                    return urllib.request.urlopen(
                        url + "/view?" + urllib.parse.urlencode(im), timeout=60
                    ).read()
            st = e.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(f"exec error: {json.dumps(st)[:300]}")
            if st.get("completed"):
                empty += 1
                if empty >= 8:
                    raise RuntimeError(f"completed no image: {json.dumps(st)[:300]}")
        time.sleep(poll)
    raise TimeoutError(pid)


def generate(
    base_tag: str,
    prompt: str,
    *,
    count: int = 5,
    negative: str = DEFAULT_NEG,
    char_lora: Optional[str] = None,
    style_lora: Optional[str] = None,
    style_strength: float = 0.85,
    improve: bool = True,
    url: str = "http://127.0.0.1:8191",
    outdir: str = "/workspace/prod_out",
    seed0: Optional[int] = None,
    steps: int = 28,
    cfg: float = 6.5,
    on_event: Optional[Callable[[str, dict], None]] = None,
) -> list[str]:
    """Generate `count` images; return saved PNG paths. on_event(kind, data) for progress."""
    if base_tag not in BASES:
        raise ValueError(f"base must be one of {list(BASES)}")
    os.makedirs(outdir, exist_ok=True)
    char_lora = char_lora or resolve_char_lora(url, base_tag)
    seed0 = seed0 if seed0 is not None else int(time.time()) & 0x7FFFFFF
    paths: list[str] = []
    for i in range(count):
        seed = seed0 + i * 3
        graph = build_graph(
            base_tag,
            char_lora,
            prompt,
            negative,
            seed,
            improve=improve,
            style_lora=style_lora,
            style_strength=style_strength,
            steps=steps,
            cfg=cfg,
        )
        try:
            pid = _submit(url, graph)
            if on_event:
                on_event("start", {"index": i, "total": count, "prompt_id": pid, "seed": seed})
            png = _await_image(url, pid)
            p = os.path.join(outdir, f"{int(time.time())}_{i:02d}.png")
            with open(p, "wb") as fh:
                fh.write(png)
            paths.append(p)
            if on_event:
                on_event("done", {"index": i, "total": count, "path": p})
        except Exception as exc:  # noqa: BLE001 — report and continue the batch
            if on_event:
                on_event("fail", {"index": i, "total": count, "error": str(exc)[:200]})
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=list(BASES))
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--no-improve", action="store_true")
    ap.add_argument("--style-lora", default=None)
    ap.add_argument("--url", default="http://127.0.0.1:8191")
    ap.add_argument("--outdir", default="/workspace/prod_out")
    args = ap.parse_args()
    paths = generate(
        args.base,
        args.prompt,
        count=args.count,
        negative=args.negative,
        style_lora=args.style_lora,
        improve=not args.no_improve,
        url=args.url,
        outdir=args.outdir,
        on_event=lambda k, d: print(f"[{k}] {d}", flush=True),
    )
    print(f"generated {len(paths)} -> {args.outdir}")


if __name__ == "__main__":
    main()
