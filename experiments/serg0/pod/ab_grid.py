#!/usr/bin/env python3
"""A/B comparison grid: same prompt+seed rendered as base SDXL / base Illustrious /
SDXL+serg0-LoRA / Illustrious+serg0-LoRA, across EVERY R1 checkpoint (400..2400).

Raw txt2img (NO improver) so the comparison isolates the model: identical prompt and
identical seed per prompt-row across all variants — only the base / LoRA / checkpoint
changes. Builds a self-contained page with a checkpoint switcher (the LoRA columns swap
per selected checkpoint) and zips it for offline download.

Layout per prompt = one row: [База SDXL | База Illustrious | SDXL+LoRA | Illustrious+LoRA].

Run via run_ab.sh (it symlinks the checkpoint LoRAs + brings up two fresh ComfyUI
instances). Output -> /workspace/ab_out/{variant}/{i:03d}.png + index.html + ab_compare.zip.
"""

import json
import os
import threading
import time
import urllib.parse
import urllib.request
import zipfile

WS = "/workspace"
OUT = f"{WS}/ab_out"
HB = f"{WS}/agent_heartbeat"
STEPS = [400, 800, 1200, 1600, 2000, 2400]
# base tag -> (checkpoint, lora-stem for the R1 checkpoint files, ComfyUI url)
SDXL = ("sd_xl_base_1.0.safetensors", "serg0_sdxl_r1", "http://127.0.0.1:8190")
ILL = ("Illustrious-XL-v1.0.safetensors", "serg0_ill_r1", "http://127.0.0.1:8189")
NEG = (
    "lowres, bad anatomy, bad hands, extra fingers, extra limbs, deformed, "
    "blank eyes, white eyes, deformed eyes, watermark, text, signature, blurry"
)
# >=20 character-agnostic prompts spanning everyday + fantasy/cyberpunk/postapoc/scifi
# and a mix of framing (portrait / full body / close-up) and lighting.
PROMPTS = [
    "close-up portrait, neutral expression, soft studio light",
    "full body standing on a city street, golden hour",
    "sitting at a cafe with a laptop, medium shot, indoor warm light",
    "three-quarter portrait, slight smile, window light",
    "walking in a park, full body, autumn daylight",
    "standing on castle ramparts overlooking a valley, wide shot, golden hour",
    "browsing a rain-slick cyberpunk market under neon holograms, full body, night",
    "huddled under a corrugated metal lean-to during ashfall, medium shot, dim twilight",
    "on the bridge of a starship looking at a viewscreen, medium shot, cool light",
    "a medieval traveler resting by a campfire in a forest, full body, night",
    "close-up profile, dramatic side lighting, dark background",
    "sitting on a park bench reading a book, medium shot, morning",
    "standing in a snowy street, full body, cold overcast light",
    "leaning against a brick wall, arms crossed, three-quarter view, daylight",
    "cooking at a kitchen stove, medium shot, warm light",
    "portrait laughing with head slightly back, bright daylight",
    "standing under a streetlamp at night, full body, dramatic shadow",
    "hiking on a forest trail, full body, dappled sunlight",
    "in a desert wasteland wearing goggles, full body, harsh sun",
    "close-up, thoughtful expression, rim light, dark background",
]
SEED0 = 90000


def sample(graph, url, timeout=300, poll=1.5):
    req = urllib.request.Request(
        url + "/prompt",
        data=json.dumps({"prompt": graph}).encode(),
        headers={"Content-Type": "application/json"},
    )
    pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    end = time.monotonic() + timeout
    empty = 0
    while time.monotonic() < end:
        e = json.loads(urllib.request.urlopen(url + f"/history/{pid}", timeout=15).read()).get(pid)
        if e:
            for n in e.get("outputs", {}).values():
                for im in n.get("images", []):
                    return urllib.request.urlopen(
                        url + "/view?" + urllib.parse.urlencode(im), timeout=60
                    ).read()
            st = e.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(f"err {json.dumps(st)[:200]}")
            if st.get("completed"):
                empty += 1
                if empty >= 6:
                    raise RuntimeError("completed no image")
        time.sleep(poll)
    raise TimeoutError(pid)


def graph(ckpt, lora, prompt, seed):
    g = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}}
    if lora:
        g["2"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": lora,
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        }
        m, c = ["2", 0], ["2", 1]
    else:
        m, c = ["1", 0], ["1", 1]
    pos = f"serg0, 1boy, solo, {prompt}, masterpiece, best quality"
    g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": c, "text": pos}}
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": c, "text": NEG}}
    g["5"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    }
    g["6"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": m,
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
    }
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}}
    g["8"] = {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "ab"}}
    return g


def variants(tag):
    ckpt, stem, url = SDXL if tag == "sdxl" else ILL
    vs = [(f"base_{tag}", None)]
    vs += [(f"{tag}_s{s}", f"{stem}-step{s:08d}.safetensors") for s in STEPS]
    return ckpt, url, vs


def run_base(tag):
    """One base: render base (no LoRA) + every checkpoint, for all prompts."""
    ckpt, url, vs = variants(tag)
    for vname, lora in vs:
        d = f"{OUT}/{vname}"
        os.makedirs(d, exist_ok=True)
        for i, p in enumerate(PROMPTS):
            try:
                open(HB, "w").write(str(int(time.time())))
            except OSError:
                pass
            o = f"{d}/{i:03d}.png"
            if os.path.exists(o):
                continue
            t = time.time()
            try:
                png = sample(graph(ckpt, lora, p, SEED0 + i), url)
                open(o, "wb").write(png)
                print(f"{vname} {i}: ok {time.time() - t:.0f}s", flush=True)
            except Exception as exc:
                print(f"{vname} {i}: FAIL {str(exc)[:120]}", flush=True)


def build_page():
    cols = ["База SDXL", "База Illustrious", "SDXL + LoRA", "Illustrious + LoRA"]
    btns = "".join(
        f'<button class="ck" data-s="{s}" onclick="setck({s})">шаг {s}</button>' for s in STEPS
    )
    rows = []
    for i, p in enumerate(PROMPTS):
        f = f"{i:03d}.png"
        cells = (
            f'<a href="base_sdxl/{f}" target="_blank"><img loading="lazy" src="base_sdxl/{f}"></a>'
            f'<a href="base_ill/{f}" target="_blank"><img loading="lazy" src="base_ill/{f}"></a>'
            f'<a class="ls" href="sdxl_s2400/{f}" target="_blank"><img loading="lazy" class="li-sdxl" data-f="{f}" src="sdxl_s2400/{f}"></a>'
            f'<a class="li" href="ill_s2400/{f}" target="_blank"><img loading="lazy" class="li-ill" data-f="{f}" src="ill_s2400/{f}"></a>'
        )
        rows.append(f'<div class="prompt">#{i + 1}. {p}</div><div class="row">{cells}</div>')
    head = "".join(f'<div class="h">{c}</div>' for c in cols)
    html = f"""<!doctype html><html lang=ru><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Serg0 R1 — A/B (base vs LoRA, по чекпойнтам)</title>
<style>
body{{font-family:system-ui,Arial;margin:0;background:#111;color:#eee}}
header{{position:sticky;top:0;background:#1a1a1a;padding:10px 14px;border-bottom:1px solid #333;z-index:5}}
h1{{font-size:17px;margin:0 0 8px}}
.ck{{background:#2a2a2a;color:#ccc;border:1px solid #444;border-radius:6px;padding:6px 10px;margin:2px;cursor:pointer;font-size:13px}}
.ck.on{{background:#3b82f6;color:#fff;border-color:#3b82f6}}
.dl{{float:right;background:#10b981;color:#fff;text-decoration:none;padding:7px 12px;border-radius:6px;font-size:13px}}
.note{{color:#999;font-size:12px;margin-top:6px}}
.heads{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;padding:6px 14px;color:#9bd;font-size:12px;font-weight:600}}
.prompt{{padding:14px 14px 4px;color:#ffd479;font-size:14px}}
.row{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;padding:0 14px 6px}}
.row img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#222;display:block}}
.ls{{outline:2px solid #3b82f6;border-radius:8px}} .li{{outline:2px solid #a855f7;border-radius:8px}}
</style>
<header>
<a class=dl href="ab_compare.zip" download>⬇ Скачать страницу (zip)</a>
<h1>Serg0 R1 — сравнение: <b>База</b> vs <b>+LoRA</b>, по чекпойнтам обучения</h1>
<div>Чекпойнт LoRA: {btns}</div>
<div class=note>Один и тот же промт и сид во всех 4 столбцах — отличается только база / наличие LoRA / чекпойнт. Без улучшайзера (честное сравнение). Клик по картинке — полный размер. Колонки с LoRA меняются кнопками выше.</div>
</header>
<div class=heads>{head}</div>
{"".join(rows)}
<script>
function setck(s){{
 document.querySelectorAll('.li-sdxl').forEach(im=>{{im.src='sdxl_s'+s+'/'+im.dataset.f;im.closest('a').href='sdxl_s'+s+'/'+im.dataset.f;}});
 document.querySelectorAll('.li-ill').forEach(im=>{{im.src='ill_s'+s+'/'+im.dataset.f;im.closest('a').href='ill_s'+s+'/'+im.dataset.f;}});
 document.querySelectorAll('.ck').forEach(b=>b.classList.toggle('on',+b.dataset.s===s));
}}
setck(2400);
</script>
</html>"""
    with open(f"{OUT}/index.html", "w", encoding="utf-8") as fh:
        fh.write(html)


def zip_page():
    z = f"{OUT}/ab_compare.zip"
    if os.path.exists(z):
        os.remove(z)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for root, _dirs, files in os.walk(OUT):
            for fn in files:
                fp = os.path.join(root, fn)
                if fp == z:
                    continue
                zf.write(fp, os.path.relpath(fp, os.path.dirname(OUT)))


def main():
    os.makedirs(OUT, exist_ok=True)
    ths = [threading.Thread(target=run_base, args=(t,)) for t in ("sdxl", "ill")]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    build_page()
    zip_page()
    print("=== A/B grid done ===", flush=True)


if __name__ == "__main__":
    main()
