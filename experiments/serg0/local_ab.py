#!/usr/bin/env python3
"""LOCAL A/B comparison grid (RTX 5070 Ti, single ComfyUI on :8188).

Same prompt+seed rendered as base SDXL / base Illustrious / SDXL+serg0-LoRA /
Illustrious+serg0-LoRA, across EVERY R2 checkpoint discovered on disk (no hardcoded
step list — whatever kohya actually wrote drives both the render loop AND the page).

Raw txt2img (NO improver) so the comparison isolates the model: identical prompt and
identical seed per prompt-row across all variants — only the base / LoRA / checkpoint
changes. Builds a self-contained page with a checkpoint switcher (the two LoRA columns
swap to the selected checkpoint) and zips it for offline download.

Layout per prompt = one row: [База SDXL | База Illustrious | SDXL+LoRA | Illustrious+LoRA].

Single GPU ⇒ ONE ComfyUI, ONE checkpoint resident at a time ⇒ render outer-loop-by-base
(all SDXL variants, then all Illustrious) so the ~6.5GB checkpoint loads twice, not 220×.
Resumable: existing PNGs are skipped, so an interrupted 440-render run just continues.

Run via run_local_ab.sh (symlinks the R2 checkpoints into ComfyUI/models/loras first).
Output -> /home/serg/serg0_ab_out/{variant}/{i:03d}.png + index.html + ab_compare.zip.
"""
# ruff: noqa: E501 - this module embeds a self-contained HTML/CSS/JS page template

import json
import os
import re
import time
import urllib.parse
import urllib.request
import zipfile

OUT = "/home/serg/serg0_ab_out"
URL = "http://127.0.0.1:8188"
LORA_SRC = {
    "sdxl": "/home/serg/serg0_lora_out/sdxl_r2",
    "ill": "/home/serg/serg0_lora_out/ill_r2",
}
COMFY_LORA = "/home/serg/ComfyUI/models/loras"
CKPT = {"sdxl": "sd_xl_base_1.0.safetensors", "ill": "Illustrious-XL-v1.0.safetensors"}
FINAL_STEP = 4000  # the bare serg0_<base>_r2.safetensors == final == this step
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


def discover(base):
    """Map step -> lora filename for one base, from whatever kohya wrote.

    Parses serg0_<base>_r2-stepNNNNNNNN.safetensors; the bare serg0_<base>_r2.safetensors
    (kohya's final save) is canonicalised to FINAL_STEP, overriding any step04000 dupe.
    """
    src = LORA_SRC[base]
    steps: dict[int, str] = {}
    if os.path.isdir(src):
        pat = re.compile(rf"^serg0_{base}_r2-step0*(\d+)\.safetensors$")
        for f in sorted(os.listdir(src)):
            m = pat.match(f)
            if m:
                steps[int(m.group(1))] = f
        final = f"serg0_{base}_r2.safetensors"
        if os.path.exists(os.path.join(src, final)):
            steps[FINAL_STEP] = final  # dedup final vs step04000 -> prefer the named final
    return dict(sorted(steps.items()))


def link_loras():
    """Flat-symlink every discovered R2 checkpoint into ComfyUI/models/loras (idempotent)."""
    os.makedirs(COMFY_LORA, exist_ok=True)
    linked = 0
    for base in ("sdxl", "ill"):
        for fn in discover(base).values():
            src = os.path.join(LORA_SRC[base], fn)
            dst = os.path.join(COMFY_LORA, fn)
            if not os.path.exists(dst):
                os.symlink(src, dst)
                linked += 1
    return linked


def loras_listed():
    """The LoRA filenames ComfyUI currently exposes. Hitting /object_info/LoraLoader also
    forces ComfyUI to rescan the loras folder, picking up just-symlinked files."""
    try:
        info = json.loads(
            urllib.request.urlopen(URL + "/object_info/LoraLoader", timeout=20).read()
        )
        return set(info["LoraLoader"]["input"]["required"]["lora_name"][0])
    except Exception:
        return set()


def ensure_visible(want):
    """Confirm the freshly-symlinked r2 loras are actually visible to ComfyUI.

    ComfyUI caches its lora list; a stale list means every LoRA cell 404s while base cells
    pass — a page of broken columns after a 70-min run. We rescan (twice) and report.
    """
    have = loras_listed()
    missing = want - have
    if missing:
        have = loras_listed()  # one more rescan pass
        missing = want - have
    if missing:
        print(
            f"WARNING: {len(missing)} r2 loras NOT visible to ComfyUI "
            f"(e.g. {sorted(missing)[:2]}). Restart ComfyUI / check models/loras symlinks.",
            flush=True,
        )
    else:
        print(f"all {len(want)} r2 loras visible to ComfyUI", flush=True)
    return not missing


def smoke(base):
    """Render ONE real-checkpoint cell up front; raise on failure so a stale-lora 404 aborts
    the run in seconds instead of after a 70-minute grid of broken LoRA columns."""
    steps = discover(base)
    if not steps:
        raise RuntimeError(f"no {base} r2 checkpoints found")
    step, fn = next(iter(steps.items()))
    png = sample(graph(CKPT[base], fn, PROMPTS[0], SEED0))
    print(f"smoke OK: {base} s{step} ({fn}) -> {len(png)} bytes", flush=True)


def sample(graph, timeout=300, poll=1.5):
    req = urllib.request.Request(
        URL + "/prompt",
        data=json.dumps({"prompt": graph}).encode(),
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


def render_one(ckpt, lora, vname):
    """Render all prompts for one variant (base or one checkpoint). Resumable."""
    d = f"{OUT}/{vname}"
    os.makedirs(d, exist_ok=True)
    for i, p in enumerate(PROMPTS):
        o = f"{d}/{i:03d}.png"
        if os.path.exists(o):
            continue
        t = time.time()
        try:
            png = sample(graph(ckpt, lora, p, SEED0 + i))
            with open(o, "wb") as fh:
                fh.write(png)
            print(f"{vname} {i}: ok {time.time() - t:.0f}s", flush=True)
        except Exception as exc:  # log and continue, never abort the whole grid
            print(f"{vname} {i}: FAIL {str(exc)[:120]}", flush=True)


def render_base(base):
    """One base, outer loop: base (no LoRA) + every checkpoint, all in ckpt-resident order."""
    ckpt = CKPT[base]
    render_one(ckpt, None, f"base_{base}")
    for step, fn in discover(base).items():
        render_one(ckpt, fn, f"{base}_s{step}")


def build_page(steps):
    cols = ["База SDXL", "База Illustrious", "SDXL + LoRA", "Illustrious + LoRA"]
    default = steps[-1] if steps else FINAL_STEP
    btns = "".join(
        f'<button class="ck" data-s="{s}" onclick="setck({s})">шаг {s}</button>' for s in steps
    )
    rows = []
    for i, p in enumerate(PROMPTS):
        f = f"{i:03d}.png"
        cells = (
            f'<a href="base_sdxl/{f}" target="_blank"><img loading="lazy" src="base_sdxl/{f}"></a>'
            f'<a href="base_ill/{f}" target="_blank"><img loading="lazy" src="base_ill/{f}"></a>'
            f'<a class="ls" href="sdxl_s{default}/{f}" target="_blank">'
            f'<img loading="lazy" class="li-sdxl" data-f="{f}" src="sdxl_s{default}/{f}"></a>'
            f'<a class="li" href="ill_s{default}/{f}" target="_blank">'
            f'<img loading="lazy" class="li-ill" data-f="{f}" src="ill_s{default}/{f}"></a>'
        )
        rows.append(f'<div class="prompt">#{i + 1}. {p}</div><div class="row">{cells}</div>')
    head = "".join(f'<div class="h">{c}</div>' for c in cols)
    html = f"""<!doctype html><html lang=ru><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Serg0 R2 — A/B (base vs LoRA, по чекпойнтам)</title>
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
<h1>Serg0 R2 — сравнение: <b>База</b> vs <b>+LoRA</b>, по чекпойнтам обучения</h1>
<div>Чекпойнт LoRA: {btns}</div>
<div class=note>Один и тот же промт и сид во всех 4 столбцах — отличается только база / наличие LoRA / чекпойнт. Без улучшайзера (честное сравнение). Клик по картинке — полный размер. Колонки с LoRA меняются кнопками выше. Ищем пик: похожесть × гибкость (поза/фон слушаются промт, цвета не «жареные»).</div>
</header>
<div class=heads>{head}</div>
{"".join(rows)}
<script>
function setck(s){{
 document.querySelectorAll('.li-sdxl').forEach(im=>{{im.src='sdxl_s'+s+'/'+im.dataset.f;im.closest('a').href='sdxl_s'+s+'/'+im.dataset.f;}});
 document.querySelectorAll('.li-ill').forEach(im=>{{im.src='ill_s'+s+'/'+im.dataset.f;im.closest('a').href='ill_s'+s+'/'+im.dataset.f;}});
 document.querySelectorAll('.ck').forEach(b=>b.classList.toggle('on',+b.dataset.s===s));
}}
setck({default});
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
    n = link_loras()
    print(f"symlinked {n} new R2 checkpoints into {COMFY_LORA}", flush=True)
    s_steps, i_steps = discover("sdxl"), discover("ill")
    steps = sorted(set(s_steps) | set(i_steps))
    print(f"sdxl steps={list(s_steps)}", flush=True)
    print(f"ill  steps={list(i_steps)}", flush=True)
    # guard: the symlinked r2 loras must be visible, and a real LoRA cell must render,
    # BEFORE the 70-min grid — else stale-list 404s yield a page of broken LoRA columns.
    ensure_visible(set(s_steps.values()) | set(i_steps.values()))
    if s_steps:
        smoke("sdxl")
    if i_steps:
        smoke("ill")
    # outer loop by base: one checkpoint resident at a time, no thrashing
    render_base("sdxl")
    render_base("ill")
    build_page(steps)
    zip_page()
    print(f"=== local A/B grid done ({len(steps)} checkpoints/base) ===", flush=True)


if __name__ == "__main__":
    main()
