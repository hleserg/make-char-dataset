#!/usr/bin/env python3
"""Serg0 comic panel-factory (MVP) — the AI-delegated part of the comic pipeline (①/②
from research/comic-pipelines.md): render panels with the char-LoRA (identity anchor) +
style-LoRA + improver, ControlNet for per-panel pose/composition, IP-Adapter to pull a
face on problem panels, then per-panel re-detail / regen, and export panels + a page grid
for lettering by hand in Clip Studio Paint.

Dependency-free stdlib server (reuses ui_single's plumbing: busy-gate, ws step-progress,
ComfyUI submit, Telegram) so cu128 is never touched. Identity = char-LoRA (NOT seed-lock —
seed-lock is just a reproducibility toggle). Aspect maps to SDXL training buckets. Batch
render is ONE sequential job with a concurrency guard (the «повисло» lesson). v1 «доводка»
re-runs the validated FaceDetailer (face+hands) on a panel — no mask canvas; manual-mask
inpaint is stage 2.

Run:  python3 ui_comic.py            # serves http://localhost:8770
"""
# ruff: noqa: E501 - this module embeds a self-contained HTML/CSS/JS page template

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import threading
import time
import urllib.parse
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ui_single as us

gi = us.gi

HOST, PORT = "0.0.0.0", 8770
WORK = "/home/serg/serg0_comic_out/work"
PAGE_JSON = os.path.join(WORK, "page.json")
COMFY_INPUT = us.COMFY_INPUT
# aspect -> SDXL training bucket (arbitrary tall/wide degrades SDXL anatomy)
ASPECT = {"portrait": (832, 1216), "square": (1024, 1024), "landscape": (1216, 832)}

DEFAULT_SETTINGS = {
    "base": "illustrious",
    "style_lora": us.STYLE_NONE,
    "style_strength": 0.85,
    "improve": True,
    "negative": gi.DEFAULT_NEG,
    "aspect": "portrait",
}


def _blank_panel():
    return {
        "prompt": "",
        "aspect": "portrait",
        "seed": 0,
        "lock_seed": False,
        "cn_ref": None,  # ControlNet pose/composition ref (one filename in ComfyUI/input)
        "ip_refs": [],  # IP-Adapter face refs (filenames)
        "image": None,  # rendered panel filename in WORK
        "v": 0,  # bump to cache-bust the thumbnail
    }


def load_page():
    if os.path.exists(PAGE_JSON):
        try:
            with open(PAGE_JSON) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"settings": dict(DEFAULT_SETTINGS), "panels": [_blank_panel()]}


PAGE = load_page()
PAGE_LOCK = threading.Lock()


def save_page():
    os.makedirs(WORK, exist_ok=True)
    with open(PAGE_JSON, "w") as fh:
        json.dump(PAGE, fh, indent=2)


# ------------------------------------------------------------------------- graph builders
def build_panel_graph(panel, settings, seed):
    base = settings["base"] if settings["base"] in gi.BASES else "illustrious"
    char_lora = gi.resolve_char_lora(us.COMFY_URL, base)
    style = settings.get("style_lora") or None
    if style == us.STYLE_NONE:
        style = None
    w, h = ASPECT.get(panel.get("aspect") or settings.get("aspect", "portrait"), (832, 1216))
    g = gi.build_graph(
        base,
        char_lora,
        panel.get("prompt", ""),
        settings.get("negative") or gi.DEFAULT_NEG,
        seed,
        improve=bool(settings.get("improve", True)),
        style_lora=style,
        style_strength=float(settings.get("style_strength", 0.85)),
        width=w,
        height=h,
    )
    modes = us.available_ref_modes()
    if panel.get("ip_refs") and modes["ipadapter"]:
        g = us.apply_ipadapter(g, panel["ip_refs"], 0.8)
    if panel.get("cn_ref") and modes["controlnet"]:
        g = us.apply_controlnet(g, panel["cn_ref"], 0.8)
    return g


def build_redetail(panel, settings, img_name, seed, denoise=0.42):
    """v1 «доводка»: re-run FaceDetailer(face)+FaceDetailer(hands) on an existing panel image."""
    base = settings["base"] if settings["base"] in gi.BASES else "illustrious"
    ckpt = gi.BASES[base][0]
    char_lora = gi.resolve_char_lora(us.COMFY_URL, base)
    pos = f"{gi.TRIGGER}, {panel.get('prompt', '')}, {gi.QUALITY}"
    neg = settings.get("negative") or gi.DEFAULT_NEG
    g = {
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
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": pos}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": neg}},
        "7": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "10": {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": "bbox/face_yolov8m.pt"},
        },
        "11": {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": "bbox/hand_yolov8s.pt"},
        },
    }
    g["20"] = gi._fd(
        ["7", 0],
        ["2", 0],
        ["2", 1],
        ["1", 2],
        ["3", 0],
        ["4", 0],
        ["10", 0],
        seed,
        512,
        denoise,
        0.50,
    )
    g["21"] = gi._fd(
        ["20", 0],
        ["2", 0],
        ["2", 1],
        ["1", 2],
        ["3", 0],
        ["4", 0],
        ["11", 0],
        seed + 1,
        384,
        max(0.2, denoise - 0.05),
        0.40,
    )
    g["40"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["21", 0], "filename_prefix": "comicdetail"},
    }
    return g


# ------------------------------------------------------------------------------- job/SSE
class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.finished = False

    def emit(self, kind, **data):
        with self.cond:
            self.events.append({"kind": kind, "t": time.strftime("%H:%M:%S"), **data})
            self.cond.notify_all()

    def done(self):
        with self.cond:
            self.finished = True
            self.cond.notify_all()


ACTIVE: Job | None = None  # the single in-flight render/regen/redetail job (concurrency guard)


def _img_to_input(src_path, tag):
    """Copy a rendered panel into ComfyUI/input so LoadImage can read it (for re-detail)."""
    os.makedirs(COMFY_INPUT, exist_ok=True)
    name = f"comicpanel_{tag}.png"
    with open(src_path, "rb") as a, open(os.path.join(COMFY_INPUT, name), "wb") as b:
        b.write(a.read())
    return name


def render_worker(job, indices, mode):
    """mode: 'render'|'regen' build+sample; 'redetail' re-run detailer on existing image."""
    global ACTIVE
    try:
        improve = bool(PAGE["settings"].get("improve", True))
        ok, reason, _v, _r = us.busy_check(improve)
        waited = 0
        while not ok:
            job.emit("busy", reason=reason, waited=waited, retick=us.RETICK_SEC)
            time.sleep(us.RETICK_SEC)
            waited += us.RETICK_SEC
            ok, reason, _v, _r = us.busy_check(improve)
        job.emit("ready", reason=reason)
        us.tg(f"🟢 Serg0 комикс: рендер {len(indices)} кадр(ов), режим {mode}.")
        cid = uuid.uuid4().hex
        ws = us.ComfyWS(us.COMFY_HOST, us.COMFY_PORT, cid)
        try:
            for n, idx in enumerate(indices):
                with PAGE_LOCK:
                    if idx >= len(PAGE["panels"]):
                        continue
                    panel = dict(PAGE["panels"][idx])
                    settings = dict(PAGE["settings"])
                seed = (
                    panel["seed"]
                    if panel.get("lock_seed") and panel.get("seed")
                    else (int(time.time()) & 0x7FFFFFF) + idx * 7
                )
                if mode == "redetail":
                    if not panel.get("image"):
                        job.emit(
                            "panel_done", index=idx, pos=n, total=len(indices), skipped="no image"
                        )
                        continue
                    in_name = _img_to_input(os.path.join(WORK, panel["image"]), f"{idx}")
                    graph = build_redetail(panel, settings, in_name, seed)
                else:
                    graph = build_panel_graph(panel, settings, seed)
                pid = us.submit(graph, cid)
                job.emit("panel_start", index=idx, pos=n, total=len(indices), seed=seed)
                node = None
                while True:
                    op, raw = ws.recv()
                    if op == 0x8:
                        raise ConnectionError("ws closed")
                    if op != 0x1:
                        continue
                    msg = json.loads(raw)
                    t, d = msg.get("type"), msg.get("data", {})
                    if t == "progress" and d.get("prompt_id") == pid:
                        job.emit(
                            "step",
                            index=idx,
                            pos=n,
                            total=len(indices),
                            value=d.get("value", 0),
                            max=d.get("max", 1),
                            node=node,
                        )
                    elif t == "executing" and d.get("prompt_id") == pid:
                        if d.get("node") is None:
                            break
                        node = d.get("node")
                    elif t == "execution_error" and d.get("prompt_id") == pid:
                        raise RuntimeError(str(d.get("exception_message", "exec error"))[:200])
                png = us.history_image(pid)
                fn = f"panel_{idx}.png"
                with open(os.path.join(WORK, fn), "wb") as fh:
                    fh.write(png)
                with PAGE_LOCK:
                    if idx < len(PAGE["panels"]):
                        PAGE["panels"][idx]["image"] = fn
                        PAGE["panels"][idx]["seed"] = seed
                        PAGE["panels"][idx]["v"] = PAGE["panels"][idx].get("v", 0) + 1
                        save_page()
                job.emit(
                    "panel_done",
                    index=idx,
                    pos=n,
                    total=len(indices),
                    file=fn,
                    v=PAGE["panels"][idx]["v"],
                )
        finally:
            ws.close()
        job.emit("all_done", count=len(indices))
        us.tg(f"✅ Serg0 комикс: готово {len(indices)} кадр(ов).")
        with PAGE_LOCK:
            imgs = [
                os.path.join(WORK, PAGE["panels"][i]["image"])
                for i in indices
                if i < len(PAGE["panels"]) and PAGE["panels"][i].get("image")
            ]
        us.tg_album(imgs, f"Serg0 комикс: {len(imgs)} кадр(ов)")
    except Exception as exc:
        job.emit("error", error=str(exc)[:300])
        us.tg(f"❌ Serg0 комикс: ошибка — {str(exc)[:160]}")
    finally:
        job.done()
        ACTIVE = None


def start_job(indices, mode):
    global ACTIVE
    if ACTIVE and not ACTIVE.finished:
        return None
    job = Job()
    ACTIVE = job
    threading.Thread(target=render_worker, args=(job, indices, mode), daemon=True).start()
    return job


# ------------------------------------------------------------------------------- handlers
def save_ref_b64(data_url, tag):
    if not data_url or "," not in data_url:
        return None
    raw = base64.b64decode(data_url.split(",", 1)[1])
    name = f"comicref_{tag}_{uuid.uuid4().hex[:8]}.png"
    os.makedirs(COMFY_INPUT, exist_ok=True)
    with open(os.path.join(COMFY_INPUT, name), "wb") as fh:
        fh.write(raw)
    return name


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE_HTML.encode(), "text/html; charset=utf-8")
        elif path == "/api/options":
            self._send(200, json.dumps(us.options_payload()).encode())
        elif path == "/api/page":
            with PAGE_LOCK:
                self._send(200, json.dumps(PAGE).encode())
        elif path.startswith("/api/progress/"):
            self._sse(path.rsplit("/", 1)[-1])
        elif path.startswith("/out/"):
            self._file(path[len("/out/") :])
        elif path == "/api/export-zip":
            self._export_zip()
        else:
            self._send(404, b'{"error":"not found"}')

    def _file(self, rel):
        rel = urllib.parse.unquote(rel)
        full = os.path.normpath(os.path.join(WORK, rel))
        if not full.startswith(WORK) or not os.path.isfile(full):
            self._send(404, b"nope", "text/plain")
            return
        ctype = "application/zip" if full.endswith(".zip") else "image/png"
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def _export_zip(self):
        z = os.path.join(WORK, "comic_panels.zip")
        with PAGE_LOCK:
            imgs = [p["image"] for p in PAGE["panels"] if p.get("image")]
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            for i, fn in enumerate(imgs):
                fp = os.path.join(WORK, fn)
                if os.path.isfile(fp):
                    zf.write(fp, f"panel_{i + 1:02d}.png")
        self._send(200, json.dumps({"zip": "comic_panels.zip", "count": len(imgs)}).encode())

    def _sse(self, job_id):
        job = ACTIVE if (ACTIVE and ACTIVE.id == job_id) else None
        if not job:
            self._send(404, b'{"error":"no job"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        idx = 0
        try:
            while True:
                with job.cond:
                    while idx >= len(job.events) and not job.finished:
                        job.cond.wait(timeout=15)
                    new = job.events[idx:]
                    idx = len(job.events)
                    fin = job.finished
                for ev in new:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
                if fin and idx >= len(job.events):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        ln = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(ln) or b"{}")
        except Exception:
            self._send(400, b'{"error":"bad json"}')
            return
        if path == "/api/page":
            with PAGE_LOCK:
                PAGE["settings"] = body.get("settings", PAGE["settings"])
                PAGE["panels"] = body.get("panels", PAGE["panels"])
                save_page()
            self._send(200, b'{"ok":true}')
        elif path == "/api/upload":
            name = save_ref_b64(body.get("data"), body.get("tag", "ref"))
            self._send(200, json.dumps({"name": name}).encode())
        elif path in ("/api/render", "/api/regen", "/api/redetail"):
            mode = {"/api/render": "render", "/api/regen": "regen", "/api/redetail": "redetail"}[
                path
            ]
            with PAGE_LOCK:
                npan = len(PAGE["panels"])
                if path == "/api/render":
                    only_empty = bool(body.get("only_empty"))
                    indices = [
                        i
                        for i in range(npan)
                        if (not only_empty or not PAGE["panels"][i].get("image"))
                    ]
                else:
                    indices = [int(body["index"])] if "index" in body else []
                indices = [i for i in indices if 0 <= i < npan]
            if not indices:
                self._send(200, json.dumps({"error": "nothing to render"}).encode())
                return
            job = start_job(indices, mode)
            if not job:
                self._send(
                    200,
                    json.dumps(
                        {"busy": True, "msg": "уже идёт рендер — дождись окончания"}
                    ).encode(),
                )
                return
            self._send(200, json.dumps({"job_id": job.id, "count": len(indices)}).encode())
        else:
            self._send(404, b'{"error":"not found"}')


PAGE_HTML = r"""<!doctype html><html lang=ru><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Serg0 — фабрика кадров комикса</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--bd:#2a2f3a;--fg:#e8eaed;--mut:#9aa0aa;--acc:#3b82f6;--ok:#10b981}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,Arial;background:var(--bg);color:var(--fg)}
.wrap{max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mut);font-size:12px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;margin-bottom:12px}
label{display:block;font-size:12px;color:var(--mut);margin:6px 0 3px}
input[type=text],textarea,select{width:100%;background:#0d0f14;color:var(--fg);border:1px solid var(--bd);border-radius:7px;padding:7px;font:inherit;font-size:13px}
textarea{min-height:48px;resize:vertical}
.row{display:flex;gap:10px;flex-wrap:wrap}.row>*{flex:1;min-width:130px}
button{background:#2a2f3a;color:var(--fg);border:1px solid var(--bd);border-radius:7px;padding:7px 12px;cursor:pointer;font-size:13px}
button.acc{background:var(--acc);border-color:var(--acc)} button.go{background:var(--ok);color:#04110b;border:0;font-weight:700;padding:10px 16px}
button:disabled{opacity:.5;cursor:wait}
.panels{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.panel{border:1px solid var(--bd);border-radius:9px;padding:10px;background:#0d0f14}
.panel h3{margin:0 0 6px;font-size:13px;color:var(--mut)}
.thumb{width:100%;aspect-ratio:3/4;object-fit:cover;border-radius:7px;background:#1a1a1a;border:1px solid var(--bd);display:block;margin-top:8px}
.mini{display:flex;gap:6px;margin-top:6px}.mini button{flex:1;padding:6px;font-size:12px}
.zones{display:flex;gap:8px;margin-top:6px}.zones>div{flex:1}
.zones small{color:var(--mut);font-size:11px}
.bar{height:10px;background:#0d0f14;border:1px solid var(--bd);border-radius:6px;overflow:hidden;margin:3px 0}
.bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .2s}
.busy{background:#3a2a0d;border:1px solid #6b4e16;color:#ffd479;padding:8px;border-radius:7px;font-size:12px;margin:6px 0}
.thumbs{display:flex;gap:4px;flex-wrap:wrap}.thumbs img{width:34px;height:34px;object-fit:cover;border-radius:4px;border:1px solid var(--bd)}
.hint{color:var(--mut);font-size:12px}
</style>
<div class=wrap>
<h1>Serg0 — фабрика кадров комикса</h1>
<div class=sub>char-LoRA(2400) — якорь личности · стиль · улучшайзер · ControlNet поза · IP-Adapter лицо · <span id=gpu></span></div>

<div class=card>
  <div class=row>
    <div><label>База</label><select id=base></select></div>
    <div><label>Стиль-LoRA</label><select id=style></select></div>
    <div><label>Сила стиля <span id=ssv>0.85</span></label><input type=range id=ss min=0 max=1.2 step=0.05 value=0.85></div>
    <div><label>Аспект по умолч.</label><select id=aspect><option value=portrait>портрет 832×1216</option><option value=square>квадрат 1024</option><option value=landscape>ландшафт 1216×832</option></select></div>
  </div>
  <label>Негатив</label><textarea id=neg></textarea>
  <div class=row style="align-items:center;margin-top:6px">
    <div style="flex:0 0 auto"><label style="display:inline">Улучшайзер</label> <input type=checkbox id=improve checked></div>
    <div style="flex:1"></div>
    <button class=acc onclick=addPanel()>+ Кадр</button>
    <button class=go id=renderAll onclick="render(false)">Рендерить всё</button>
    <button id=renderEmpty onclick="render(true)">Рендерить пустые</button>
  </div>
</div>

<div class=card id=progc style="display:none">
  <div id=busy class="busy" style="display:none"></div>
  <div class=row><b style="font-size:13px">Кадр <span id=pnum>—</span></b><span class=hint id=ovrt></span></div>
  <div class=bar><i id=ovr></i></div>
  <div class=bar><i id=stp></i></div>
</div>

<div class=panels id=panels></div>

<div class=card style="margin-top:12px">
  <div class=row style="align-items:center">
    <b style="flex:1">Экспорт</b>
    <button onclick=exportPage()>⬇ Страница PNG (грид)</button>
    <button onclick=exportZip()>⬇ Кадры (zip, полный размер)</button>
    <span class=hint>Леттеринг/баблы — руками в Clip Studio Paint.</span>
  </div>
</div>
</div>
<canvas id=pagecanvas style="display:none"></canvas>
<script>
const $=s=>document.querySelector(s), api=async(u,o)=>(await fetch(u,o)).json();
let OPT=null, P={settings:{},panels:[]}, REFMODES={};
const toDataURL=f=>new Promise(r=>{const fr=new FileReader();fr.onload=()=>r(fr.result);fr.readAsDataURL(f)});

async function init(){
  OPT=await api('/api/options'); REFMODES=OPT.ref_modes;
  $('#gpu').textContent=OPT.gpu.ok?('✅ '+OPT.gpu.reason):('⚠ '+OPT.gpu.reason);
  $('#base').innerHTML=OPT.bases.map(b=>`<option value="${b}">${b==='sdxl'?'Vanilla SDXL':'Illustrious'}</option>`).join('');
  $('#style').innerHTML=OPT.styles.map(s=>`<option>${s}</option>`).join('');
  P=await api('/api/page'); applySettingsToUI(); renderPanels();
}
function applySettingsToUI(){const s=P.settings;$('#base').value=s.base;$('#style').value=s.style_lora;$('#ss').value=s.style_strength;$('#ssv').textContent=s.style_strength;$('#aspect').value=s.aspect;$('#neg').value=s.negative;$('#improve').checked=s.improve;}
function gatherSettings(){P.settings={base:$('#base').value,style_lora:$('#style').value,style_strength:+$('#ss').value,aspect:$('#aspect').value,negative:$('#neg').value,improve:$('#improve').checked};}
$('#ss').oninput=e=>$('#ssv').textContent=e.target.value;
['base','style','aspect','neg','improve'].forEach(id=>$('#'+id).onchange=()=>{gatherSettings();sync();});

function addPanel(){P.panels.push({prompt:"",aspect:P.settings.aspect||'portrait',seed:0,lock_seed:false,cn_ref:null,ip_refs:[],image:null,v:0});renderPanels();sync();}
function delPanel(i){P.panels.splice(i,1);renderPanels();sync();}
async function sync(){await fetch('/api/page',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(P)});}

function renderPanels(){
  const el=$('#panels'); el.innerHTML='';
  P.panels.forEach((p,i)=>{
    const d=document.createElement('div'); d.className='panel';
    const cn = REFMODES.controlnet?`<div><small>Поза→ControlNet</small><input type=file accept=image/* onchange="upRef(${i},'cn',this)">${p.cn_ref?'<small>✓</small>':''}</div>`:'';
    const ip = REFMODES.ipadapter?`<div><small>Лицо→IP-Adapter</small><input type=file accept=image/* multiple onchange="upRef(${i},'ip',this)"><div class=thumbs id=ipt${i}></div></div>`:'';
    d.innerHTML=`<h3>Кадр ${i+1} <button style="float:right;padding:2px 8px" onclick="delPanel(${i})">✕</button></h3>
      <textarea placeholder="что в кадре: сцена, поза, эмоция, ракурс (front view / close-up …)" onchange="P.panels[${i}].prompt=this.value;sync()">${p.prompt||''}</textarea>
      <div class=row style="margin-top:6px">
        <div><label>Аспект</label><select onchange="P.panels[${i}].aspect=this.value;sync()">
          <option value=portrait ${p.aspect==='portrait'?'selected':''}>портрет</option>
          <option value=square ${p.aspect==='square'?'selected':''}>квадрат</option>
          <option value=landscape ${p.aspect==='landscape'?'selected':''}>ландшафт</option></select></div>
        <div><label>Сид (lock) <input type=checkbox ${p.lock_seed?'checked':''} onchange="P.panels[${i}].lock_seed=this.checked;sync()"></label>
          <input type=text value="${p.seed||0}" onchange="P.panels[${i}].seed=+this.value;sync()"></div>
      </div>
      <div class=zones>${cn}${ip}</div>
      <img class=thumb id=th${i} src="${p.image?('/out/'+p.image+'?v='+(p.v||0)):''}" ${p.image?'':'style=visibility:hidden'}>
      <div class=mini><button onclick="regen(${i})">⟳ Перегенерировать</button><button onclick="redetail(${i})" title="заново прогнать FaceDetailer по лицу/рукам">✦ Передетейлить</button></div>`;
    el.appendChild(d);
    if(REFMODES.ipadapter&&p.ip_refs&&p.ip_refs.length){$('#ipt'+i).innerHTML=p.ip_refs.map(()=>'<img src="" style="background:#333">').join('');}
  });
}
async function upRef(i,kind,inp){
  const files=[...inp.files]; if(!files.length)return;
  const names=[];
  for(const f of files){const du=await toDataURL(f);const r=await api('/api/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({data:du,tag:kind})});names.push(r.name);}
  if(kind==='cn'){P.panels[i].cn_ref=names[0];}else{P.panels[i].ip_refs=names;}
  sync(); renderPanels();
}

function listen(job,total){
  $('#progc').style.display='block';$('#ovr').style.width='0';$('#stp').style.width='0';$('#busy').style.display='none';
  const es=new EventSource('/api/progress/'+job);
  es.onmessage=ev=>{const m=JSON.parse(ev.data);
    if(m.kind==='busy'){$('#busy').style.display='block';$('#busy').textContent=`⏳ Комп занят: ${m.reason}. Жду, проверю через 5 мин (${Math.round(m.waited/60)} мин).`;}
    else if(m.kind==='ready'){$('#busy').style.display='none';}
    else if(m.kind==='panel_start'){$('#pnum').textContent=`${m.pos+1}/${m.total} (№${m.index+1})`;$('#stp').style.width='0';}
    else if(m.kind==='step'){const f=m.value/Math.max(1,m.max);$('#stp').style.width=(f*100)+'%';const o=(m.pos+f)/m.total;$('#ovr').style.width=(o*100)+'%';$('#ovrt').textContent=Math.round(o*100)+'%';}
    else if(m.kind==='panel_done'){const o=(m.pos+1)/m.total;$('#ovr').style.width=(o*100)+'%';$('#ovrt').textContent=Math.round(o*100)+'%';
      if(m.file){const im=$('#th'+m.index);if(im){im.src='/out/'+m.file+'?v='+(m.v||Date.now());im.style.visibility='visible';P.panels[m.index].image=m.file;P.panels[m.index].v=m.v;}}}
    else if(m.kind==='all_done'){$('#ovr').style.width='100%';$('#ovrt').textContent='готово';setBusy(false);es.close();}
    else if(m.kind==='error'){$('#ovrt').textContent='❌ '+m.error;setBusy(false);es.close();}
  };
  es.onerror=()=>{setBusy(false);};
}
function setBusy(b){['renderAll','renderEmpty'].forEach(id=>$('#'+id).disabled=b);}
async function render(onlyEmpty){gatherSettings();await sync();setBusy(true);
  const r=await api('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({only_empty:onlyEmpty})});
  if(r.busy){alert(r.msg);setBusy(false);return;} if(r.error){alert(r.error);setBusy(false);return;} listen(r.job_id,r.count);}
async function regen(i){gatherSettings();await sync();setBusy(true);
  const r=await api('/api/regen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:i})});
  if(r.busy){alert(r.msg);setBusy(false);return;} listen(r.job_id,1);}
async function redetail(i){if(!P.panels[i].image){alert('сначала отрендери кадр');return;}gatherSettings();await sync();setBusy(true);
  const r=await api('/api/redetail',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:i})});
  if(r.busy){alert(r.msg);setBusy(false);return;} listen(r.job_id,1);}

async function exportZip(){const r=await api('/api/export-zip');if(r.zip){const a=document.createElement('a');a.href='/out/'+r.zip;a.download='comic_panels.zip';a.click();}}
function exportPage(){
  const imgs=P.panels.filter(p=>p.image);if(!imgs.length){alert('нет отрендеренных кадров');return;}
  const cols=Math.min(2,imgs.length),rows=Math.ceil(imgs.length/cols),cw=900,gap=16;
  const ch=Math.ceil(cw/cols*4/3); // portrait-ish cells
  const cv=$('#pagecanvas');cv.width=cw;cv.height=rows*ch+(rows+1)*gap;const x=cv.getContext('2d');
  x.fillStyle='#fff';x.fillRect(0,0,cv.width,cv.height);let loaded=0;
  imgs.forEach((p,k)=>{const im=new Image();im.crossOrigin='anonymous';im.onload=()=>{
    const c=k%cols,r=Math.floor(k/cols),cwid=(cw-(cols+1)*gap)/cols;
    const px=gap+c*(cwid+gap),py=gap+r*(ch+gap);
    const s=Math.min(cwid/im.width,ch/im.height),dw=im.width*s,dh=im.height*s;
    x.drawImage(im,px+(cwid-dw)/2,py+(ch-dh)/2,dw,dh);
    if(++loaded===imgs.length){const a=document.createElement('a');a.href=cv.toDataURL('image/png');a.download='comic_page.png';a.click();}
  };im.src='/out/'+p.image+'?v='+(p.v||0);});
}
init();
</script>
</html>"""


def main():
    os.makedirs(WORK, exist_ok=True)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Serg0 comic factory -> http://localhost:{PORT}  (bind {HOST}:{PORT})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
