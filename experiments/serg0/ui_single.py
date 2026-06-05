#!/usr/bin/env python3
"""Serg0 single-generation UI (HLE-923) — local prod generation on the RTX 5070 Ti.

Dependency-free: stdlib HTTP server + a minimal ComfyUI websocket client for real
step-level progress. Talks to ComfyUI on :8188 over HTTP/WS, shells `nvidia-smi`/`free`
for the busy-gate, and `tg.sh` for Telegram. No pip installs -> no cu128 risk.

Spec (HLE-923), all in one page:
  - style LoRA (or none) + strength
  - base model (vanilla SDXL / Illustrious)
  - prompt + negative prompt
  - THREE labelled reference zones, each with a clear "что зачем":
      • IP-Adapter — похожесть/стиль по рефам (несколько)
      • ControlNet — поза/композиция (один реф)
      • img2img    — перерисовать вход (один реф) + сила denoise
  - improver toggle (FaceDetailer×2 → UltimateSDUpscale), default ON
  - count (default 5)
  - on Generate: Telegram(click) → busy-gate (VRAM/RAM; else «комп занят» + retick 5 мин,
    показывается в UI) → live per-frame + per-step + overall progress → Telegram(start) →
    previews (click = full res) → Telegram(done)+zip → «Скачать все (zip)» / «Сгенерировать ещё».

The generation graph is reused from gen_improve.build_graph (txt2img + char-LoRA +
optional style LoRA + improver). img2img is layered on via a tiny, ID-stable graph edit
(core LoadImage+VAEEncode nodes). IP-Adapter / ControlNet zones light up once their custom
nodes are installed (detected live via /object_info).

Run:  python3 ui_single.py            # serves on http://localhost:8765
"""
# ruff: noqa: E501 - this module embeds a self-contained HTML/CSS/JS page template

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pod"))
import contextlib

import gen_improve as gi

COMFY_HOST, COMFY_PORT = "127.0.0.1", 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_INPUT = "/home/serg/ComfyUI/input"
OUT_ROOT = "/home/serg/serg0_gen_out"
TG = "/home/serg/tg.sh"
HOST, PORT = "0.0.0.0", 8765
# busy-gate: minimum free VRAM / available RAM (MiB) before we dare start a job
VRAM_NEED = {True: 9000, False: 6500}  # keyed by improver on/off
RAM_NEED = 2500
RETICK_SEC = 300  # «комп занят» recheck cadence
STYLE_NONE = "— без стиля —"


# ----------------------------------------------------------------------------- helpers
def _sh(cmd: list[str], timeout: int = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def nvidia_free_mb() -> int:
    out = _sh(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    try:
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def ram_avail_mb() -> int:
    out = _sh(["free", "-m"])
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 7:
                return int(parts[6])
    return 0


def busy_check(improve: bool) -> tuple[bool, str, int, int]:
    """Return (ok, reason, vram_free, ram_avail). ok=True means safe to start."""
    vram, ram = nvidia_free_mb(), ram_avail_mb()
    need_v = VRAM_NEED[improve]
    if vram < need_v:
        return False, f"GPU занят: свободно {vram} МиБ < нужно {need_v} МиБ", vram, ram
    if ram < RAM_NEED:
        return False, f"RAM занята: доступно {ram} МиБ < нужно {RAM_NEED} МиБ", vram, ram
    return True, f"свободно: GPU {vram} МиБ, RAM {ram} МиБ", vram, ram


def tg(msg: str) -> None:
    with contextlib.suppress(Exception):
        subprocess.run([TG, msg], capture_output=True, timeout=25)


def tg_doc(path: str, caption: str) -> None:
    try:
        if os.path.getsize(path) <= 49 * 1024 * 1024:
            subprocess.run([TG, "-doc", path, caption], capture_output=True, timeout=180)
        else:
            tg(caption + " (zip > 50 МБ — забери из UI)")
    except Exception:
        pass


def _tg_creds() -> tuple[str | None, str | None]:
    tok = chat = None
    with contextlib.suppress(Exception), open("/home/serg/.serg0_tg.env") as fh:
        for ln in fh:
            if ln.startswith("TG_TOKEN="):
                tok = ln.strip().split("=", 1)[1]
            elif ln.startswith("TG_CHAT="):
                chat = ln.strip().split("=", 1)[1]
    return tok, chat


def tg_album(paths: list[str], caption: str = "") -> None:
    """Send results to Telegram as inline PHOTOS (sendMediaGroup, ≤10/group), not a zip."""
    paths = [p for p in paths if os.path.exists(p)]
    tok, chat = _tg_creds()
    if not paths or not tok or not chat:
        return
    for i in range(0, len(paths), 10):
        chunk = paths[i : i + 10]
        media, fargs = [], []
        for j, p in enumerate(chunk):
            key = f"photo{j}"
            item = {"type": "photo", "media": f"attach://{key}"}
            if i == 0 and j == 0 and caption:
                item["caption"] = caption
            media.append(item)
            fargs += ["-F", f"{key}=@{p}"]
        cmd = [
            "curl",
            "-s",
            "-m",
            "200",
            f"https://api.telegram.org/bot{tok}/sendMediaGroup",
            "-F",
            f"chat_id={chat}",
            "-F",
            "media=" + json.dumps(media, ensure_ascii=False),
            *fargs,
        ]
        with contextlib.suppress(Exception):
            subprocess.run(cmd, capture_output=True, timeout=210)


def comfy_get(path: str, timeout: int = 15):
    return json.loads(urllib.request.urlopen(COMFY_URL + path, timeout=timeout).read())


def node_classes() -> set[str]:
    try:
        return set(comfy_get("/object_info").keys())
    except Exception:
        return set()


# Which ref modes the BACKEND actually wires into the graph today. img2img is live (core
# nodes); IP-Adapter / ControlNet flip to True once their nodes are installed AND wired
# (post-training, task HLE-923/#21). Node presence alone is not enough — we AND the two.
BACKEND_REF = {"ipadapter": True, "controlnet": True, "img2img": True}


def available_ref_modes() -> dict[str, bool]:
    cls = node_classes()
    present = {
        "ipadapter": "IPAdapterAdvanced" in cls or "IPAdapterApply" in cls,
        "controlnet": "ControlNetApplyAdvanced" in cls or "ControlNetApply" in cls,
        "img2img": "VAEEncode" in cls,  # core — always present
    }
    return {k: present[k] and BACKEND_REF[k] for k in present}


# ------------------------------------------------------- minimal ComfyUI websocket client
class ComfyWS:
    """Just enough WebSocket to read ComfyUI's JSON progress frames (text, server->client)."""

    def __init__(self, host: str, port: int, client_id: str, timeout: int = 600):
        self.client_id = client_id
        self.sock = socket.create_connection((host, port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET /ws?clientId={client_id} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("ws handshake closed")
            buf += chunk
        self._buf = buf.split(b"\r\n\r\n", 1)[1]
        self.sock.settimeout(timeout)

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("ws closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv(self) -> tuple[int, bytes]:
        b0b1 = self._read(2)
        opcode = b0b1[0] & 0x0F
        masked = b0b1[1] & 0x80
        ln = b0b1[1] & 0x7F
        if ln == 126:
            ln = struct.unpack(">H", self._read(2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", self._read(8))[0]
        mask = self._read(4) if masked else b""
        data = self._read(ln) if ln else b""
        if masked and data:
            data = bytes(d ^ mask[i % 4] for i, d in enumerate(data))
        return opcode, data

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.sock.close()


def submit(graph: dict, client_id: str) -> str:
    body = json.dumps({"prompt": graph, "client_id": client_id}).encode()
    req = urllib.request.Request(
        COMFY_URL + "/prompt", data=body, headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]


def history_image(pid: str) -> bytes:
    e = comfy_get(f"/history/{pid}").get(pid, {})
    for n in e.get("outputs", {}).values():
        for im in n.get("images", []):
            if im.get("type") == "temp":
                continue
            return urllib.request.urlopen(
                COMFY_URL + "/view?" + urllib.parse.urlencode(im), timeout=60
            ).read()
    # fall back to any image
    for n in e.get("outputs", {}).values():
        for im in n.get("images", []):
            return urllib.request.urlopen(
                COMFY_URL + "/view?" + urllib.parse.urlencode(im), timeout=60
            ).read()
    raise RuntimeError("no image in history")


def apply_img2img(graph: dict, img_name: str, denoise: float) -> dict:
    """ID-stable edit: feed an input image (LoadImage→VAEEncode) into KSampler '6' at denoise<1."""
    graph["50"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
    graph["51"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["50", 0], "vae": ["1", 2]}}
    graph["6"]["inputs"]["latent_image"] = ["51", 0]
    graph["6"]["inputs"]["denoise"] = float(denoise)
    return graph


# ControlNet model filename that install_refs.sh downloads (used via the CORE Canny node).
CN_MODEL = "controlnet-canny-sdxl-1.0.safetensors"


def apply_controlnet(graph: dict, img_name: str, strength: float, cn_model: str = CN_MODEL) -> dict:
    """Condition KSampler '6' on a reference's structure: ref → Canny → ControlNetApplyAdvanced.

    Schema-verified against ComfyUI core: ControlNetApplyAdvanced(positive, negative, control_net,
    image, strength, start_percent, end_percent, vae) → (positive, negative); '6' repoints onto them.
    """
    graph["60"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
    graph["61"] = {
        "class_type": "Canny",
        "inputs": {"image": ["60", 0], "low_threshold": 0.4, "high_threshold": 0.8},
    }
    graph["62"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cn_model}}
    graph["63"] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": ["3", 0],
            "negative": ["4", 0],
            "control_net": ["62", 0],
            "image": ["61", 0],
            "strength": float(strength),
            "start_percent": 0.0,
            "end_percent": 1.0,
            "vae": ["1", 2],
        },
    }
    graph["6"]["inputs"]["positive"] = ["63", 0]
    graph["6"]["inputs"]["negative"] = ["63", 1]
    return graph


# IP-Adapter model + CLIP-vision files that install_refs.sh / dl_models download.
IP_MODEL = "ip-adapter_sdxl_vit-h.safetensors"
IP_CLIP = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"


def apply_ipadapter(graph: dict, img_names: list[str], weight: float) -> dict:
    """Push reference appearance/style into the model via cubiq IPAdapterAdvanced.

    Schema from the node source: IPAdapterModelLoader(ipadapter_file)→IPADAPTER;
    IPAdapterAdvanced(model, ipadapter, image, weight, weight_type, combine_embeds, start_at,
    end_at, embeds_scaling, clip_vision)→MODEL. Multiple refs are stacked with core ImageBatch.
    The IP-Adapter sits AFTER the LoRA chain; every model consumer (KSampler + improver) is
    repointed onto its output so identity-from-refs flows through the whole graph.
    """
    last = graph["6"]["inputs"]["model"][0]  # end of the checkpoint→char→style LoRA chain
    # load refs, batch if more than one
    img_ids = []
    for k, name in enumerate(img_names):
        nid = f"ipImg{k}"
        graph[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        img_ids.append(nid)
    image_ref = [img_ids[0], 0]
    for k, nid in enumerate(img_ids[1:], 1):
        bid = f"ipB{k}"
        graph[bid] = {
            "class_type": "ImageBatch",
            "inputs": {"image1": image_ref, "image2": [nid, 0]},
        }
        image_ref = [bid, 0]
    graph["ipL"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": IP_MODEL}}
    graph["ipC"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": IP_CLIP}}
    graph["ipA"] = {
        "class_type": "IPAdapterAdvanced",
        "inputs": {
            "model": [last, 0],
            "ipadapter": ["ipL", 0],
            "image": image_ref,
            "clip_vision": ["ipC", 0],
            "weight": float(weight),
            "weight_type": "linear",
            "combine_embeds": "concat",
            "start_at": 0.0,
            "end_at": 1.0,
            "embeds_scaling": "V only",
        },
    }
    # repoint every OTHER model consumer ([last,0]) onto the IP-Adapter output
    for nid, node in graph.items():
        if nid == "ipA":
            continue
        if node.get("inputs", {}).get("model") == [last, 0]:
            node["inputs"]["model"] = ["ipA", 0]
    return graph


# ----------------------------------------------------------------------------------- jobs
class Job:
    def __init__(self, job_id: str, params: dict):
        self.id = job_id
        self.params = params
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.finished = False
        self.images: list[str] = []
        self.zip: str | None = None
        self.outdir = os.path.join(OUT_ROOT, job_id)
        os.makedirs(self.outdir, exist_ok=True)

    def emit(self, kind: str, **data) -> None:
        with self.cond:
            self.events.append({"kind": kind, "t": time.strftime("%H:%M:%S"), **data})
            self.cond.notify_all()

    def done(self) -> None:
        with self.cond:
            self.finished = True
            self.cond.notify_all()


JOBS: dict[str, Job] = {}


def _save_ref(job: Job, data_url: str, tag: str) -> str | None:
    """Decode a base64 data-URL, write into ComfyUI/input, return the filename."""
    if not data_url or "," not in data_url:
        return None
    raw = base64.b64decode(data_url.split(",", 1)[1])
    name = f"serg0ui_{job.id}_{tag}.png"
    os.makedirs(COMFY_INPUT, exist_ok=True)
    with open(os.path.join(COMFY_INPUT, name), "wb") as fh:
        fh.write(raw)
    return name


def worker(job: Job) -> None:
    p = job.params
    improve = bool(p.get("improve", True))
    count = max(1, min(int(p.get("count", 5)), 20))
    base = p.get("base", "sdxl")
    base = base if base in gi.BASES else "sdxl"
    try:
        # ---- busy-gate: wait, reticking every RETICK_SEC, surfacing status in the UI ----
        waited = 0
        while True:
            ok, reason, vram, ram = busy_check(improve)
            if ok:
                job.emit("ready", reason=reason, vram=vram, ram=ram)
                break
            job.emit("busy", reason=reason, vram=vram, ram=ram, waited=waited, retick=RETICK_SEC)
            if waited == 0:
                tg(f"⏳ Serg0 UI: комп занят ({reason}). Жду и проверяю каждые 5 мин.")
            time.sleep(RETICK_SEC)
            waited += RETICK_SEC

        char_lora = gi.resolve_char_lora(COMFY_URL, base)
        style = p.get("style_lora") or None
        if style == STYLE_NONE:
            style = None
        neg = p.get("negative") or gi.DEFAULT_NEG
        modes = available_ref_modes()
        img2img_name = _save_ref(job, p.get("img2img"), "i2i")
        cn_name = _save_ref(job, p.get("controlnet"), "cn") if modes["controlnet"] else None
        ip_names: list[str] = []
        if modes["ipadapter"]:
            for k, du in enumerate(p.get("ipadapter") or []):
                nm = _save_ref(job, du, f"ip{k}")
                if nm:
                    ip_names.append(nm)
        seed0 = int(time.time()) & 0x7FFFFFF

        tg(
            f"🟢 Serg0 UI: старт генерации — {count} шт, база {base}, улучшайзер {'ON' if improve else 'OFF'}."
        )
        job.emit("start_all", count=count, base=base, char_lora=char_lora, style=style)

        cid = uuid.uuid4().hex
        ws = ComfyWS(COMFY_HOST, COMFY_PORT, cid)
        try:
            for i in range(count):
                seed = seed0 + i * 3
                graph = gi.build_graph(
                    base,
                    char_lora,
                    p.get("prompt", ""),
                    neg,
                    seed,
                    improve=improve,
                    style_lora=style,
                    style_strength=float(p.get("style_strength", 0.85)),
                )
                if ip_names:
                    graph = apply_ipadapter(graph, ip_names, float(p.get("ip_weight", 0.8)))
                if cn_name:
                    graph = apply_controlnet(graph, cn_name, float(p.get("cn_strength", 0.8)))
                if img2img_name:
                    graph = apply_img2img(graph, img2img_name, float(p.get("img2img_denoise", 0.6)))
                pid = submit(graph, cid)
                job.emit("frame_start", index=i, total=count, seed=seed)
                node = None
                while True:
                    op, raw = ws.recv()
                    if op == 0x8:
                        raise ConnectionError("ws closed mid-gen")
                    if op != 0x1:  # skip binary previews / ping
                        continue
                    msg = json.loads(raw)
                    t, d = msg.get("type"), msg.get("data", {})
                    if t == "progress" and d.get("prompt_id") == pid:
                        job.emit(
                            "step",
                            index=i,
                            total=count,
                            value=d.get("value", 0),
                            max=d.get("max", 1),
                            node=node,
                        )
                    elif t == "executing" and d.get("prompt_id") == pid:
                        if d.get("node") is None:
                            break  # this prompt finished
                        node = d.get("node")
                    elif t == "execution_error" and d.get("prompt_id") == pid:
                        raise RuntimeError(str(d.get("exception_message", "exec error"))[:200])
                png = history_image(pid)
                fn = f"{i:02d}.png"
                with open(os.path.join(job.outdir, fn), "wb") as fh:
                    fh.write(png)
                job.images.append(fn)
                job.emit("frame_done", index=i, total=count, file=fn)
        finally:
            ws.close()

        # ---- zip + finish ----
        z = os.path.join(job.outdir, "serg0_gen.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            for fn in job.images:
                zf.write(os.path.join(job.outdir, fn), fn)
        job.zip = "serg0_gen.zip"
        job.emit("all_done", count=len(job.images), zip=job.zip)
        tg(f"✅ Serg0 UI: готово {len(job.images)}/{count}.")
        # send results to Telegram as inline PHOTOS (not the zip); zip stays as a UI download
        tg_album(
            [os.path.join(job.outdir, fn) for fn in job.images], f"Serg0: {len(job.images)} шт"
        )
    except Exception as exc:  # surface, never crash the server thread
        job.emit("error", error=str(exc)[:300])
        tg(f"❌ Serg0 UI: ошибка генерации — {str(exc)[:160]}")
    finally:
        job.done()


# ------------------------------------------------------------------------------- HTTP/UI
def options_payload() -> dict:
    loras = gi.list_loras(COMFY_URL)
    styles = [STYLE_NONE, *sorted(x for x in loras if x.lower().startswith("cmc"))]
    chars = {b: gi.resolve_char_lora(COMFY_URL, b) for b in gi.BASES}
    ok_sdxl, reason, vram, ram = busy_check(True)
    return {
        "bases": list(gi.BASES),
        "styles": styles,
        "char_loras": chars,
        "ref_modes": available_ref_modes(),
        "default_negative": gi.DEFAULT_NEG,
        "gpu": {"ok": ok_sdxl, "reason": reason, "vram": vram, "ram": ram},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/options":
            self._send(200, json.dumps(options_payload()).encode())
        elif path.startswith("/api/progress/"):
            self._sse(path.rsplit("/", 1)[-1])
        elif path.startswith("/out/"):
            self._serve_file(path[len("/out/") :])
        else:
            self._send(404, b'{"error":"not found"}')

    def _serve_file(self, rel: str) -> None:
        rel = urllib.parse.unquote(rel)
        full = os.path.normpath(os.path.join(OUT_ROOT, rel))
        if not full.startswith(OUT_ROOT) or not os.path.isfile(full):
            self._send(404, b"nope", "text/plain")
            return
        ctype = "application/zip" if full.endswith(".zip") else "image/png"
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def _sse(self, job_id: str) -> None:
        job = JOBS.get(job_id)
        if not job:
            self._send(404, b'{"error":"no job"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        idx = 0
        try:
            while True:
                with job.cond:
                    while idx >= len(job.events) and not job.finished:
                        job.cond.wait(timeout=15)
                    new = job.events[idx:]
                    idx = len(job.events)
                    finished = job.finished
                for ev in new:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
                if finished and idx >= len(job.events):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path != "/api/generate":
            self._send(404, b'{"error":"not found"}')
            return
        ln = int(self.headers.get("Content-Length", 0))
        try:
            params = json.loads(self.rfile.read(ln) or b"{}")
        except Exception:
            self._send(400, b'{"error":"bad json"}')
            return
        tg("👆 Serg0 UI: нажата генерация.")
        job = Job(uuid.uuid4().hex[:12], params)
        JOBS[job.id] = job
        threading.Thread(target=worker, args=(job,), daemon=True).start()
        self._send(200, json.dumps({"job_id": job.id}).encode())


PAGE = r"""<!doctype html><html lang=ru><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Serg0 — одиночная генерация</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--bd:#2a2f3a;--fg:#e8eaed;--mut:#9aa0aa;--acc:#3b82f6;--ok:#10b981}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,Arial;background:var(--bg);color:var(--fg)}
.wrap{max-width:1080px;margin:0 auto;padding:18px}
h1{font-size:19px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px;margin-bottom:12px}
label{display:block;font-size:13px;color:var(--mut);margin:8px 0 4px}
input[type=text],textarea,select{width:100%;background:#0d0f14;color:var(--fg);border:1px solid var(--bd);border-radius:7px;padding:8px;font:inherit;font-size:14px}
textarea{min-height:64px;resize:vertical}
.row{display:flex;gap:12px;flex-wrap:wrap}.row>*{flex:1;min-width:160px}
.seg{display:inline-flex;border:1px solid var(--bd);border-radius:8px;overflow:hidden}
.seg button{background:#0d0f14;color:var(--fg);border:0;padding:8px 14px;cursor:pointer;font-size:14px}
.seg button.on{background:var(--acc)}
.refs{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.refz{border:1px dashed var(--bd);border-radius:9px;padding:10px;background:#0d0f14}
.refz h3{margin:0 0 2px;font-size:14px}.refz p{margin:0 0 8px;color:var(--mut);font-size:12px;min-height:30px}
.refz.dis{opacity:.45}.thumbs{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.thumbs img{width:54px;height:54px;object-fit:cover;border-radius:6px;border:1px solid var(--bd)}
.gen{background:var(--ok);color:#04110b;border:0;border-radius:9px;padding:12px 20px;font-size:16px;font-weight:700;cursor:pointer;width:100%}
.gen:disabled{opacity:.5;cursor:wait}
.bar{height:12px;background:#0d0f14;border:1px solid var(--bd);border-radius:7px;overflow:hidden;margin:4px 0}
.bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .2s}
.bar.ovr>i{background:var(--ok)}
.busy{background:#3a2a0d;border:1px solid #6b4e16;color:#ffd479;padding:9px 12px;border-radius:8px;font-size:13px}
.log{font-size:12px;color:var(--mut);white-space:pre-wrap;max-height:120px;overflow:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.grid a img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;border:1px solid var(--bd)}
.hidden{display:none}.flex{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.note{color:var(--mut);font-size:12px}.chk{display:flex;align-items:center;gap:8px}
small.r{color:var(--mut)}
</style>
<div class=wrap>
<h1>Serg0 — одиночная генерация</h1>
<div class=sub>Локально на 5070 Ti · char-LoRA + улучшайзер · <span id=gpu></span></div>

<div class=card>
  <label>Базовая модель</label>
  <div class=seg id=base></div>
  <div class=row>
    <div><label>Стиль-LoRA</label><select id=style></select></div>
    <div><label>Сила стиля <span id=ssv>0.85</span></label><input type=range id=ss min=0 max=1.2 step=0.05 value=0.85></div>
  </div>
  <label>Промт</label><textarea id=prompt placeholder="на крыше на закате, в полный рост, динамичная поза"></textarea>
  <label>Негатив</label><textarea id=neg></textarea>
</div>

<div class=card>
  <div class=flex style="justify-content:space-between">
    <b style="font-size:14px">Референсы (по желанию) — три разных способа</b>
    <span class=note>каждый со своим назначением ↓</span>
  </div>
  <div class=refs>
    <div class=refz id=z_ip>
      <h3>IP-Adapter</h3><p>Похожесть / стиль по рефам. Можно <b>несколько</b> — модель подтянет облик и палитру.</p>
      <input type=file id=f_ip accept=image/* multiple>
      <label>Сила <span id=ipv>0.8</span></label><input type=range id=ip_s min=0 max=1.5 step=0.05 value=0.8>
      <div class=thumbs id=t_ip></div>
    </div>
    <div class=refz id=z_cn>
      <h3>ControlNet</h3><p>Поза / композиция. <b>Один</b> реф — копируется только структура (скелет/контур).</p>
      <input type=file id=f_cn accept=image/*>
      <label>Сила <span id=cnv>0.8</span></label><input type=range id=cn_s min=0 max=1 step=0.05 value=0.8>
      <div class=thumbs id=t_cn></div>
    </div>
    <div class=refz id=z_i2i>
      <h3>img2img</h3><p>Перерисовать вход. <b>Один</b> реф — берётся как основа, denoise задаёт силу изменений.</p>
      <input type=file id=f_i2i accept=image/*>
      <label>Denoise <span id=i2v>0.60</span></label><input type=range id=i2_s min=0.2 max=0.95 step=0.05 value=0.6>
      <div class=thumbs id=t_i2i></div>
    </div>
  </div>
</div>

<div class=card>
  <div class=row>
    <div class=chk><input type=checkbox id=improve checked><label style="margin:0">Улучшайзер (лицо+руки+апскейл) — рекомендуется</label></div>
    <div><label>Сколько картинок</label><input type=text id=count value=5 style="max-width:90px"></div>
  </div>
  <button class=gen id=go>Сгенерировать</button>
</div>

<div class=card id=prog style="display:none">
  <div id=busy class="busy hidden"></div>
  <div class=flex><b style="font-size:13px">Общий прогресс</b><small class=r id=ovrt></small></div>
  <div class="bar ovr"><i id=ovr></i></div>
  <div class=flex><b style="font-size:13px">Текущий кадр</b><small class=r id=frmt></small></div>
  <div class=bar><i id=frm></i></div>
  <div class=log id=log></div>
</div>

<div class=card id=results style="display:none">
  <div class=flex style="justify-content:space-between">
    <b>Результат</b>
    <div class=flex><a id=dl class=note href=# download>⬇ Скачать все (zip)</a>
    <button id=again class=seg style="padding:6px 12px;cursor:pointer">Сгенерировать ещё</button></div>
  </div>
  <div class=grid id=grid></div>
</div>
</div>
<script>
const $=s=>document.querySelector(s), api=async(u,o)=>(await fetch(u,o)).json();
let OPT=null, BASE='sdxl', ipFiles=[], cnFile=null, i2iFile=null;
const toDataURL=f=>new Promise(r=>{const fr=new FileReader();fr.onload=()=>r(fr.result);fr.readAsDataURL(f)});
function thumbs(el,files){el.innerHTML='';files.forEach(f=>{const i=new Image();i.src=URL.createObjectURL(f);el.appendChild(i)})}

async function init(){
  OPT=await api('/api/options');
  $('#gpu').textContent=OPT.gpu.ok?('✅ '+OPT.gpu.reason):('⚠ '+OPT.gpu.reason);
  $('#base').innerHTML=OPT.bases.map(b=>`<button data-b="${b}">${b==='sdxl'?'Vanilla SDXL':'Illustrious'}</button>`).join('');
  document.querySelectorAll('#base button').forEach(b=>b.onclick=()=>{BASE=b.dataset.b;document.querySelectorAll('#base button').forEach(x=>x.classList.toggle('on',x===b))});
  document.querySelector('#base button').click();
  $('#style').innerHTML=OPT.styles.map(s=>`<option>${s}</option>`).join('');
  $('#neg').value=OPT.default_negative;
  // ref-mode availability
  if(!OPT.ref_modes.ipadapter){dis('#z_ip','IP-Adapter ноды ставятся после обучения')}
  if(!OPT.ref_modes.controlnet){dis('#z_cn','ControlNet ноды ставятся после обучения')}
}
function dis(sel,msg){const z=$(sel);z.classList.add('dis');z.querySelector('input[type=file]').disabled=true;z.querySelector('p').innerHTML+=`<br><b style="color:#ffd479">${msg}</b>`}

$('#ss').oninput=e=>$('#ssv').textContent=e.target.value;
$('#cn_s').oninput=e=>$('#cnv').textContent=e.target.value;
$('#i2_s').oninput=e=>$('#i2v').textContent=e.target.value;
$('#ip_s').oninput=e=>$('#ipv').textContent=e.target.value;
$('#f_ip').onchange=e=>{ipFiles=[...e.target.files];thumbs($('#t_ip'),ipFiles)};
$('#f_cn').onchange=e=>{cnFile=e.target.files[0];thumbs($('#t_cn'),cnFile?[cnFile]:[])};
$('#f_i2i').onchange=e=>{i2iFile=e.target.files[0];thumbs($('#t_i2i'),i2iFile?[i2iFile]:[])};
$('#again').onclick=()=>{$('#results').style.display='none';$('#prog').style.display='none';window.scrollTo(0,0)};

$('#go').onclick=async()=>{
  $('#go').disabled=true;$('#prog').style.display='block';$('#results').style.display='none';
  $('#log').textContent='';$('#ovr').style.width='0';$('#frm').style.width='0';$('#busy').classList.add('hidden');
  const body={base:BASE,style_lora:$('#style').value,style_strength:+$('#ss').value,
    prompt:$('#prompt').value,negative:$('#neg').value,improve:$('#improve').checked,count:+$('#count').value,
    img2img:i2iFile?await toDataURL(i2iFile):null,img2img_denoise:+$('#i2_s').value,
    controlnet:cnFile?await toDataURL(cnFile):null,cn_strength:+$('#cn_s').value,
    ipadapter:ipFiles.length?await Promise.all(ipFiles.map(toDataURL)):null,ip_weight:+$('#ip_s').value};
  const {job_id}=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  listen(job_id);
};

function logln(s){const l=$('#log');l.textContent+=s+'\n';l.scrollTop=l.scrollHeight}
function listen(job){
  const es=new EventSource('/api/progress/'+job);
  let total=+$('#count').value||5;
  es.onmessage=ev=>{
    const m=JSON.parse(ev.data);
    if(m.kind==='busy'){const b=$('#busy');b.classList.remove('hidden');
      b.textContent=`⏳ Комп занят: ${m.reason}. Жду, проверю снова через 5 мин (ждём ${Math.round(m.waited/60)} мин).`;logln('занят: '+m.reason)}
    else if(m.kind==='ready'){$('#busy').classList.add('hidden');logln('свободно — старт ('+m.reason+')')}
    else if(m.kind==='start_all'){total=m.count;logln(`старт: ${m.count} шт, база ${m.base}, lora ${m.char_lora}`+(m.style?(' +стиль '+m.style):''))}
    else if(m.kind==='frame_start'){$('#frm').style.width='0';$('#frmt').textContent=`кадр ${m.index+1}/${m.total}`}
    else if(m.kind==='step'){const f=m.value/Math.max(1,m.max);$('#frm').style.width=(f*100)+'%';
      const o=(m.index+f)/total;$('#ovr').style.width=(o*100)+'%';$('#ovrt').textContent=Math.round(o*100)+'%'+(m.node?(' · '+m.node):'')}
    else if(m.kind==='frame_done'){const o=(m.index+1)/total;$('#ovr').style.width=(o*100)+'%';$('#ovrt').textContent=Math.round(o*100)+'%';
      addImg(job,m.file);logln('готов кадр '+(m.index+1))}
    else if(m.kind==='all_done'){$('#ovr').style.width='100%';$('#ovrt').textContent='100%';$('#go').disabled=false;
      $('#dl').href='/out/'+job+'/'+m.zip;$('#results').style.display='block';logln('✅ готово '+m.count);es.close()}
    else if(m.kind==='error'){logln('❌ '+m.error);$('#go').disabled=false;es.close()}
  };
  es.onerror=()=>{logln('(поток прерван)');$('#go').disabled=false};
}
function addImg(job,file){$('#results').style.display='block';const g=$('#grid');
  const a=document.createElement('a');a.href='/out/'+job+'/'+file;a.target='_blank';
  const im=new Image();im.src='/out/'+job+'/'+file;a.appendChild(im);g.appendChild(a)}
$('#again').addEventListener('click',()=>{$('#grid').innerHTML=''});
init();
</script>
</html>"""


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Serg0 single-gen UI -> http://localhost:{PORT}  (bind {HOST}:{PORT})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
