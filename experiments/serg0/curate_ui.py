#!/usr/bin/env python3
"""Curation UI for the iterative char-dataset loop — stdlib only (runs anywhere).

View generated frames from any browser (served via the RunPod HTTP proxy), accept
them into the dataset or reject them, then kick the pipe onward: request more
generations or a retrain on the accepted set. Decisions persist to a JSON; the
"pipe" buttons drop an intent file that the orchestrator (or the agent) acts on.

Run:  python curate_ui.py            # serves 0.0.0.0:8090, scans $CURATE_ROOT
Env:  CURATE_ROOT (default /workspace/out), CURATE_PORT (8090),
      CURATE_STATE (default /workspace/curate_decisions.json),
      CURATE_CMD   (default /workspace/ui_command.json)
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TARGET = 30  # optimal keeper count to aim for across rounds (goal 30-40)

ACCESS = os.environ.get("CURATE_ACCESS", "/workspace/ui_last_access")

ROOT = os.environ.get("CURATE_ROOT", "/workspace/out")
PORT = int(os.environ.get("CURATE_PORT", "8090"))
STATE = os.environ.get("CURATE_STATE", "/workspace/curate_decisions.json")
CMD = os.environ.get("CURATE_CMD", "/workspace/ui_command.json")
EXTS = (".png", ".jpg", ".jpeg", ".webp")

PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Serg0 curate</title>
<style>
 body{background:#14161b;color:#e6e6e6;font:14px system-ui,Segoe UI,Roboto;margin:0}
 header{position:sticky;top:0;background:#191c22;border-bottom:1px solid #2a2e37;padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 select,button{background:#262a33;color:#e6e6e6;border:1px solid #3a3f4b;border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer}
 button:hover{background:#30353f}
 .pill{padding:2px 8px;border-radius:10px;font-size:12px}
 .a{background:#1d3a23;color:#8fe39b}.r{background:#3a1d1d;color:#e39b9b}.p{background:#2a2e37;color:#aab}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;padding:14px}
 .card{background:#1c1f27;border:2px solid #2a2e37;border-radius:8px;overflow:hidden}
 .card.acc{border-color:#3a9b54}.card.rej{border-color:#9b3a3a;opacity:.55}
 .card img{width:100%;display:block;cursor:zoom-in}
 .row{display:flex}.row button{flex:1;border-radius:0;border:0;border-top:1px solid #2a2e37}
 .nm{padding:4px 8px;color:#8aa;font-size:11px;word-break:break-all}
 .ok{background:#1d3a23}.no{background:#3a1d1d}
 #act{margin-left:auto;display:flex;gap:8px}
</style></head><body>
<header>
 <b>Serg0 curate</b>
 <select id=grp onchange=load()></select>
 <span class=pill id=ca>✓ 0</span><span class=pill id=cr>✗ 0</span><span class=pill id=cp>· 0</span>
 <span id=act>
   <button onclick="pipe('generate')">💾 Сохранить + ещё круг</button>
   <button onclick="pipe('done')">✅ Хватит — собрать датасет</button>
 </span>
</header>
<div class=grid id=grid></div>
<script>
let DATA={images:[],groups:[]};
async function load(){
  const r=await fetch('/list');DATA=await r.json();
  const g=document.getElementById('grp');
  if(g.children.length===0){g.innerHTML='<option value="">все ('+DATA.images.length+')</option>'+DATA.groups.map(x=>`<option>${x}</option>`).join('');}
  const f=g.value;
  let a=0,rj=0,p=0;const grid=document.getElementById('grid');grid.innerHTML='';
  for(const it of DATA.images){
    if(f && !it.path.startsWith(f+'/'))continue;
    if(it.d==='accept')a++;else if(it.d==='reject')rj++;else p++;
    const c=document.createElement('div');
    c.className='card'+(it.d==='accept'?' acc':it.d==='reject'?' rej':'');
    const u='/img?p='+encodeURIComponent(it.path);
    c.innerHTML=`<a href="${u}" target=_blank><img loading=lazy src="${u}"></a>
      <div class=row><button class="${it.d==='accept'?'ok':''}" onclick="mark('${it.path}','accept')">✓ взять</button>
      <button class="${it.d==='reject'?'no':''}" onclick="mark('${it.path}','reject')">✗ брак</button></div>
      <div class=nm>${it.path}</div>`;
    grid.appendChild(c);
  }
  document.getElementById('ca').textContent='✓ '+a;
  document.getElementById('cr').textContent='✗ '+rj;
  document.getElementById('cp').textContent='· '+p;
}
async function mark(p,d){await fetch('/mark?p='+encodeURIComponent(p)+'&d='+d,{method:'POST'});
  const it=DATA.images.find(x=>x.path===p);if(it)it.d=(it.d===d?'':d);load();}
async function pipe(action){if(action==='done'&&!confirm('Собрать финальный датасет из всех отобранных?'))return;
  const r=await fetch('/pipe?action='+action,{method:'POST'});const j=await r.json();
  alert(action==='done'?('Датасет: '+j.accepted+' кадров — собираю на поде, итог в телегу.')
    :('Всего отобрано: '+j.accepted+'. Запускаю следующий круг — ссылка придёт в телегу.'));}
load();
</script></body></html>"""


def scan():
    out = []
    for dp, _, fs in os.walk(ROOT):
        for f in sorted(fs):
            if f.lower().endswith(EXTS):
                out.append(os.path.relpath(os.path.join(dp, f), ROOT))
    return sorted(out)


def groups(imgs):
    return sorted({p.split("/")[0] for p in imgs if "/" in p})


def latest_round():
    """Highest generation-round index N for which an r{N}_sdxl dir exists (else 0)."""
    rs = []
    try:
        for d in os.listdir(ROOT):
            m = re.match(r"r(\d+)_sdxl$", d)
            if m and os.path.isdir(os.path.join(ROOT, d)):
                rs.append(int(m.group(1)))
    except OSError:
        pass
    return max(rs) if rs else 0


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


def save_state(s):
    json.dump(s, open(STATE, "w"), indent=0)


class H(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _touch(self):
        try:
            with open(ACCESS, "w") as fh:
                fh.write(str(int(time.time())))
        except OSError:
            pass

    def do_GET(self):
        self._touch()
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            return self._send(200, "text/html; charset=utf-8", PAGE.encode())
        if u.path == "/list":
            st = load_state()
            imgs = [{"path": p, "d": st.get(p, "")} for p in scan()]
            return self._send(
                200,
                "application/json",
                json.dumps({"images": imgs, "groups": groups([i["path"] for i in imgs])}).encode(),
            )
        if u.path == "/img":
            rel = q.get("p", [""])[0]
            full = os.path.normpath(os.path.join(ROOT, rel))
            if not full.startswith(os.path.realpath(ROOT)) and not full.startswith(ROOT):
                return self._send(403, "text/plain", b"no")
            try:
                with open(full, "rb") as fh:
                    return self._send(200, "image/png", fh.read())
            except OSError:
                return self._send(404, "text/plain", b"not found")
        if u.path == "/compare":
            # default to the latest generation round; ?a/?b override for older rounds
            rn = latest_round()
            a = q.get("a", [f"r{rn}_sdxl"])[0]
            b = q.get("b", [f"r{rn}_illustrious"])[0]
            try:
                scenes = json.load(open(os.path.join(ROOT, f"r{rn}_scenes.json")))
            except (OSError, ValueError):
                scenes = []
            try:
                names = sorted(
                    f for f in os.listdir(os.path.join(ROOT, a)) if f.lower().endswith(".png")
                )
            except OSError:
                names = []
            st = load_state()
            keepers = sum(1 for v in st.values() if v == "accept")
            rows = []
            for i, n in enumerate(names):
                cap = scenes[i] if i < len(scenes) else n
                rows.append(
                    f"<div class=cap>{i}. {cap}</div><div class=pair>"
                    f'<div class=fig data-p="{a}/{n}" onclick="toggle(this)">'
                    f'<a class=open target=_blank onclick="event.stopPropagation()" href="/img?p={a}/{n}">↗</a>'
                    f'<img loading=lazy src="/img?p={a}/{n}"></div>'
                    f'<div class=fig data-p="{b}/{n}" onclick="toggle(this)">'
                    f'<a class=open target=_blank onclick="event.stopPropagation()" href="/img?p={b}/{n}">↗</a>'
                    f'<img loading=lazy src="/img?p={b}/{n}"></div></div>'
                )
            css = (
                "body{background:#14161b;color:#e6e6e6;font:14px system-ui;margin:0}"
                ".top{position:sticky;top:0;z-index:3;background:#191c22;border-bottom:1px solid #2a2e37}"
                ".bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 14px}"
                ".bar b{font-size:15px}.cnt{color:#9aa}.tot{margin-left:auto;color:#bff5c6;font-weight:700}"
                ".go,.done{border-radius:8px;padding:8px 14px;font-weight:700;cursor:pointer;font-size:14px}"
                ".go{background:#1d3a23;color:#bff5c6;border:1px solid #3a9b54}"
                ".done{background:#26303a;color:#bfe0ff;border:1px solid #4a78a8}"
                ".cols{display:grid;grid-template-columns:1fr 1fr}.cols div{padding:6px;text-align:center;font-weight:700}"
                ".l{color:#8cf}.r{color:#fc8}.cap{padding:10px 12px 4px;color:#9aa}"
                ".pair{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:0 8px 12px}"
                ".fig{position:relative;cursor:pointer}"
                ".fig img{width:100%;display:block;border-radius:8px;border:2px solid #2a2e37}"
                ".fig.picked img{border-color:#3a9b54;box-shadow:0 0 0 2px #3a9b54 inset}"
                ".fig.picked::after{content:'\\2713';position:absolute;top:8px;left:8px;background:#3a9b54;color:#fff;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700}"
                ".open{position:absolute;top:8px;right:8px;background:#000a;color:#fff;border-radius:6px;padding:1px 7px;text-decoration:none}"
            )
            js = (
                "const picked=new Set();"
                "function render(){document.querySelectorAll('.fig').forEach(f=>f.classList.toggle('picked',picked.has(f.dataset.p)));"
                "document.getElementById('cnt').textContent='Выбрано в этом круге: '+picked.size;}"
                "function toggle(el){const p=el.dataset.p;picked.has(p)?picked.delete(p):picked.add(p);render();}"
                "async function init(){try{const d=await(await fetch('/list')).json();"
                "for(const it of d.images){if(it.d==='accept'&&(it.path.startsWith(A+'/')||it.path.startsWith(B+'/')))picked.add(it.path);}}catch(e){}render();}"
                "async function commit(action){"
                "if(action==='done'&&!confirm('Собрать финальный датасет из всех отобранных и закончить циклы?'))return;"
                "const j=await(await fetch('/commit',{method:'POST',headers:{'Content-Type':'application/json'},"
                "body:JSON.stringify({selected:[...picked],groups:[A,B],action})})).json();"
                "if(action==='done'){alert('Готово: '+j.accepted+' кадров в датасет. Сборка на поде — итог придёт в телегу.');}"
                "else{alert('Сохранено в круге: '+picked.size+' · всего отобрано: '+j.accepted+'/'+j.target+'. "
                "Запускаю следующий круг — ссылка придёт в телегу.');}}"
                "init();"
            )
            page = (
                "<!doctype html><meta charset=utf-8>"
                "<meta name=viewport content='width=device-width,initial-scale=1'>"
                f"<title>compare r{rn}</title><style>{css}</style>"
                "<div class=top><div class=bar>"
                f"<b>Круг {rn} — отметь верные кадры (клик)</b>"
                "<span id=cnt class=cnt>Выбрано в этом круге: 0</span>"
                f"<span class=tot>Отобрано всего: {keepers}/{TARGET}</span>"
                "<button class=go onclick=\"commit('generate')\">💾 Сохранить + ещё круг</button>"
                "<button class=done onclick=\"commit('done')\">✅ Хватит — собрать датасет</button>"
                f"</div><div class=cols><div class=l>{a}</div><div class=r>{b}</div></div></div>"
                + "".join(rows)
                + "<script>const A="
                + json.dumps(a)
                + ";const B="
                + json.dumps(b)
                + ";"
                + js
                + "</script>"
            )
            return self._send(200, "text/html; charset=utf-8", page.encode())
        return self._send(404, "text/plain", b"?")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/mark":
            p, d = q.get("p", [""])[0], q.get("d", [""])[0]
            st = load_state()
            if st.get(p) == d:
                st.pop(p, None)  # toggle off
            else:
                st[p] = d
            save_state(st)
            return self._send(200, "application/json", b'{"ok":1}')
        if u.path == "/pipe":
            action = q.get("action", ["generate"])[0]
            st = load_state()
            accepted = [p for p, d in st.items() if d == "accept"]
            json.dump(
                {"action": action, "accepted": accepted, "root": ROOT}, open(CMD, "w"), indent=2
            )
            return self._send(
                200,
                "application/json",
                json.dumps({"ok": 1, "accepted": len(accepted), "target": TARGET}).encode(),
            )
        if u.path == "/commit":
            # Compare-page: selected -> accept (keepers for the dataset), every other
            # frame in the compared groups -> reject. action=generate kicks the next
            # round; action=done assembles the dataset. Consumed once by pipe_worker.
            length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, OSError):
                data = {}
            selected = set(data.get("selected", []))
            grps = data.get("groups", [])
            action = data.get("action", "generate")
            universe = (
                [p for p in scan() if any(p.startswith(g + "/") for g in grps)]
                if grps
                else list(selected)
            )
            st = load_state()
            for p in universe:
                st[p] = "accept" if p in selected else "reject"
            save_state(st)
            accepted = [p for p, d in st.items() if d == "accept"]
            json.dump(
                {"action": action, "accepted": accepted, "root": ROOT}, open(CMD, "w"), indent=2
            )
            rejected = sum(1 for p in universe if p not in selected)
            return self._send(
                200,
                "application/json",
                json.dumps(
                    {"ok": 1, "accepted": len(accepted), "rejected": rejected, "target": TARGET}
                ).encode(),
            )
        return self._send(404, "text/plain", b"?")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    print(f"curate UI on 0.0.0.0:{PORT}  root={ROOT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
