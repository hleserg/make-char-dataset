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
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
   <button onclick="pipe('regenerate')">↻ Сгенерить ещё</button>
   <button onclick="pipe('retrain')">⇪ На дообучение (принятые)</button>
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
async function pipe(action){const r=await fetch('/pipe?action='+action,{method:'POST'});const j=await r.json();
  alert(action+': записано ('+j.accepted+' принятых). Агент подхватит.');}
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
            action = q.get("action", [""])[0]
            st = load_state()
            accepted = [p for p, d in st.items() if d == "accept"]
            json.dump(
                {"action": action, "accepted": accepted, "root": ROOT}, open(CMD, "w"), indent=2
            )
            return self._send(
                200, "application/json", json.dumps({"ok": 1, "accepted": len(accepted)}).encode()
            )
        return self._send(404, "text/plain", b"?")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    print(f"curate UI on 0.0.0.0:{PORT}  root={ROOT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
