"""Blind gold-set labelling tool. Run: .venv/bin/python gold/app.py  ->  http://localhost:8001

Labels append to gold/labels_<annotator>.jsonl (latest entry per dish wins). No model output is
ever sent to the page. ?set=overlap shows only the 20 overlap dishes (for the second annotator).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CANDIDATES = HERE / "gold_candidates.json"
DIMS_FILE = ROOT / "taste_dimensions.json"
DIMS = ["spice", "sweet", "sour", "bitter", "umami", "richness", "crunch", "creaminess"]
BLIND_FIELDS = ["uid", "name", "description", "cuisine", "restaurant", "ingredients", "overlap"]

app = FastAPI()


def safe_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9_-]", "", name.strip().lower().replace(" ", "_"))
    if not n:
        raise HTTPException(400, "annotator name required")
    return n[:40]


def labels_path(name: str) -> Path:
    return HERE / f"labels_{safe_name(name)}.jsonl"


def latest_labels(name: str) -> dict[str, dict]:
    p = labels_path(name)
    out: dict[str, dict] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["uid"]] = rec
    return out


class Label(BaseModel):
    annotator: str
    uid: str
    taste: dict[str, int] = {}
    skip: bool = False
    unsure: bool = False
    seconds: float = Field(default=0, ge=0)


@app.get("/api/dishes")
def dishes(set: str = "all"):
    if not CANDIDATES.exists():
        raise HTTPException(503, "gold_candidates.json not built yet: run gold/select_candidates.py")
    cands = json.loads(CANDIDATES.read_text())
    if set == "overlap":
        cands = [c for c in cands if c.get("overlap")]
    return [{k: c.get(k) for k in BLIND_FIELDS} for c in cands]


@app.get("/api/dims")
def dims():
    return json.loads(DIMS_FILE.read_text())


@app.get("/api/labels")
def get_labels(annotator: str):
    return latest_labels(annotator)


@app.post("/api/label")
def post_label(lab: Label):
    if not lab.skip:
        if set(lab.taste) != set(DIMS):
            raise HTTPException(400, "all 8 dims required")
        if any(not (0 <= v <= 10) for v in lab.taste.values()):
            raise HTTPException(400, "values must be 0-10")
    rec = lab.model_dump()
    rec["annotator"] = safe_name(lab.annotator)
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with labels_path(lab.annotator).open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return {"ok": True}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold Labeller</title>
<style>
:root{--bg:#fafaf7;--card:#fff;--ink:#1c1c1a;--mut:#6b6b66;--line:#e4e2dc;--acc:#c2410c;--ok:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#1f1f1c;--ink:#ecebe6;--mut:#9a998f;--line:#34332e;--acc:#fb923c;--ok:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:16px}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header input{padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
.bar{height:6px;background:var(--line);border-radius:3px;flex:1;min-width:120px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:14px}
h2{margin:0 0 4px;font-size:20px}.meta{color:var(--mut);font-size:13px}.desc{margin:8px 0}.ing{font-size:13px;color:var(--mut)}
.tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:9px;border:1px solid var(--line);margin-left:6px}
.dim{display:grid;grid-template-columns:96px 1fr 34px;gap:10px;align-items:center;padding:6px 4px;border-radius:6px}
.dim.focus{background:color-mix(in srgb,var(--acc) 10%,transparent)}
.dim label{font-weight:600}.dim output{font-variant-numeric:tabular-nums;font-weight:700;text-align:right}
input[type=range]{width:100%;accent-color:var(--acc)}
.anch{grid-column:2;position:relative;height:16px;font-size:11px;color:var(--mut);margin-top:-6px}
.anch span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.btns{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}button{padding:8px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
.keys{font-size:12px;color:var(--mut);margin-top:10px}kbd{border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:11px}
.untouched output{color:var(--mut)}
</style></head><body><div class="wrap">
<header><strong>Gold labeller</strong><input id="who" placeholder="your name"><span id="prog" class="meta"></span><div class="bar"><i id="pbar"></i></div></header>
<div id="main" class="card">Enter your name to start.</div>
<div class="keys"><kbd>0</kbd>-<kbd>9</kbd> set focused slider · <kbd>=</kbd> sets 10 · <kbd>←</kbd><kbd>→</kbd> ±1 · <kbd>Tab</kbd>/<kbd>↓</kbd> next slider · <kbd>↑</kbd> previous · <kbd>Enter</kbd> save + next · <kbd>U</kbd> unsure · <kbd>S</kbd> skip · <kbd>[</kbd> <kbd>]</kbd> prev/next dish</div>
</div>
<script>
const qs=new URLSearchParams(location.search), SET=qs.get('set')||'all';
let DISHES=[],DIMS=[],LABELS={},idx=0,focus=0,vals={},touched=new Set(),unsure=false,t0=Date.now();
const $=s=>document.querySelector(s), who=$('#who');
try{who.value=localStorage.getItem('gold_who')||''}catch(e){}
async function j(u,o){const r=await fetch(u,o);if(!r.ok)throw new Error(await r.text());return r.json()}
async function start(){
  const name=who.value.trim(); if(!name)return; try{localStorage.setItem('gold_who',name)}catch(e){}
  try{DISHES=await j('/api/dishes?set='+SET)}catch(e){$('#main').textContent='No candidates yet: '+e.message;return}
  DIMS=(await j('/api/dims')).dimensions; LABELS=await j('/api/labels?annotator='+encodeURIComponent(name));
  idx=DISHES.findIndex(d=>!LABELS[d.uid]); if(idx<0)idx=DISHES.length; render();
}
who.addEventListener('change',start);
function prog(){const n=DISHES.filter(d=>LABELS[d.uid]).length;$('#prog').textContent=`${n}/${DISHES.length} done`;$('#pbar').style.width=(100*n/Math.max(1,DISHES.length))+'%'}
function render(){
  prog(); t0=Date.now(); focus=0; touched=new Set(); unsure=false;
  if(idx>=DISHES.length){$('#main').innerHTML='<h2>All done 🎉</h2><p class="meta">Use [ to revisit dishes. Run gold/build_gold.py to merge.</p>';return}
  const d=DISHES[idx], prev=LABELS[d.uid];
  vals={}; DIMS.forEach(x=>vals[x.key]=prev&&prev.taste&&prev.taste[x.key]!=null?prev.taste[x.key]:0);
  if(prev&&!prev.skip)DIMS.forEach(x=>touched.add(x.key)); unsure=!!(prev&&prev.unsure);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  $('#main').innerHTML=`<div class="meta">#${idx+1} · ${esc(d.cuisine)} · ${esc(d.restaurant)}${d.overlap?'<span class="tag">overlap</span>':''}${prev?'<span class="tag">'+(prev.skip?'skipped':'labelled')+'</span>':''}</div>
  <h2>${esc(d.name)}</h2><div class="desc">${esc(d.description)||'<span class="meta">(no description)</span>'}</div>
  <div class="ing">${d.ingredients&&d.ingredients.length?'Ingredients: '+esc(d.ingredients.join(', ')):''}</div>
  <div id="dims" style="margin-top:12px">${DIMS.map((x,i)=>`<div class="dim" data-i="${i}"><label for="s_${x.key}">${x.key}</label>
   <input type="range" min="0" max="10" step="1" id="s_${x.key}" value="${vals[x.key]}"><output id="o_${x.key}">${vals[x.key]}</output>
   <div class="anch">${x.anchors.map(a=>`<span style="left:${a.score*10}%">${esc(a.ingredient)}</span>`).join('')}</div></div>`).join('')}</div>
  <div class="btns"><button id="prevb">[ Prev</button><label><input type="checkbox" id="uns" ${unsure?'checked':''}> unsure (U)</label>
   <button id="skipb">Skip (S)</button><button class="pri" id="saveb">Save + next (Enter)</button></div>`;
  DIMS.forEach((x,i)=>{const s=$('#s_'+x.key);s.addEventListener('input',()=>{set(x.key,+s.value)});s.addEventListener('focus',()=>{focus=i;hl()})});
  $('#uns').onchange=e=>unsure=e.target.checked; $('#saveb').onclick=()=>save(false); $('#skipb').onclick=()=>save(true); $('#prevb').onclick=()=>go(-1);
  hl(); $('#s_'+DIMS[0].key).focus();
}
function set(k,v){vals[k]=Math.max(0,Math.min(10,v));touched.add(k);$('#s_'+k).value=vals[k];$('#o_'+k).textContent=vals[k];hl()}
function hl(){document.querySelectorAll('.dim').forEach((el,i)=>{el.classList.toggle('focus',i===focus);el.classList.toggle('untouched',!touched.has(DIMS[i].key))})}
function go(dl){idx=Math.max(0,Math.min(DISHES.length,idx+dl));render()}
async function save(skip){
  const d=DISHES[idx]; if(!d)return;
  if(!skip&&touched.size<DIMS.length&&!confirm(`${DIMS.length-touched.size} slider(s) untouched (left at their value). Save anyway?`))return;
  const rec={annotator:who.value.trim(),uid:d.uid,taste:skip?{}:{...vals},skip,unsure,seconds:(Date.now()-t0)/1000};
  try{await j('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)})}catch(e){alert('Save failed: '+e.message);return}
  LABELS[d.uid]=rec; go(1);
}
document.addEventListener('keydown',e=>{
  if(document.activeElement===who||!DIMS.length)return;
  const k=DIMS[focus]&&DIMS[focus].key;
  if(/^[0-9]$/.test(e.key)){set(k,+e.key);e.preventDefault()}
  else if(e.key==='='||e.key==='+'){set(k,10);e.preventDefault()}
  else if(e.key==='ArrowRight'||e.key==='ArrowLeft'){set(k,vals[k]+(e.key==='ArrowRight'?1:-1));e.preventDefault()}
  else if(e.key==='ArrowDown'||(e.key==='Tab'&&!e.shiftKey)){focus=(focus+1)%DIMS.length;$('#s_'+DIMS[focus].key).focus();e.preventDefault()}
  else if(e.key==='ArrowUp'||(e.key==='Tab'&&e.shiftKey)){focus=(focus-1+DIMS.length)%DIMS.length;$('#s_'+DIMS[focus].key).focus();e.preventDefault()}
  else if(e.key==='Enter'){save(false);e.preventDefault()}
  else if(e.key==='s'||e.key==='S'){save(true)}
  else if(e.key==='u'||e.key==='U'){unsure=!unsure;const c=$('#uns');if(c)c.checked=unsure}
  else if(e.key==='['){go(-1)}else if(e.key===']'){go(1)}
});
if(who.value)start();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
