package main

// controlPanelHTML is the entire self-contained control panel — one file, no
// external assets (CSP-free, works offline). It polls /api/status for the
// listener table + throughput and streams /api/events for the live request log
// and in-flight indicator, mirroring the terminal dashboard. Theme-aware.
const controlPanelHTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llama-dyn-proxy control panel</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --fg:#e6e9ef; --muted:#98a0b3;
    --line:#2a2f3a; --accent:#5b9dff; --ok:#3fb950; --off:#6e7681; --warn:#e3b341; --err:#f85149;
  }
  @media (prefers-color-scheme: light){
    :root{ --bg:#f6f7f9; --panel:#fff; --panel2:#f0f2f5; --fg:#1b1f27; --muted:#5b6370;
           --line:#e2e5ea; --accent:#2f6fed; --ok:#1a7f37; --off:#a0a6b0; --warn:#9a6700; --err:#cf222e; }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
  header h1{font-size:16px;margin:0;font-weight:650}
  header .sub{color:var(--muted);font-size:12px}
  main{max-width:1100px;margin:0 auto;padding:20px;display:grid;gap:20px;grid-template-columns:1fr 1fr}
  .full{grid-column:1/-1}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
  .card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 12px}
  table{width:100%;border-collapse:collapse}
  th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
  th{color:var(--muted);font-weight:600;font-size:12px}
  tr:last-child td{border-bottom:none}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
  .dot.on{background:var(--ok)} .dot.offd{background:var(--off)}
  .pill{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;border:1px solid var(--line);color:var(--muted)}
  .switch{position:relative;display:inline-block;width:40px;height:22px}
  .switch input{opacity:0;width:0;height:0}
  .slider{position:absolute;inset:0;background:var(--off);border-radius:999px;transition:.15s;cursor:pointer}
  .slider:before{content:"";position:absolute;height:16px;width:16px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.15s}
  input:checked+.slider{background:var(--accent)}
  input:checked+.slider:before{transform:translateX(18px)}
  input:disabled+.slider{opacity:.35;cursor:not-allowed}
  .btn{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:6px 11px;font-size:12.5px;cursor:pointer}
  .btn:hover{border-color:var(--accent)}
  .btn.active{background:var(--accent);border-color:var(--accent);color:#fff}
  .bucketrow{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
  #inflight{font-size:13px;color:var(--muted)}
  #inflight.active{color:var(--fg)}
  .bars{display:grid;grid-template-columns:auto 1fr auto;gap:5px 10px;align-items:center;font-size:12.5px}
  .bar{height:8px;background:var(--panel2);border-radius:999px;overflow:hidden}
  .bar>span{display:block;height:100%;background:var(--accent)}
  .log{max-height:280px;overflow:auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
  .log .row{padding:3px 0;border-bottom:1px solid var(--line);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .muted{color:var(--muted)} .err{color:var(--err)} .warn{color:var(--warn)}
  .tablewrap{max-height:380px;overflow:auto}
  .tablewrap table{width:100%}
  .tablewrap thead th{position:sticky;top:0;background:var(--panel)}
  th.sortable{cursor:pointer;user-select:none}
  th.sortable:hover{color:var(--fg)}
  th.sortable.sort-active{color:var(--accent)}
  .toolbar{display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
  select.btn{appearance:auto}
  @media (max-width:760px){ main{grid-template-columns:1fr} }
</style>
</head>
<body>
<header>
  <h1>llama-dyn-proxy</h1>
  <span class="sub" id="conn">connecting…</span>
  <span class="sub">· control panel</span>
</header>
<main>
  <section class="card full">
    <h2 id="listeners-toggle" style="cursor:pointer;user-select:none"><span id="listeners-caret">▾</span> Listeners</h2>
    <div id="listeners-wrap">
    <table id="listeners"><thead><tr>
      <th style="width:90px">Enabled</th><th>Name</th><th>Kind</th><th>Port</th>
      <th>Model</th><th>Pass-through sampling</th><th>Alert (local)</th><th></th>
    </tr></thead><tbody></tbody></table>
    </div>
  </section>

  <section class="card">
    <h2>Classification mode</h2>
    <div class="bucketrow" id="buckets"></div>
    <p class="muted" style="margin:10px 0 0;font-size:12px">Forcing a mode applies to every backend. "Auto-detect" restores per-request classification.</p>
    <h2 style="margin-top:18px">Vision describe</h2>
    <label class="switch"><input type="checkbox" id="vision-toggle" onchange="post('/api/vision?on='+this.checked)"><span class="slider"></span></label>
    <p class="muted" style="margin:10px 0 0;font-size:12px">Global: images in any request (every backend) are described once by the configured VLM and replaced with text before reaching the target model.</p>
  </section>

  <section class="card">
    <h2>In-flight</h2>
    <div id="inflight">idle</div>
    <div class="bars" id="bars" style="margin-top:12px"></div>
  </section>

  <section class="card full">
    <h2>Recent requests</h2>
    <div class="log" id="log"><div class="muted">waiting for traffic…</div></div>
  </section>

  <section class="card full">
    <h2>Throughput (all-time, per provider · model · mode)</h2>
    <div class="toolbar">
      <label class="muted" style="font-size:12px">Model:
        <select id="throughput-model-filter" class="btn"><option value="">All models</option></select>
      </label>
      <span class="muted" style="font-size:12px">Click a column header to sort.</span>
    </div>
    <div class="tablewrap">
    <table id="throughput"><thead><tr>
      <th class="sortable" data-key="provider"><span class="th-label">Provider</span><span class="th-arrow"></span></th>
      <th class="sortable" data-key="model"><span class="th-label">Model</span><span class="th-arrow"></span></th>
      <th class="sortable" data-key="bucket"><span class="th-label">Mode</span><span class="th-arrow"></span></th>
      <th class="sortable" data-key="samples"><span class="th-label">Samples</span><span class="th-arrow"></span></th>
      <th class="sortable" data-key="avg"><span class="th-label">Avg tok/s</span><span class="th-arrow"></span></th>
      <th class="sortable" data-key="sum_prompt_tokens"><span class="th-label">Prompt</span><span class="th-arrow"></span></th>
      <th class="sortable" data-key="sum_completion_tokens"><span class="th-label">Completion</span><span class="th-arrow"></span></th>
    </tr></thead><tbody></tbody></table>
    </div>
  </section>
</main>

<script>
const BUCKETS = ["strict_code","exploratory_code","explanation","architecture"];
const $ = s => document.querySelector(s);

async function post(url){ try{ await fetch(url,{method:"POST"}); }catch(e){} refresh(); }

function renderListeners(list){
  const tb = $("#listeners tbody"); tb.innerHTML = "";
  for(const l of list){
    const tr = document.createElement("tr");
    const enabled = "<label class=switch><input type=checkbox "+(l.running?"checked":"")+
      " onchange=\"post('/api/listener?name="+encodeURIComponent(l.name)+"&action='+(this.checked?'start':'stop'))\"><span class=slider></span></label>";
    let bypass = "<span class=muted>—</span>";
    if(l.supports_bypass){
      bypass = "<label class=switch><input type=checkbox "+(l.bypass_sampling?"checked":"")+(l.running?"":" ")+
        " onchange=\"post('/api/bypass?name="+encodeURIComponent(l.name)+"&on='+this.checked)\"><span class=slider></span></label>";
    }
    const err = l.last_err ? "<span class=err title='"+l.last_err.replace(/'/g,"")+"'>error</span>" : "";
    // Model cell: clinepass gets a live dropdown from the catalog; other
    // backends show their configured/effective model as text.
    let model;
    if(l.name==="clinepass"){
      window._clinepassModel = l.model || "";
      model = "<select id=model-clinepass class=btn onchange=\"post('/api/model?name=clinepass&model='+encodeURIComponent(this.value))\"><option>loading…</option></select>";
    } else {
      model = l.model ? "<code>"+l.model+"</code>" : "<span class=muted>—</span>";
    }
    // Alert-continuation is local-only (llama-server use case): switch on the
    // local row, "—" elsewhere.
    let alert = "<span class=muted>—</span>";
    if(l.name==="local"){
      alert = "<label class=switch><input type=checkbox "+(l.alert?"checked":"")+
        " onchange=\"post('/api/alert?name=local&on='+this.checked)\"><span class=slider></span></label>";
    }
    tr.innerHTML = "<td>"+enabled+"</td>"+
      "<td><span class='dot "+(l.running?"on":"offd")+"'></span><code>"+l.name+"</code></td>"+
      "<td><span class=pill>"+l.kind+"</span></td>"+
      "<td><code>"+l.port+"</code></td>"+
      "<td>"+model+"</td>"+
      "<td>"+bypass+"</td><td>"+alert+"</td><td>"+err+"</td>";
    tb.appendChild(tr);
  }
  populateClinepassModels();
}

// Populate the clinepass model dropdown from the catalog (grouped by billing).
let _catalog = null;
async function populateClinepassModels(){
  const sel = document.getElementById("model-clinepass");
  if(!sel) return;
  const current = (window._clinepassModel||"");
  if(!_catalog){
    try{ _catalog = (await (await fetch("/api/models")).json()).models || []; }
    catch(e){ sel.innerHTML = "<option>catalog unavailable</option>"; return; }
  }
  const groups = {};
  for(const m of _catalog){ (groups[m.group] = groups[m.group]||[]).push(m); }
  let html = "<option value=''>(config default)</option>";
  const labels = {subscription:"ClinePass subscription", "pay-as-you-go":"pay-as-you-go", free:"free"};
  for(const g of ["subscription","pay-as-you-go","free"]){
    if(!groups[g]) continue;
    html += "<optgroup label='"+(labels[g]||g)+"'>";
    for(const m of groups[g]){
      const seldd = m.id===current ? " selected" : "";
      html += "<option value='"+m.id+"'"+seldd+">"+m.id+"</option>";
    }
    html += "</optgroup>";
  }
  sel.innerHTML = html;
}

function renderBuckets(forced){
  const box = $("#buckets"); box.innerHTML = "";
  const mk = (label,val)=>{
    const b = document.createElement("button");
    b.className = "btn"+((forced||"")===val?" active":"");
    b.textContent = label;
    b.onclick = ()=>post("/api/bucket?bucket="+encodeURIComponent(val));
    return b;
  };
  box.appendChild(mk("auto-detect",""));
  for(const bk of BUCKETS) box.appendChild(mk(bk,bk));
}

// Throughput table state: raw rows from the last /api/status, the active
// sort (key/dir), and the active model filter. Sorting/filtering re-render
// from _throughputRows without a round-trip.
let _throughputRows = [];
let _throughputSort = {key:null, dir:1};
let _throughputModelFilter = "";

function throughputAvg(r){
  return r.sum_latency_ms>0 ? r.sum_completion_tokens/(r.sum_latency_ms/1000) : 0;
}

// Rebuilds the model-filter <select> options from the current rows. Skipped
// by the caller while the dropdown itself is focused/open, for the same
// reason the listeners table skips re-rendering mid-interaction — clobbering
// the open menu's options would close it and could commit the wrong option.
function populateThroughputModelFilter(rows){
  const sel = $("#throughput-model-filter");
  if(!sel) return;
  const models = Array.from(new Set(rows.map(r=>r.model).filter(Boolean))).sort();
  const current = sel.value;
  let html = "<option value=''>All models</option>";
  for(const m of models) html += "<option value='"+m+"'"+(m===current?" selected":"")+">"+m+"</option>";
  sel.innerHTML = html;
}

function renderThroughput(rows){
  _throughputRows = rows || [];
  const filterSel = $("#throughput-model-filter");
  if(document.activeElement !== filterSel){
    populateThroughputModelFilter(_throughputRows);
  }
  renderThroughputFiltered();
}

function renderThroughputFiltered(){
  const tb = $("#throughput tbody");
  let rows = _throughputRows;
  if(_throughputModelFilter) rows = rows.filter(r=>r.model===_throughputModelFilter);

  const {key, dir} = _throughputSort;
  if(key){
    rows = rows.slice().sort((a,b)=>{
      const av = key==="avg" ? throughputAvg(a) : a[key];
      const bv = key==="avg" ? throughputAvg(b) : b[key];
      if(typeof av === "string" || typeof bv === "string"){
        return dir*String(av||"").localeCompare(String(bv||""));
      }
      return dir*((av||0)-(bv||0));
    });
  }

  document.querySelectorAll("#throughput th.sortable").forEach(th=>{
    const arrowEl = th.querySelector(".th-arrow");
    if(th.dataset.key===key){ th.classList.add("sort-active"); arrowEl.textContent = dir===1?" ▲":" ▼"; }
    else{ th.classList.remove("sort-active"); arrowEl.textContent = ""; }
  });

  tb.innerHTML = "";
  if(!rows.length){ tb.innerHTML = "<tr><td colspan=7 class=muted>no data</td></tr>"; return; }
  for(const r of rows){
    const avg = throughputAvg(r);
    const tr = document.createElement("tr");
    tr.innerHTML = "<td><code>"+r.provider+"</code></td><td><code>"+(r.model||"—")+"</code></td>"+
      "<td>"+r.bucket+"</td><td>"+r.samples+"</td><td>"+(avg?avg.toFixed(1):"—")+"</td>"+
      "<td>"+r.sum_prompt_tokens+"</td><td>"+r.sum_completion_tokens+"</td>";
    tb.appendChild(tr);
  }
}

// Bound once at load: header clicks toggle sort (text fields default
// ascending, numeric fields default descending — highest-first is the more
// useful first look at samples/throughput), and the filter dropdown re-renders
// from the cached rows without a fetch.
function initThroughputControls(){
  const textKeys = new Set(["provider","model","bucket"]);
  document.querySelectorAll("#throughput th.sortable").forEach(th=>{
    th.addEventListener("click", ()=>{
      const key = th.dataset.key;
      if(_throughputSort.key===key){
        _throughputSort.dir *= -1;
      } else {
        _throughputSort.key = key;
        _throughputSort.dir = textKeys.has(key) ? 1 : -1;
      }
      renderThroughputFiltered();
    });
  });
  const filterSel = $("#throughput-model-filter");
  if(filterSel){
    filterSel.addEventListener("change", (e)=>{
      _throughputModelFilter = e.target.value;
      renderThroughputFiltered();
    });
  }
}

async function refresh(){
  try{
    const s = await (await fetch("/api/status")).json();
    // Don't rebuild the listeners table while the user has a dropdown open
    // (activeElement is the SELECT) — re-rendering would destroy the open
    // menu and commit whatever option is under the cursor. Skip just the
    // table this cycle; the rest still updates.
    const ae = document.activeElement;
    if(!(ae && ae.tagName === "SELECT")){
      renderListeners(s.listeners||[]);
    }
    renderBuckets(s.forced_bucket);
    renderThroughput(s.throughput||[]);
    const vt = $("#vision-toggle");
    if(vt && document.activeElement !== vt) vt.checked = !!s.vision_describe;
  }catch(e){}
}

// Collapse/expand the listeners section (persisted).
function initListenersToggle(){
  const t = $("#listeners-toggle"), wrap = $("#listeners-wrap"), caret = $("#listeners-caret");
  const apply = (collapsed)=>{ wrap.style.display = collapsed?"none":""; caret.textContent = collapsed?"▸":"▾"; };
  let collapsed = localStorage.getItem("listenersCollapsed")==="1";
  apply(collapsed);
  t.onclick = ()=>{ collapsed=!collapsed; localStorage.setItem("listenersCollapsed", collapsed?"1":"0"); apply(collapsed); };
}

// live log + in-flight via SSE
const logEl = $("#log"); let logInit=false;
function addLog(ev){
  if(!logInit){ logEl.innerHTML=""; logInit=true; }
  const t = new Date(ev.Timestamp||Date.now()).toLocaleTimeString();
  const host = ev.Host? " "+ev.Host : "";
  const model = ev.Model? " "+ev.Model : "";
  // retry= and issue= are always shown (matching the TUI's formatLogLine),
  // including retry=0 / issue=clean for an ordinary clean request — these
  // were previously only shown when non-zero/present, which silently hid
  // the common case instead of confirming "nothing went wrong".
  let msg;
  if(ev.Error){
    msg = "<span class=muted>retry="+(ev.RetryCount||0)+"</span> <span class=err>ERR "+ev.Error+"</span>"+model+host;
  } else {
    const issue = ev.Issue || "clean";
    const alert = ev.AlertRounds? " <span class=warn>⚠"+ev.AlertRounds+"</span>" : "";
    msg = "<span class=muted>"+ev.Bucket+"  retry="+(ev.RetryCount||0)+"  issue="+issue+"</span>"+model+host+
      " <span class=muted>"+(ev.LatencyMs||0)+"ms · "+(ev.CompletionTokens||0)+"tok</span>"+alert;
  }
  const row = document.createElement("div");
  row.className="row"; row.innerHTML = "<span class=muted>"+t+"</span> "+msg;
  logEl.prepend(row);
  while(logEl.childElementCount>60) logEl.removeChild(logEl.lastChild);
  refresh(); // update throughput after each completed request
}

function bar(label,val,max){
  const pct = Math.max(0,Math.min(100, (val/max)*100));
  return "<span class=muted>"+label+"</span><span class=bar><span style='width:"+pct+"%'></span></span><span>"+val+"</span>";
}
function showBars(ev){
  const b = $("#bars");
  b.innerHTML = bar("temp",(ev.Temperature||0).toFixed(2),2)+
    bar("top_p",(ev.TopP||0).toFixed(2),1)+
    bar("top_k",ev.TopK||0,100)+
    bar("budget",ev.ThinkingBudgetTokens||0,8192)+
    bar("rep_pen",(ev.RepeatPenalty||0).toFixed(2),2);
}

const inflight = $("#inflight");
function showInflight(p){
  if(p.Done){ inflight.textContent="idle"; inflight.className=""; return; }
  inflight.className="active";
  const label = p.Label || (p.Bucket+" · attempt "+(p.Attempt||0));
  const model = p.Model ? " ["+p.Model+"]" : "";
  const rate = p.GenerationElapsedMs>0 ? " · "+(p.ApproxTokens/(p.GenerationElapsedMs/1000)).toFixed(1)+" tok/s" : "";
  inflight.textContent = "▶ "+label+model+" · "+((p.ElapsedMs||0)/1000).toFixed(1)+"s · "+(p.ApproxTokens||0)+" tok"+rate;
}

function connect(){
  const es = new EventSource("/api/events");
  es.onopen = ()=>{ $("#conn").textContent="● live"; };
  es.onerror = ()=>{ $("#conn").textContent="○ reconnecting…"; };
  es.onmessage = (m)=>{
    let d; try{ d=JSON.parse(m.data); }catch(e){ return; }
    if(d.kind==="event"){ addLog(d.data); showBars(d.data); }
    else if(d.kind==="progress"){ showInflight(d.data); }
  };
}

initListenersToggle(); initThroughputControls(); refresh(); connect(); setInterval(refresh, 5000);
</script>
</body>
</html>`
