
const players = {
  a: { tag:"", imageName:"", file:null, width:0, height:0, quality:"—", analysis:null },
  b: { tag:"", imageName:"", file:null, width:0, height:0, quality:"—", analysis:null }
};

function normalizeTag(v){
  let s=(v||"").trim().toUpperCase().replace(/\s+/g,"");
  if(s && !s.startsWith("#")) s="#"+s;
  return s;
}

function buildingName(id){
  const hit = window.CLASHCOMPARE_BUILDINGS?.buildings?.find(b => b.id === id);
  return hit?.name || id;
}

function formatLevels(levels){
  return Object.entries(levels)
    .sort((a,b) => {
      if(a[0]==="inconnu") return 1;
      if(b[0]==="inconnu") return -1;
      return Number(b[0]) - Number(a[0]);
    })
    .map(([level, n]) => level==="inconnu" ? `${n} niv. inconnu` : `${n}× niv.${level}`)
    .join(", ");
}

function renderInventory(el, inventory){
  if(!inventory || !Object.keys(inventory).length){
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = Object.entries(inventory).map(([id, slot]) =>
    `<div class="inv-row"><b>${buildingName(id)}</b><span>${slot.total} — ${formatLevels(slot.levels)}</span></div>`
  ).join("");
}

async function pingEngine(){
  const el = document.querySelector("#engine-status");
  try{
    const r = await fetch("/api/health");
    if(!r.ok) throw new Error();
    const data = await r.json();
    if(el) el.textContent = `Moteur local prêt : ${data.engine} • imgsz ${data.imgsz}. Niveaux : air-defense / town-hall seulement (≥ 60 %).`;
    return true;
  }catch{
    if(el) el.textContent = "Moteur local absent. Lance : .\\.venv\\Scripts\\python.exe training\\scripts\\serve_compare.py";
    return false;
  }
}

function buildPlayerPanel(key, label){
  const tpl=document.querySelector("#player-template").content.cloneNode(true);
  const panel=document.querySelector("#panel-"+key);
  panel.appendChild(tpl);

  const title=panel.querySelector(".player-title");
  const tagInput=panel.querySelector(".tag-input");
  const imageInput=panel.querySelector(".image-input");
  const uploadBtn=panel.querySelector(".upload-btn");
  const preview=panel.querySelector(".preview");
  const analyzeBtn=panel.querySelector(".analyze-btn");
  const clearBtn=panel.querySelector(".clear-btn");
  const area=panel.querySelector(".analysis-area");
  const status=panel.querySelector(".status-pill");
  const notice=panel.querySelector(".r-notice");
  const inventoryEl=panel.querySelector(".r-inventory");
  title.textContent=label;

  uploadBtn.addEventListener("click",()=>imageInput.click());

  imageInput.addEventListener("change",()=>{
    const file=imageInput.files?.[0];
    if(!file) return;
    const url=URL.createObjectURL(file);
    const img=new Image();
    img.onload=()=>{
      players[key].file=file;
      players[key].imageName=file.name;
      players[key].width=img.naturalWidth;
      players[key].height=img.naturalHeight;
      players[key].analysis=null;
      const px=img.naturalWidth*img.naturalHeight;
      let q=px>=1800000 ? "Très bonne" : px>=900000 ? "Bonne" : "Faible";
      players[key].quality=q;
      preview.src=url;
      preview.hidden=false;
      status.textContent="Image ajoutée";
      renderCompare();
    };
    img.src=url;
  });

  analyzeBtn.addEventListener("click", async ()=>{
    const tag=normalizeTag(tagInput.value);
    players[key].tag=tag;
    tagInput.value=tag;

    if(!players[key].file){
      alert("Ajoute d'abord la capture du village.");
      return;
    }

    area.hidden=false;
    panel.querySelector(".r-tag").textContent=tag || "Non renseigné";
    panel.querySelector(".r-image").textContent=`${players[key].width} × ${players[key].height}`;
    panel.querySelector(".r-quality").textContent=players[key].quality;
    panel.querySelector(".r-count").textContent="…";
    notice.textContent="Analyse YOLO en cours (GTX 1050, quelques secondes)…";
    analyzeBtn.disabled=true;
    status.textContent="Analyse…";

    try{
      const body=new FormData();
      body.append("image", players[key].file, players[key].imageName);
      const response=await fetch("/api/analyze", { method:"POST", body });
      const data=await response.json();
      if(!response.ok) throw new Error(data.error || "analyse impossible");
      players[key].analysis=data;
      panel.querySelector(".r-count").textContent=String(data.count);
      notice.textContent=data.level_note || "Types détectés par YOLO V5. Niveaux seulement si classifieur fiable.";
      renderInventory(inventoryEl, data.inventory);
      status.textContent="Analysé";
    }catch(err){
      players[key].analysis=null;
      panel.querySelector(".r-count").textContent="—";
      notice.textContent = String(err.message||err) + " — lance le serveur local serve_compare.py si besoin.";
      inventoryEl.hidden=true;
      status.textContent="Erreur";
    }finally{
      analyzeBtn.disabled=false;
      renderCompare();
    }
  });

  clearBtn.addEventListener("click",()=>{
    players[key]={ tag:"", imageName:"", file:null, width:0, height:0, quality:"—", analysis:null };
    tagInput.value="";
    imageInput.value="";
    preview.removeAttribute("src");
    preview.hidden=true;
    area.hidden=true;
    inventoryEl.hidden=true;
    inventoryEl.innerHTML="";
    status.textContent="En attente";
    renderCompare();
  });
}

function countOf(analysis, id){
  return analysis?.inventory?.[id]?.total ?? 0;
}

function renderCompare(){
  const el=document.querySelector("#compare-content");
  const A=players.a, B=players.b;
  const rows=[
    ["# joueur", A.tag||"—", B.tag||"—"],
    ["Capture", A.imageName?"Ajoutée":"—", B.imageName?"Ajoutée":"—"],
    ["Résolution", A.width?`${A.width}×${A.height}`:"—", B.width?`${B.width}×${B.height}`:"—"],
    ["Qualité", A.quality, B.quality],
    ["Bâtiments détectés", A.analysis?String(A.analysis.count):"—", B.analysis?String(B.analysis.count):"—"]
  ];
  const ids=new Set([
    ...Object.keys(A.analysis?.inventory||{}),
    ...Object.keys(B.analysis?.inventory||{})
  ]);
  const buildingRows=[...ids].sort((x,y)=> buildingName(x).localeCompare(buildingName(y),"fr"))
    .map(id => ({
      label: buildingName(id),
      a: A.analysis ? String(countOf(A.analysis, id)) : "—",
      b: B.analysis ? String(countOf(B.analysis, id)) : "—",
      diff: A.analysis && B.analysis && countOf(A.analysis,id)!==countOf(B.analysis,id)
    }));

  el.innerHTML=`
    <div class="compare-row"><b>Critère</b><b>Joueur A</b><b>Joueur B</b></div>
    ${rows.map(r=>`<div class="compare-row"><span class="label">${r[0]}</span><b>${r[1]}</b><b>${r[2]}</b></div>`).join("")}
    ${buildingRows.map(r=>`<div class="compare-row${r.diff?" diff":""}"><span class="label">${r.label}</span><b>${r.a}</b><b>${r.b}</b></div>`).join("")}
  `;
}

document.querySelectorAll(".tab").forEach(btn=>{
  btn.addEventListener("click",()=>{
    document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p=>p.classList.remove("active"));
    btn.classList.add("active");
    document.querySelector("#panel-"+btn.dataset.tab).classList.add("active");
  });
});

buildPlayerPanel("a","Joueur A");
buildPlayerPanel("b","Joueur B");
renderCompare();
pingEngine();

if("serviceWorker" in navigator){
  window.addEventListener("load",()=>navigator.serviceWorker.register("./service-worker.js").catch(()=>{}));
}

fetch("./data/buildings.json")
  .then(r => r.json())
  .then(catalog => {
    const levels = catalog.buildings.reduce((n,b)=>n+b.levels.length,0);
    const el = document.querySelector("#catalog-count");
    if(el) el.textContent = `${catalog.buildings.length} types de bâtiments • ${levels} emplacements niveau`;
    window.CLASHCOMPARE_BUILDINGS = catalog;
    renderCompare();
  })
  .catch(()=>{ const el=document.querySelector("#catalog-count"); if(el) el.textContent="Catalogue indisponible"; });
