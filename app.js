const STORAGE_KEY = "clashcompare:players:v1";

const players = {
  a: emptyPlayer(),
  b: emptyPlayer()
};

let exportMapping = null;

function emptyPlayer(){
  return {
    tag:"",
    imageName:"",
    file:null,
    width:0,
    height:0,
    quality:"—",
    analysis:null,
    villageExport:null
  };
}

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

function effectiveFor(player){
  const api = window.ClashCompareVillageExport;
  if(!api){
    const inv = player.analysis?.inventory || {};
    return {
      inventory: inv,
      source: Object.keys(inv).length ? "yolo" : "none",
      complementIds: []
    };
  }
  return api.mergeInventories(player.villageExport?.inventory, player.analysis?.inventory);
}

function sourceLabel(source){
  if(source === "export") return "Export JSON officiel";
  if(source === "yolo") return "YOLO (complément / fallback)";
  if(source === "mixed") return "Export JSON + complément YOLO";
  return "Aucune donnée";
}

function renderInventory(el, inventory){
  if(!inventory || !Object.keys(inventory).length){
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = Object.entries(inventory)
    .sort((a,b)=> buildingName(a[0]).localeCompare(buildingName(b[0]), "fr"))
    .map(([id, slot]) =>
      `<div class="inv-row"><b>${buildingName(id)}</b><span>${slot.total} — ${formatLevels(slot.levels)}</span></div>`
    ).join("");
}

function refreshPlayerResult(key){
  const panel = document.querySelector("#panel-"+key);
  if(!panel) return;
  const player = players[key];
  const area = panel.querySelector(".analysis-area");
  const notice = panel.querySelector(".r-notice");
  const inventoryEl = panel.querySelector(".r-inventory");
  const status = panel.querySelector(".status-pill");
  const eff = effectiveFor(player);
  const hasData = Object.keys(eff.inventory).length > 0 || player.villageExport || player.analysis;

  if(!hasData){
    area.hidden = true;
    return;
  }

  area.hidden = false;
  panel.querySelector(".r-tag").textContent = player.tag || player.villageExport?.tag || "Non renseigné";
  panel.querySelector(".r-image").textContent = player.width
    ? `${player.width} × ${player.height}`
    : (player.villageExport ? "Non requis (export)" : "—");
  panel.querySelector(".r-quality").textContent = player.villageExport
    ? sourceLabel(eff.source)
    : player.quality;
  panel.querySelector(".r-count").textContent = Object.keys(eff.inventory).length
    ? String(Object.values(eff.inventory).reduce((n,s)=>n+(s.total||0),0))
    : "—";

  const bits = [];
  bits.push(`Source : ${sourceLabel(eff.source)}.`);
  if(player.villageExport?.townHallLevel != null){
    bits.push(`HDV export : niv.${player.villageExport.townHallLevel}.`);
  }
  if(player.villageExport?.unresolved?.length){
    bits.push(`${player.villageExport.unresolved.length} entrée(s) Non détectée(s) (hors mapping).`);
  }
  if(eff.complementIds.length){
    bits.push(`Complément YOLO : ${eff.complementIds.map(buildingName).join(", ")}.`);
  }
  if(player.analysis?.level_note){
    bits.push(player.analysis.level_note);
  }
  if(!player.villageExport && !player.analysis){
    bits.push("Aucune donnée importée.");
  }
  notice.textContent = bits.join(" ");
  renderInventory(inventoryEl, eff.inventory);

  if(player.villageExport && player.analysis) status.textContent = "Export + YOLO";
  else if(player.villageExport) status.textContent = "Export importé";
  else if(player.analysis) status.textContent = "Analysé";
  else status.textContent = "En attente";
}

function persistState(){
  try{
    const payload = {
      a: {
        tag: players.a.tag || "",
        exportRaw: players.a.villageExport?.raw || null
      },
      b: {
        tag: players.b.tag || "",
        exportRaw: players.b.villageExport?.raw || null
      }
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }catch{
    /* quota / private mode */
  }
}

function applyExport(key, rawText, panel){
  const api = window.ClashCompareVillageExport;
  if(!api || !exportMapping){
    throw new Error("Parseur d'export indisponible");
  }
  const parsed = api.parseVillageExport(rawText, exportMapping);
  if(!parsed.ok) throw new Error(parsed.error || "import impossible");

  players[key].villageExport = parsed;
  if(!players[key].tag && parsed.tag){
    players[key].tag = normalizeTag(parsed.tag);
    const tagInput = panel.querySelector(".tag-input");
    if(tagInput) tagInput.value = players[key].tag;
  }
  const exportStatus = panel.querySelector(".export-status");
  if(exportStatus){
    exportStatus.textContent = parsed.note || "Export importé.";
    exportStatus.classList.remove("bad");
    exportStatus.classList.add("ok");
  }
  persistState();
  refreshPlayerResult(key);
  renderCompare();
}

async function pingEngine(){
  const el = document.querySelector("#engine-status");
  try{
    const r = await fetch("/api/health");
    if(!r.ok) throw new Error();
    const data = await r.json();
    if(el){
      el.textContent =
        `Moteur local prêt : ${data.engine} • imgsz ${data.imgsz}` +
        ` • conf ${data.conf ?? 0.25}` +
        ` • max_det ${data.max_det ?? 1000}` +
        ` • politique ${data.inference_policy || "baseline-conf25"}.` +
        ` Niveaux YOLO : désactivés tant que tous les niveaux récents ne sont pas couverts.` +
        ` JSON = toi/ami ; capture YOLO = adversaire.`;
    }
    return true;
  }catch{
    if(el) el.textContent = "Moteur local absent (YOLO optionnel). L'import JSON fonctionne hors ligne. Pour YOLO : .\\.venv\\Scripts\\python.exe training\\scripts\\serve_compare.py";
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
  const exportInput=panel.querySelector(".export-input");
  const exportFile=panel.querySelector(".export-file");
  const exportBtn=panel.querySelector(".export-btn");
  const exportClearBtn=panel.querySelector(".export-clear-btn");
  const exportStatus=panel.querySelector(".export-status");
  const status=panel.querySelector(".status-pill");
  title.textContent=label;

  tagInput.addEventListener("change",()=>{
    players[key].tag = normalizeTag(tagInput.value);
    tagInput.value = players[key].tag;
    persistState();
    refreshPlayerResult(key);
    renderCompare();
  });

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
      refreshPlayerResult(key);
      renderCompare();
    };
    img.src=url;
  });

  exportBtn.addEventListener("click",()=>{
    try{
      applyExport(key, exportInput.value, panel);
    }catch(err){
      exportStatus.textContent = String(err.message||err);
      exportStatus.classList.remove("ok");
      exportStatus.classList.add("bad");
    }
  });

  exportFile.addEventListener("change", async ()=>{
    const file = exportFile.files?.[0];
    if(!file) return;
    try{
      const text = await file.text();
      exportInput.value = text;
      applyExport(key, text, panel);
    }catch(err){
      exportStatus.textContent = String(err.message||err);
      exportStatus.classList.remove("ok");
      exportStatus.classList.add("bad");
    }
  });

  exportClearBtn.addEventListener("click",()=>{
    players[key].villageExport = null;
    exportInput.value = "";
    exportFile.value = "";
    exportStatus.textContent = "Export effacé.";
    exportStatus.classList.remove("ok","bad");
    persistState();
    refreshPlayerResult(key);
    renderCompare();
  });

  analyzeBtn.addEventListener("click", async ()=>{
    const tag=normalizeTag(tagInput.value);
    players[key].tag=tag;
    tagInput.value=tag;
    persistState();

    if(!players[key].file){
      if(players[key].villageExport){
        refreshPlayerResult(key);
        renderCompare();
        return;
      }
      alert("Importe d'abord l'export JSON du village, ou ajoute une capture pour YOLO.");
      return;
    }

    analyzeBtn.disabled=true;
    status.textContent="Analyse…";
    const notice=panel.querySelector(".r-notice");
    panel.querySelector(".analysis-area").hidden=false;
    notice.textContent="Analyse YOLO en cours (GTX 1050, quelques secondes)…";

    try{
      const body=new FormData();
      body.append("image", players[key].file, players[key].imageName);
      const response=await fetch("/api/analyze", { method:"POST", body });
      const data=await response.json();
      if(!response.ok) throw new Error(data.error || "analyse impossible");
      players[key].analysis=data;
      refreshPlayerResult(key);
    }catch(err){
      players[key].analysis=null;
      if(players[key].villageExport){
        refreshPlayerResult(key);
        notice.textContent = `YOLO indisponible (${err.message||err}). L'export JSON reste affiché.`;
      }else{
        panel.querySelector(".r-count").textContent="—";
        notice.textContent = String(err.message||err) + " — lance le serveur local serve_compare.py si besoin.";
        panel.querySelector(".r-inventory").hidden=true;
        status.textContent="Erreur";
      }
    }finally{
      analyzeBtn.disabled=false;
      renderCompare();
    }
  });

  clearBtn.addEventListener("click",()=>{
    players[key]=emptyPlayer();
    tagInput.value="";
    imageInput.value="";
    exportInput.value="";
    exportFile.value="";
    exportStatus.textContent="Colle le JSON exporté depuis le jeu, ou choisis un fichier .json.";
    exportStatus.classList.remove("ok","bad");
    preview.removeAttribute("src");
    preview.hidden=true;
    panel.querySelector(".analysis-area").hidden=true;
    panel.querySelector(".r-inventory").hidden=true;
    panel.querySelector(".r-inventory").innerHTML="";
    status.textContent="En attente";
    persistState();
    renderCompare();
  });
}

function countOf(player, id){
  return effectiveFor(player).inventory?.[id]?.total ?? 0;
}

function renderCompare(){
  const el=document.querySelector("#compare-content");
  const A=players.a, B=players.b;
  const effA=effectiveFor(A);
  const effB=effectiveFor(B);
  const rows=[
    ["# joueur", A.tag||A.villageExport?.tag||"—", B.tag||B.villageExport?.tag||"—"],
    ["Source données", sourceLabel(effA.source), sourceLabel(effB.source)],
    ["Export JSON", A.villageExport?"Importé":"—", B.villageExport?"Importé":"—"],
    ["Capture", A.imageName?"Ajoutée":"—", B.imageName?"Ajoutée":"—"],
    ["Résolution", A.width?`${A.width}×${A.height}`:"—", B.width?`${B.width}×${B.height}`:"—"],
    ["HDV (export)", A.villageExport?.townHallLevel!=null?`niv.${A.villageExport.townHallLevel}`:"Non détecté", B.villageExport?.townHallLevel!=null?`niv.${B.villageExport.townHallLevel}`:"Non détecté"],
    ["Bâtiments / pièges", Object.keys(effA.inventory).length?String(Object.values(effA.inventory).reduce((n,s)=>n+(s.total||0),0)):"—", Object.keys(effB.inventory).length?String(Object.values(effB.inventory).reduce((n,s)=>n+(s.total||0),0)):"—"]
  ];
  const ids=new Set([
    ...Object.keys(effA.inventory||{}),
    ...Object.keys(effB.inventory||{})
  ]);
  const buildingRows=[...ids].sort((x,y)=> buildingName(x).localeCompare(buildingName(y),"fr"))
    .map(id => {
      const aHas = !!effA.inventory[id];
      const bHas = !!effB.inventory[id];
      const aText = aHas
        ? `${countOf(A,id)} — ${formatLevels(effA.inventory[id].levels)}`
        : "Non disponible";
      const bText = bHas
        ? `${countOf(B,id)} — ${formatLevels(effB.inventory[id].levels)}`
        : "Non disponible";
      return {
        label: buildingName(id),
        a: aText,
        b: bText,
        diff: aHas && bHas && countOf(A,id)!==countOf(B,id)
      };
    });

  el.innerHTML=`
    <div class="compare-row"><b>Critère</b><b>Joueur A</b><b>Joueur B</b></div>
    ${rows.map(r=>`<div class="compare-row"><span class="label">${r[0]}</span><b>${r[1]}</b><b>${r[2]}</b></div>`).join("")}
    ${buildingRows.map(r=>`<div class="compare-row${r.diff?" diff":""}"><span class="label">${r.label}</span><b>${r.a}</b><b>${r.b}</b></div>`).join("")}
  `;
}

function restorePersisted(panelByKey){
  try{
    const raw = localStorage.getItem(STORAGE_KEY);
    if(!raw) return;
    const data = JSON.parse(raw);
    for(const key of ["a","b"]){
      const saved = data?.[key];
      if(!saved) continue;
      const panel = panelByKey[key];
      if(saved.tag){
        players[key].tag = normalizeTag(saved.tag);
        panel.querySelector(".tag-input").value = players[key].tag;
      }
      if(saved.exportRaw){
        panel.querySelector(".export-input").value = saved.exportRaw;
        try{
          applyExport(key, saved.exportRaw, panel);
        }catch(err){
          const exportStatus = panel.querySelector(".export-status");
          exportStatus.textContent = "Export sauvegardé illisible : " + (err.message||err);
          exportStatus.classList.add("bad");
        }
      }
    }
  }catch{
    /* ignore */
  }
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

Promise.all([
  fetch("./data/buildings.json").then(r => r.json()),
  fetch("./data/coc-export-mapping.json").then(r => r.json())
])
  .then(([catalog, mapping]) => {
    exportMapping = mapping;
    window.CLASHCOMPARE_BUILDINGS = catalog;
    const levels = catalog.buildings.reduce((n,b)=>n+b.levels.length,0);
    const el = document.querySelector("#catalog-count");
    if(el) el.textContent = `${catalog.buildings.length} types de bâtiments • ${levels} emplacements niveau • import JSON prêt`;
    restorePersisted({
      a: document.querySelector("#panel-a"),
      b: document.querySelector("#panel-b")
    });
    renderCompare();
  })
  .catch(()=>{
    const el=document.querySelector("#catalog-count");
    if(el) el.textContent="Catalogue / mapping indisponible";
  });
