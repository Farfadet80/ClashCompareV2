
const players = {
  a: { tag:"", imageName:"", width:0, height:0, quality:"—" },
  b: { tag:"", imageName:"", width:0, height:0, quality:"—" }
};

function normalizeTag(v){
  let s=(v||"").trim().toUpperCase().replace(/\s+/g,"");
  if(s && !s.startsWith("#")) s="#"+s;
  return s;
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
  title.textContent=label;

  uploadBtn.addEventListener("click",()=>imageInput.click());

  imageInput.addEventListener("change",()=>{
    const file=imageInput.files?.[0];
    if(!file) return;
    const url=URL.createObjectURL(file);
    const img=new Image();
    img.onload=()=>{
      players[key].imageName=file.name;
      players[key].width=img.naturalWidth;
      players[key].height=img.naturalHeight;
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

  analyzeBtn.addEventListener("click",()=>{
    const tag=normalizeTag(tagInput.value);
    players[key].tag=tag;
    tagInput.value=tag;

    if(!players[key].imageName){
      alert("Ajoute d'abord la capture du village.");
      return;
    }

    area.hidden=false;
    panel.querySelector(".r-tag").textContent=tag || "Non renseigné";
    panel.querySelector(".r-image").textContent=`${players[key].width} × ${players[key].height}`;
    panel.querySelector(".r-quality").textContent=players[key].quality;
    panel.querySelector(".r-state").textContent="Capture prête pour reconnaissance";
    status.textContent="Prêt";
    renderCompare();
  });

  clearBtn.addEventListener("click",()=>{
    players[key]={ tag:"", imageName:"", width:0, height:0, quality:"—" };
    tagInput.value="";
    imageInput.value="";
    preview.removeAttribute("src");
    preview.hidden=true;
    area.hidden=true;
    status.textContent="En attente";
    renderCompare();
  });
}

function renderCompare(){
  const el=document.querySelector("#compare-content");
  const A=players.a, B=players.b;
  const rows=[
    ["# joueur", A.tag||"—", B.tag||"—"],
    ["Capture", A.imageName?"Ajoutée":"—", B.imageName?"Ajoutée":"—"],
    ["Résolution", A.width?`${A.width}×${A.height}`:"—", B.width?`${B.width}×${B.height}`:"—"],
    ["Qualité", A.quality, B.quality]
  ];
  el.innerHTML=`
    <div class="compare-row"><b>Critère</b><b>Joueur A</b><b>Joueur B</b></div>
    ${rows.map(r=>`<div class="compare-row"><span class="label">${r[0]}</span><b>${r[1]}</b><b>${r[2]}</b></div>`).join("")}
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
  })
  .catch(()=>{ const el=document.querySelector("#catalog-count"); if(el) el.textContent="Catalogue indisponible"; });
