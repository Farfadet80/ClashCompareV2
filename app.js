const $ = id => document.getElementById(id);
const p1 = $('player1'), p2 = $('player2'), statusEl = $('status');
const profilesEl = $('profiles'), comparisonEl = $('comparison'), rowsEl = $('compareRows');

const normalize = v => {
  v = (v || '').trim().toUpperCase().replace(/\s/g,'');
  return v && !v.startsWith('#') ? '#' + v : v;
};
const valid = v => /^#[0289PYLQGRJCUV]{3,14}$/i.test(v) || /^#[A-Z0-9]{3,14}$/.test(v);

function esc(s=''){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function save(){localStorage.setItem('cc_p1',normalize(p1.value));localStorage.setItem('cc_p2',normalize(p2.value))}
function n(v){return new Intl.NumberFormat('fr-FR').format(Number(v||0))}
function clanName(p){return p.clan?.name || 'Aucun clan'}
function leagueName(p){return p.league?.name || 'Non classé'}
function heroSummary(p){
  const h=(p.heroes||[]).filter(x=>x.village==='home');
  return h.length ? h.map(x=>`${x.name} ${x.level}/${x.maxLevel}`).join(' • ') : '—';
}
function troopProgress(p){
  const arr=(p.troops||[]).filter(x=>x.village==='home' && Number.isFinite(x.level) && Number.isFinite(x.maxLevel));
  if(!arr.length)return null;
  const cur=arr.reduce((s,x)=>s+x.level,0), max=arr.reduce((s,x)=>s+x.maxLevel,0);
  return max?Math.round(cur/max*100):null;
}
function spellProgress(p){
  const arr=(p.spells||[]).filter(x=>Number.isFinite(x.level) && Number.isFinite(x.maxLevel));
  if(!arr.length)return null;
  const cur=arr.reduce((s,x)=>s+x.level,0), max=arr.reduce((s,x)=>s+x.maxLevel,0);
  return max?Math.round(cur/max*100):null;
}
async function fetchPlayer(tag){
  const base=(window.CLASHCOMPARE_API||'').replace(/\/$/,'');
  if(!base || base.includes('COLLE_ICI')){
    throw new Error("L'URL du Cloudflare Worker n'est pas encore configurée dans config.js.");
  }
  const res=await fetch(`${base}/player?tag=${encodeURIComponent(tag)}`);
  const data=await res.json().catch(()=>({}));
  if(!res.ok) throw new Error(data.message || data.reason || `Erreur API (${res.status})`);
  return data;
}
function profile(p,label){
  const troop=troopProgress(p), spell=spellProgress(p);
  return `<article class="profile">
    <div class="tag">${esc(label)}</div>
    <h3>${esc(p.name)}</h3>
    <div class="tag">${esc(p.tag)}</div>
    <div class="th">HDV ${p.townHallLevel ?? '—'}</div>
    <div class="clan">🛡️ ${esc(clanName(p))}</div>
    <div class="stats">
      <div class="row"><span>Trophées</span><strong>${n(p.trophies)}</strong></div>
      <div class="row"><span>Record trophées</span><strong>${n(p.bestTrophies)}</strong></div>
      <div class="row"><span>Niveau XP</span><strong>${n(p.expLevel)}</strong></div>
      <div class="row"><span>Étoiles de guerre</span><strong>${n(p.warStars)}</strong></div>
      <div class="row"><span>Dons</span><strong>${n(p.donations)}</strong></div>
      <div class="row"><span>Ligue</span><strong>${esc(leagueName(p))}</strong></div>
      <div class="row"><span>Progression troupes*</span><strong>${troop===null?'—':troop+' %'}</strong></div>
      <div class="row"><span>Progression sorts*</span><strong>${spell===null?'—':spell+' %'}</strong></div>
      <div class="row"><span>Héros</span><strong>${esc(heroSummary(p))}</strong></div>
    </div>
  </article>`;
}
function compRow(label,a,b,higher=true,suffix=''){
  const aa=Number(a||0), bb=Number(b||0);
  const ca=aa===bb?'equal':((higher?aa>bb:aa<bb)?'winner':'loser');
  const cb=aa===bb?'equal':((higher?bb>aa:bb<aa)?'winner':'loser');
  return `<div class="row"><span>${esc(label)}</span><strong><span class="${ca}">${n(aa)}${suffix}</span> / <span class="${cb}">${n(bb)}${suffix}</span></strong></div>`;
}
function render(a,b){
  profilesEl.innerHTML=profile(a,'Joueur A')+profile(b,'Joueur B');
  profilesEl.classList.remove('hidden');
  rowsEl.innerHTML=
    compRow('HDV',a.townHallLevel,b.townHallLevel)+
    compRow('Trophées',a.trophies,b.trophies)+
    compRow('Record trophées',a.bestTrophies,b.bestTrophies)+
    compRow('Niveau XP',a.expLevel,b.expLevel)+
    compRow('Étoiles de guerre',a.warStars,b.warStars)+
    compRow('Dons',a.donations,b.donations)+
    compRow('Progression troupes*',troopProgress(a),troopProgress(b),true,' %')+
    compRow('Progression sorts*',spellProgress(a),spellProgress(b),true,' %');
  comparisonEl.classList.remove('hidden');
}
async function compare(){
  const a=normalize(p1.value), b=normalize(p2.value);
  p1.value=a;p2.value=b;
  if(!valid(a)||!valid(b)){statusEl.className='hint error';statusEl.textContent='Entre deux tags joueurs valides.';return}
  save(); profilesEl.classList.add('hidden');comparisonEl.classList.add('hidden');
  statusEl.className='hint loading';statusEl.textContent='Récupération des joueurs…';
  try{
    const [pa,pb]=await Promise.all([fetchPlayer(a),fetchPlayer(b)]);
    render(pa,pb);
    statusEl.className='hint';statusEl.textContent='Données récupérées avec succès.';
  }catch(e){
    statusEl.className='hint error';statusEl.textContent=e.message || 'Impossible de récupérer les joueurs.';
  }
}
$('compareBtn').addEventListener('click',compare);
$('clearBtn').addEventListener('click',()=>{p1.value='';p2.value='';localStorage.removeItem('cc_p1');localStorage.removeItem('cc_p2');profilesEl.classList.add('hidden');comparisonEl.classList.add('hidden');statusEl.className='hint';statusEl.textContent='Les tags restent enregistrés sur cet appareil.'});
[p1,p2].forEach(x=>x.addEventListener('input',()=>{x.value=x.value.toUpperCase().replace(/\s/g,'');save()}));
p1.value=localStorage.getItem('cc_p1')||'';p2.value=localStorage.getItem('cc_p2')||'';
if('serviceWorker' in navigator) addEventListener('load',()=>navigator.serviceWorker.register('./service-worker.js'));
