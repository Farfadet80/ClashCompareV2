const p1 = document.getElementById('player1');
const p2 = document.getElementById('player2');
const result = document.getElementById('result');

function normalizeTag(value) {
  let v = value.trim().toUpperCase();
  if (v && !v.startsWith('#')) v = '#' + v;
  return v;
}

function validTag(tag) {
  return /^#[A-Z0-9]{3,14}$/.test(tag);
}

function saveTags() {
  localStorage.setItem('coc_player1', normalizeTag(p1.value));
  localStorage.setItem('coc_player2', normalizeTag(p2.value));
}

function loadTags() {
  p1.value = localStorage.getItem('coc_player1') || '';
  p2.value = localStorage.getItem('coc_player2') || '';
}

function comparePlayers() {
  const a = normalizeTag(p1.value);
  const b = normalizeTag(p2.value);

  p1.value = a;
  p2.value = b;

  if (!validTag(a) || !validTag(b)) {
    result.classList.remove('hidden');
    result.innerHTML = '<p class="error">⚠️ Entre deux tags valides, par exemple #ABC123.</p>';
    return;
  }

  saveTags();

  const demo = [
    ['Niveau HDV', '16', '15'],
    ['Trophées', '2 850', '2 640'],
    ['XP', '210', '184'],
    ['Statut guerre', 'Actif', 'Actif']
  ];

  result.classList.remove('hidden');
  result.innerHTML = `
    <h2>📊 Comparaison</h2>
    <div class="comparison">
      <div class="player">
        <strong>Joueur 1</strong>
        <div class="tag">${a}</div>
        <span class="badge">Démonstration</span>
      </div>
      <div class="player">
        <strong>Joueur 2</strong>
        <div class="tag">${b}</div>
        <span class="badge">Démonstration</span>
      </div>
    </div>
    ${demo.map(row => `<div class="stat"><span>${row[0]}</span><strong>${row[1]} / ${row[2]}</strong></div>`).join('')}
    <p class="hint">Ces statistiques sont provisoires. La prochaine version pourra récupérer les vraies données via un backend sécurisé.</p>
  `;
}

document.getElementById('compareBtn').addEventListener('click', comparePlayers);

document.getElementById('clearBtn').addEventListener('click', () => {
  p1.value = '';
  p2.value = '';
  localStorage.removeItem('coc_player1');
  localStorage.removeItem('coc_player2');
  result.classList.add('hidden');
  result.innerHTML = '';
});

[p1, p2].forEach(input => {
  input.addEventListener('input', () => {
    input.value = input.value.toUpperCase().replace(/\s/g, '');
    saveTags();
  });
});

loadTags();

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./service-worker.js');
  });
}
