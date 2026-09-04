# ClashCompare — mémoire technique

Dernière mise à jour : 2026-09-04 (promotion inférence imgsz 800 + allowlist niveaux)

## État actuel

### Ce qui fonctionne

- Marqueur release : models/ACTIVE.json → uilding-detector-v5s-infer800 (imgsz 800).
- PWA locale : `index.html` / `app.js` / `style.css` (onglets Joueur A, Joueur B, Comparatif, tags, import d’image, catalogue 50 bâtiments / 543 niveaux).
- **Analyse réelle** : `training/scripts/serve_compare.py` sert la PWA et `POST /api/analyze` (détecteur V5, **imgsz 800**). Plus de fake de métadonnées.
- Pipeline YOLO local : détection des **types** de bâtiments, puis classifieurs de niveaux partiels (jamais inventés).
- Environnement `.venv` : Python 3.12.10, **PyTorch 2.7.1+cu118**, Ultralytics **8.4.137**, CUDA 11.8, GPU **GTX 1050 CC 6.1 (`sm_61`)** — inférence et entraînement CUDA OK.
- Détecteur **V5 promu** : poids V5 inchangés ; alias PT/ONNX -> `models/releases/building-detector-v5s-infer800/` (**imgsz 800**). Archive train : `building-detector-v5s-distilled-640/`.
- Run V5 `building-detector-v5s-distilled-640` : **terminé 80/80** le 2026-09-02 18:14 (pas un arrêt à 71).

### Ce qui ne fonctionne pas encore
- L’API officielle Clash of Clans n’est **pas** branchée (`config.js` : `window.CLASHCOMPARE_API = ""`). Elle ne fournit de toute façon pas les niveaux individuels des bâtiments.
- Le navigateur n’exécute pas ONNX lui-même : il envoie la capture au serveur Python local.
- Classifieurs de niveaux : seulement `air-defense` (8–11) et `town-hall` (10–14).
- `town-hall-guardian` : 0 annotation.
- Pas de dépôt Git (pas de dossier `.git`, `git` absent du PATH).

## Architecture

- **Frontend** : PWA statique (HTML/JS/CSS).
- **Vision** : Ultralytics YOLO11, entraînement Windows + GTX 1050.
- **API CoC** : prévue plus tard via backend ; pas de clé dans le repo.
- **Données village détaillées** : vision (YOLO), pas l’API.

## Installation / lancement

```powershell
cd C:\Users\jimmy\Desktop\Clashcompare\ClashCompareV2-main
.\.venv\Scripts\python.exe training\scripts\check_environment.py
.\.venv\Scripts\python.exe training\scripts\serve_compare.py
# puis ouvrir http://127.0.0.1:8765/
# Ne pas lancer `python -m http.server 8765` en parallèle (ça masque /api/analyze).
```

Recréer le venv (ne pas le faire tant que l’actuel marche) :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r training\requirements-gtx1050.txt
```

## YOLO

| Élément | Valeur |
|---|---|
| Dernier run | `training/runs/building-detector-v5s-distilled-640/` |
| Modèle de base | YOLO11s (`yolo11s.pt`, ~9,45 M params) |
| Dataset YAML | `training/dataset.yaml` (runtime : `training/runs/dataset.resolved.yaml`) |
| Images | train 931 / val 118 / test 60 |
| Classes | 50 (49 annotées ; `town-hall-guardian` vide) |
| Params V5 | imgsz 640, batch 4, epochs 80, patience 20, `cls_pw=0.25`, cos LR, distillation depuis V4 `models/building-detector.pt` |
| Ultralytics | 8.4.137 |

### Vérification 71 vs 80 (2026-09-03)

| Fichier | Date | Taille | Champ `epoch` | Optimizer |
|---|---|---:|---:|---|
| `results.csv` | — | 81 lignes (header + **1→80**) | dernière ligne = **80** | — |
| `epoch71.pt` | 02/09 17:06 | 78 Mo | **71** (0-indexé ≈ CSV 72) | présent |
| `epoch79.pt` | 02/09 18:14 | 78 Mo | **79** (0-indexé = CSV **80**) | présent |
| `last.pt` | 02/09 18:14 | 19 Mo | **-1** (run terminé, strippé) | **absent** |
| `best.pt` | 02/09 18:14 (poids figés ~15:19) | 19 Mo | -1 | absent |

**Arrêt réel : 80/80 (run terminé), pas 71.** `epoch71.pt` existe mais n’est pas le dernier état. **Ne pas relancer un entraînement depuis zéro.** **Ne pas forcer 20 epochs de plus** sauf demande explicite de Jimmy.

### Checkpoints (ne pas supprimer)

- `weights/last.pt` — état **final strippé** après 80/80 (`epoch=-1`, **pas d’optimizer**). Inutilisable pour `resume=True`. Ne pas le modifier.
- `weights/best.pt` — meilleur fitness (mAP50-95 0,59838). Pas l’état complet.
- `weights/epoch79.pt` — **dernier checkpoint complet**.
- `weights/epoch71.pt` — save intermédiaire, **plus ancien** que `epoch79.pt`.
- `weights/epoch0.pt` … `epoch79.pt` — historiques `save_period=1`.

Métriques val du log d’entraînement V5 (CSV epoch 80) : P 0,841 / R 0,796 / mAP50 0,826 / mAP50-95 0,595.

### Évaluation V5 vs V4 (lancée 2026-09-03, GTX 1050, batch 1 ou 2)

Commandes :

```powershell
cd C:\Users\jimmy\Desktop\Clashcompare\ClashCompareV2-main
$env:YOLO_CONFIG_DIR = (Get-Location).Path
$env:MPLCONFIGDIR = Join-Path (Get-Location) ".matplotlib"
.\.venv\Scripts\python.exe training\scripts\evaluate_detector.py --model models\building-detector.pt --split val --imgsz 512 --batch 2 --name v4-alias-val
.\.venv\Scripts\python.exe training\scripts\evaluate_detector.py --model models\building-detector.pt --split test --imgsz 512 --batch 2 --name v4-alias-test
.\.venv\Scripts\python.exe training\scripts\evaluate_detector.py --model training\runs\building-detector-v5s-distilled-640\weights\best.pt --split val --imgsz 640 --batch 1 --name v5s-distilled-640-val
.\.venv\Scripts\python.exe training\scripts\evaluate_detector.py --model training\runs\building-detector-v5s-distilled-640\weights\best.pt --split test --imgsz 640 --batch 1 --name v5s-distilled-640-test
```

(Les deux commandes V4 ci-dessus ont été lancées **avant** la copie des alias V5, sur l’ancien `models/building-detector.pt` V4.)

| Split | Modèle | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| Val | V4 alias, imgsz 512 | 0,6729 | 0,3940 | 0,4382 | 0,3055 |
| Val | V5 `best.pt`, imgsz 640 | 0,8519 | 0,7890 | 0,8241 | 0,6022 |
| Test réservé | V4 alias, imgsz 512 | 0,6765 | 0,3669 | 0,4284 | 0,2880 |
| Test réservé | V5 `best.pt`, imgsz 640 | **0,7878** | **0,7915** | **0,8022** | **0,5788** |

JSON : `training/runs/evaluations/v4-alias-val|test/metrics.json` et `v5s-distilled-640-val|test/metrics.json`.

**Décision : promouvoir V5.** Elle bat V4 sur les 4 métriques du test réservé.

Export :

```powershell
.\.venv\Scripts\python.exe training\scripts\export_detector_release.py --model training\runs\building-detector-v5s-distilled-640\weights\best.pt --release building-detector-v5s-distilled-640 --imgsz 640
Copy-Item models\releases\building-detector-v5s-distilled-640\building-detector-v5s-distilled-640.pt models\building-detector.pt
Copy-Item models\releases\building-detector-v5s-distilled-640\building-detector-v5s-distilled-640.onnx models\building-detector.onnx
```

Parité PT/ONNX OK (43 détections, SHA-256 PT `866595fad39a5b7dfdf87076332faadc40a88bc55eae1b02f093d996362fb93d`). ONNX 640×640. V4 reste dans `models/releases/building-detector-v4-pedro-no-guardian/`. Inférence : `--imgsz 640`.

## GPU

- GTX 1050, Compute Capability **6.1** (sm_61).
- On utilise volontairement une **stack Python/PyTorch/CUDA figée** pour rester compatible avec ce GPU et entraîner YOLO en CUDA.
- Les builds PyTorch récents (ex. cu128 / GPU à partir du Compute Capability **7.5**) **ne supportent plus** la GTX 1050 : risque de bascule CPU ou d'environnement cassé (voir .venv-broken-20260831).
- **Ne mets pas** Python, PyTorch, CUDA ou Ultralytics à jour sans vérifier sm_61. Le but n'est **pas** de mettre Python à jour à tout prix.
- Avant toute modification d'environnement : lire les versions installées (	raining/scripts/check_environment.py) et partir de la stack validée, ne pas recréer un venv « latest ».
- Stack validée : Python **3.12.10**, 	orch==2.7.1+cu118, 	orchvision==0.22.1+cu118, Ultralytics **8.4.137** (	raining/requirements-gtx1050.txt).

## Reprise d’entraînement

Le run V5 est **terminé à 80/80**. Il n’y a **pas** d’entraînement à reprendre.

- `last.pt` est **inutilisable** pour `resume=True` (epoch=-1, pas d’optimizer). Ne pas le modifier.
- Le dernier checkpoint **complet** est `epoch79.pt` (optimizer présent).
- Relancer 80 epochs depuis `yolo11s.pt` ou forcer +20 epochs **uniquement si Jimmy le demande**. Dans ce cas, partir de `epoch79.pt`, pas de `last.pt`, et ne supprimer aucun checkpoint.

## Travail effectué (session Cursor 2026-09-03)

- Audit du projet, CUDA GTX 1050, lancement PWA, smoke `analyze_village.py` (V4, 34 détections).
- Correctif CSS aperçu image + cache service worker (`clashcompare-v3-4-yolo`).
- Éval V5 vs V4 (val + test) : V5 gagne largement le test → **V5 promu**.
- Export `export_detector_release.py` + copie des alias. `analyze_village.py` défaut imgsz 640.
- Backend local `serve_compare.py` + `POST /api/analyze` branché dans `app.js`. Test : 30 détections V5 sur une image test (pas de niveaux inventés).

## Fichiers modifiés

- `app.js`, `index.html`, `style.css`, `service-worker.js`, `HANDOFF.md`, `models/README.md`, `training/scripts/analyze_village.py`, `training/scripts/serve_compare.py`
- Alias `models/building-detector.pt` / `.onnx` (V5)
- Release `models/releases/building-detector-v5s-distilled-640/`
- Évals `training/runs/evaluations/v5s-distilled-640-*` et `v4-alias-*`
- Export a aussi écrit `training/runs/building-detector-v5s-distilled-640/weights/best.onnx` (aucun `epoch*.pt` supprimé)

## Bugs connus

- Si `python -m http.server 8765` tourne déjà, `/api/analyze` ne répond pas.
- Preview cassé si l’ancien service worker sert encore `style.css` (cache `clashcompare-v3-3`).
- `last.pt` final sans optimizer : piège classique Ultralytics.

## Point 1+2 (2026-09-03 soir)

- Import `--as-train` des splits restants : `coc-all-traps` (+42 images / 3984 boîtes), glenn, evans, base-finder. Train = **999 images / 40222 boîtes**. Val/test ClashCompare **inchangés**.
- Fine-tune **V6** terminé après 17 epochs : `training/runs/building-detector-v6-weakclass-ft` depuis V5, imgsz 640, batch 2, `cls_pw=0.4`, distillation V5.
- Meilleure validation V6 : P 0,8347 / R 0,7847 / mAP50 0,8258 / mAP50-95 0,5893. Elle reste sous la V5 en mAP50-95 (0,6022), donc **V6 non promue** et test réservé non consulté.
- Classifieur **canon** expérimental : `models/level-cannon.pt` (niveaux 1–21). Val interne top1 **75 %** sur 36 crops — dataset minuscule (2–7 images/niveau), **à ne pas traiter comme fiable**. HDV 10–14 et anti-air 8–11 inchangés.
- `serve_compare.py` arrêté le temps du fine-tune GPU. Relancer après V6.

## Évaluation SAHI/tiling V5 (2026-09-04)

Script reproductible ajouté : `training/scripts/evaluate_tiled_detector.py`.

- Passe globale YOLO11 à `imgsz=640`, puis tuiles de 640 pixels source.
- Overlap 20 % ou 25 %, zones de propriété définies par le centre des boîtes pour écarter les prédictions tronquées aux bords.
- Fusion classe par classe avec `torchvision.ops.batched_nms`, IoU 0,55.
- Inférence à `conf=0.001`, NMS interne IoU 0,70, `max_det=300`, comme la validation Ultralytics pour ne pas tronquer les courbes AP.
- Métriques Ultralytics : AP interpolée par classe, IoU 0,50:0,95 ; P/R au seuil maximisant F1. Le script mesure aussi la passe globale seule avec exactement le même évaluateur.
- Garde-fou : le split `test` est refusé sans option explicite `--allow-test`.

Commandes exécutées sur les 118 images VAL :

```powershell
.\.venv\Scripts\python.exe training\scripts\evaluate_tiled_detector.py --model models\building-detector.pt --split val --imgsz 640 --tile-size 640 --overlap 0.20 --merge-iou 0.55 --device 0 --name v5-sahi-val-tiles640-o20-nms55
.\.venv\Scripts\python.exe training\scripts\evaluate_tiled_detector.py --model models\building-detector.pt --split val --imgsz 640 --tile-size 640 --overlap 0.25 --merge-iou 0.55 --device 0 --name v5-sahi-val-tiles640-o25-nms55
```

| Configuration VAL | P | R | mAP50 | mAP50-95 | Temps appels inférence |
|---|---:|---:|---:|---:|---:|
| Baseline V5 publiée (`model.val`) | 0,8519 | 0,7890 | 0,8241 | 0,6022 | 31,6 ms/image (ancienne mesure) |
| Contrôle global du nouveau script | 0,8299 | 0,7978 | 0,8242 | 0,6025 | 5,7–7,3 s / 118 |
| Global + tuiles 640, overlap 20 % | 0,8395 | 0,7756 | 0,8224 | 0,5994 | 25,6 s / 118 (216,9 ms/image) |
| Global + tuiles 640, overlap 25 % | 0,8238 | 0,7891 | 0,8220 | 0,5990 | 23,9 s / 118 (202,4 ms/image) |

Le contrôle reproduit les mAP publiées à `+0,00006` (mAP50) et `+0,00027` (mAP50-95). Les P/R au point F1 varient davantage ; la décision repose donc sur les AP comparables, pas sur un pourcentage inventé.

Classes faibles, meilleur cas overlap 20 %, mAP50-95 contrôle → tiling :

- `hidden-tesla` : 0,3533 → 0,3488 ; `bomb` : 0,1717 → 0,1477.
- `spring-trap` : 0,1745 → 0,1707 ; `air-bomb` : 0,1205 → 0,1100.
- `seeking-air-mine` : 0,0508 → 0,0492 ; `skeleton-trap` : 0,0446 → 0,0445.
- `spell-tower` progresse isolément : 0,4913 → 0,5136, insuffisant face au recul global et des pièges.

JSON complets, métriques par classe incluses :
`training/runs/evaluations/v5-sahi-val-tiles640-o20-nms55/metrics.json` et
`training/runs/evaluations/v5-sahi-val-tiles640-o25-nms55/metrics.json`.

**Décision : ne pas promouvoir le tiling.** Les deux configurations reculent en mAP50 et mAP50-95, les petits pièges ne progressent pas globalement et le coût est nettement supérieur. Le test réservé n'a pas été évalué. `serve_compare.py` et l'alias V5 restent inchangés.



## Session 2026-09-04 (Cursor) — imgsz 800 + classifieurs

### Tests exécutés
- check_environment.py : CUDA GTX 1050 OK, torch 2.7.1+cu118, Ultralytics 8.4.137.
- alidate_dataset.py : train 999 / val 118 / test 60.
- smoke_test_yolo.py : OK.
- Classifieurs : air-defense val 99,3 % / test 100 % ; town-hall val 100 % / test 98 %.
- Comparaison V5 **même poids**, imgsz 640 vs 800 :

| Split | imgsz | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Val | 640 | 0,8519 | 0,7890 | 0,8241 | 0,6022 |
| Val | 800 | 0,8259 | 0,8236 | 0,8416 | 0,6247 |
| Test réservé | 640 | 0,7878 | 0,7915 | 0,8022 | 0,5788 |
| Test réservé | **800** | **0,8273** | **0,8281** | **0,8390** | **0,6121** |

JSON : 	raining/runs/evaluations/v5-alias-*-imgsz{640,800}*/metrics.json.

Pièges VAL 640→800 (mAP50-95) : air-bomb 0,12→0,22 ; bomb 0,18→0,24 ; spring-trap 0,17→0,21 ; hidden-tesla 0,33→0,39 ; skeleton-trap 0,05→0,10.

### Décisions (règle : promouvoir seulement si mieux que baseline)
- **Promouvoir imgsz 800** pour l'inférence (test réservé meilleur sur les 4 métriques). Poids PT inchangés.
- Export ONNX 800×800 + parité OK → models/releases/building-detector-v5s-infer800/ + alias.
- **Ne pas** réactiver V6 ni SAHI (déjà sous baseline).
- Quarantaine level-cannon.pt → models/experimental/ ; allowlist production ir-defense, 	own-hall dans nalyze_village.py.
- Smoke analyse test : 35 détections, niveaux AD seulement, imgsz 800.

### Fichiers touchés
- 	raining/scripts/analyze_village.py, serve_compare.py
- models/building-detector.pt|.onnx, models/releases/building-detector-v5s-infer800/
- models/experimental/level-cannon.pt + README
- models/README.md, service-worker.js (clashcompare-v3-5-infer800), HANDOFF.md

## TODO

1. Inférence **imgsz 800** promue (2026-09-04) — poids V5 inchangés ; archive ONNX 640 conservée.
2. SAHI/tiling et V6 : non promus (sous baseline).
3. Plus de captures Mode photo (pièges / teslas) — datasets publics épuisés localement.
4. Plus de crops **niveaux** (canons surtout) avant de sortir level-cannon de la quarantaine.
5. API CoC pour héros / labo / sorts — jamais inventer les bâtiments.
6. Réinstaller Git si versionnage souhaité.
