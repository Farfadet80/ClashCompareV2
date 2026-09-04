# ClashCompare

PWA locale de comparaison de villages Clash of Clans, avec analyse YOLO réelle via un serveur Python.

## État actuel (2026-09-04)

- **Détecteur** : V5 (YOLO11s), poids `models/building-detector.pt`
- **Inférence** : **imgsz 800** (promue : bat 640 sur le test réservé)
- **ONNX** : `models/building-detector.onnx` entrée 800×800, parité PT vérifiée
- **Niveaux** : classifieurs production `air-defense` (8–11) et `town-hall` (10–14) uniquement — jamais inventés
- **Non promus** : V6, SAHI/tiling, classifieur canon (quarantaine `models/experimental/`)

| Split | Config | mAP50 | mAP50-95 |
|---|---|---:|---:|
| Test réservé | V5 imgsz 640 | 0,802 | 0,579 |
| Test réservé | V5 imgsz **800** | **0,839** | **0,612** |

Détail technique : `HANDOFF.md` et `models/README.md`.

## Lancement

```powershell
cd C:\Users\jimmy\Desktop\Clashcompare\ClashCompareV2-main
.\.venv\Scripts\python.exe training\scripts\check_environment.py
.\.venv\Scripts\python.exe training\scripts\serve_compare.py
```

Ouvrir http://127.0.0.1:8765/

Ne pas lancer `python -m http.server 8765` en parallèle (ça masque `/api/analyze`).

## Fonctionnalités UI

- Joueur A / Joueur B / Comparatif
- Import Mode photo, aperçu, catalogue bâtiments
- Analyse via `POST /api/analyze` (serveur local, pas d’ONNX dans le navigateur)
- PWA installable (cache `clashcompare-v3-5-infer800`)

## Environnement GPU (GTX 1050)

On utilise volontairement une **stack figée** (pas « latest ») pour rester compatible avec la **GTX 1050 (Compute Capability 6.1 / `sm_61`)** et entraîner YOLO en CUDA. Les PyTorch récents (souvent CC ≥ 7.5, ex. cu128) cassent ce GPU.

Stack validée : Python 3.12.10, `torch==2.7.1+cu118`, Ultralytics 8.4.137.

```powershell
.\.venv\Scripts\python.exe training\scripts\check_environment.py
# Recréer le venv seulement si nécessaire, avec CE fichier (pas les dernières versions) :
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r training\requirements-gtx1050.txt
```

**Ne mets pas** Python / PyTorch / CUDA / Ultralytics à jour sans vérifier `sm_61`. En cas de doute, partir des versions déjà installées.

## API Clash of Clans

`config.js` : `window.CLASHCOMPARE_API = ""` — non branchée. L’API officielle ne fournit de toute façon pas les niveaux individuels des bâtiments ; la vision reste la source pour l’inventaire détaillé.

## Licence / disclaimer

Application fan, non officielle, sans soutien de Supercell.
