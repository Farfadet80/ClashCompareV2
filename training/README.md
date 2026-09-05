# Entraînement local — GTX 1050

## État de l'environnement

L'environnement `.venv` utilise Python 3.12.10, PyTorch 2.7.1 avec CUDA 11.8 et
Ultralytics. Cette combinaison conserve le support `sm_61` de la GTX 1050.

**Ne pas upgrader** Python / PyTorch / CUDA / Ultralytics « à la dernière version » :
les builds récents (souvent Compute Capability ≥ 7.5) ne supportent plus la GTX 1050
et peuvent forcer le CPU ou casser le venv. Toujours vérifier
`training/scripts/check_environment.py` et `training/requirements-gtx1050.txt` avant
toute modification d'environnement.

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe training\scripts\check_environment.py
.\.venv\Scripts\python.exe training\scripts\validate_dataset.py
```

Le validateur bloque volontairement l'entraînement si les dossiers sont vides, si les
annotations YOLO sont invalides ou s'il y a moins de 100 images d'entraînement et 20 de
validation. Ce seuil évite surtout un lancement accidentel : il ne garantit pas à lui seul
un modèle fiable. Il faut des captures variées, chaque bâtiment encadré, toutes les classes
utiles représentées et un jeu de validation séparé.

La couverture des niveaux se contrôle avec :

```powershell
.\.venv\Scripts\python.exe training\scripts\validate_level_dataset.py
```

Le catalogue contient 543 combinaisons bâtiment/niveau. Le minimum de 10 images par niveau
est uniquement un garde-fou (5 430 crops au total) ; viser 20 à 50 crops variés par niveau
est préférable pour généraliser aux téléphones, zooms, obstacles et décorations différents.

## Datasets publics d'amorçage

Les sources compatibles repérées et leurs correspondances de classes sont enregistrées dans
`training/sources/public-datasets.json`. Après avoir exporté une source en format YOLO et
décompressé son ZIP dans `training/imports/<source>`, l'import se fait ainsi :

```powershell
.\.venv\Scripts\python.exe training\scripts\import_public_dataset.py training\imports\nothing-clash-of-clans nothing-clash-of-clans
```

Les labels non compatibles sont ignorés, les polygones de segmentation sont convertis en
boîtes et l'attribution de licence est ajoutée à `training/dataset/SOURCES.md`. Les splits
`val` et `test` sont protégés : même avec `--as-train`, une image déjà réservée n'est plus
recopiée dans le train.

## Intégrité des splits

Avant tout fine-tune, générer le train assaini sans supprimer les fichiers sources :

```powershell
.\.venv\Scripts\python.exe training\scripts\audit_split_integrity.py
.\.venv\Scripts\python.exe training\scripts\train_detector.py --prepare-only
```

L'audit compare les stems et SHA-256 entre train/val/test, écrit le rapport
`training/reports/split-integrity.json` et génère
`training/dataset/detector/train-clean.txt`. `train_detector.py` utilise cette liste par
défaut. L'option `--allow-split-overlap` existe uniquement pour un diagnostic explicite.

## Commandes prêtes

Test technique synthétique d'une époque :

```powershell
.\.venv\Scripts\python.exe training\scripts\smoke_test_yolo.py
```

Fine-tune réel, uniquement après validation et depuis un checkpoint existant :

```powershell
.\.venv\Scripts\python.exe training\scripts\train_detector.py `
  --model training\runs\building-detector-v5s-distilled-640\weights\epoch79.pt `
  --epochs 20 --imgsz 640 --batch 2 --name building-detector-v7-experiment
```

La GTX 1050 ne dispose que de 3 Go. Si la mémoire est insuffisante, utiliser `--batch 1`
ou `--imgsz 512`. Ne jamais relancer de zéro ni promouvoir sans battre la baseline
V5 + imgsz 800 + TTA sur le protocole documenté dans `HANDOFF.md`.

## Classifieurs de niveaux et analyse complète

Préparer puis entraîner un bâtiment dont les crops sont suffisamment nombreux :

```powershell
.\.venv\Scripts\python.exe training\scripts\prepare_level_classifier_dataset.py air-defense
.\.venv\Scripts\python.exe training\scripts\train_level_classifier.py air-defense
.\.venv\Scripts\python.exe training\scripts\evaluate_level_classifier.py air-defense
```

Une fois les meilleurs poids copiés dans `models/level-<bâtiment>.pt` et le détecteur dans
`models/building-detector.pt`, une capture se teste de bout en bout avec :

```powershell
.\.venv\Scripts\python.exe training\scripts\analyze_village.py C:\chemin\capture.png
```

Le script produit un JSON et une image annotée dans `training/runs/inference`. Le niveau
reste `null` lorsqu'aucun classifieur fiable n'est encore disponible pour ce bâtiment.

Évaluer explicitement un détecteur et figer une version PT/ONNX :

```powershell
.\.venv\Scripts\python.exe training\scripts\evaluate_detector.py --split val --name v3-val
.\.venv\Scripts\python.exe training\scripts\evaluate_detector.py --split test --name v3-test
.\.venv\Scripts\python.exe training\scripts\export_detector_release.py --release building-detector-v3-49classes
```

Les imports publics conservent désormais leur split `test`. La commande suivante a servi à
restaurer les anciens tests qui avaient été fusionnés dans `val` :

```powershell
.\.venv\Scripts\python.exe training\scripts\reserve_public_test_split.py --apply
```

Ce lot de 60 images et 3 630 boîtes est réservé aux comparaisons V4+. Il ne constitue pas
une mesure indépendante de la V3, car celle-ci l'avait déjà vu pendant sa validation.

## Modèles d'amorçage produits

Fichiers stables dans `models/` :

- `building-detector.pt` / `.onnx` : détecteur V5, inférence **800** (voir `ACTIVE.json`) ;
- `level-air-defense.pt` / `.onnx` : niveaux 8 à 11 ;
- `level-town-hall.pt` / `.onnx` : niveaux 10 à 14 ;
- `experimental/level-cannon.pt` : **non production**.

La V3 a été entraînée avec 759 images, 30 324 boîtes et 49 classes représentées. Après
restauration du split public réservé, le dataset contient 118 images/6 731 boîtes de
validation et 60 images/3 630 boîtes de test. `town-hall-guardian` est l'unique classe sans
annotation d'entraînement ; elle sera traitée dans une phase ultérieure.

## V4 finale sans gardien

La V4 a repris la V3 avec 84 captures Pedro supplémentaires et 1 560 boîtes compatibles,
pour un total de 843 images et 31 884 boîtes d'entraînement. Les troupes, sorts, éléments
d'interface et `Little_Guardian` ont été exclus. Le cycle a terminé 50 époques.

Sur la validation fixe, la V4 obtient précision 0,6729, rappel 0,3940, mAP50 0,4382 et
mAP50-95 0,3055. Sur le test réservé, elle obtient 0,6765, 0,3669, 0,4284 et 0,2880. Le
rapport complet est dans `training/reports/v4-final-report.md`. La release PT/ONNX validée
reste archivée dans `models/releases/building-detector-v4-pedro-no-guardian/` (remplacée
ensuite par V5 puis inférence 800).

## V5 promue + inférence 800

La V5 (`building-detector-v5s-distilled-640`) bat la V4 sur le test réservé. Les poids PT
sont figés (SHA-256 `866595…`). Le **4 septembre 2026**, l’inférence à **imgsz 800** a
battu 640 sur le même test (mAP50-95 0,6121 vs 0,5788) : les alias actifs pointent vers
`models/releases/building-detector-v5s-infer800/` (ONNX 800×800). Marqueur :
`models/ACTIVE.json`.

Évaluer la config active :

```powershell
.\.venv\Scripts\python.exe training\scripts\evaluate_detector.py --split val --name v5-infer800-val
.\.venv\Scripts\python.exe training\scripts\analyze_village.py C:\chemin\capture.png
.\.venv\Scripts\python.exe training\scripts\serve_compare.py
```

Classifieurs de niveaux en production : aucun. `air-defense` et `town-hall`
restent expérimentaux tant que les niveaux récents et le rejet hors distribution
ne sont pas couverts. Le canon expérimental
reste dans `models/experimental/`.

Non promus : V6 weakclass-ft, SAHI/tiling.

Pour recréer l'environnement, installer d'abord Python 3.12.10 puis :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r training\requirements-gtx1050.txt
```
