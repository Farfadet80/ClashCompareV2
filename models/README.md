# Modèles ClashCompare

Modèles PyTorch utilisés par le pipeline local :

- `building-detector.pt` : détection des types de bâtiments ;
- `level-air-defense.pt` : niveaux 8 à 11 de la défense antiaérienne ;
- `level-town-hall.pt` : niveaux 10 à 14 de l'hôtel de ville.

Ces deux classifieurs sont conservés pour expérimentation, mais ne sont plus
allowlistés en production : leurs classes fermées ne couvrent pas les niveaux
récents et peuvent retourner un ancien niveau avec une confiance élevée.
Un niveau absent de leur liste d'entraînement ne peut pas être reconnu correctement.
`level-cannon.pt` est en quarantaine dans `models/experimental/` (dataset trop petit).

Les exports ONNX validés avec ONNX Runtime sont également présents :

- `building-detector.onnx` (entrée **800 × 800**, mêmes poids V5) ;
- `level-air-defense.onnx` (entrée 224 × 224) ;
- `level-town-hall.onnx` (entrée 224 × 224).

## Version active V5 (inférence imgsz 800)

Poids entraînés V5 (`building-detector-v5s-distilled-640`). Alias actifs :

- PT/ONNX → `models/releases/building-detector-v5s-infer800/` (inférence **800**)
- Archive entraînement 640 → `models/releases/building-detector-v5s-distilled-640/`
- V4 → `models/releases/building-detector-v4-pedro-no-guardian/`

Décision 2026-09-03 : V5 bat V4 sur le **test réservé** (imgsz 640).
Décision 2026-09-04 : **inférence 800** bat 640 sur le même test réservé (mêmes poids).

| Split | Config | Précision | Rappel | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| Val | V4 alias 512 | 0,6729 | 0,3940 | 0,4382 | 0,3055 |
| Val | V5 imgsz 640 | 0,8519 | 0,7890 | 0,8241 | 0,6022 |
| Val | V5 imgsz 800 | 0,8259 | 0,8236 | 0,8416 | 0,6247 |
| Test | V4 alias 512 | 0,6765 | 0,3669 | 0,4284 | 0,2880 |
| Test | V5 imgsz 640 | 0,7878 | 0,7915 | 0,8022 | 0,5788 |
| Test | V5 imgsz **800** | **0,8273** | **0,8281** | **0,8390** | **0,6121** |
| Test | V5 imgsz **800 + TTA** | **0,8525** | 0,8235 | **0,8573** | **0,6222** |

Parité PT/ONNX 800 (2026-09-04) : 36 détections, classes identiques, Δconf max < 0,000002.
SHA-256 PT : `866595fad39a5b7dfdf87076332faadc40a88bc55eae1b02f093d996362fb93d` (inchangé).
Inférence : `analyze_village.py` / `serve_compare.py` utilisent `--imgsz 800` et **TTA on** par défaut (`--no-tta` pour désactiver).

Non promu (sous baseline AP) : V6 weakclass-ft, SAHI/tiling (640 et 800), imgsz 704/832/896/960.
`town-hall-guardian` n'a toujours aucune annotation.

## Version V3 — 49 classes annotées

La version figée se trouve dans `models/releases/building-detector-v3-49classes/`, avec :

- le poids PyTorch versionné ;
- l'export ONNX statique 512 × 512 ;
- `release.json`, qui contient les empreintes SHA-256 et le résultat du contrôle de parité.

Le contrôle PT/ONNX donne les mêmes classes et boîtes sur l'image de référence, avec un
écart de confiance inférieur à 0,000002. La V3 atteint environ 0,318 de mAP50 sur son ancien
jeu de validation. Le lot public désormais réservé donne 0,310, mais n'est pas un test
indépendant pour la V3 puisqu'il avait été inclus dans sa validation. Il devient le benchmark
réservé des versions V4 et suivantes.

La classe `town-hall-guardian` ne possède toujours aucune annotation et ne doit pas être
présentée comme apprise. Aucun classifieur de niveau n'est actuellement allowlisté :
`analyze_village.py` n'affiche donc aucun niveau issu d'une capture.
Le marqueur de release active est `models/ACTIVE.json`.
