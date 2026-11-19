# Modèles ClashCompare

Modèles PyTorch utilisés par le pipeline local :

- `building-detector.pt` : détection des types de bâtiments ;
- `level-air-defense.pt` : niveaux 8 à 11 de la défense antiaérienne ;
- `level-town-hall.pt` : niveaux 1 à 18 de l'hôtel de ville, actif avec seuil 0,92.

Le classifieur de l'hôtel de ville est allowlisté en production. Il obtient
140/141 (99,3 %) sur le test réservé TH4–TH18 et 3/3 sur les contrôles séparés
TH1–TH3. Son seuil 0,92 rejette l'unique erreur du test réservé : 138/141
prédictions sont affichées avec 100 % de justesse parmi les prédictions acceptées.
L'ancien modèle TH10–TH14 est archivé sous `models/experimental/`.
Le classifieur de défense antiaérienne reste expérimental : sa plage est incomplète.
`level-cannon.pt` est en quarantaine dans `models/experimental/` (dataset trop petit).
Le candidat `level-wall` est également conservé dans `models/experimental/` :
98,4 % sur un test indépendant de 372 crops, mais seulement pour les niveaux
1–6, 16 et 17. Il n'est donc pas chargé en production.

Les exports ONNX validés avec ONNX Runtime sont également présents :

- `building-detector.onnx` (entrée **800 × 800**, poids actifs FT1) ;
- `level-air-defense.onnx` (entrée 224 × 224) ;
- `level-town-hall.onnx` (entrée 224 × 224).

## Version active FT1 (inférence imgsz 800 + TTA)

Poids : fine-tune depuis V5 (`building-detector-v5-th1018-ft1`, best epoch 35). Alias actifs :

- PT/ONNX → `models/releases/building-detector-v5-th1018-ft1-infer800/` (inférence **800**)
- Précédent V5 → `models/releases/building-detector-v5s-infer800/`
- Archive entraînement V5 640 → `models/releases/building-detector-v5s-distilled-640/`

Décision 2026-11-19 : FT1 bat V5 sur le **même test réservé actuel**, imgsz **800 + TTA**.

| Split | Config | Précision | Rappel | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| Test actuel | V5 800+TTA | 0,7431 | 0,7658 | 0,7523 | 0,5416 |
| Test actuel | **FT1 800+TTA** | **0,8210** | **0,7826** | **0,8053** | **0,5800** |

SHA-256 PT FT1 : `a06eebc79506788fbbbb6ed054faf649a65318afcc65840d00f8373e4a7b6fc5`.
Inférence : `analyze_village.py` / `serve_compare.py` utilisent `--imgsz 800` et **TTA on** par défaut (`--no-tta` pour désactiver).

### Historique V5 (ancien test réservé, 2026-09-04)

| Split | Config | Précision | Rappel | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| Test | V5 imgsz **800 + TTA** | **0,8525** | 0,8235 | **0,8573** | **0,6222** |

Non promu (sous baseline AP) : V6 weakclass-ft, SAHI/tiling (640 et 800), imgsz 704/832/896/960, V9 coc-analyzer-ft.
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
présentée comme apprise. Le classifieur de niveau `town-hall` est allowlisté ;
les autres niveaux restent inconnus tant que leurs modèles ne passent pas les mêmes contrôles.
Le marqueur de release active est `models/ACTIVE.json`.
