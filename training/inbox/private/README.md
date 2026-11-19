# Collecte privée élargie (locale, non publiée)

Dossier **gitignoré pour les images** : captures Mode photo / screens amis /
référence TH19 téléchargée localement. **Ne jamais pousser les binaires** sur
le remote public (ni LFS public). Les `.session.json` / `.provenance.json`
peuvent être versionnés (annotation sans image).

Préférence Jimmy (2026-11-18) : collecte privée OK pour datasets locaux ;
ne pas redistribuer les images. **Focus TH10–18** ; TH19 (dataIds + classes
YOLO 50–53) **à la fin**.

## Arborescence

```
training/inbox/private/
  README.md                          # ce fichier (versionné)
  drop/                              # déposer ici les captures brutes
  <class-id>/                        # staging annotateur (wall, workshop, …)
  p0-mode-photo/                     # P0 assist V5 (exhaustive=false) wall/tornado/giga/crafted
  p0-tesla-hardneg/                  # FP hidden-tesla documentés (rejected)
  p0-wall-tiles/                     # tuiles mur Mode photo (revue visuelle stricte)
  reference-only/th19-visual/        # blog / guides — hors train YOLO
  exports-json/                      # exports village JSON amis (dataIds)
```

## P0 pré-train (2026-11-19)

1. Assist villages : `assist_fill_village_sessions.py` → `p0-mode-photo/`
2. Ne **jamais** `exhaustive=true` sans revue annotator + `promote_p0_assisted_sessions.py --i-reviewed`
3. Hard-neg tesla : `mine_tesla_hard_negatives.py` (pas baisser conf inférence)
4. Val/test wall+giga : `rebalance_weak_val_test.py` (dataset local gitignoré)
5. Synthèse : `training/reports/p0-priority-pass-summary-20261119.json`

## Règles

1. Licence / consentement : `consentement local — dataset privé` **ou** CC BY
   clairement sidecaré. Blog Supercell = `reference-only` seulement.
2. Pas de fine-tune tant que `check_finetune_readiness` rouge **et** sans OK Jimmy.
3. Pas d’invention de `dataId` : coller un export TH19 réel dans `exports-json/`.
4. Annotation **exhaustive** via `http://127.0.0.1:8765/training/annotator/`
   avant `import_annotation_session.py` (villages). Crops isolés :
   `promote_isolated_crop_sessions.py`.
5. Ignorer classes YOLO `spiky-tower` / `firecracker-tower` /
   `guardian-shrine` / `squad-barracks` jusqu’à la phase TH19.

## Staging

Depuis une capture déposée dans `drop/` :

```powershell
.\.venv\Scripts\python.exe training\scripts\stage_private_visual_candidates.py `
  --from-drop --class-focus wall
```

Élargir la file murs depuis imports CC BY locaux (copie privée, pas republier) :

```powershell
.\.venv\Scripts\python.exe training\scripts\stage_private_visual_candidates.py `
  --from-wall-imports --max-per-source 60
```

Élargir P1/P2 TH10–18 (tornado / giga / workshop / pet-house / …) :

```powershell
.\.venv\Scripts\python.exe training\scripts\stage_private_visual_candidates.py `
  --from-priority-imports --max-per-class 40
```

Référence visuelle TH19 (hors train, fin de roadmap) :

```powershell
.\.venv\Scripts\python.exe training\scripts\stage_private_visual_candidates.py `
  --fetch-th19-reference
```

## Priorités (TH10–18)

| P1 | `wall` (+ val/test), `tornado-trap`, `giga-bomb`, vrai `town-hall-guardian` |
| P2 | workshop / pet-house / blacksmith / hero-hall / crafted-defense |
| Fin | classes YOLO 50–53 + dataIds TH19 (hors scope actuel) |

## Séparation licence

- `training/inbox/public-reference/` = manifeste CC BY C1–C2
- `training/inbox/private/` = privé / élargi — ne pas mélanger dans un même
  commit d’images
