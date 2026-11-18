# Collecte privée élargie (locale, non publiée)

Dossier **gitignoré pour les images** : captures Mode photo / screens amis /
référence TH19 téléchargée localement. **Ne jamais pousser les binaires** sur
le remote public (ni LFS public).

Préférence Jimmy (2026-11-18) : collecte privée OK pour datasets locaux ;
ne pas redistribuer les images.

## Arborescence

```
training/inbox/private/
  README.md                          # ce fichier (versionné)
  drop/                              # déposer ici les captures brutes
  <class-id>/                        # staging annotateur (wall, spiky-tower, …)
  reference-only/th19-visual/        # blog / guides — hors train YOLO
  exports-json/                      # exports village JSON amis (dataIds)
```

## Règles

1. Licence / consentement : `consentement local — dataset privé` **ou** CC BY
   clairement sidecaré. Blog Supercell = `reference-only` seulement.
2. Pas de fine-tune tant que `check_finetune_readiness` rouge **et** sans OK Jimmy.
3. Pas d’invention de `dataId` : coller un export TH19 réel dans `exports-json/`.
4. Annotation **exhaustive** via `http://127.0.0.1:8765/training/annotator/`
   avant `import_annotation_session.py`.

## Staging

Depuis une capture déposée dans `drop/` :

```powershell
.\.venv\Scripts\python.exe training\scripts\stage_private_visual_candidates.py `
  --from-drop --class-focus wall
```

Élargir la file murs depuis imports CC BY locaux (copie privée, pas republier) :

```powershell
.\.venv\Scripts\python.exe training\scripts\stage_private_visual_candidates.py `
  --from-wall-imports --max-per-source 40
```

Référence visuelle TH19 (hors train) :

```powershell
.\.venv\Scripts\python.exe training\scripts\stage_private_visual_candidates.py `
  --fetch-th19-reference
```

## Priorités

| P0 | `spiky-tower`, `firecracker-tower`, `guardian-shrine`, `squad-barracks`, vrai `town-hall-guardian` |
| P1 | `wall` (+ val/test), `tornado-trap`, `giga-bomb` |
| P2 | workshop / pet-house / blacksmith / hero-hall / crafted-defense |

## Séparation licence

- `training/inbox/public-reference/` = manifeste CC BY C1–C2
- `training/inbox/private/` = privé / élargi — ne pas mélanger dans un même
  commit d’images
