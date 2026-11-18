# Collecte visuelle autonome (sans captures Jimmy)

Ce dossier reçoit les **candidats publics** pour renforcer YOLO sur les classes
faibles et réserver les classes TH19. Jimmy n’a **rien** à capturer.

## Règles

1. **Licence d’abord** — uniquement CC BY / CC BY-SA / MIT / consentement auteur
   documenté. Assets blog Supercell = `reference-only`, pas le train.
2. **Pas d’entraînement ici** — staging + annotation seulement. V5 reste actif.
3. **Pas de dataId inventé** — ce pipeline est visuel ; le mapping JSON reste
   `pendingNewBuildings` jusqu’à observation d’un vrai export.
4. **Annotation exhaustive** — `training/annotator/` puis
   `import_annotation_session.py` (refuse les sessions partielles).

## Arborescence

```
training/inbox/public-reference/<class-id>/
  _provenance.template.json
  <image>.png|+ .json sidecar
training/reference-only/th19-visual/<class-id>/   # guides / blog, hors train
```

## Préparer l’inbox

```powershell
.\.venv\Scripts\python.exe training\scripts\stage_public_visual_collection.py
.\.venv\Scripts\python.exe training\scripts\stage_public_visual_collection.py --check-only
.\.venv\Scripts\python.exe training\scripts\stage_priority_visual_candidates.py
```

Le script `stage_priority_visual_candidates.py` copie les images CC BY locales
(déjà dans `training/imports/`) vers `<class-id>/` avec :
- `*.provenance.json` (licence, SHA, source)
- `*.session.json` brouillon annotateur (`exhaustive: false`, boîtes `pending`)

Les binaires `.jpg/.png` de l’inbox sont gitignorés (recréables). Les JSON restent
versionnés. Rapport C1–C2 :
`training/reports/visual-collection-c1c2-20261118.json`

Manifeste sources / licences :
`training/sources/visual-collection-manifest.json`

Registre Roboflow déjà connu :
`training/sources/public-datasets.json`

## Flux recommandé (agent / machine locale)

1. Lire le manifeste → cibler `very_weak` + `reserved_th19_yolo`.
2. Pour une source CC BY Roboflow : télécharger YOLO → `training/imports/<key>/`.
3. `audit_public_dataset.py` → blockers vides.
4. `stage_priority_visual_candidates.py` → inbox + brouillons.
5. Revue humaine exhaustive (annotator) ; accepter/rejeter chaque boîte ; cocher exhaustive.
6. `import_annotation_session.py` **ou**
   `import_public_dataset.py ... --train-only --confirm-exhaustive-review` seulement après revue.
7. `audit_split_integrity.py` + `check_finetune_readiness.py`.
8. **Stop** : ne lancer un fine-tune que sur OK Jimmy + readiness vert + mesure vs V5.

## Seuils avant de demander un fine-tune

| Objectif | Minimum |
|---|---|
| Nouvelle classe TH19 | ≥ 20 boîtes acceptées, ≥ 8 villages indépendants |
| Classe très faible (`wall`, pièges rares, guardian) | ≥ 20 boîtes train + présence VAL |
| Promotion | reserved test ≥ baseline V5 800+TTA |

## Affichage app

Les détections issues uniquement de la capture apparaissent comme
**« Reconnu visuellement (YOLO) »**. L’export JSON reste prioritaire.
