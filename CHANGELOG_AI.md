# CHANGELOG_AI — suivi Cursor ↔ ChatGPT

Journal des modifications faites par l’agent Cursor.  
À lire **avec** `HANDOFF.md` (état technique) et `PASSATION-CHATGPT.md` (vision produit).  
Règle : chaque entrée note **quoi / pourquoi / décision**, jamais de secrets.

Format d’entrée :

```
## YYYY-MM-DD — titre court
- Branche / commits :
- Fait :
- Décisions :
- À faire ensuite :
```

---

## 2026-09-05 — V8 Stage1 échouée ; V5 conservée

- Branche / commits :
  - `feature/village-json-import`
  - `0540377` Add conservative fine-tune controls and promotion gate
  - `3fedad2` Expose conservative V8 augmentation controls
  - Tag : `savepoint-before-v8-process-2026-09-05`
- Fait :
  - Protocole V8 : départ V5 promu, Stage1 backbone gelé, LR AdamW `5e-5`, warmup 0, mosaic 0, augmentations légères, 4 epochs.
  - Stage1 VAL 640 interne : meilleur epoch 3 (mAP50-95 0,589) ; epoch 4 en recul.
  - Stage1 VAL 800+TTA : P **0,855** / R **0,817** / mAP50 **0,847** / mAP50-95 **0,619**.
  - Gate vs V5 : **échoué** (mAP50 −0,006 ; mAP50-95 −0,013).
- Décisions :
  - **Pas de Stage2**, pas de TEST réservé, **pas de promotion**.
  - Arrêt des sweeps d’entraînement : le goulot est la **données** (pièges/Teslas Mode photo).
  - V5 `building-detector-v5s-infer800` reste le modèle actif.
- À faire ensuite :
  - Collecte/annotation Mode photo ciblée.
  - Garder le gate `gate_detector_candidate.py` pour tout futur candidat.

---

## 2026-09-05 — Suivi AI + fine-tune V7 propre

- Branche / commits :
  - `feature/village-json-import`
  - `c53680b` Add local village JSON import
  - `0c2dc44` Support fine-tuning distilled checkpoints
  - Tag savepoint : `savepoint-before-village-json-import-2026-09-04`
- Fait :
  - Créé ce fichier `CHANGELOG_AI.md` pour le suivi ChatGPT.
  - Confirmé limite produit : le JSON officiel n’est récupérable **que** par le propriétaire du compte (pas via tag # d’un adversaire). YOLO reste indispensable pour les villages tiers.
  - Dataset assaini : **931** images train (68 overlaps exclus, non détruits).
  - Baseline VAL reproduite `imgsz 800 + TTA` : P 0,870 / R 0,826 / mAP50 **0,853** / mAP50-95 **0,632**.
  - Fine-tune `building-detector-v7-clean-ft2-20260905` lancé depuis élève de `epoch79.pt` (checkpoint distillé extrait, original intact). Prévu 20 epochs, arrêté proprement à **9/20** par early-stop patience 8.
  - Meilleur checkpoint interne : epoch 1. Évaluation comparable VAL `800+TTA` : P **0,855** / R **0,837** / mAP50 **0,857** / mAP50-95 **0,629**.
  - Baseline V5 VAL `800+TTA` : P **0,870** / R **0,826** / mAP50 **0,853** / mAP50-95 **0,632**.
  - V7 améliore le rappel (+0,010), mAP50 (+0,004) et les pièges sur VAL (`skeleton-trap` +0,064 ; `air-bomb` +0,036 ; `spring-trap` +0,028), mais baisse précision (−0,015) et mAP50-95 globale (−0,003).
- Décisions :
  - Ne pas importer WalkStation (annotations corrompues / classes parasites).
  - Ne pas entraîner sur icônes `clash-of-clans-data` (pas des captures village).
  - **V7 non promue** : elle ne bat pas V5 sur les critères globaux VAL. TEST réservé volontairement non consulté.
  - Ne pas upgrader Python / PyTorch / CUDA (GTX 1050 sm_61).
- À faire ensuite :
  - Collecter captures Mode photo (pièges / Teslas) — vrai levier YOLO.
  - Conserver V5 comme modèle actif.

---

## 2026-09-04 — Import JSON village + hygiène splits

- Branche / commits :
  - `feature/village-json-import` (puis travail local)
  - Splits : `4a44cc9`, `3429dde`
  - Collecte Photo Mode : `5ea8fce`
- Fait :
  - Parseur `village-export.js` + mapping `data/coc-export-mapping.json`.
  - UI A/B : coller / fichier JSON ; priorité export > YOLO ; `localStorage` tags+export.
  - Tests `training/scripts/test_village_export.py` OK ; E2E PWA OK.
  - Audit fuites train↔val/test → `train-clean.txt` (931) + garde dans `import_public_dataset.py`.
- Décisions :
  - JSON = inventaire propriétaire / ami qui partage.
  - YOLO = fallback / villages sans JSON.
  - Architecture prod (cloud / local / ONNX navigateur) toujours **non tranchée**.
- À faire ensuite :
  - Fine-tune propre (démarré 2026-09-05).
  - UX maxage / icônes / API profil plus tard.

---

## Convention pour Cursor

1. Après chaque lot de travail notable : mettre à jour **ce fichier** + la section état de `HANDOFF.md`.
2. Commit Git dédié ou inclus dans le commit de feature (pas de secrets, pas de dumps JSON joueur).
3. Ne pas remplacer `HANDOFF.md` : `CHANGELOG_AI` = chronologie ; `HANDOFF` = vérité technique actuelle.
4. Si contradiction ChatGPT ↔ fichiers : **fichiers + Git gagnent** ; noter la question dans une entrée.
