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
  - Fine-tune `building-detector-v7-clean-ft2-20260905` lancé depuis élève de `epoch79.pt` (checkpoint distillé extrait, original intact). 20 epochs, batch 2, `cls_pw=0.25`, cos LR, distillation V5.
  - Epochs 1–5 : meilleur mAP50-95 provisoire = **epoch 1 (0,588)** ; epochs 2–4 en baisse puis remontée à l’epoch 5 (0,577).
- Décisions :
  - Ne pas importer WalkStation (annotations corrompues / classes parasites).
  - Ne pas entraîner sur icônes `clash-of-clans-data` (pas des captures village).
  - Ne pas promouvoir V7 sans battre baseline VAL puis TEST `800+TTA`.
  - Ne pas upgrader Python / PyTorch / CUDA (GTX 1050 sm_61).
- À faire ensuite :
  - Finir V7 (ou early-stop patience 8).
  - Évaluer meilleur checkpoint en VAL 800+TTA vs baseline.
  - TEST réservé seulement si VAL gagne.
  - Collecter captures Mode photo (pièges / Teslas) — vrai levier YOLO.

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
