# Modèles expérimentaux

Ces poids ne sont **pas** chargés par `analyze_village.py` / `serve_compare.py`.

## `level-cannon.pt`

- Classifieur de niveaux 1–21 entraîné sur un dataset minuscule (~117 train / 36 val).
- Top-1 interne ~75 % — **non fiable** pour la production.
- Remis en quarantaine le 2026-09-04.
- `air-defense` reste hors production : sa plage de niveaux est incomplète.
- `town-hall` a depuis été remplacé par le classifieur TH1–TH18 validé et
  promu dans `models/level-town-hall.*`.
- Pour le réactiver : enrichir les crops, réentraîner, battre une baseline claire sur un split test, puis l’ajouter explicitement à `PRODUCTION_LEVEL_CLASSIFIERS`.

## Candidat murs `level-wall-v2-9classes`

- Poids figés : `models/experimental/level-wall.pt` et
  `models/experimental/level-wall.onnx` (entrée 224 × 224).
- Validation : 97,09 % au meilleur point (époque 5), après suppression de 66
  doublons exacts et séparation des captures source entre train/validation.
- Test indépendant : 375/383, soit 97,91 %, sur les niveaux 1–7, 16 et 17.
- Niveau 7 : 11/11 corrects sur le test réservé.
- Avec un seuil de 0,87 : 99,46 % de précision sur 97,39 % des tests acceptés.
- Parité PT/ONNX : 32/32 prédictions identiques, écart de confiance maximal
  inférieur à 0,000001.
- Hors production : les niveaux 8–15, 18 et 19 sont absents et ne peuvent pas
  être rejetés correctement par ce classifieur fermé.
