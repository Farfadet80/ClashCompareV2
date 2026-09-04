# Modèles expérimentaux

Ces poids ne sont **pas** chargés par `analyze_village.py` / `serve_compare.py`.

## `level-cannon.pt`

- Classifieur de niveaux 1–21 entraîné sur un dataset minuscule (~117 train / 36 val).
- Top-1 interne ~75 % — **non fiable** pour la production.
- Remis en quarantaine le 2026-09-04 : seuls `air-defense` et `town-hall` restent dans l’allowlist de production.
- Pour le réactiver : enrichir les crops, réentraîner, battre une baseline claire sur un split test, puis l’ajouter explicitement à `PRODUCTION_LEVEL_CLASSIFIERS`.
