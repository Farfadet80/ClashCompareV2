# Guide d’annotation — classes rares / absentes

Mise à jour : 2026-09-05. Ce document sert à reconnaître visuellement les classes,
pas à fabriquer des annotations. Une boîte n’est créée que si l’objet est réellement
visible sur la capture.

## Limite fondamentale des captures adversaires

- Les pièges cachés et la Tesla cachée ne sont normalement pas visibles avant activation.
- Une capture d’adversaire ne permet donc pas de prouver leur absence ni leur quantité.
- Pour ces classes, collecter le **mode édition / mode photo du propriétaire**.
- Le `Giga Bomb` constitue une exception utile : Supercell le décrit comme toujours visible.
- Dans l’application, une classe non visible reste `Non détecté` ; aucun remplissage par
  quantité théorique, position probable ou niveau d’HDV.

Source officielle :

- TH17 / Giga Bomb :
  https://supercell.com/en/games/clashofclans/blog/game-updates/the-town-hall-17-update-is-here-2/

## Classes catalogue export-only — `bob-hut` / `helper-hut`

- `bob-hut` : cabane de B.O.B, `dataId` export vérifié `1000064`, niveau 1.
- `helper-hut` : cabane des assistants, `dataId` export vérifié `1000093`,
  niveau 1.
- Elles portent le catalogue applicatif à 52 entrées, mais ne sont pas ajoutées
  artificiellement aux 50 classes YOLO sans annotations.
- V5 peut confondre la cabane de B.O.B avec `builders-hut` : l'export HDV16 réel
  contient 5 cabanes d'ouvrier + 1 cabane de B.O.B, tandis que V5 a annoncé
  6 `builders-hut`.
- Références :
  - données statiques `coc.py` :
    https://github.com/mathsman5133/coc.py
  - Supercell, cabane des assistants :
    https://support.supercell.com/clash-of-clans/en/articles/helpers-2.html

## Classe 40 — `town-hall-guardian` (0 annotation)

Noms officiels à rechercher : `Guardian`, `Smasher`, `Longshot`, `Logger`.

- Débloqués à l’HDV 18 ; un seul Guardian défend à la fois.
- `Smasher` : Guardian de mêlée, grand personnage avec masse.
- `Longshot` : Guardian à distance.
- `Logger` : Guardian à distance ajouté en avril 2026, armé d’un rondin ;
  cinq niveaux confirmés par l’annonce officielle.
- Annoter le Guardian visible, **pas** l’HDV complet.
- Garder une seule classe agrégée pour l’instant ; conserver le nom exact
  `smasher` / `longshot` dans les métadonnées de l’image.
- Ne pas auto-étiqueter une zone autour d’un HDV 18 : les planches
  `training/curation/guardians/` ne sont que des candidats de revue.

Sources :

- Supercell TH18 :
  https://supercell.com/en/games/clashofclans/blog/release-notes/town-hall-18-crash-lands-update/
- Supercell Logger :
  https://supercell.com/en/games/clashofclans/blog/release-notes/the-sound-of-clash-update/
- Wiki Guardians :
  https://clashofclans.fandom.com/wiki/Town_Hall/Guardians

## Classe 19 — `wall` (12 boîtes train, aucune VAL/TEST)

- Une annotation détection doit représenter **une pièce de mur physique**, pas le
  réseau complet ni une ligne entière.
- Varier les niveaux, thèmes d’HDV, orientations isométriques et murs en cours
  d’amélioration.
- Source publique vérifiée : `CoC Wall Detection`, 50 images, CC BY 4.0 :
  https://universe.roboflow.com/kruegerp-stu-proton-me/coc-wall-detection
- Cette source est en **instance segmentation de réseaux de murs**. Elle ne doit pas
  passer directement dans `import_public_dataset.py` : la boîte englobante d’un masque
  de réseau serait un faux exemple pour le détecteur de pièces.
- Usage envisagé : pipeline segmentation séparé, ou réannotation manuelle pièce par
  pièce. Le téléchargement Roboflow demande une connexion.

## Classes 48–49 — pièges rares

### `tornado-trap`

- Un seul exemplaire peut exister ; il est caché en attaque.
- Collecte prioritaire en mode édition/photo propriétaire.
- Ne pas utiliser de datasets `Tornado` issus de Clash Royale ou d’autres jeux.
- Référence : https://clashofclans.fandom.com/wiki/Tornado_Trap

### `giga-bomb`

- Piège HDV 17, toujours visible, grosse zone et fort recul selon Supercell.
- Un seul exemplaire ; ne pas confondre avec `giant-bomb`.
- Références :
  - https://supercell.com/en/games/clashofclans/blog/game-updates/the-town-hall-17-update-is-here-2/
  - https://clashofclans.fandom.com/wiki/Giga_Bomb

## Pièges faibles mais annotés

Classes : `bomb`, `spring-trap`, `air-bomb`, `giant-bomb`,
`seeking-air-mine`, `skeleton-trap`.

- Niveaux maximaux recoupés au 5 septembre 2026 :
  `bomb` 14, `spring-trap` 13, `air-bomb` 13, `giant-bomb` 12,
  `seeking-air-mine` 8, `skeleton-trap` 5, `tornado-trap` 3,
  `giga-bomb` 4.
- Photographier la vue propriétaire où tous les pièges sont visibles.
- Ne pas utiliser une capture d’attaque pour produire des annotations négatives.
- Vérifier les confusions `bomb` / décorations, `giant-bomb` / `giga-bomb`,
  `air-bomb` / `seeking-air-mine`.
- Annoter exhaustivement les autres bâtiments visibles dans la même image afin de ne
  pas créer de faux négatifs.

## Bâtiments récents à diversifier

### `revenge-tower`

- Nouvelle défense HDV 18, devient plus forte lorsque des bâtiments sont détruits.
- Source officielle :
  https://supercell.com/en/games/clashofclans/blog/release-notes/town-hall-18-crash-lands-update/

### `multi-gear-tower`

- Défense HDV 17 confirmée par Supercell.
- Source officielle :
  https://supercell.com/en/games/clashofclans/blog/release-notes/welcome-to-clash-anytime-update/

### `crafted-defense`

- Classe agrégée pour des défenses temporaires qui changent par phase.
- Phase 1 : `Hook Tower`, `Flame Spinner`,
  `Crusher Mortar`.
- Phase 2 : `Light Beam`, `Hero Bell`, `Bomb Hive`.
- Phase 3 : `Roaster`, `Air Bombs`, `Lava Launcher`.
- Phase 4 active au 5 septembre 2026 : `Cake-A-Pult`, `Hero Hunter`,
  `Hot Candle`.
- Stocker le nom de variante dans les métadonnées même si YOLO conserve la classe
  agrégée.
- Sources :
  - https://supercell.com/en/games/clashofclans/blog/release-notes/welcome-to-lets-get-crafty-update/
  - https://supercell.com/en/games/clashofclans/blog/release-notes/town-hall-18-crash-lands-update/
  - https://supercell.com/en/games/clashofclans/blog/news/its-the-sound-of-clash/
  - https://supercell.com/en/games/clashofclans/blog/news/the-awesome-quest-is-here/

### `hero-hall`, `workshop`, `pet-house`, `blacksmith`

- Classes réelles déjà apprises mais avec peu de diversité.
- Chercher des captures village complètes récentes ; éviter les icônes, pages wiki
  détourées et écrans de menu comme données du détecteur.

## Sources examinées mais non retenues

- `MyCOCHere` : 407 images / 57 libellés, CC BY 4.0 ; bâtiments standards et
  Tesla, mais aucun mur, Guardian, piège rare ou défense TH18 utile.
- Kaggle `Clash Of Clan Object/Item Detection` : miroir exact de
  `targetArch v6` / WalkStation (800 images), déjà rejeté.
- Find This Base / miroir Hugging Face : défenses TH13, déjà importé ; aucune
  classe rare ciblée.
- WalkStation : annotations corrompues / classes parasites, rejet maintenu.
- Datasets Clash Royale et Clan Capital : domaine visuel incorrect.
- Wiki/Fandom et pages Supercell : références d’identification seulement, pas
  importées automatiquement comme dataset d’entraînement.

## Gate avant apprentissage

1. Revue humaine et licence explicite.
2. Déduplication hash + perceptuelle.
3. Un village et ses variantes dans un seul split.
4. VAL/TEST actuels inchangés ; nouveaux exemples d’évaluation séparés si besoin.
5. Mesure contre V5, `imgsz=800`, TTA activée.
6. Aucune promotion si mAP50 et mAP50-95 ne battent pas la baseline comparable.
