# Captures à annoter

Déposer ici des captures complètes prises en **mode photo**, sans menu ouvert et à la
résolution originale. Conserver plusieurs villages, niveaux d'hôtel de ville, zooms et
appareils différents. Ne pas recadrer les bâtiments avant l'annotation du détecteur.

Pour aider ensuite à étiqueter les niveaux, utiliser si possible un nom descriptif :

`th14-village-ami-001.png`

Les captures dont les niveaux exacts sont connus sont prioritaires. Les données publiques
amorcent le détecteur, mais des captures récentes sont indispensables pour les bâtiments et
niveaux ajoutés après leur publication.

## Campagne prioritaire

Les classes qui bloquent encore une couverture « tous les bâtiments » :

1. `wall` (quasi absent en YOLO — préférer export JSON pour soi/ami)
2. `town-hall-guardian` (**0** annotation : impossible à apprendre sans labels manuels)
3. `giga-bomb` / `tornado-trap`
4. `seeking-air-mine` / `skeleton-trap` / `air-bomb` / `spring-trap` / `bomb`
5. `hidden-tesla`
6. bâtiments TH élevés rares : `workshop`, `pet-house`, `hero-hall`, `crafted-defense`

Privilégier des villages propriétaires en mode édition/photo où ces éléments sont
réellement visibles. Une absence non vérifiable ne doit jamais être annotée comme zéro.

Après dépôt, option revue rapide (pseudo-labels **non importés**) :

```powershell
.\.venv\Scripts\python.exe training\scripts\prepare_active_learning_candidates.py
```

Une capture jointe hors de l'inbox peut être conservée explicitement :

```powershell
.\.venv\Scripts\python.exe training\scripts\prepare_active_learning_candidates.py `
  --image "C:\chemin\village.jpg" --town-hall-level 16 --local-training-consent
```

Sortie : `training/curation/active-learning/` (JSON + overlays). Annotation manuelle
exhaustive obligatoire avant tout import dataset.

Pour valider les suggestions :

```powershell
.\.venv\Scripts\python.exe training\scripts\serve_compare.py
```

Puis ouvrir `http://127.0.0.1:8765/training/annotator/`, charger l'image depuis
`sources/`, la session `.annotation-session.json` et la vérité terrain éventuelle.
Les boîtes orange sont des suggestions non acceptées. L'import final passe par
`import_annotation_session.py`, qui refuse une session partielle.

## Règles de collecte

- Obtenir l'accord du propriétaire de la capture.
- Masquer les notifications, discussions, noms réels et autres données personnelles.
- Garder l'image originale : pas de compression par messagerie ni de redimensionnement.
- Varier HDV, décor, obstacles, zoom, résolution, appareil et densité du village.
- Conserver aussi des images difficiles/négatives avec décorations ressemblant aux pièges.
- Annoter **tous** les bâtiments visibles, pas uniquement les classes faibles : une
  annotation partielle crée de faux négatifs.
- Ne jamais placer deux vues ou augmentations du même village dans des splits différents.
- Ne jamais ajouter une image de VAL/TEST au train.

## Nom et métadonnées

Nom recommandé :

`<groupe-village>__th<niveau>__<appareil>__<vue>.png`

Exemple :

`jimmy-base-01__th14__iphone14__vue-01.png`

Ajouter à côté un JSON du même nom lorsque l'information est connue :

```json
{
  "village_group": "jimmy-base-01",
  "town_hall": 14,
  "device": "iPhone 14",
  "capture_mode": "mode-photo",
  "levels_verified": true,
  "consent": true
}
```

`village_group` sert à garantir que toutes les vues du même village restent dans un seul
split. Si un niveau n'est pas vérifié, utiliser `false` et ne pas l'inventer.

## Avant import

1. contrôler visuellement la capture ;
2. dédupliquer par hash/perception ;
3. attribuer le groupe village à un seul split ;
4. annoter exhaustivement ;
5. exécuter `audit_split_integrity.py` ;
6. mesurer toute nouvelle version contre V5 + imgsz 800 + TTA.

