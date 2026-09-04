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

Les classes qui limitent actuellement V5 sont :

1. `seeking-air-mine`
2. `skeleton-trap`
3. `air-bomb`
4. `spring-trap`
5. `bomb`
6. `hidden-tesla`
7. `tornado-trap`

Privilégier des villages propriétaires en mode édition/photo où ces éléments sont
réellement visibles. Une absence non vérifiable ne doit jamais être annotée comme zéro.

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

