# Format d'annotation du détecteur

Chaque image `.jpg`/`.png` dans `dataset/detector/images/train` possède un fichier `.txt`
du même nom dans `dataset/detector/labels/train`.

Format YOLO par ligne :

`class_id x_center y_center width height`

Les coordonnées sont normalisées entre 0 et 1.

Exemple :
`1 0.512 0.424 0.061 0.079`

Utilise `classes.json` pour faire correspondre l'identifiant de classe au bâtiment.

Pour le niveau, place ensuite le crop du bâtiment dans :
`dataset/levels/<building-id>/level-<n>/`

Exemple :
`dataset/levels/cannon/level-21/cannon_001.jpg`
