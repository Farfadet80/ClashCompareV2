# ClashCompare V3

PWA iPhone de comparaison de villages Clash of Clans.

## V3
- # joueur conservé
- Joueur A / Joueur B
- Import d'une capture complète du village
- Conseil sous le bouton d'import :
  Modifier le village -> Mode photo -> capture complète
- Aperçu de l'image
- Vérification de la résolution de la capture
- Onglet Comparatif
- PWA installable depuis Safari

## Important
Cette version met en place l'interface et le flux d'import/analyse.
Elle ne prétend pas encore reconnaître de façon fiable tous les bâtiments et niveaux :
le moteur de vision devra être ajouté ensuite.


## V3.1 — bibliothèque de reconnaissance
- Catalogue JSON des bâtiments du village principal
- Arborescence `references/` par bâtiment et niveau
- Emplacements prêts pour plusieurs images de référence
- Compteur du catalogue dans l'interface
- Mention fan/non officielle ajoutée

Les images de référence ne sont volontairement pas redistribuées dans le ZIP. Utilise des captures/recadrages que tu es autorisé à employer.

## V3.2 — catalogue au 31 août 2026
Catalogue étendu avec les catégories visuelles récentes du village principal, les pièges,
les défenses fabriquées, le gardien d'HDV et les états de supercharge.
La mise à jour du 31/08/2026 (supercharges Cabane d'ouvrier + Monolithe) est prise en compte.


## V3.3 — AI Training Ready
- Pipeline en 2 étapes : détection du type de bâtiment puis classification du niveau
- Structure de dataset YOLO prête
- `classes.json` généré depuis le catalogue V3.2
- Notebook Google Colab inclus
- Export ONNX prévu pour exécution dans le navigateur
- Dossier `models/` prêt à recevoir le modèle entraîné

### Pourquoi 2 étapes ?
Détecter directement des centaines de combinaisons bâtiment+niveau est beaucoup moins robuste.
ClashCompare détecte d'abord `cannon`, `archer-tower`, etc., puis un classifieur dédié estime
le niveau du crop détecté.
