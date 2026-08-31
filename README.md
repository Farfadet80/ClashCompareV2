# ClashCompare V2

Contenu :
- PWA GitHub Pages
- récupération de vraies données joueur
- profils Joueur A / Joueur B
- comparaison HDV, trophées, XP, étoiles de guerre, dons
- héros
- estimation de progression des troupes et sorts disponibles dans l'API
- Cloudflare Worker séparé pour protéger le token API

## Fichiers à mettre à la racine de ton dépôt GitHub

index.html
style.css
app.js
config.js
manifest.json
service-worker.js
icons/

Le dossier `cloudflare-worker/` n'a pas besoin d'être servi par GitHub Pages. Il contient le code à copier dans Cloudflare.

## Mise à jour depuis V1

Tu peux remplacer les anciens fichiers par ceux de la V2.

IMPORTANT : configure ensuite `config.js` avec l'URL de ton Worker.

Lis :
cloudflare-worker/README-CLOUDFLARE.md

## Limite actuelle

L'endpoint joueur Clash of Clans fournit beaucoup de données (profil, troupes, héros, sorts, etc.), mais pas la liste complète des bâtiments et murs du village principal. Cette partie demandera donc une autre méthode.
