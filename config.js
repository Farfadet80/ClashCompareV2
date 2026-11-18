// ClashCompare
// Inventaire prioritaire : export JSON officiel du jeu (import local, jamais uploadé).
// Le # joueur reste dans l'interface pour un futur enrichissement profil via API CoC
// (héros, lab, ligue, équipements — jamais l'inventaire bâtiment/mur/piège).
// YOLO reste un complément / fallback local.
//
// Stub API profil :
// - Laisser CLASHCOMPARE_API = "" (normal, pas de clé dans le dépôt).
// - Plus tard : URL d'un backend proxy qui appelle l'API officielle Supercell
//   avec une clé stockée hors repo (jamais committer de token).
// - Voir docs dans HANDOFF.md / CHANGELOG_AI.md (section API CoC).
window.CLASHCOMPARE_API = "";

// Scope de reconnaissance Home Village : TH min inclus → TH max catalogue.
// Jimmy : pas d'export TH19 requis ; viser tous les bâtiments à partir de TH10.
// Les dataIds non sourcés restent « Non détecté » (jamais inventés).
window.CLASHCOMPARE_RECOGNITION_MIN_TH = 10;
