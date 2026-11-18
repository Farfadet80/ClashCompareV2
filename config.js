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
