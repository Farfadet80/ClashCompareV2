# PASSATION COMPLÈTE CHATGPT → CURSOR — CLASHCOMPARE

Ce message complète `HANDOFF.md`.  
**Ne repars pas de zéro.** Le repo, `HANDOFF.md`, les poids YOLO, datasets, métriques/evals et l'état réel des fichiers locaux restent la source de vérité lorsqu'un détail technique diffère de ce message.

## 1. OBJECTIF GLOBAL

ClashCompare doit permettre de **comparer facilement deux comptes/villages Clash of Clans**.

L'utilisateur entre deux tags joueurs :

`#JOUEUR_A`  
`#JOUEUR_B`

Les tags doivent être **modifiables directement dans l'application** afin que n'importe quel utilisateur puisse comparer les comptes qu'il souhaite.

Il ne s'agit donc pas d'une application codée uniquement pour deux comptes précis.

**Ne jamais demander/utiliser les identifiants Supercell ID des joueurs.**

L'identification des comptes doit se faire via les **tags publics Clash of Clans** et les données accessibles légalement/publiquement.

---

## 2. INTERFACE VOULUE

L'application doit être agréable visuellement, dans l'univers Clash of Clans, tout en restant simple.

Il faut au minimum trois vues principales :

**Joueur A**

Récapitulatif détaillé du village/compte A.

**Joueur B**

Même chose pour le joueur B.

**Comparaison**

Comparaison directe A ↔ B.

Les petites **images/icônes des bâtiments** doivent être utilisées lorsque c'est possible pour rendre la comparaison beaucoup plus lisible qu'un simple tableau de texte.

L'interface doit être utilisable aussi bien sur ordinateur que sur téléphone.

---

## 3. MOBILE / IPHONE

Jimmy utilise notamment un **iPhone 14**.

Dès le début du projet, l'objectif était de pouvoir utiliser ClashCompare comme une application sur iPhone **sans forcément payer un compte Apple Developer**.

La direction retenue est donc notamment une **PWA** :

Safari → Partager → Ajouter à l'écran d'accueil.

Le projet contient déjà la partie PWA ; ne pas la supprimer inutilement.

GitHub Pages avait notamment été envisagé comme solution gratuite d'hébergement, mais aucun déploiement définitif n'est confirmé actuellement.

---

## 4. COMPARAISON FINALE SOUHAITÉÉ

À terme, ClashCompare doit comparer le plus précisément possible :

- HDV
- bâtiments
- défenses
- murs
- héros
- troupes
- sorts
- familiers
- équipements lorsqu'ils sont disponibles
- progression générale du compte

Pour les bâtiments, l'objectif est notamment d'obtenir des comparaisons du genre :

**Canons**

Joueur A :
- 3 × niveau 20
- 4 × niveau 21

Joueur B :
- 1 × niveau 20
- 6 × niveau 21

Même logique pour les autres bâtiments lorsque les données peuvent être déterminées de façon fiable.

---

## 5. POURCENTAGE DE MAXAGE

Une fonctionnalité importante voulue depuis le début est une estimation du type :

**Village A : 82 % max pour son HDV**

**Village B : 91 % max pour son HDV**

Ce pourcentage doit être calculé par rapport au **niveau d'HDV du joueur**, pas simplement par rapport au maximum absolu du jeu.

L'application doit également pouvoir expliquer **ce qu'il reste à améliorer**.

Exemple :

- défenses : 94 %
- murs : 78 %
- héros : 88 %
- troupes : 96 %
- progression globale : 89 %

Les valeurs ne doivent évidemment pas être inventées : le calcul doit être basé uniquement sur des données réellement récupérées/détectées.

---

## 6. API OFFICIELLE CLASH OF CLANS

L'API officielle Clash of Clans doit être utilisée pour récupérer tout ce qu'elle permet d'obtenir proprement à partir d'un tag joueur.

Mais une limitation fondamentale avait été identifiée :

**l'API officielle ne fournit pas l'état complet du village, notamment la liste/niveau de tous les bâtiments et murs.**

C'est précisément pour cette raison que le projet s'est orienté vers une deuxième source d'information :

**analyse visuelle / screenshots + YOLO.**

Ne tente donc pas de remplacer toute la partie vision par l'API CoC en supposant que l'API donne les bâtiments.

Actuellement, d'après l'état constaté :

`CLASHCOMPARE_API = ""`

L'API est **prévue mais pas encore branchée**.

Aucune clé API CoC fiable n'est actuellement fournie dans cette passation.

Ne mets jamais de clé/token directement dans un dépôt public.

---

## 7. ARCHITECTURE VISÉE

À terme, le fonctionnement doit être hybride :

**API Clash of Clans**
→ informations disponibles officiellement.

**Vision / YOLO**
→ informations que l'API ne fournit pas, notamment l'analyse visuelle du village.

**ClashCompare**
→ fusion des informations.

Puis :

**calcul progression / comparaison A ↔ B / interface utilisateur.**

Ne fais pas dépendre une donnée de YOLO lorsqu'elle est déjà disponible de manière plus fiable via l'API.

---

## 8. ÉTAT YOLO À CONSERVER

D'après l'état actuel transmis :

**V5**
- entraînement terminé : **80/80**
- version promue
- baseline actuelle

**Inference**
- `800`
- **TTA actif**

Des essais ultérieurs ont été effectués :

**V6 / SAHI / cannon**

Ils **n'ont pas été promus**.

Ne considère donc pas automatiquement qu'une version plus récente numériquement est meilleure.

La V5 promue reste la référence tant qu'une nouvelle version n'obtient pas de meilleurs résultats mesurables.

---

## 9. RÈGLE ABSOLUE POUR YOLO

**NE PAS RECOMMENCER L'ENTRAÎNEMENT DE ZÉRO.**

Avant tout nouvel entraînement :

1. inspecter les runs existants ;
2. identifier `best.pt`, `last.pt` et les checkpoints ;
3. inspecter les datasets ;
4. regarder les métriques/evals existantes ;
5. comprendre pourquoi V5 a été promue ;
6. comprendre pourquoi V6/SAHI/cannon n'ont pas été promus ;
7. reprendre uniquement à partir de l'état existant.

Chaque amélioration doit être comparée objectivement à la baseline V5.

**Pas de promotion d'un modèle uniquement parce qu'il est nouveau.**

---

## 10. ENVIRONNEMENT PC — TRÈS IMPORTANT

Le PC utilise une :

**NVIDIA GeForce GTX 1050**

Compute Capability :

**6.1**

L'environnement existant a été volontairement configuré pour rester compatible avec cette ancienne carte.

L'état transmis indique notamment :

`torch 2.7.1+cu118`

et un `.venv` spécifique fonctionnel.

**NE PAS faire automatiquement :**

`pip install --upgrade torch`

ou une mise à niveau globale de :

- Python
- PyTorch
- CUDA
- torchvision
- dépendances YOLO critiques

Les versions modernes de PyTorch/CUDA peuvent abandonner la prise en charge de la Compute Capability 6.1.

On avait justement rencontré ce problème avec une version de PyTorch construite pour des GPU **CC >= 7.5**, incompatible avec la GTX 1050 CC 6.1.

**L'environnement fonctionnel actuel est plus important que le fait d'avoir les dernières versions.**

---

## 11. WORKSPACE

Le workspace principal est :

`C:\Users\jimmy\Desktop\Clashcompare\ClashCompareV2-main`

Il contient également le `.venv` à préserver.

Inspecte ce dossier avant de créer ou déplacer quoi que ce soit.

---

## 12. GITHUB

Repository :

`Farfadet80/ClashCompareV2`

Git doit servir de sécurité.

Avant une modification importante :

- vérifier `git status`
- regarder l'historique
- identifier le dernier état fonctionnel
- faire des commits propres

Ne détruis pas une version fonctionnelle pour tester une nouvelle approche.

---

## 13. PIPELINE ACTUEL

Les informations récupérées indiquent notamment :

**PWA + `serve_compare.py`**

et actuellement :

**niveaux AD + HDV seulement**

Ne suppose donc pas que la reconnaissance complète de tous les bâtiments est déjà terminée.

L'objectif est de faire progresser cette reconnaissance **progressivement et avec validation**, plutôt que de prétendre reconnaître des éléments non fiables.

---

## 14. DONNÉES FICTIVES / ANCIENNE MAQUETTE

La toute première maquette de ClashCompare contenait des **données fictives** uniquement pour montrer l'interface et le fonctionnement attendu.

Ces données ne doivent surtout pas être interprétées comme des données réelles provenant de Clash of Clans.

Le projet final doit remplacer progressivement ces données par :

API officielle + analyse vision fiable.

**Ne jamais afficher une information inventée comme si elle provenait du compte réel.**

Si une donnée n'est pas disponible :

afficher par exemple :

`Non disponible`

ou

`Non détecté`

plutôt que de l'estimer arbitrairement.

---

## 15. TAGS JOUEURS

Les tags A et B doivent rester modifiables.

L'ancienne maquette prévoyait également leur **persistance locale**, afin que l'utilisateur ne soit pas obligé de les retaper à chaque ouverture.

Préserver cette logique si elle existe encore.

L'application doit pouvoir être utilisée par **plusieurs personnes avec leurs propres tags**.

---

## 16. PHILOSOPHIE DE DÉVELOPPEMENT

La priorité n'est pas d'ajouter énormément de fonctionnalités rapidement.

Priorités :

**1. Fiabilité des données**

**2. Fiabilité YOLO**

**3. Pipeline complet fonctionnel**

**4. Comparaison correcte**

**5. UX/UI**

**6. Déploiement mobile/public**

Une fonctionnalité qui affiche une mauvaise information est pire qu'une fonctionnalité temporairement absente.

---

## 17. CE QU'IL NE FAUT PAS FAIRE

Ne pas :

- repartir de zéro ;
- supprimer les poids YOLO existants ;
- supprimer les datasets ;
- écraser les résultats/evals ;
- upgrader aveuglément Python/PyTorch/CUDA ;
- remplacer V5 sans preuve qu'un modèle est meilleur ;
- inventer les niveaux des bâtiments ;
- considérer les données fictives de la maquette comme réelles ;
- utiliser des identifiants Supercell ID ;
- exposer une clé API dans GitHub ;
- supposer que l'API CoC fournit la totalité du village ;
- casser la PWA existante inutilement ;
- modifier massivement l'architecture avant d'avoir compris le code existant.

---

## 18. AVANT DE CONTINUER LE DÉVELOPPEMENT

Fais d'abord un audit du workspace.

Vérifie :

- branche Git actuelle ;
- `git status` ;
- derniers commits ;
- fichiers non suivis ;
- `.gitignore` ;
- `.venv` ;
- version Python ;
- version PyTorch ;
- CUDA réellement utilisée ;
- détection effective de la GTX 1050 ;
- datasets YOLO ;
- structure/classes des datasets ;
- poids `best.pt` / `last.pt` ;
- runs V5/V6 ;
- métriques/evals ;
- configuration infer 800/TTA ;
- SAHI ;
- tests cannon ;
- `serve_compare.py` ;
- PWA ;
- état actuel AD/HDV ;
- variables d'environnement ;
- configuration API CoC ;
- éventuelles clés/secrets locaux non versionnés.

Ensuite seulement, indique ce qui fonctionne réellement et ce qui reste à faire.

---

## 19. OBJECTIF IMMÉDIAT

**Ne commence pas immédiatement une grosse nouvelle fonctionnalité.**

Première mission :

**faire fonctionner ClashCompare dans son état actuel de bout en bout et établir une baseline reproductible.**

Ensuite, reprendre l'amélioration YOLO **exactement à partir de la V5 promue**, en utilisant les expériences V6/SAHI/cannon comme historique plutôt que de les refaire aveuglément.

Chaque changement doit pouvoir répondre à :

**« Est-ce réellement meilleur que V5 sur les mêmes données de validation ? »**

Si la réponse n'est pas démontrée par les métriques/evals, ne pas promouvoir.

---

## 20. OBJECTIF FINAL

À terme, l'expérience souhaitée est approximativement :

**Utilisateur ouvre ClashCompare**

↓

**Entre Joueur A + Joueur B**

↓

**ClashCompare récupère automatiquement tout ce qui est disponible via l'API**

↓

**Complète les informations nécessaires via le pipeline vision lorsqu'elles peuvent être obtenues de manière fiable**

↓

**Construit les deux profils**

↓

**Calcule leur progression par rapport à leur HDV**

↓

**Affiche Joueur A / Joueur B / Comparaison**

↓

**Montre clairement qui est devant, dans quelles catégories, de combien, et ce qu'il reste à améliorer.**

---

**IMPORTANT POUR CURSOR :** si `HANDOFF.md`, les métriques, Git ou les fichiers réels contredisent une information technique de cette passation, **les fichiers et résultats réels du projet sont prioritaires**. Ce message sert surtout à préserver la vision produit et les décisions prises avec ChatGPT.

## Notes de réception (2026-09-04)

- Passation enregistrée telle quelle dans le repo.
- Clé API CoC : absente — `CLASHCOMPARE_API = ""` normal.
- Baseline E2E déjà vérifiée + tag Git `savepoint-v5-infer800-tta-2026-09-04`.
- Aucune grosse feature démarrée sur la seule base de ce document.
