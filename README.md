# Maillages et précision FEM d'une onde acoustique

Projet CentraleSupélec, cours 2SC5700. Les trois bibliothèques scientifiques utilisées sont NumPy, SciPy et Matplotlib. Les versions ayant produit les résultats sont enregistrées dans `requirements.txt` et les JSON ; l'environnement vérifié utilise Python **3.14.6**. Les dépendances indirectes installées sont celles de ces trois bibliothèques, sans autre bibliothèque scientifique directement utilisée par les scripts.

## Installation et exécution

Depuis ce dossier :

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python mesh_implementation.py
.venv/bin/python fem_implementation.py
.venv/bin/python fem_implementation_shifted.py
```

Les commandes exécutent les vérifications avant le calcul.

La variante FEM déformée importe le solveur commun ; les trois scripts doivent être conservés ensemble. Les sorties sont indépendantes du répertoire depuis lequel les commandes sont lancées : `report/figures/` pour les images et `report/data/` pour les CSV, JSON et tableaux LaTeX. Les caches locaux vont dans `build/` et ne sont pas livrés.

Le protocole par défaut utilise `(16,4)`, `(32,8)`, `(64,16)`, `(128,32)` et `k=π,2π,3π`. Les calculs régénèrent les données, ajustements et tableaux de chaque famille. Si le protocole change, adapter aussi les descriptions et chiffres du rapport et de la présentation : les tableaux sont générés, mais les paragraphes d'analyse ne sont pas rédigés automatiquement.

## Rapport, présentation et archive

Une distribution LaTeX contenant `latexmk`, `babel` français, `lmodern` et `beamer` est nécessaire **uniquement pour recompiler les documents**, pas pour exécuter les simulations.

```sh
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
cd ../presentation
latexmk -pdf -interaction=nonstopmode -halt-on-error presentation.tex
```

Les PDF et toutes leurs sources, y compris les tableaux inclus, sont fournis.

L'archive contient les trois scripts, les dépendances déclarées, les données, les figures, les sources et PDF du rapport et de la présentation, le README. Elle exclut `.venv`, `.git`, les caches, les auxiliaires LaTeX et les diapositives du professeur. Les références pédagogiques restent identifiées dans le rapport.

## Conventions et correspondance aux questions

- Maillage Q1–Q11 : `mesh_implementation.py`, commentaires portant le numéro réel de chaque question.
- FEM Q1 : `fem_implementation.py`. FEM Q2 : `fem_implementation_shifted.py`, avec fonctions communes marquées Q1/Q2.
- Les énoncés FEM n'ont pas de sous-questions officiellement lettrées : les commentaires utilisent les intitulés des demandes.
- Géométrie plane, indices entiers commençant à zéro, connectivité à pseudo-pointeurs ; triangles et quadrangles ordonnés et non dégénérés. Le solveur exige des triangles orientés positivement.
- Rapport d'aspect : `(sqrt(3)/6)*h_max/rho`. Facteur d'arête : minimum/moyenne arithmétique ; l'option `convention="median"` permet la longueur intermédiaire si une précision du professeur l'exige.
- Frontière Q10 : **alternative sinusoïdale autorisée**, non une fractale ; niveau entier, amplitude inférieure à la hauteur, au moins quatre intervalles par période. Aire polygonale constante démontrée et vérifiée.
- Déplacement interne : direction `(1,0)`, amplitude relative `0.1`, mêmes paramètres et coordonnées pour tous les k d'une résolution.
- `h` : diamètre moyen des triangles, différent de `1/nx`. `h_max` est exporté séparément.
- L² et H¹ : erreurs **continues intégrées**, avec quadrature d'ordre 6 contrôlée à l'ordre 8 ; la semi-norme H¹ est aussi exportée.
- L∞ : maximum **estimé** sur grilles barycentriques imbriquées, stabilisation relative à 0,5 %. Le maximum nodal est conservé séparément ; aucun supremum exact n'est certifié.
- Matrices K/M assemblées une seule fois par géométrie en COO/CSR ; factorisation de `K-k²M` sur les inconnues intérieures ; résidu exigé inférieur à `1e-9`.
- Les régressions séparées et conjointes sont empiriques. L'erreur maximale du modèle est publiée avec R² ; les puissances ne sont pas imposées à 2 et 3.

`compute_norms` a une signature corrigée : `(node_coords, p_elem2nodes, elem2nodes, u_approx, wavenumber, ...)`. Les seuls vecteurs nodaux et un h global ne suffisent pas à intégrer les normes demandées. Les vecteurs FEM sont complexes de forme `(n,)`. `get_boundary_mesh` renvoie un `BoundaryMesh` avec coordonnées, pseudo-pointeurs, connectivité locale, `l2g` et éléments adjacents. `remove_node` renvoie les trois tableaux et refuse un nœud encore référencé.

## Format de chargement

`load_mesh(nodes_file, elements_file)` lit des fichiers texte :

```text
# nœuds : identifiant x y
10 0 0
30 1 0
80 0 1
```

```text
# éléments : identifiant n1 n2 n3 [n4]
99 10 30 80
```

Les identifiants sont convertis en indices locaux ; ils ne doivent pas être nécessairement contigus. Un fichier d'éléments contient soit trois soit quatre sommets par ligne. Les fonctions de connectivité acceptent aussi les tableaux mixtes construits en mémoire. Les suppressions et ajouts ne remeshingent pas automatiquement le domaine ; les contrôles locaux ne détectent pas tous les recouvrements possibles de cellules arbitrairement insérées.

## Validation et limites

Les scripts vérifient les critères analytiques, les barycentres, les niveaux et l'aire, le déplacement, le sous-maillage de bord, les modifications topologiques, le chargement d'identifiants, les matrices élémentaires, la solution affine exacte, les normes d'une erreur constante, les gradients, Dirichlet, la quadrature, la régression sur une loi synthétique et la reproductibilité.

Les résultats publiés comprennent 24 solutions. Ils montrent les tendances pour une onde suivant x sur le carré unité. Ils ne constituent pas une simulation d'atténuation environnementale, une preuve de loi universelle de pollution, ni un résultat valable pour toutes les déformations ou toutes les fréquences. Les autres conventions possibles du terme « medium » doivent être annoncées si elles sont retenues.


Les versions utilisées sont NumPy 2.5.3, SciPy 1.18.1 et Matplotlib 3.11.2. Les CSV contiennent les erreurs et les diagnostics de chaque cas ; les JSON ajoutent les paramètres, les versions et les ajustements. Les tableaux LaTeX sont générés à partir de ces mesures.

## Comparaison des choix numériques

`.venv/bin/python compare_choices.py` reproduit les comparaisons complémentaires du rapport : stockage théorique plein et tableaux CSR de K/M, erreur continue et norme nodale de masse, quadratures d’ordres 2/3/8. Le spectre généralisé est calculé en dense uniquement sur le petit maillage 16×4 pour vérifier l’indéfinitude. Les résultats sont exportés dans `report/data/choices.csv` et `choices.json`, avec `report/figures/choices.png`. Le stockage mesuré exclut les facteurs LU et les tableaux temporaires ; aucun gain de temps n’est déduit de cette mesure. Le livre n’étant pas fourni dans le projet, le rapprochement avec ses chapitres repose sur les thèmes indiqués ; les supports fournis restent la référence vérifiée.

## Origine des fichiers

Les documents pédagogiques fournis par le professeur sont les PDF des supports et des énoncés. Les maillages utilisés dans les simulations sont construits directement en Python : aucun fichier externe de nœuds ou d’éléments n’est requis. Les CSV, JSON, tableaux LaTeX et figures sont des sorties produites pour ce projet. `compare_choices.py` reproduit les comparaisons complémentaires, sans constituer un outil ou une donnée fournis par le professeur. La fonction `load_mesh` est un utilitaire facultatif, non utilisé par les simulations du devoir.
