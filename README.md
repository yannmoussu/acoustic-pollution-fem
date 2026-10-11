# Maillages et précision FEM d’une onde acoustique

Projet CentraleSupélec, cours 2SC5700. Les seules bibliothèques scientifiques utilisées sont NumPy, SciPy et Matplotlib.

## Installation

Python **3.12 ou plus récent** est nécessaire pour les versions indiquées dans `requirements.txt`. L’installation et les calculs ont été vérifiés avec Python 3.14.6, NumPy 2.5.3, SciPy 1.18.1 et Matplotlib 3.11.2.

Après extraction du ZIP, ouvrir un terminal dans le dossier `Yann MOUSSU`.

Sous Linux ou macOS :

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Sous Windows, dans l’invite de commandes :

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## Exécution

Dans le même terminal, lancer les programmes dans cet ordre :

```sh
python mesh_implementation.py
python fem_implementation.py
python fem_implementation_shifted.py
python appendix.py
```

- `mesh_implementation.py` construit et analyse les maillages, puis exécute les opérations de Q1 à Q11.
- `fem_implementation.py` résout le problème sur les maillages réguliers et calcule les erreurs et ajustements de Q1.
- `fem_implementation_shifted.py` utilise le même solveur après déplacement interne pour Q2.
- `appendix.py` produit les comparaisons de stockage, les contrôles complémentaires, les graphiques d’exposants et les rapports d’erreur des deux familles.

Les vérifications numériques sont exécutées par les trois scripts principaux avant les calculs. Les fonctions portent les numéros des questions dans leurs docstrings. Les quatre fichiers Python doivent rester dans le même dossier.

Les maillages sont générés directement : aucun fichier de maillage extérieur n’est nécessaire. Les résultats sont écrits dans `report/data/` (CSV, JSON et tableaux LaTeX) et `report/figures/` (images). Ces chemins sont calculés à partir de l’emplacement des scripts. Les fichiers JSON fournis contiennent les résultats utilisés pour les graphiques complémentaires ; ils sont régénérés par les études FEM. Les caches créés dans `build/` peuvent être supprimés.

Le protocole utilise les maillages `16×4`, `32×8`, `64×16`, `128×32` et les nombres d’onde `π`, `2π`, `3π`, soit 24 solutions. Sur la machine de vérification, l’exécution complète prend environ 30 secondes, y compris la création initiale des caches graphiques ; la durée dépend de la machine.

## Documents

Les PDF se lisent directement, sans installation de LaTeX :

- `report/report.pdf` : rapport et analyses.
- `presentation/presentation.pdf` : présentation.

Pour recompiler les sources, une distribution LaTeX contenant `latexmk`, `babel` français, `lmodern`, `needspace` et `beamer` est nécessaire :

```sh
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
cd ../presentation
latexmk -pdf -interaction=nonstopmode -halt-on-error presentation.tex
```

L’archive contient les deux PDF, leurs sources et les figures et tableaux utilisés, les quatre programmes, les données de calcul, `requirements.txt` et ce README. Les supports du professeur, anciens dossiers extraits, caches, auxiliaires LaTeX et outils de développement sont exclus.

## Conventions et vérifications

Le facteur d’arête utilise la moyenne arithmétique des longueurs. Les quadrangles subdivisés sont convexes et ordonnés. Le bord fractal est construit à niveau fixé, avec aire conservée ; le déplacement interne laisse les nœuds du bord inchangés.

Les erreurs L² et H¹ sont intégrées dans les triangles. La norme L∞ est estimée par échantillonnage intérieur. Les contrôles portent notamment sur les aires, les matrices élémentaires, une solution affine exacte, les conditions de Dirichlet, le résidu, la quadrature et la reproductibilité. Les exposants sont ajustés aux mesures, sans imposer une loi universelle en h et k.
