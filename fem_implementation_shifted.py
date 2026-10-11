"""Partie 2 : réponse FEM Q2, variante déformée du solveur commun.

Les commentaires FEM Q1/Q2 du module commun identifient les demandes non
lettrées de l'énoncé : normes, pentes séparées, ajustement conjoint, surface 3D
et quantité h**alpha*k**beta. Aucune numérotation de sous-question n'est inventée.
"""
from fem_implementation import (
    run_convergence_study, main, SHIFT_DIRECTION, SHIFT_MAGNITUDE,
)


# Réponse FEM Q2 — Maillage régulier puis déplacement interne fixe et reproductible.
# Réponse FEM Q2 — h moyen recalculé sur le maillage déformé (module commun).
# Réponse FEM Q2 — Normes ∞ estimée, L² et H¹ continues (module commun).
# Réponse FEM Q2 — Alpha selon h, bêta selon k et estimation conjointe (module commun).
# Réponse FEM Q2 — Surface 3D et erreur selon h**alpha*k**beta (module commun).
def run_convergence_study_shifted():
    """Direction (1,0), amplitude relative 0,1, identiques entre exécutions et k."""
    return run_convergence_study(shifted=True)


if __name__ == "__main__":
    main(shifted=True)
