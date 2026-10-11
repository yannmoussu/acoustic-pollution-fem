"""On lance l’étude FEM sur les maillages à nœuds internes déplacés."""
from fem_implementation import (
    run_convergence_study, main,
)


def run_convergence_study_shifted():
    """Q2 : on calcule les erreurs après déplacement interne de direction (1, 0) et d’amplitude 0,1."""
    return run_convergence_study(shifted=True)


if __name__ == "__main__":
    main(shifted=True)
