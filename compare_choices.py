"""Comparaisons pédagogiques pour Q1/Q2 FEM ; mêmes matrices et onde de référence.

Les alternatives évaluées sont des comparaisons nouvelles, pas un historique
supposé de tentatives. Aucun calcul dense de grande taille n'est effectué.
"""
import csv
import json
from pathlib import Path
import numpy as np
import scipy.linalg as la
import fem_implementation as fem
from mesh_implementation import generate_irregular_boundary_mesh
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def compare_choices():
    """Q1/Q2 : stockage plein/CSR et erreur nodale interpolée/continue."""
    rows = []
    for nx in (16, 32, 64, 128):
        xy, p, conn = generate_irregular_boundary_mesh(0, 1, 0, 1, nx, nx//4, level=0)
        geo = fem.prepare_geometry(xy, p, conn)
        K, M, F = fem.fem_assembly(p, conn, xy, geometry=geo)
        k = 3*np.pi
        u, exact, diagnostics = fem.solve_helmholtz(xy, p, conn, k, (K, M, F), True)
        nodal = exact-u
        continuous = fem.integrate_errors(geo, u, k, order=8)['err_l2']
        interpolated = float(np.sqrt(np.vdot(nodal, M@nodal).real))
        interpolation = fem.integrate_errors(geo, exact, k, order=8)['err_l2']
        rows.append({'nx': nx, 'ny': nx//4, 'nodes': len(xy),
                     'h': float(geo['diameters'].mean()), 'k': float(k),
                     'dense_K_M_bytes': 2*len(xy)**2*8,
                     'csr_K_M_bytes': sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in (K,M)),
                     'err_l2_continuous': continuous,
                     'err_l2_nodal_mass': interpolated,
                     'interpolation_l2': interpolation,
                     'quadrature_q2_relative': abs(fem.integrate_errors(geo, u, k, order=2)['err_l2']/continuous-1),
                     'quadrature_q3_relative': abs(fem.integrate_errors(geo, u, k, order=3)['err_l2']/continuous-1)})
        if nx == 16:
            I = diagnostics['interior']
            spectrum = la.eigh(K[I][:,I].toarray(), M[I][:,I].toarray(), eigvals_only=True)
            spectral = {'nx': nx, 'ny': nx//4, 'lambda_min': float(spectrum[0]),
                        'lambda_max': float(spectrum[-1]),
                        'negative_directions_k_2pi': int(np.sum(spectrum<(2*np.pi)**2)),
                        'positive_directions_k_2pi': int(np.sum(spectrum>(2*np.pi)**2))}
    data = ROOT/'report/data'
    with (data/'choices.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (data/'choices.json').write_text(json.dumps({'scope': 'Matrices réelles K/M ; stockage hors facteurs LU et temporaires. Onde régulière k=3π.',
                                              'results': rows, 'spectrum_small_mesh': spectral}, indent=2, ensure_ascii=False)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    nodes = [r['nodes'] for r in rows]
    for key, label in [('dense_K_M_bytes', 'Stockage plein théorique'), ('csr_K_M_bytes', 'Stockage CSR mesuré')]:
        axes[0].loglog(nodes, [r[key]/2**20 for r in rows], 'o-', label=label)
    axes[0].set(xlabel='Nombre de nœuds', ylabel='Mémoire de K et M (Mio)', title='Stockage des matrices assemblées')
    for key, label in [('err_l2_continuous', 'Erreur continue'), ('err_l2_nodal_mass', 'Erreur nodale interpolée'), ('interpolation_l2', 'Interpolation de l’onde exacte')]:
        axes[1].loglog([r['h'] for r in rows], [r[key] for r in rows], 'o-', label=label)
    axes[1].set(xlabel='Diamètre moyen', ylabel='Norme L²', title='Maillage régulier, k = 3π')
    for ax in axes:
        ax.grid(True, which='both', alpha=.25); ax.legend(fontsize=8)
    fig.savefig(ROOT/'report/figures/choices.png', dpi=180); plt.close(fig)
    print(json.dumps({'finest': rows[-1], 'spectrum': spectral}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    compare_choices()
