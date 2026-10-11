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
from mesh_implementation import (generate_fractal_boundary_mesh, remove_element, add_element,
                                  add_node, remove_node, get_boundary_mesh, compute_areas)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def compare_choices():
    """Q1/Q2 : stockage plein/CSR et erreur nodale interpolée/continue."""
    rows = []
    for nx in (16, 32, 64, 128):
        xy, p, conn = generate_fractal_boundary_mesh(0, 1, 0, 1, nx, nx//4, level=0)
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


def illustrate_checks():
    """Q9 et contrôles FEM : figures et mesures indépendantes des calculs d'erreur."""
    mesh = generate_fractal_boundary_mesh(0, 1, 0, 1, 2, 2, level=0)
    removed = remove_element(*mesh, len(mesh[1])-2)
    restored = add_element(*removed, mesh[2][-3:])
    isolated = add_node(*mesh, [1.15, .5])
    final = remove_node(*isolated, len(isolated[0])-1)
    assert all(np.array_equal(a, b) for a, b in zip(mesh, restored))
    assert all(np.array_equal(a, b) for a, b in zip(mesh, final))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.7), constrained_layout=True)
    stages = [mesh, removed, restored, mesh, isolated, final]
    titles = ['État initial : 8 triangles', 'Retrait du dernier : 7 triangles', 'Réinsertion : état initial retrouvé',
              'État initial : 9 nœuds', 'Ajout : 10 nœuds, dont un isolé', 'Retrait du nœud : état initial retrouvé']
    for ax, state, title in zip(axes.ravel(), stages, titles):
        xy, _, conn = state
        ax.triplot(xy[:, 0], xy[:, 1], conn.reshape(-1, 3), lw=.8)
        ax.scatter(xy[:, 0], xy[:, 1], s=12)
        if len(xy) == 10: ax.scatter(*xy[-1], color='red', s=35)
        ax.set(title=title, xlim=(-.1, 1.25), ylim=(-.1, 1.1), aspect='equal')
        ax.tick_params(labelsize=8); ax.title.set_fontsize(9)
    fig.savefig(ROOT/'report/figures/topology_checks.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, n in zip(axes, (8, 16)):
        pts = np.array([(i/n, j/n) for i in range(n+1) for j in range(n+1-i)])
        ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color='black')
        ax.scatter(pts[:, 0], pts[:, 1], s=12)
        ax.set(title=f'{n} divisions par côté : {len(pts)} points', xlabel='ξ', ylabel='η', aspect='equal')
    fig.savefig(ROOT/'report/figures/sampling_grids.png', dpi=180); plt.close(fig)
    mesh = generate_fractal_boundary_mesh(0, 1, 0, 1, 8, 4, level=0)
    xy, p, conn = mesh
    K, M, F = fem.fem_assembly(p, conn, xy)
    boundary = get_boundary_mesh(*mesh).l2g
    exact = 1+2*xy[:, 0]-3*xy[:, 1]
    u, residual, _ = fem.solve_dirichlet_system(K, M, F, boundary, exact[boundary], 0)
    bary, weights = fem.triangle_quadrature(6)
    geo = fem.prepare_geometry(*mesh)
    const = lambda q: np.ones(q.shape[:-1], complex)
    zero = lambda q: np.zeros(q.shape, complex)
    errors = fem.integrate_errors(geo, np.zeros(len(xy)), 0, exact_function=const, exact_gradient=zero)
    measures = {'topology_element_roundtrip': True, 'topology_node_roundtrip': True,
                'translated_triangle_area': float(compute_areas(np.array([[1e9,1e9],[1e9+1,1e9],[1e9,1e9+1]]), np.array([0,3]), np.array([0,1,2]))[0]),
                'affine_max_nodal_error': float(np.max(np.abs(u-exact))), 'affine_relative_residual': residual,
                'reference_triangle_weight_sum': float(weights.sum()),
                'barycentric_integrals': (weights@bary).tolist(),
                'constant_error_l2': errors['err_l2'], 'constant_error_h1': errors['err_h1'],
                'constant_error_h1_semi': errors['err_h1_semi']}
    (ROOT/'report/data/verification.json').write_text(json.dumps(measures, indent=2, ensure_ascii=False)+'\n')
    headers = ['Contrôle', 'Valeur attendue', 'Résultat du code']
    table_rows = [['Ajout/retrait d’un élément', 'Tableaux identiques', 'Oui'],
                  ['Ajout/retrait d’un nœud isolé', 'Tableaux identiques', 'Oui'],
                  ['Triangle translaté de $10^9$ : aire', '0,5', f"{measures['translated_triangle_area']:.12f}"],
                  ['Solution affine : erreur nodale max.', '0', f"{measures['affine_max_nodal_error']:.2e}"],
                  ['Erreur constante : norme L²', '1', f"{errors['err_l2']:.12f}"],
                  ['Erreur constante : norme H¹', '1', f"{errors['err_h1']:.12f}"],
                  ['Erreur constante : semi-norme H¹', '0', f"{errors['err_h1_semi']:.2e}"],
                  ['Somme des poids sur le triangle', '0,5', f"{weights.sum():.12f}"],
                  ['Intégrale de chaque coordonnée barycentrique', '1/6', f"{(weights@bary)[0]:.12f}"]]
    # Tableau compact, produit directement à partir des mesures.
    tex = ['\\begin{tabular}{p{7cm}rr}', '\\toprule', ' & '.join(headers)+r' \\', '\\midrule']
    tex += [' & '.join(row)+r' \\' for row in table_rows]
    tex += ['\\bottomrule', '\\end{tabular}']
    (ROOT/'report/data/verification.tex').write_text('\n'.join(tex)+'\n')
    comparisons = json.loads((ROOT/'report/data/choices.json').read_text())
    tex = [r'\begin{tabular}{rrr}', r'\toprule', r'$n_x$ & Écart ordre 2/8 & Écart ordre 3/8 \\', r'\midrule']
    tex += [f"{r['nx']} & {r['quadrature_q2_relative']:.3e} & {r['quadrature_q3_relative']:.3e}"+r' \\' for r in comparisons['results']]
    tex += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'report/data/quadrature_checks.tex').write_text('\n'.join(tex)+'\n')
    tex = [r'\begin{tabular}{rrr}', r'\toprule', r'$n_x$ & Diamètre moyen & Déplacement fixe / diamètre \\', r'\midrule']
    tex += [f"{r['nx']} & {r['h']:.5f} & {0.01/r['h']:.5f}"+r' \\' for r in comparisons['results']]
    tex += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'report/data/displacement_scaling.tex').write_text('\n'.join(tex)+'\n')

    print(json.dumps(measures, ensure_ascii=False))


if __name__ == '__main__':
    compare_choices()
    illustrate_checks()
