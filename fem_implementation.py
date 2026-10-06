import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from mesh_implementation import generate_irregular_boundary_mesh

# Implémentation de la méthode des éléments finis (FEM)

def get_trifem_elementary_stiffness(coords, area):
    mat_e = np.zeros((3, 3), dtype=np.float64)
    for i in range(0, 3):
        for j in range(0, 3):
            mat_e[i, j] = (coords[(j + 1) % 3, 1] - coords[(j + 2) % 3, 1]) * \
                           (coords[(i + 1) % 3, 1] - coords[(i + 2) % 3, 1]) + \
                           (coords[(j + 2) % 3, 0] - coords[(j + 1) % 3, 0]) * \
                           (coords[(i + 2) % 3, 0] - coords[(i + 1) % 3, 0])
    mat_e *= 0.25 / area
    return mat_e

def get_trifem_elementary_mass(coords, area):
    mat_e = np.ones((3, 3), dtype=np.float64)
    mat_e *= area / 12.0
    mat_e[range(3), range(3)] *= 2.0
    return mat_e

def get_trifem_elementary_term(coords, area, f):
    mat_e = get_trifem_elementary_mass(coords, area)
    return mat_e.dot(f)

def fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m):
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes) - 1
    K = scipy.sparse.lil_matrix((nnodes, nnodes), dtype=np.complex128)
    M = scipy.sparse.lil_matrix((nnodes, nnodes), dtype=np.complex128)
    F = np.zeros((nnodes, 1), dtype=np.complex128)

    for e in range(nelems):
        nodes = elem2nodes[p_elem2nodes[e]:p_elem2nodes[e+1]]
        coords = node_coords[nodes]
        Pe = np.ones((3, 3))
        Pe[:, 1:3] = coords[:, 0:2]
        area = abs(np.linalg.det(Pe)) / 2.0
        Ke = get_trifem_elementary_stiffness(coords, area)
        Me = get_trifem_elementary_mass(coords, area)
        Fe = get_trifem_elementary_term(coords, area, f_unassembled[nodes])

        for i in range(3):
            F[nodes[i]] += Fe[i]
            for j in range(3):
                K[nodes[i], nodes[j]] += coef_k[e] * Ke[i, j]
                M[nodes[i], nodes[j]] += coef_m[e] * Me[i, j]

    return K, M, F

def apply_dirichlet_condition(nodes, values, mat, vec):
    nodes = np.array(nodes)
    values = np.array(values).reshape(-1, 1)
    mat_csr = mat.tocsr()
    temp = mat_csr[:, nodes].dot(values)
    vec = vec - temp

    mat_lil = mat_csr.tolil()
    for i, node in enumerate(nodes):
        vec[node] = values[i]
        mat_lil[node, :] = 0.0
        mat_lil[:, node] = 0.0
        mat_lil[node, node] = 1.0

    return mat_lil.tocsr(), vec

def compute_average_h(node_coords, p_elem2nodes, elem2nodes):
    nelems = len(p_elem2nodes) - 1
    h_sum = 0.0
    for e in range(nelems):
        nodes = elem2nodes[p_elem2nodes[e]:p_elem2nodes[e+1]]
        p = node_coords[nodes]
        L = [np.linalg.norm(p[0]-p[1]), np.linalg.norm(p[1]-p[2]), np.linalg.norm(p[2]-p[0])]
        h_sum += max(L)
    return h_sum / nelems if nelems > 0 else 0.0

def compute_norms(u_exact, u_approx, h):
    error = u_approx - u_exact
    norm_inf = np.max(np.abs(error))
    norm_l2 = np.sqrt(np.sum(np.abs(error)**2)) * h
    norm_h1 = norm_l2 / h 
    return norm_inf, norm_l2, norm_h1

# Résolution de l'équation de Helmholtz

def solve_helmholtz(node_coords, p_elem2nodes, elem2nodes, wavenumber):
    r"""
    Résout l'équation de Helmholtz en utilisant des raw strings
    pour éviter les SyntaxWarnings (ex: (-\Delta - k^2) u = f).
    """
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes) - 1

    solexact = np.zeros((nnodes, 1), dtype=np.complex128)
    f_unassembled = np.zeros((nnodes, 1), dtype=np.complex128)

    for i in range(nnodes):
        x = node_coords[i, 0]
        solexact[i] = np.exp(1j * wavenumber * x)
        f_unassembled[i] = 0.0

    coef_k = np.ones(nelems, dtype=np.complex128)
    coef_m = np.ones(nelems, dtype=np.complex128)
    K, M, F = fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)

    A = K - (wavenumber**2) * M
    B = F

    nodes_on_boundary = np.where((node_coords[:,0] == 0) | (node_coords[:,0] == 1) |
                                 (node_coords[:,1] == 0) | (node_coords[:,1] == 1))[0]
    values_boundary = solexact[nodes_on_boundary]
    
    A_csr, B = apply_dirichlet_condition(nodes_on_boundary, values_boundary, A, B)
    u_approx = scipy.sparse.linalg.spsolve(A_csr, B).reshape(-1, 1)

    return u_approx, solexact

def run_convergence_study():
    """
    Analyse l'erreur en fonction de h et k uniquement sur le maillage régulier.
    """
    hs_target = [1/16, 1/32, 1/64] 
    ks_vals = [np.pi, 2*np.pi, 3*np.pi]
    results = []

    for h_t in hs_target:
        nx = int(1/h_t)
        ny = nx // 4 

        # Uniquement le maillage régulier (pas de shift)
        node_coords, p_elem2nodes, elem2nodes = generate_irregular_boundary_mesh(0,1,0,1,nx,ny,level=0)
        h_real = compute_average_h(node_coords, p_elem2nodes, elem2nodes)

        for k in ks_vals:
            u_approx, u_exact = solve_helmholtz(node_coords, p_elem2nodes, elem2nodes, k)
            norm_inf, norm_l2, norm_h1 = compute_norms(u_exact, u_approx, h_real)
            
            # Sauvegarde de toutes les normes et de h_nom pour faciliter le post-traitement
            results.append({
                'h_nom': h_t, 'h': h_real, 'k': k, 
                'err_l2': norm_l2, 'err_inf': norm_inf, 'err_h1': norm_h1
            })

    return results

if __name__ == "__main__":
    print("--- Implémentation de la Partie 2 : FEM & Erreurs ---")

    print("Test de validation sur maillage 16x4 (k=pi)...")
    node_coords, p_elem2nodes, elem2nodes = generate_irregular_boundary_mesh(0,1,0,1,16,4,level=0)
    h_test = compute_average_h(node_coords, p_elem2nodes, elem2nodes)
    u_approx, u_exact = solve_helmholtz(node_coords, p_elem2nodes, elem2nodes, wavenumber=np.pi)

    inf, l2, h1 = compute_norms(u_exact, u_approx, h_test)
    print(f"Erreur validée -> L2: {l2:.4e} | L_inf: {inf:.4e} | H1: {h1:.4e}")
    
    print("\nLancement de l'étude de convergence globale (Maillage Régulier uniquement)...")
    results = run_convergence_study()

    h_noms = np.array([res['h_nom'] for res in results])
    hs_vals = np.array([res['h'] for res in results])
    ks_vals = np.array([res['k'] for res in results])
    errs_l2 = np.array([res['err_l2'] for res in results])

    # Affichage Console des 3 Normes
    print(f"\n--- Validation des Normes extraites ---")
    for res in results:
        if res['h_nom'] == 1/16 and res['k'] == np.pi:
            print(f"Pour h=1/16 et k=pi : L2 = {res['err_l2']:.4e} | L_inf = {res['err_inf']:.4e} | H1 = {res['err_h1']:.4e}")

    # Figure 5 : Erreur vs h (k fixe)
    plt.figure(figsize=(6, 5))
    k_target = ks_vals[0]
    mask_k = (ks_vals == k_target)
    plt.loglog(hs_vals[mask_k], errs_l2[mask_k], 'o-', lw=2, label=r"k = \(\pi\)")
    p_alpha = np.polyfit(np.log(hs_vals[mask_k]), np.log(errs_l2[mask_k]), 1)
    plt.plot(hs_vals[mask_k], np.exp(p_alpha[1]) * hs_vals[mask_k]**p_alpha[0], '--', label=rf"Pente \(\alpha\) = {p_alpha[0]:.2f}")
    plt.xlabel("Taille moyenne du maillage h")
    plt.ylabel(r"Erreur \(L_2\)")
    plt.title("Convergence Spatiale (Régulier)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure5.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 6 : Erreur vs k (h fixe)
    plt.figure(figsize=(6, 5))
    h_target_nom = 1/32
    mask_h = (h_noms == h_target_nom)
    plt.loglog(ks_vals[mask_h], errs_l2[mask_h], 's-', color='red', lw=2, label=r"h nominal = 1/32")
    p_beta = np.polyfit(np.log(ks_vals[mask_h]), np.log(errs_l2[mask_h]), 1)
    plt.plot(ks_vals[mask_h], np.exp(p_beta[1]) * ks_vals[mask_h]**p_beta[0], 'k--', label=rf"Pente \(\beta\) = {p_beta[0]:.2f}")
    plt.xlabel("Nombre d'onde k")
    plt.ylabel(r"Erreur \(L_2\)")
    plt.title("Erreur de Dispersion (Régulier)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure6.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 7 : Surface 3D de l'erreur vs (h, k)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    H_unq = np.unique(hs_vals)
    K_unq = np.unique(ks_vals)
    H_grid, K_grid = np.meshgrid(H_unq, K_unq)
    E_grid = np.zeros_like(H_grid)
    for i in range(len(K_unq)):
        for j in range(len(H_unq)):
            val = errs_l2[(hs_vals == H_unq[j]) & (ks_vals == K_unq[i])]
            if len(val) > 0:
                E_grid[i, j] = val[0]
                
    surf = ax.plot_surface(H_grid, K_grid, E_grid, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel('h')
    ax.set_ylabel('k')
    ax.set_zlabel(r'Erreur \(L_2\)')
    ax.set_title("Surface d'Erreur Numérique (Régulier)")
    plt.tight_layout()
    plt.savefig("Figure7.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 8 : Erreur vs Quantité de pollution h^2 k^3
    plt.figure(figsize=(6, 5))
    pollution_qty = (hs_vals**2) * (ks_vals**3)
    plt.loglog(pollution_qty, errs_l2, 'd-', color='green', lw=2)
    plt.xlabel(r"Quantité \(h^2 k^3\)")
    plt.ylabel(r"Erreur \(L_2\)")
    plt.title("Analyse de la Pollution Acoustique (Régulier)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figure8.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nÉvaluation théorique (Régulier) : alpha = {p_alpha[0]:.2f}, beta = {p_beta[0]:.2f}")
    print("Succès : Les figures 5 à 8 ont été sauvegardées avec succès.")