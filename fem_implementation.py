"""Partie 2 : FEM Q1 et fonctions communes à FEM Q2.

Éléments triangulaires P1, onde u*=exp(i*k*x), f=0 et Dirichlet sur tout
le bord. Vecteurs complexes de forme (n,), assemblage COO puis CSR.
"""
from pathlib import Path
import argparse
import csv
import json
import platform
import warnings
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import linalg as spla
from mesh_implementation import (
    ROOT, FIGURES, DATA, validate_geometry, triangle_geometry, get_boundary_mesh,
    generate_irregular_boundary_mesh, shift_internal_nodes, mesh_statistics,
    check_mesh_implementation,
)
import matplotlib.pyplot as plt

NORM_KEYS = ("err_inf", "err_l2", "err_h1")
NORM_LABELS = (r"$L^\infty$ estimée", r"$L^2$", r"$H^1$")
DEFAULT_NX = (16, 32, 64, 128)
DEFAULT_K = (np.pi, 2*np.pi, 3*np.pi)
SHIFT_DIRECTION = (1., 0.)
SHIFT_MAGNITUDE = 0.1


def _element_gradients(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (3, 2) or not np.isfinite(coords).all():
        raise ValueError("Un élément P1 doit contenir trois points plans finis.")
    signed2 = ((coords[1, 0]-coords[0, 0])*(coords[2, 1]-coords[0, 1])
               -(coords[1, 1]-coords[0, 1])*(coords[2, 0]-coords[0, 0]))
    scale = np.ptp(coords, axis=0)
    if abs(signed2) <= 64*np.finfo(float).eps*max(float(scale @ scale), np.finfo(float).tiny):
        raise ValueError("Triangle dégénéré.")
    b = np.roll(coords[:, 1], -1)-np.roll(coords[:, 1], -2)
    c = np.roll(coords[:, 0], -2)-np.roll(coords[:, 0], -1)
    return np.column_stack((b, c))/signed2, abs(signed2)/2

# Réponse FEM Q1/Q2 — Matrices élémentaires P1 de rigidité et de masse.
def get_trifem_elementary_stiffness(coords, area):
    gradients, actual = _element_gradients(coords)
    if not np.isfinite(area) or not np.isclose(area, actual, rtol=1e-12, atol=0):
        raise ValueError("Aire incohérente avec les coordonnées du triangle.")
    return area*(gradients @ gradients.T)

def get_trifem_elementary_mass(coords, area):
    _, actual = _element_gradients(coords)
    if not np.isfinite(area) or not np.isclose(area, actual, rtol=1e-12, atol=0):
        raise ValueError("Aire incohérente avec les coordonnées du triangle.")
    return area/12*(np.ones((3, 3))+np.eye(3))

def get_trifem_elementary_term(coords, area, f):
    """Charge interpolée P1 ; exacte ici puisque f=0."""
    f = np.asarray(f, dtype=complex).reshape(-1)
    if f.size != 3 or not np.isfinite(f).all():
        raise ValueError("Trois valeurs finies de charge attendues.")
    return get_trifem_elementary_mass(coords, area) @ f

def prepare_geometry(node_coords, p_elem2nodes, elem2nodes):
    validate_geometry(node_coords, p_elem2nodes, elem2nodes)
    tris, q, lengths, signed = triangle_geometry(node_coords, p_elem2nodes, elem2nodes)
    if len(tris) == 0:
        raise ValueError("Un calcul FEM exige au moins un triangle.")
    b = np.roll(q[:, :, 1], -1, axis=1)-np.roll(q[:, :, 1], -2, axis=1)
    c = np.roll(q[:, :, 0], -2, axis=1)-np.roll(q[:, :, 0], -1, axis=1)
    gradients = np.stack((b, c), axis=2)/(2*signed[:, None, None])
    return {"triangles": tris, "points": q, "areas": signed, "gradients": gradients,
            "diameters": lengths.max(axis=1), "nnodes": len(node_coords)}

# Réponse FEM Q1/Q2 — Assemblage local-global du cours, triplets COO vers CSR.
def fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled=None,
                 coef_k=None, coef_m=None, geometry=None):
    geo = geometry if geometry is not None else prepare_geometry(node_coords, p_elem2nodes, elem2nodes)
    tris, area, grad = geo["triangles"], geo["areas"], geo["gradients"]
    nelems, nnodes = len(tris), len(node_coords)
    f = np.zeros(nnodes, complex) if f_unassembled is None else np.asarray(f_unassembled, complex).reshape(-1)
    ck = np.ones(nelems) if coef_k is None else np.asarray(coef_k).reshape(-1)
    cm = np.ones(nelems) if coef_m is None else np.asarray(coef_m).reshape(-1)
    if f.size != nnodes or ck.size != nelems or cm.size != nelems:
        raise ValueError("Dimensions des coefficients ou de la charge incohérentes.")
    if not all(np.isfinite(x).all() for x in (f, ck, cm)):
        raise ValueError("Charge et coefficients doivent être finis.")
    Ke = area[:, None, None]*np.einsum("eia,eja->eij", grad, grad)
    Me = area[:, None, None]/12*(np.ones((3, 3))+np.eye(3))
    rows = np.broadcast_to(tris[:, :, None], Ke.shape).ravel()
    cols = np.broadcast_to(tris[:, None, :], Ke.shape).ravel()
    K = sparse.coo_matrix(((ck[:, None, None]*Ke).ravel(), (rows, cols)), shape=(nnodes, nnodes)).tocsr()
    M = sparse.coo_matrix(((cm[:, None, None]*Me).ravel(), (rows, cols)), shape=(nnodes, nnodes)).tocsr()
    Fe = np.einsum("eij,ej->ei", Me, f[tris])
    F = np.zeros(nnodes, complex)
    np.add.at(F, tris.ravel(), Fe.ravel())
    return K, M, F

def _boundary_values(nodes, values, nnodes):
    raw = np.asarray(nodes)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("Les indices de Dirichlet doivent être un vecteur entier.")
    nodes = raw.astype(np.int64, copy=False)
    values = np.asarray(values, dtype=complex).reshape(-1)
    if (values.size != nodes.size or len(np.unique(nodes)) != len(nodes)
            or np.any(nodes < 0) or np.any(nodes >= nnodes) or not np.isfinite(values).all()):
        raise ValueError("Conditions de Dirichlet invalides.")
    return nodes, values

# Réponse FEM Q1/Q2 — Condition de Dirichlet avec report du second membre.
def apply_dirichlet_condition(nodes, values, mat, vec):
    """Version complète, conservée pour validation ; résolution courante sur le bloc intérieur."""
    A = sparse.csr_matrix(mat, dtype=complex)
    nodes, values = _boundary_values(nodes, values, A.shape[0])
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    if A.shape[0] != A.shape[1] or vec.size != A.shape[0]:
        raise ValueError("Dimensions du système incohérentes.")
    rhs = vec-A[:, nodes] @ values
    free = np.ones(A.shape[0])
    free[nodes] = 0
    D = sparse.diags(free)
    constrained = (D @ A @ D+sparse.diags(1-free)).tocsr()
    rhs[nodes] = values
    return constrained, rhs

def solve_dirichlet_system(K, M, F, boundary, values, k):
    if not np.isfinite(k) or k < 0:
        raise ValueError("Le nombre d'onde doit être réel, fini et positif ou nul.")
    n = K.shape[0]
    boundary, values = _boundary_values(boundary, values, n)
    interior = np.setdiff1d(np.arange(n), boundary)
    A = (K-k*k*M).tocsr()
    rhs = np.asarray(F, dtype=complex).reshape(-1)[interior]-A[interior][:, boundary] @ values
    u = np.zeros(n, dtype=complex)
    u[boundary] = values
    relative_residual = 0.0
    if len(interior):
        Aii = A[interior][:, interior].tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            try:
                u[interior] = spla.spsolve(Aii, rhs)
            except spla.MatrixRankWarning as exc:
                raise ValueError("Système singulier : fréquence propre discrète.") from exc
        if not np.isfinite(u).all():
            raise ValueError("Solution FEM non finie.")
        residual = Aii @ u[interior]-rhs
        relative_residual = float(np.linalg.norm(residual)/max(np.linalg.norm(rhs), np.finfo(float).tiny))
        if relative_residual > 1e-9:
            raise ValueError("Le résidu relatif dépasse 10⁻⁹.")
    return u, relative_residual, interior

# Réponse FEM Q1/Q2 — Onde exacte du cours et résolution de Helmholtz.
def solve_helmholtz(node_coords, p_elem2nodes, elem2nodes, wavenumber, assembled=None,
                    return_diagnostics=False):
    exact = np.exp(1j*wavenumber*np.asarray(node_coords)[:, 0])
    K, M, F = assembled if assembled is not None else fem_assembly(p_elem2nodes, elem2nodes, node_coords)
    boundary = get_boundary_mesh(node_coords, p_elem2nodes, elem2nodes).l2g
    u, residual, interior = solve_dirichlet_system(K, M, F, boundary, exact[boundary], wavenumber)
    if return_diagnostics:
        return u, exact, {"relative_residual": residual, "boundary_error": float(np.max(np.abs(u[boundary]-exact[boundary]))),
                          "interior": interior}
    return u, exact

def triangle_quadrature(order):
    """Gauss-Legendre tensorisé par (r,s)->(r,(1-r)s) ; poids de somme 1/2."""
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("Ordre de quadrature entier supérieur ou égal à 2 requis.")
    t, w = np.polynomial.legendre.leggauss(order)
    t, w = (t+1)/2, w/2
    r, s = np.meshgrid(t, t, indexing="ij")
    wr, ws = np.meshgrid(w, w, indexing="ij")
    xi, eta = r.ravel(), ((1-r)*s).ravel()
    bary = np.column_stack((1-xi-eta, xi, eta))
    return bary, (wr*ws*(1-r)).ravel()

def _exact_fields(points, k, exact_function=None, exact_gradient=None):
    value = np.exp(1j*k*points[..., 0]) if exact_function is None else exact_function(points)
    gradient = np.zeros(points.shape, dtype=complex)
    if exact_gradient is None:
        gradient[..., 0] = 1j*k*value
    else:
        gradient = exact_gradient(points)
    return value, gradient

# Réponse FEM Q1/Q2 — Intégration de l'erreur continue et de son gradient, pas de I_h(e).
def integrate_errors(geometry, u_approx, k, order=6, exact_function=None, exact_gradient=None):
    bary, weights = triangle_quadrature(order)
    tris, q, areas, gradients = (geometry[name] for name in ("triangles", "points", "areas", "gradients"))
    u = np.asarray(u_approx, dtype=complex).reshape(-1)
    if u.size != geometry["nnodes"] or not np.isfinite(u).all():
        raise ValueError("Vecteur FEM invalide.")
    l2_squared = h1_squared = interpolation_squared = interpolation_gradient_squared = 0.0
    amplitude_max = phase_max = 0.0
    for start in range(0, len(tris), 512):
        sl = slice(start, start+512)
        points = np.einsum("qi,eia->eqa", bary, q[sl])
        exact, exact_grad = _exact_fields(points, k, exact_function, exact_gradient)
        ue = u[tris[sl]]
        approx = ue @ bary.T
        grad_approx = np.einsum("ei,eia->ea", ue, gradients[sl])
        jac_weights = 2*areas[sl, None]*weights[None, :]
        l2_squared += np.sum(jac_weights*np.abs(exact-approx)**2)
        h1_squared += np.sum(jac_weights*np.sum(np.abs(exact_grad-grad_approx[:, None, :])**2, axis=2))
        exact_nodes, _ = _exact_fields(q[sl], k, exact_function, exact_gradient)
        interpolated = exact_nodes @ bary.T
        interpolation_grad = np.einsum("ei,eia->ea", exact_nodes, gradients[sl])
        interpolation_squared += np.sum(jac_weights*np.abs(exact-interpolated)**2)
        interpolation_gradient_squared += np.sum(jac_weights*np.sum(np.abs(exact_grad-interpolation_grad[:, None, :])**2, axis=2))
        if exact_function is None:
            amplitude_max = max(amplitude_max, float(np.max(np.abs(np.abs(approx)-1))))
            # Phase principale relative ; pas une constante de dispersion universelle.
            if np.min(np.abs(approx)) < 1e-12:
                raise ValueError("Phase indéfinie : amplitude quasi nulle.")
            phase_max = max(phase_max, float(np.max(np.abs(np.angle(approx*np.conj(exact))))))
    return {"err_l2": float(np.sqrt(l2_squared)), "err_h1_semi": float(np.sqrt(h1_squared)),
            "err_h1": float(np.sqrt(l2_squared+h1_squared)),
            "interpolation_l2": float(np.sqrt(interpolation_squared)),
            "interpolation_h1": float(np.sqrt(interpolation_squared+interpolation_gradient_squared)),
            "amplitude_error_sampled": amplitude_max, "phase_error_sampled": phase_max}

def sample_supremum(geometry, u_approx, k, subdivisions, exact_function=None):
    """Maximum sur une grille barycentrique comprenant les sommets et les arêtes."""
    if not isinstance(subdivisions, (int, np.integer)) or subdivisions < 1:
        raise ValueError("Nombre de subdivisions entier positif requis.")
    bary = np.array([(i/subdivisions, j/subdivisions, 1-(i+j)/subdivisions)
                     for i in range(subdivisions+1) for j in range(subdivisions+1-i)])
    u = np.asarray(u_approx, dtype=complex).reshape(-1)
    maximum = 0.0
    for start in range(0, len(geometry["triangles"]), 256):
        sl = slice(start, start+256)
        q = geometry["points"][sl]
        points = np.einsum("qi,eia->eqa", bary, q)
        exact = np.exp(1j*k*points[..., 0]) if exact_function is None else exact_function(points)
        approx = u[geometry["triangles"][sl]] @ bary.T
        maximum = max(maximum, float(np.max(np.abs(exact-approx))))
    return maximum

def compute_norms(node_coords, p_elem2nodes, elem2nodes, u_approx, wavenumber,
                  quadrature_order=6, sample_subdivisions=16):
    """API corrigée : la géométrie est indispensable aux normes L² et H¹."""
    geometry = prepare_geometry(node_coords, p_elem2nodes, elem2nodes)
    errors = integrate_errors(geometry, u_approx, wavenumber, quadrature_order)
    infinity = sample_supremum(geometry, u_approx, wavenumber, sample_subdivisions)
    return infinity, errors["err_l2"], errors["err_h1"]

def evaluate_errors(geometry, u, exact_nodes, k):
    errors = integrate_errors(geometry, u, k, order=6)
    refined = integrate_errors(geometry, u, k, order=8)
    delta = max(abs(refined[name]-errors[name])/max(refined[name], 1e-15)
                for name in ("err_l2", "err_h1", "interpolation_l2", "interpolation_h1"))
    if delta > 1e-7:
        raise ValueError("La quadrature 6/8 n'est pas convergée : augmenter son ordre.")
    # Comparaison de grilles imbriquées ; maximum estimé, jamais présenté comme exact.
    previous = sample_supremum(geometry, u, k, 8)
    for subdivisions in (16, 32, 64):
        current = sample_supremum(geometry, u, k, subdivisions)
        change = abs(current-previous)/max(current, 1e-15)
        if change <= 0.005:
            break
        previous = current
    if change > 0.005:
        raise ValueError("Maximum échantillonné non stabilisé à 0,5 % : raffiner la grille.")
    errors.update({"err_inf": current, "err_inf_nodal": float(np.max(np.abs(u-exact_nodes))),
                   "quadrature_relative_change": float(delta), "sup_relative_change": float(change),
                   "sample_subdivisions": subdivisions})
    return errors

def nearest_spectral_gap(K, M, interior, k):
    """Cours 6 : distance à une valeur propre du problème généralisé intérieur."""
    if len(interior) < 2:
        return float("nan")
    Kii, Mii = K[interior][:, interior], M[interior][:, interior]
    nearest = spla.eigsh(Kii, M=Mii, k=1, sigma=k*k, which="LM", return_eigenvectors=False,
                         v0=np.random.default_rng(0).standard_normal(len(interior)), tol=1e-9)[0]
    return float(abs(nearest-k*k))

# Réponse FEM Q1 — Étude régulière ; FEM Q2 utilise exactement le même protocole.
def run_convergence_study(shifted=False, nx_values=DEFAULT_NX, k_values=DEFAULT_K):
    results = []
    for nx in nx_values:
        if not isinstance(nx, (int, np.integer)) or nx < 4 or nx % 4:
            raise ValueError("nx doit être un multiple de 4, supérieur ou égal à 4.")
        ny = nx//4
        mesh = generate_irregular_boundary_mesh(0, 1, 0, 1, nx, ny, level=0)
        xy, p, conn = mesh
        # Réponse FEM Q2 — Déplacement déterministe construit une seule fois par résolution.
        if shifted:
            xy = shift_internal_nodes(xy, p, conn, direction=SHIFT_DIRECTION, magnitude=SHIFT_MAGNITUDE)
        frozen_coordinates = xy.copy()
        geo = prepare_geometry(xy, p, conn)
        assembled = fem_assembly(p, conn, xy, geometry=geo)  # K et M réutilisées pour tous les k.
        quality = mesh_statistics(xy, p, conn)
        for k in k_values:
            u, exact, diagnostics = solve_helmholtz(xy, p, conn, k, assembled, True)
            np.testing.assert_array_equal(xy, frozen_coordinates)
            errors = evaluate_errors(geo, u, exact, k)
            row = {"nx": nx, "ny": ny, "nodes": len(xy), "elements": len(conn)//3,
                   "h_nom": 1/nx, "h": compute_average_h(xy, p, conn),
                   "h_max": quality["h_max"], "k": float(k), "kh_max": float(k*quality["h_max"]),
                   "aspect_max": quality["aspect_max"], "angle_min_deg": quality["angle_min_deg"],
                   "relative_residual": diagnostics["relative_residual"],
                   "boundary_error": diagnostics["boundary_error"],
                   "spectral_gap": nearest_spectral_gap(assembled[0], assembled[1], diagnostics["interior"], k),
                   **errors}
            results.append(row)
            print(f"{'Déformé' if shifted else 'Régulier'} {nx}×{ny}, k/π={k/np.pi:g} : "
                  f"L²={row['err_l2']:.3e}, H¹={row['err_h1']:.3e}, résidu={row['relative_residual']:.2e}", flush=True)
    return results

def _power_fit(x, error):
    x, error = np.asarray(x), np.asarray(error)
    if len(x) < 2 or np.any(x <= 0) or np.any(error <= 0):
        raise ValueError("Une régression logarithmique exige au moins deux mesures positives.")
    slope, intercept = np.polyfit(np.log(x), np.log(error), 1)
    order = np.argsort(x)[::-1]
    local = np.diff(np.log(error[order]))/np.diff(np.log(x[order]))
    return {"exponent": float(slope), "C": float(np.exp(intercept)), "local_slopes": local.tolist()}

# Réponse FEM Q1/Q2 — Alpha à k fixé, bêta à h fixé, puis estimation conjointe.
def fit_convergence(results):
    fits = {}
    h, k = np.array([r["h"] for r in results]), np.array([r["k"] for r in results])
    design = np.column_stack((np.ones(len(results)), np.log(h), np.log(k)))
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("Le plan expérimental ne permet pas d'identifier alpha et bêta.")
    for name in NORM_KEYS:
        error = np.array([r[name] for r in results])
        coef, _, _, _ = np.linalg.lstsq(design, np.log(error), rcond=None)
        residual = np.log(error)-design @ coef
        variance = np.sum((np.log(error)-np.log(error).mean())**2)
        by_k = {}
        for value in np.unique(k):
            mask = k == value
            by_k[str(float(value))] = _power_fit(h[mask], error[mask])
        by_nx = {}
        for nx in sorted({r["nx"] for r in results}):
            mask = np.array([r["nx"] == nx for r in results])
            by_nx[str(nx)] = _power_fit(k[mask], error[mask])
        fits[name] = {"alpha": float(coef[1]), "beta": float(coef[2]), "C": float(np.exp(coef[0])),
                      "r2_log": float(1-np.sum(residual**2)/variance),
                      "rmse_log": float(np.sqrt(np.mean(residual**2))),
                      "max_relative_fit_error": float(np.max(np.abs(np.exp(-residual)-1))),
                      "alpha_by_k": by_k, "beta_by_nx": by_nx}
    return fits

# Réponse FEM Q1/Q2 — Quatre familles de figures, chacune pour les trois normes.
def plot_convergence(results, fits, shifted=False, output_dir=FIGURES):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first = 9 if shifted else 5
    title = "Maillage déformé" if shifted else "Maillage régulier"
    h, k = np.array([r["h"] for r in results]), np.array([r["k"] for r in results])
    # Erreur selon h : toutes les fréquences, droites ajustées et pentes.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, name, label in zip(axes, NORM_KEYS, NORM_LABELS):
        error = np.array([r[name] for r in results])
        for value in np.unique(k):
            mask = k == value
            fit = fits[name]["alpha_by_k"][str(float(value))]
            ax.loglog(h[mask], error[mask], "o-", label=rf"$k/\pi={value/np.pi:g}$, $\alpha={fit['exponent']:.2f}$")
            ax.loglog(h[mask], fit["C"]*h[mask]**fit["exponent"], "--", alpha=.55)
        ax.set(xlabel="Diamètre moyen h", ylabel=f"Erreur {label}")
        ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle(title+" — erreur selon h")
    fig.tight_layout()
    fig.savefig(output_dir/f"Figure{first}.png", dpi=180)
    plt.close(fig)
    # Erreur selon k : toutes les résolutions et leurs pentes.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, name, label in zip(axes, NORM_KEYS, NORM_LABELS):
        error = np.array([r[name] for r in results])
        for nx in sorted({r["nx"] for r in results}):
            mask = np.array([r["nx"] == nx for r in results])
            fit = fits[name]["beta_by_nx"][str(nx)]
            ax.loglog(k[mask], error[mask], "o-", label=rf"${nx}\times{nx//4}$, $\beta={fit['exponent']:.2f}$")
            ax.loglog(k[mask], fit["C"]*k[mask]**fit["exponent"], "--", alpha=.55)
        ax.set(xlabel="Nombre d'onde k", ylabel=f"Erreur {label}")
        ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle(title+" — erreur selon k")
    fig.tight_layout()
    fig.savefig(output_dir/f"Figure{first+1}.png", dpi=180)
    plt.close(fig)
    # Surface expérimentale et plan du modèle en coordonnées log10.
    fig = plt.figure(figsize=(16, 5.5))
    h_values, k_values = np.unique(h), np.unique(k)
    H, K = np.meshgrid(h_values, k_values)
    for index, (name, label) in enumerate(zip(NORM_KEYS, NORM_LABELS), 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        E = np.array([[next(r[name] for r in results if r["h"] == hh and r["k"] == kk)
                       for hh in h_values] for kk in k_values])
        fit = fits[name]
        predicted = fit["C"]*H**fit["alpha"]*K**fit["beta"]
        ax.plot_surface(np.log10(H), np.log10(K), np.log10(E), cmap="viridis", alpha=.65)
        ax.plot_wireframe(np.log10(H), np.log10(K), np.log10(predicted), color="black", linewidth=.8)
        ax.set(xlabel="log₁₀(h)", ylabel="log₁₀(k)", zlabel="log₁₀(erreur)", title=label)
        ax.text2D(.02, .93, rf"$\alpha={fit['alpha']:.2f}$, $\beta={fit['beta']:.2f}$", transform=ax.transAxes, fontsize=9)
    fig.suptitle(title+" — surface mesurée et modèle ajusté (grille noire)")
    fig.subplots_adjust(left=.02, right=.94, bottom=.15, top=.85, wspace=.32)
    fig.savefig(output_dir/f"Figure{first+2}.png", dpi=180)
    plt.close(fig)
    # Quantité demandée construite avec les exposants estimés, sans relier des cas distincts.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, name, label in zip(axes, NORM_KEYS, NORM_LABELS):
        fit = fits[name]
        quantity = h**fit["alpha"]*k**fit["beta"]
        error = np.array([r[name] for r in results])
        for nx in sorted({r["nx"] for r in results}):
            mask = np.array([r["nx"] == nx for r in results])
            ax.loglog(quantity[mask], error[mask], "o", label=rf"${nx}\times{nx//4}$")
        order = np.argsort(quantity)
        ax.loglog(quantity[order], fit["C"]*quantity[order], "k--", label="Modèle ajusté")
        ax.set(xlabel=rf"$h^{{{fit['alpha']:.2f}}}k^{{{fit['beta']:.2f}}}$", ylabel=f"Erreur {label}")
        ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle(title+" — quantité combinée estimée")
    fig.tight_layout()
    fig.savefig(output_dir/f"Figure{first+3}.png", dpi=180)
    plt.close(fig)

def export_results(results, fits, shifted=False, checks=None):
    DATA.mkdir(parents=True, exist_ok=True)
    name = "shifted" if shifted else "regular"
    with (DATA/f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    metadata = {"family": name, "nx_values": sorted({r["nx"] for r in results}),
                "k_values": sorted({r["k"] for r in results}), "quadrature_orders": [6, 8],
                "supremum_relative_tolerance": .005,
                "shift_direction": list(SHIFT_DIRECTION), "shift_magnitude": SHIFT_MAGNITUDE if shifted else 0.,
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    import matplotlib
    metadata["matplotlib"] = matplotlib.__version__
    payload = {"metadata": metadata, "results": results, "fits": fits, "checks": checks}
    (DATA/f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    write_report_tables(payload, name)

def _latex_number(value, digits=3):
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"

def _latex_table(headers, rows):
    columns = "l"+"r"*(len(headers)-1)
    return (rf"\begin{{tabular}}{{{columns}}}"+"\n\\toprule\n"+" & ".join(headers)+r" \\"+
            "\n\\midrule\n"+"\n".join(" & ".join(row)+r" \\" for row in rows)+
            "\n\\bottomrule\n\\end{tabular}\n")

def write_report_tables(payload, name):
    """Les tableaux du PDF proviennent des mêmes mesures que les CSV/JSON."""
    results, fits = payload["results"], payload["fits"]
    rows = [[rf"\({r['nx']}\times{r['ny']}\)", f"{r['k']/np.pi:g}",
             *[_latex_number(r[key]) for key in NORM_KEYS]] for r in results]
    (DATA/f"{name}_errors.tex").write_text(_latex_table(
        ["Maillage", r"\(k/\pi\)", r"\(L^\infty\) estimée", r"\(L^2\)", r"\(H^1\)"], rows), encoding="utf-8")
    rows = [[label, f"{fits[key]['alpha']:.4f}", f"{fits[key]['beta']:.4f}",
             _latex_number(fits[key]["C"]), f"{fits[key]['r2_log']:.5f}",
             f"{100*fits[key]['max_relative_fit_error']:.1f}\\,\\%"]
            for key, label in zip(NORM_KEYS, [r"\(L^\infty\) estimée", r"\(L^2\)", r"\(H^1\)"])]
    (DATA/f"{name}_joint.tex").write_text(_latex_table(
        ["Norme", r"\(\widehat\alpha\)", r"\(\widehat\beta\)", r"\(\widehat C\)", r"\(R^2_{\log}\)", "Écart max."], rows), encoding="utf-8")
    rows = []
    for k in payload["metadata"]["k_values"]:
        rows.append([f"{k/np.pi:g}", *[f"{fits[key]['alpha_by_k'][str(k)]['exponent']:.4f}" for key in NORM_KEYS]])
    (DATA/f"{name}_alpha.tex").write_text(_latex_table(
        [r"\(k/\pi\)", r"\(\alpha_\infty\)", r"\(\alpha_{L^2}\)", r"\(\alpha_{H^1}\)"], rows), encoding="utf-8")
    rows = [[rf"\({nx}\times{nx//4}\)",
             *[f"{fits[key]['beta_by_nx'][str(nx)]['exponent']:.4f}" for key in NORM_KEYS]]
            for nx in payload["metadata"]["nx_values"]]
    (DATA/f"{name}_beta.tex").write_text(_latex_table(
        ["Maillage", r"\(\beta_\infty\)", r"\(\beta_{L^2}\)", r"\(\beta_{H^1}\)"], rows), encoding="utf-8")
    rows = []
    for nx in payload["metadata"]["nx_values"]:
        subset = [r for r in results if r["nx"] == nx]
        row = subset[0]
        rows.append([rf"\({nx}\times{nx//4}\)", f"{row['nodes']}/{row['elements']}",
                     f"{row['h']:.6f}", f"{row['h_max']:.6f}",
                     _latex_number(max(r["relative_residual"] for r in subset), 2)])
    (DATA/f"{name}_geometry.tex").write_text(_latex_table(
        ["Maillage", "Nœuds/triangles", r"\(\bar h\)", r"\(h_{\max}\)", "Résidu max."], rows), encoding="utf-8")
    rows = []
    for k in payload["metadata"]["k_values"]:
        for key, label in zip(NORM_KEYS, [r"\(L^\infty\)", r"\(L^2\)", r"\(H^1\)"]):
            slopes = fits[key]["alpha_by_k"][str(k)]["local_slopes"]
            rows.append([f"{k/np.pi:g}", label, *[f"{v:.4f}" for v in slopes]])
    levels = payload["metadata"]["nx_values"]
    headers = [r"\(k/\pi\)", "Norme"]+[rf"\({a}\to{b}\)" for a, b in zip(levels[:-1], levels[1:])]
    (DATA/f"{name}_local.tex").write_text(_latex_table(headers, rows), encoding="utf-8")
    rows = [[rf"\({r['nx']}\times{r['ny']}\)", f"{r['k']/np.pi:g}",
             f"{r['spectral_gap']:.3f}", _latex_number(r['interpolation_l2'], 2),
             _latex_number(r['phase_error_sampled'], 2), _latex_number(r['amplitude_error_sampled'], 2)]
            for r in results]
    (DATA/f"{name}_diagnostics.tex").write_text(_latex_table(
        ["Maillage", r"\(k/\pi\)", r"\(\delta_h\)", r"\(\|u^*-I_hu^*\|_{L^2}\)", "Phase (rad)", "Amplitude"], rows), encoding="utf-8")

# Réponse FEM Q2 — Fonction de diamètre moyen, également utilisée pour Q1.
def compute_average_h(node_coords, p_elem2nodes, elem2nodes):
    _, _, lengths, _ = triangle_geometry(node_coords, p_elem2nodes, elem2nodes)
    if len(lengths) == 0:
        raise ValueError("Le diamètre moyen d'un maillage vide n'est pas défini.")
    return float(lengths.max(axis=1).mean())

def check_fem_implementation():
    """Matrices analytiques, affine exacte, vrais gradients, bord, quadrature et régression."""
    xy = np.array([[0., 0.], [1., 0.], [0., 1.]])
    Ke = get_trifem_elementary_stiffness(xy, .5)
    Me = get_trifem_elementary_mass(xy, .5)
    np.testing.assert_allclose(Ke, [[1., -.5, -.5], [-.5, .5, 0], [-.5, 0, .5]])
    np.testing.assert_allclose(Me, (np.ones((3, 3))+np.eye(3))/24)
    np.testing.assert_allclose(Ke @ np.ones(3), 0, atol=1e-14)
    np.testing.assert_allclose(Me.sum(), .5)
    mesh = generate_irregular_boundary_mesh(0, 1, 0, 1, 8, 4, 0)
    geo = prepare_geometry(*mesh)
    K, M, F = fem_assembly(mesh[1], mesh[2], mesh[0], geometry=geo)
    np.testing.assert_allclose((K-K.T).data, 0., atol=1e-14)
    np.testing.assert_allclose((M-M.T).data, 0., atol=1e-14)
    np.testing.assert_allclose(K @ np.ones(len(mesh[0])), 0., atol=1e-13)
    np.testing.assert_allclose(M.sum(), 1., atol=1e-14)
    boundary = get_boundary_mesh(*mesh).l2g
    affine = 1+2*mesh[0][:, 0]-3*mesh[0][:, 1]
    u, residual, _ = solve_dirichlet_system(K, M, F, boundary, affine[boundary], 0)
    np.testing.assert_allclose(u, affine, atol=2e-13)
    A, rhs = apply_dirichlet_condition(boundary, affine[boundary], K, F)
    np.testing.assert_allclose(spla.spsolve(A, rhs), affine, atol=2e-13)
    const = lambda q: np.ones(q.shape[:-1], dtype=complex)
    zero_grad = lambda q: np.zeros(q.shape, dtype=complex)
    errors = integrate_errors(geo, np.zeros(len(u)), 0, exact_function=const, exact_gradient=zero_grad)
    np.testing.assert_allclose([errors["err_l2"], errors["err_h1"], errors["err_h1_semi"]], [1, 1, 0], atol=1e-13)
    affine_function = lambda q: 1+2*q[..., 0]-3*q[..., 1]
    affine_gradient = lambda q: np.broadcast_to(np.array([2., -3.]), q.shape)
    errors = integrate_errors(geo, affine, 0, exact_function=affine_function, exact_gradient=affine_gradient)
    assert errors["err_l2"] < 1e-13 and errors["err_h1"] < 1e-12
    bary, weights = triangle_quadrature(6)
    np.testing.assert_allclose(weights.sum(), .5, atol=1e-14)
    np.testing.assert_allclose(weights @ bary, np.ones(3)/6, atol=1e-14)
    wave, exact, diagnostics = solve_helmholtz(*mesh, np.pi, return_diagnostics=True)
    np.testing.assert_array_equal(wave[boundary], exact[boundary])
    assert diagnostics["relative_residual"] < 1e-12
    eval_errors = evaluate_errors(geo, wave, exact, np.pi)
    assert eval_errors["err_h1"] >= eval_errors["err_l2"]
    assert eval_errors["err_inf"] >= eval_errors["err_inf_nodal"]-1e-14
    synthetic = [{"h": h, "k": k, "nx": nx,
                  **{name: 2*h**1.7*k**2.3 for name in NORM_KEYS}}
                 for nx, h in [(16, .1), (32, .05), (64, .025)] for k in (1., 2., 3.)]
    for fit in fit_convergence(synthetic).values():
        np.testing.assert_allclose([fit["C"], fit["alpha"], fit["beta"], fit["r2_log"]], [2, 1.7, 2.3, 1], atol=1e-12)
    # Q2 : le même déplacement et le même système produisent la même solution.
    shifted = (shift_internal_nodes(*mesh), mesh[1], mesh[2])
    first, _ = solve_helmholtz(*shifted, np.pi)
    second, _ = solve_helmholtz(*shifted, np.pi)
    np.testing.assert_array_equal(first, second)
    return {"status": "réussite", "affine_relative_residual": residual,
            "wave_relative_residual": diagnostics["relative_residual"],
            "quadrature_relative_change": eval_errors["quadrature_relative_change"],
            "checks": "matrices, affine, normes continues, quadrature, Dirichlet, résidu, ajustement conjoint, reproductibilité"}

def main(shifted=False):
    parser = argparse.ArgumentParser(description="FEM Q2" if shifted else "FEM Q1")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--nx", nargs="+", type=int, default=list(DEFAULT_NX))
    args = parser.parse_args()
    checks = {"mesh": check_mesh_implementation(), "fem": check_fem_implementation()}
    print("Vérifications analytiques et numériques réussies.", flush=True)
    if args.check_only:
        return
    results = run_convergence_study(shifted=shifted, nx_values=args.nx)
    fits = fit_convergence(results)
    plot_convergence(results, fits, shifted)
    export_results(results, fits, shifted, checks)
    print(f"Résultats sauvegardés dans {DATA}; figures dans {FIGURES}.")


if __name__ == "__main__":
    main()
