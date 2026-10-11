"""Partie 1 : réponses Maillage Q1 à Q11, tableaux à pseudo-pointeurs du cours.

Dépendances scientifiques : NumPy et Matplotlib uniquement. Les coordonnées sont
réelles (n, 2), les indices entiers commencent à zéro, les cellules sont des
triangles ou quadrangles simples dont les sommets sont ordonnés sur le contour.
"""
from pathlib import Path
from typing import NamedTuple
import argparse
import json
import os
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "build" / "cache"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = ROOT / "report" / "figures"
DATA = ROOT / "report" / "data"


def validate_connectivity(node_coords, p_elem2nodes, elem2nodes):
    """Invariants de la structure du cours (support 4, pages 6–11)."""
    xy = np.asarray(node_coords, dtype=float)
    p, conn = np.asarray(p_elem2nodes), np.asarray(elem2nodes)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Les coordonnées doivent être finies et de dimension (n, 2).")
    if p.ndim != 1 or conn.ndim != 1 or p.size == 0:
        raise ValueError("Les pseudo-pointeurs et indices doivent être des vecteurs.")
    if p.dtype.kind not in "iu" or conn.dtype.kind not in "iu":
        raise ValueError("Les indices doivent être entiers.")
    if p[0] != 0 or p[-1] != conn.size or np.any(np.diff(p) < 3):
        raise ValueError("Pseudo-pointeurs incohérents ou cellule de moins de trois sommets.")
    if np.any(~np.isin(np.diff(p), [3, 4])):
        raise ValueError("Seuls les triangles et quadrangles sont pris en charge.")
    if conn.size and (conn.min() < 0 or conn.max() >= len(xy)):
        raise ValueError("Indice de nœud hors du maillage.")
    for start, stop in zip(p[:-1], p[1:]):
        if np.unique(conn[start:stop]).size != stop - start:
            raise ValueError("Un élément contient plusieurs fois le même nœud.")
    return xy, p.astype(np.int64, copy=False), conn.astype(np.int64, copy=False)

def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def _area_tolerance(points):
    span = np.ptp(points, axis=0)
    return 64 * np.finfo(float).eps * max(float(span @ span), np.finfo(float).tiny)

def _signed_polygon_area(points):
    # Translation avant le produit vectoriel : évite la cancellation loin de l'origine.
    q = points - points[0]
    return 0.5 * np.sum(_cross(q, np.roll(q, -1, axis=0)))

def _quad_triangles(points):
    """Q8 : diagonale (0,2) d'un quadrangle strictement convexe et ordonné."""
    edges = np.roll(points, -1, axis=0)-points
    turns = _cross(edges, np.roll(edges, -1, axis=0))
    tol = _area_tolerance(points)
    if not (np.all(turns > tol) or np.all(turns < -tol)):
        raise ValueError("Le découpage exige un quadrangle convexe, non dégénéré et ordonné.")
    return ((0, 1, 2), (0, 2, 3))

def validate_geometry(node_coords, p_elem2nodes, elem2nodes, positive=True):
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    signed = compute_signed_areas(xy, p, conn)
    for e, (a, b) in enumerate(zip(p[:-1], p[1:])):
        points = xy[conn[a:b]]
        if abs(signed[e]) <= _area_tolerance(points):
            raise ValueError(f"Élément {e} dégénéré.")
        if positive and signed[e] <= 0:
            raise ValueError(f"Élément {e} inversé (orientation positive attendue).")
        if b-a == 4:
            _quad_triangles(points)
    return signed

def triangle_geometry(node_coords, p_elem2nodes, elem2nodes):
    """Géométrie vectorisée après validation du type triangulaire."""
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    if np.any(np.diff(p) != 3):
        raise ValueError("Cette opération exige un maillage triangulaire.")
    tris = conn.reshape(-1, 3)
    q = xy[tris]
    lengths = np.linalg.norm(np.roll(q, -1, axis=1)-q, axis=2)
    signed = 0.5 * _cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0])
    return tris, q, lengths, signed

def load_mesh(nodes_file, elements_file):
    """Convertit les identifiants explicites des fichiers en indices locaux zéro-based."""
    nodes = np.loadtxt(nodes_file, ndmin=2)
    elements = np.loadtxt(elements_file, ndmin=2)
    if nodes.shape[1] != 3 or elements.shape[1] not in (4, 5):
        raise ValueError("Formats attendus : id x y ; id n1 n2 n3 [n4].")
    if not np.isfinite(nodes).all() or not np.isfinite(elements).all():
        raise ValueError("Données non finies.")
    if np.any(nodes[:, 0] != np.floor(nodes[:, 0])) or np.any(elements != np.floor(elements)):
        raise ValueError("Les identifiants doivent être entiers.")
    if len(np.unique(nodes[:, 0])) != len(nodes) or len(np.unique(elements[:, 0])) != len(elements):
        raise ValueError("Identifiants dupliqués.")
    mapping = {int(value): i for i, value in enumerate(nodes[:, 0])}
    try:
        conn = np.array([mapping[int(value)] for value in elements[:, 1:].ravel()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Nœud référencé absent : {exc.args[0]}.") from exc
    stride = elements.shape[1]-1
    result = (nodes[:, 1:].copy(), np.arange(0, len(conn)+1, stride, dtype=np.int64), conn)
    validate_geometry(*result)
    return result

# Réponse Maillage Q1 — Aire par élément ; contrôle distinct de l'orientation.
def compute_signed_areas(node_coords, p_elem2nodes, elem2nodes):
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    if np.all(np.diff(p) == 3):
        q = xy[conn.reshape(-1, 3)]
        return 0.5 * _cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0])
    return np.array([_signed_polygon_area(xy[conn[a:b]]) for a, b in zip(p[:-1], p[1:])])

def compute_areas(node_coords, p_elem2nodes, elem2nodes):
    """Aires positives des triangles/quadrangles ; l'orientation reste accessible."""
    return np.abs(compute_signed_areas(node_coords, p_elem2nodes, elem2nodes))

# Réponse Maillage Q2 — Q=(sqrt(3)/6)*h_max/rho (cours 3, page 75).
def compute_aspect_ratios(node_coords, p_elem2nodes, elem2nodes):
    _, _, lengths, signed = triangle_geometry(node_coords, p_elem2nodes, elem2nodes)
    rho = np.divide(np.abs(signed), lengths.sum(axis=1)/2,
                    out=np.zeros(len(signed)), where=lengths.sum(axis=1) > 0)
    return np.divide(np.sqrt(3)/6 * lengths.max(axis=1), rho,
                     out=np.full(len(signed), np.inf), where=rho > 0)

# Réponse Maillage Q3 — « medium » : moyenne arithmétique par défaut, documentée.
def compute_edge_length_factors(node_coords, p_elem2nodes, elem2nodes, convention="mean"):
    _, _, lengths, _ = triangle_geometry(node_coords, p_elem2nodes, elem2nodes)
    if convention not in ("mean", "median"):
        raise ValueError("La convention doit être 'mean' ou 'median'.")
    denominator = lengths.mean(axis=1) if convention == "mean" else np.median(lengths, axis=1)
    return np.divide(lengths.min(axis=1), denominator,
                     out=np.zeros(len(lengths)), where=denominator > 0)

# Réponse Maillage Q4 — Déplacement proportionnel aux longueurs locales.
def _shift(node_coords, p_elem2nodes, elem2nodes, direction, magnitude, fixed):
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    old_signed = validate_geometry(xy, p, conn, positive=False)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("La direction doit être un vecteur fini non nul de dimension 2.")
    if not np.isfinite(magnitude) or magnitude < 0:
        raise ValueError("L'amplitude relative doit être finie et positive ou nulle.")
    direction = direction/np.linalg.norm(direction)
    scale, count = np.zeros(len(xy)), np.zeros(len(xy))
    for a, b in zip(p[:-1], p[1:]):
        ids = conn[a:b]
        points = xy[ids]
        length = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).mean()
        np.add.at(scale, ids, length)
        np.add.at(count, ids, 1)
    local = np.divide(scale, count, out=np.zeros_like(scale), where=count > 0)
    result = xy + magnitude*local[:, None]*direction
    result[fixed] = xy[fixed]
    new_signed = validate_geometry(result, p, conn, positive=False)
    if np.any(old_signed*new_signed <= 0):
        raise ValueError("Le déplacement inverse au moins un élément : réduire l'amplitude.")
    return result

def shift_nodes(node_coords, p_elem2nodes, elem2nodes, direction=(1., 0.), magnitude=0.1):
    return _shift(node_coords, p_elem2nodes, elem2nodes, direction, magnitude, [])

# Réponse Maillage Q5 — Frontière topologique fixe, y compris sur bord irrégulier.
def shift_internal_nodes(node_coords, p_elem2nodes, elem2nodes, boundary_nodes=None,
                         direction=(1., 0.), magnitude=0.1):
    actual = get_boundary_mesh(node_coords, p_elem2nodes, elem2nodes).l2g
    if boundary_nodes is not None and not np.array_equal(np.unique(boundary_nodes), actual):
        raise ValueError("La liste fournie ne correspond pas à la frontière topologique.")
    return _shift(node_coords, p_elem2nodes, elem2nodes, direction, magnitude, actual)

def compute_angles(node_coords, p_elem2nodes, elem2nodes):
    """Complément Q6 : angles intérieurs des triangles, en radians."""
    _, q, _, _ = triangle_geometry(node_coords, p_elem2nodes, elem2nodes)
    a, b = np.roll(q, -1, axis=1)-q, np.roll(q, 1, axis=1)-q
    den = np.linalg.norm(a, axis=2)*np.linalg.norm(b, axis=2)
    cosines = np.divide(np.sum(a*b, axis=2), den, out=np.ones_like(den), where=den > 0)
    return np.arccos(np.clip(cosines, -1, 1))

def compute_pointedness(node_coords, p_elem2nodes, elem2nodes):
    """Indicateur complémentaire : cosinus du plus petit angle, hors critères Q2/Q3."""
    return np.cos(compute_angles(node_coords, p_elem2nodes, elem2nodes).min(axis=1))

def mesh_statistics(node_coords, p_elem2nodes, elem2nodes):
    signed = validate_geometry(node_coords, p_elem2nodes, elem2nodes)
    _, _, lengths, _ = triangle_geometry(node_coords, p_elem2nodes, elem2nodes)
    angles = compute_angles(node_coords, p_elem2nodes, elem2nodes)
    return {"nodes": len(node_coords), "elements": len(signed), "area": float(signed.sum()),
            "area_min": float(signed.min()), "area_max": float(signed.max()),
            "h_mean": float(lengths.max(axis=1).mean()), "h_max": float(lengths.max()),
            "aspect_max": float(compute_aspect_ratios(node_coords, p_elem2nodes, elem2nodes).max()),
            "edge_min": float(compute_edge_length_factors(node_coords, p_elem2nodes, elem2nodes).min()),
            "angle_min_deg": float(np.rad2deg(angles.min())),
            "angle_max_deg": float(np.rad2deg(angles.max())), "invalid_elements": 0}

# Réponse Maillage Q6 — Histogrammes comparés et statistiques exportées.
def plot_mesh_analysis(node_coords, p_elem2nodes, elem2nodes, output_dir=FIGURES):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = (node_coords, p_elem2nodes, elem2nodes)
    shifted = (shift_internal_nodes(*base), p_elem2nodes, elem2nodes)
    functions = [compute_areas, compute_aspect_ratios, compute_edge_length_factors, compute_pointedness]
    labels = ["Aire des triangles", "Rapport d'aspect du cours $Q_T$",
              "Facteur d'arête $E_T$ (moyenne)", "Cosinus du plus petit angle"]
    for number, (function, label) in enumerate(zip(functions, labels), 1):
        values = [function(*mesh) for mesh in (base, shifted)]
        finite = np.concatenate([v[np.isfinite(v)] for v in values])
        bins = np.histogram_bin_edges(finite, bins=20)
        fig, ax = plt.subplots(figsize=(7, 4))
        for v, name in zip(values, ["Frontière irrégulière", "Après déplacement interne"]):
            ax.hist(v[np.isfinite(v)], bins=bins, alpha=0.6, label=name, edgecolor="black")
        ax.set(xlabel=label, ylabel="Nombre d'éléments", title="Analyse de qualité — Maillage Q6")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir/f"Figure{number}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mesh, title in zip(axes, (base, shifted), ("Avant déplacement", "Après déplacement interne")):
        ax.triplot(mesh[0][:, 0], mesh[0][:, 1], mesh[2].reshape(-1, 3), lw=0.35)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    fig.tight_layout()
    fig.savefig(output_dir/"mesh_comparison.png", dpi=180)
    plt.close(fig)
    return {"before": mesh_statistics(*base), "after": mesh_statistics(*shifted)}

# Réponse Maillage Q7 — Barycentre des sommets, de poids égaux.
def compute_barycenter(node_coords, p_elem2nodes, elem2nodes, elemid):
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    if not isinstance(elemid, (int, np.integer)) or not 0 <= elemid < len(p)-1:
        raise ValueError("Indice d'élément invalide.")
    q = xy[conn[p[elemid]:p[elemid+1]]]
    if len(q) == 4:
        _quad_triangles(q)
    if abs(_signed_polygon_area(q)) <= _area_tolerance(q):
        raise ValueError("Barycentre non traité pour un élément dégénéré.")
    return q.mean(axis=0)

# Réponse Maillage Q8 — Découpage contrôlé, triangles conservés dans un maillage mixte.
def split_quads_to_tris(node_coords, p_elem2nodes, elem2nodes):
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    validate_geometry(xy, p, conn, positive=False)
    output = []
    for a, b in zip(p[:-1], p[1:]):
        ids = conn[a:b]
        if b-a == 3:
            output.extend(ids)
        else:
            for tri in _quad_triangles(xy[ids]):
                output.extend(ids[list(tri)])
    new_conn = np.array(output, dtype=np.int64)
    return xy.copy(), np.arange(0, len(new_conn)+1, 3, dtype=np.int64), new_conn

# Réponse Maillage Q9(i) — Ajout d'un élément et mise à jour des pseudo-pointeurs.
def add_element(node_coords, p_elem2nodes, elem2nodes, new_elem_nodes):
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    ids = np.asarray(new_elem_nodes)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or ids.size not in (3, 4):
        raise ValueError("Un élément doit contenir trois ou quatre indices entiers.")
    key = set(ids.tolist())
    if any(set(conn[a:b].tolist()) == key for a, b in zip(p[:-1], p[1:])):
        raise ValueError("Cet élément existe déjà.")
    result = (xy.copy(), np.append(p, p[-1]+len(ids)), np.append(conn, ids))
    validate_geometry(*result)
    get_boundary_mesh(*result)  # Incidence et orientations des arêtes.
    return result

# Réponse Maillage Q9(ii) — Suppression, décalage des pointeurs suivants.
def remove_element(node_coords, p_elem2nodes, elem2nodes, elemid):
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    if not isinstance(elemid, (int, np.integer)) or not 0 <= elemid < len(p)-1:
        raise ValueError("Indice d'élément invalide.")
    a, b = p[elemid:elemid+2]
    new_p = np.delete(p, elemid+1)
    new_p[elemid+1:] -= b-a
    return xy.copy(), new_p, np.delete(conn, np.s_[a:b])

# Réponse Maillage Q9(iii) — Ajout d'un nœud isolé, sans retriangulation implicite.
def add_node(node_coords, p_elem2nodes, elem2nodes, new_node_coords):
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    point = np.asarray(new_node_coords, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Un nœud doit avoir deux coordonnées finies.")
    return np.vstack((xy, point)), p.copy(), conn.copy()

# Réponse Maillage Q9(iv) — Suppression sûre d'un nœud isolé et renumérotation.
def remove_node(node_coords, p_elem2nodes, elem2nodes, nodeid):
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    if not isinstance(nodeid, (int, np.integer)) or not 0 <= nodeid < len(xy):
        raise ValueError("Indice de nœud invalide.")
    if np.any(conn == nodeid):
        raise ValueError("Supprimer d'abord les éléments incidents à ce nœud.")
    return np.delete(xy, nodeid, axis=0), p.copy(), conn-(conn > nodeid)

# Réponse Maillage Q10 — Frontière fractale auto-affine, niveau indépendant, aire fixe.
def generate_fractal_boundary_mesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy,
                                     level=1, amplitude=0.05, quadrangles=False):
    bounds = np.array([xmin, xmax, ymin, ymax, amplitude], dtype=float)
    if not np.isfinite(bounds).all() or xmax <= xmin or ymax <= ymin:
        raise ValueError("Bornes finies et domaine d'aire positive requis.")
    for value in (nelemsx, nelemsy, level):
        if not isinstance(value, (int, np.integer)):
            raise ValueError("Les résolutions et le niveau doivent être entiers.")
    if nelemsx < 1 or nelemsy < 1 or level < 0:
        raise ValueError("Résolutions positives et niveau positif ou nul requis.")
    # Q10 : g est polygonale ; chaque niveau réduit l'échelle horizontale
    # par 2 et l'échelle verticale par 0.6 (construction auto-affine).
    ratio = 0.6
    if amplitude < 0 or amplitude/(1-ratio) >= ymax-ymin:
        raise ValueError("La borne amplitude/(1-0.6) doit rester inférieure à la hauteur.")
    if level > 0 and (amplitude == 0 or (1 << (level+1)) > nelemsx
                      or nelemsx % (1 << (level+1))):
        raise ValueError("Le maillage doit résoudre tous les sommets : nx multiple de 2**(niveau+1).")
    x = np.linspace(xmin, xmax, nelemsx+1)
    eta = np.linspace(0, 1, nelemsy+1)
    t = (x-xmin)/(xmax-xmin)
    d = np.zeros_like(t)
    for iteration in range(level):
        z = np.mod((2**iteration)*t, 1.0)
        # Motif signé : triangle positif sur [0,1/2], négatif sur [1/2,1].
        local = np.mod(2*z, 1.0)
        tent = 1-np.abs(2*local-1)
        motif = np.where(z < .5, tent, -tent)
        d += amplitude*ratio**iteration*motif
    d[[0, -1]] = 0.0
    X, E = np.meshgrid(x, eta)
    Y = ymin+E*((ymax-ymin)+d)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    conn = []
    for j in range(nelemsy):
        for i in range(nelemsx):
            n0 = j*(nelemsx+1)+i
            n1, n2, n3 = n0+1, n0+nelemsx+1, n0+nelemsx+2
            if quadrangles:
                conn.extend([n0, n1, n3, n2])
            else:
                conn.extend([n0, n1, n2, n1, n3, n2])
    conn = np.array(conn, dtype=np.int64)
    stride = 4 if quadrangles else 3
    p = np.arange(0, len(conn)+1, stride, dtype=np.int64)
    area = validate_geometry(xy, p, conn).sum()
    expected = (xmax-xmin)*(ymax-ymin)
    if not np.isclose(area, expected, rtol=1e-12, atol=1e-14*expected):
        raise ValueError("La conservation de l'aire du domaine n'est pas vérifiée.")
    return xy, p, conn

class BoundaryMesh(NamedTuple):
    node_coords: np.ndarray
    p_elem2nodes: np.ndarray
    elem2nodes: np.ndarray
    l2g: np.ndarray
    owners: np.ndarray

# Réponse Maillage Q11 (facultative) — Sous-maillage local du bord et l2g.
def get_boundary_mesh(node_coords, p_elem2nodes, elem2nodes):
    xy, p, conn = validate_connectivity(node_coords, p_elem2nodes, elem2nodes)
    edges = {}
    for e, (a, b) in enumerate(zip(p[:-1], p[1:])):
        ids = conn[a:b]
        for i, j in zip(ids, np.roll(ids, -1)):
            key = tuple(sorted((int(i), int(j))))
            edges.setdefault(key, []).append((int(i), int(j), e))
    if any(len(v) > 2 for v in edges.values()):
        raise ValueError("Arête non conforme : plus de deux éléments incidents.")
    for occurrences in edges.values():
        if len(occurrences) == 2 and occurrences[0][:2] == occurrences[1][:2]:
            raise ValueError("Orientations incompatibles sur une arête intérieure.")
    boundary = [v[0] for v in edges.values() if len(v) == 1]
    global_edges = np.array([(i, j) for i, j, _ in boundary], dtype=np.int64).reshape(-1, 2)
    l2g = np.unique(global_edges)
    g2l = np.full(len(xy), -1, dtype=np.int64)
    g2l[l2g] = np.arange(len(l2g))
    return BoundaryMesh(xy[l2g].copy(), np.arange(0, 2*len(boundary)+1, 2, dtype=np.int64),
                        g2l[global_edges].ravel(), l2g,
                        np.array([e for _, _, e in boundary], dtype=np.int64))

def check_mesh_implementation():
    """Vérifications discriminantes Q1–Q11, indépendantes des messages affichés."""
    xy = np.array([[0., 0.], [1., 0.], [0., 1.]])
    p, conn = np.array([0, 3]), np.array([0, 1, 2])
    np.testing.assert_allclose(compute_areas(xy, p, conn), [0.5])
    expected = np.sqrt(3)/6*np.sqrt(2)/(1-np.sqrt(2)/2)
    np.testing.assert_allclose(compute_aspect_ratios(xy, p, conn), [expected])
    np.testing.assert_allclose(compute_edge_length_factors(xy, p, conn), [3/(2+np.sqrt(2))])
    equi = np.array([[0., 0.], [1., 0.], [0.5, np.sqrt(3)/2]])
    np.testing.assert_allclose(compute_aspect_ratios(equi, p, conn), [1.])
    np.testing.assert_allclose(compute_edge_length_factors(equi, p, conn), [1.])
    np.testing.assert_allclose(compute_barycenter(xy, p, conn, 0), [1/3, 1/3])
    trapezoid = np.array([[0., 0.], [2., 0.], [1., 1.], [0., 1.]])
    np.testing.assert_allclose(compute_barycenter(trapezoid, np.array([0, 4]),
                                                np.array([0, 1, 2, 3]), 0), [.75, .5])
    for level in range(5):
        mesh = generate_fractal_boundary_mesh(0, 1, 0, 1, 64, 20, level)
        np.testing.assert_allclose(compute_areas(*mesh).sum(), 1., atol=1e-13)
        boundary = get_boundary_mesh(*mesh)
        assert len(boundary.l2g) == 168 and len(boundary.owners) == 168
        assert boundary.elem2nodes.max() < len(boundary.node_coords)
        edges = boundary.l2g[boundary.elem2nodes.reshape(-1, 2)]
        owners = mesh[2].reshape(-1, 3)[boundary.owners]
        assert all(set(edge).issubset(owner) for edge, owner in zip(edges, owners))
        shifted = shift_internal_nodes(*mesh)
        np.testing.assert_array_equal(shifted[boundary.l2g], mesh[0][boundary.l2g])
        np.testing.assert_array_equal(shifted, shift_internal_nodes(*mesh))
        np.testing.assert_allclose(compute_areas(shifted, mesh[1], mesh[2]).sum(), 1., atol=1e-13)
        validate_geometry(shift_nodes(*mesh), mesh[1], mesh[2])
    coarse = generate_fractal_boundary_mesh(0, 1, 0, 1, 32, 4, level=3)
    fine = generate_fractal_boundary_mesh(0, 1, 0, 1, 64, 8, level=3)
    np.testing.assert_allclose(coarse[0][-33:, 1], fine[0][-65::2, 1])
    top = fine[0][-65:]
    np.testing.assert_allclose(top[:, 1]+top[::-1, 1], 2., atol=1e-14)
    different = generate_fractal_boundary_mesh(0, 1, 0, 1, 64, 8, level=4)
    assert not np.allclose(top[:, 1], different[0][-65:, 1])
    quad = generate_fractal_boundary_mesh(0, 1, 0, 1, 2, 2, 0, quadrangles=True)
    tri = split_quads_to_tris(*quad)
    assert len(tri[1])-1 == 8
    np.testing.assert_allclose(compute_areas(*quad).sum(), compute_areas(*tri).sum())
    concave = np.array([[0., 0.], [2., 0.], [.5, .5], [0., 2.]])
    cp, cc = np.array([0, 4]), np.array([0, 1, 2, 3])
    # Q8 : les quadrangles concaves sortent du cadre retenu et sont refusés.
    # Q9 : enlever puis réinsérer une cellule ; nœud isolé avant et après les indices utilisés.
    removed = remove_element(*tri, 0)
    restored = add_element(*removed, tri[2][:3])
    np.testing.assert_allclose(compute_areas(*restored).sum(), 1.)
    isolated = add_node(*tri, [2., 2.])
    for actual, original in zip(remove_node(*isolated, len(isolated[0])-1), tri):
        np.testing.assert_array_equal(actual, original)
    inserted = (np.vstack(([2., 2.], xy)), p, conn+1)
    for actual, original in zip(remove_node(*inserted, 0), (xy, p, conn)):
        np.testing.assert_array_equal(actual, original)
    invalid_actions = [lambda: split_quads_to_tris(concave, cp, cc),
                       lambda: remove_node(xy, p, conn, 1),
                       lambda: add_element(xy, p, conn, [0, 1, 2]),
                       lambda: generate_fractal_boundary_mesh(0, 1, 0, 1, 20, 20, 10),
                       lambda: shift_internal_nodes(*mesh, magnitude=10),
                       lambda: split_quads_to_tris(np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]]), cp, cc)]
    for action in invalid_actions:
        try:
            action()
        except ValueError:
            pass
        else:
            raise AssertionError("Un cas invalide n'a pas été refusé.")
    # Q8 : maillage mixte et maillage vide.
    mixed = (np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [2., 0.]]),
             np.array([0, 4, 7]), np.array([0, 1, 2, 3, 1, 4, 2]))
    assert len(split_quads_to_tris(*mixed)[1]) == 4
    assert len(split_quads_to_tris(np.empty((0, 2)), np.array([0]), np.array([], dtype=int))[2]) == 0
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        nodes_file, elements_file = Path(tmp)/"nodes.txt", Path(tmp)/"elements.txt"
        nodes_file.write_text("10 0 0\n30 1 0\n80 0 1\n", encoding="utf-8")
        elements_file.write_text("99 10 30 80\n", encoding="utf-8")
        loaded = load_mesh(nodes_file, elements_file)
        np.testing.assert_array_equal(loaded[2], conn)
    return {"status": "réussite", "checks": "Q1–Q11 : géométrie, critères, bord, déplacement, topologie, chargement"}

def main():
    parser = argparse.ArgumentParser(description="Maillage : vérifications et résultats Q1–Q11.")
    parser.add_argument("--check-only", action="store_true", help="Vérifier sans régénérer les figures.")
    args = parser.parse_args()
    checks = check_mesh_implementation()
    print(checks)
    if args.check_only:
        return
    DATA.mkdir(parents=True, exist_ok=True)
    mesh = generate_fractal_boundary_mesh(0, 1, 0, 1, 64, 20, level=3)
    results = plot_mesh_analysis(*mesh)
    results["levels"] = {str(level): mesh_statistics(*generate_fractal_boundary_mesh(0, 1, 0, 1, 64, 20, level))
                         for level in range(5)}
    results["parameters"] = {"nx": 64, "ny": 20, "amplitude": .05, "level": 3, "vertical_ratio": .6, "boundary": "self_affine_fractal",
                             "shift_direction": [1., 0.], "shift_magnitude": .1,
                             "edge_convention": "mean"}
    results["checks"] = checks
    (DATA/"mesh_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for level in range(5):
        points = generate_fractal_boundary_mesh(0, 1, 0, 1, 64, 1, level)[0][-65:]
        ax.plot(points[:, 0], points[:, 1], label=f"Niveau {level}", lw=1.2)
    ax.set(xlabel="x", ylabel="Hauteur du bord", title="Approximations de la frontière fractale")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(FIGURES/"fractal_levels.png", dpi=180); plt.close(fig)
    headers = r"Niveau & Aire totale & Aire min. & $Q_{\max}$ & $E_{\min}$ & Angle min. ($^\circ$)" + " " + chr(92)*2
    rows = [f"{level} & {r['area']:.12f} & {r['area_min']:.6f} & {r['aspect_max']:.4f} & "
            f"{r['edge_min']:.4f} & {r['angle_min_deg']:.2f}"+r" \\"
            for level, r in results["levels"].items()]
    (DATA/"mesh_levels.tex").write_text("\\begin{tabular}{rrrrrr}\n\\toprule\n"+headers+
        "\n\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    print(f"Résultats : {DATA}; figures : {FIGURES}")


if __name__ == "__main__":
    main()
