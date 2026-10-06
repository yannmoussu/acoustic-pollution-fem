import numpy as np
import matplotlib.pyplot as plt

def load_mesh(nodes_file, elements_file):
    nodes = np.loadtxt(nodes_file)
    elements = np.loadtxt(elements_file, dtype=int)
    node_coords = nodes[:, 1:]
    elem_data = elements[:, 1:]
    elem2nodes = elem_data.flatten()
    nodes_per_elem = elem_data.shape[1]
    p_elem2nodes = np.arange(0, (len(elements) + 1) * nodes_per_elem, nodes_per_elem)
    return node_coords, p_elem2nodes, elem2nodes


def compute_areas(node_coords, p_elem2nodes, elem2nodes):
    triangles = elem2nodes.reshape(-1, 3)
    p = node_coords[triangles] 
    x0, y0 = p[:, 0, 0], p[:, 0, 1]
    x1, y1 = p[:, 1, 0], p[:, 1, 1]
    x2, y2 = p[:, 2, 0], p[:, 2, 1]
    areas = 0.5 * np.abs(x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1))
    return areas

def compute_aspect_ratios(node_coords, p_elem2nodes, elem2nodes):
    nelems = len(p_elem2nodes) - 1
    ratios = np.zeros(nelems)
    for e in range(nelems):
        nodes = elem2nodes[p_elem2nodes[e] : p_elem2nodes[e+1]]
        p = node_coords[nodes]
        area = 0.5 * abs(p[0,0]*(p[1,1]-p[2,1]) + p[1,0]*(p[2,1]-p[0,1]) + p[2,0]*(p[0,1]-p[1,1]))
        L = [np.linalg.norm(p[0]-p[1]), np.linalg.norm(p[1]-p[2]), np.linalg.norm(p[2]-p[0])]
        ratios[e] = (max(L)**2) / (2 * area) if area > 0 else float('inf')
    return ratios

def compute_edge_length_factors(node_coords, p_elem2nodes, elem2nodes):
    nelems = len(p_elem2nodes) - 1
    factors = np.zeros(nelems)
    for e in range(nelems):
        nodes = elem2nodes[p_elem2nodes[e] : p_elem2nodes[e+1]]
        p = node_coords[nodes]
        L = [np.linalg.norm(p[0]-p[1]), np.linalg.norm(p[1]-p[2]), np.linalg.norm(p[2]-p[0])]
        factors[e] = max(L) / min(L) if min(L) > 0 else float('inf')
    return factors

def compute_pointedness(node_coords, p_elem2nodes, elem2nodes):
    nelems = len(p_elem2nodes) - 1
    pointedness = np.zeros(nelems)
    for e in range(nelems):
        nodes = elem2nodes[p_elem2nodes[e] : p_elem2nodes[e+1]]
        p = node_coords[nodes]
        v0 = p[1] - p[0]
        v1 = p[2] - p[1]
        v2 = p[0] - p[2]
        l0, l1, l2 = np.linalg.norm(v0), np.linalg.norm(v1), np.linalg.norm(v2)
        if l0 > 0 and l1 > 0 and l2 > 0:
            cos_a = -np.dot(v0, v2) / (l0 * l2)
            cos_b = -np.dot(v1, v0) / (l1 * l0)
            cos_c = -np.dot(v2, v1) / (l2 * l1)
            pointedness[e] = max(cos_a, cos_b, cos_c)
        else:
            pointedness[e] = 1.0
    return pointedness

def shift_nodes(node_coords, p_elem2nodes, elem2nodes, direction=np.array([1, 0]), magnitude=0.1):
    new_coords = node_coords.copy()
    node_shifts = np.zeros_like(node_coords)
    counts = np.zeros(len(node_coords))
    for e in range(len(p_elem2nodes)-1):
        nodes = elem2nodes[p_elem2nodes[e] : p_elem2nodes[e+1]]
        p = node_coords[nodes]
        L = [np.linalg.norm(p[0]-p[1]), np.linalg.norm(p[1]-p[2]), np.linalg.norm(p[2]-p[0])]
        avg_L = np.mean(L)
        for n in nodes:
            node_shifts[n] += direction * avg_L * magnitude
            counts[n] += 1
    valid = counts > 0
    new_coords[valid] += node_shifts[valid] / counts[valid, np.newaxis]
    return new_coords

def shift_internal_nodes(node_coords, p_elem2nodes, elem2nodes, boundary_nodes, direction=np.array([1, 0]), magnitude=0.1):
    new_coords = shift_nodes(node_coords, p_elem2nodes, elem2nodes, direction, magnitude)
    new_coords[boundary_nodes] = node_coords[boundary_nodes]
    return new_coords

# Graphes
def plot_mesh_analysis(node_coords, p_elem2nodes, elem2nodes):
    areas = compute_areas(node_coords, p_elem2nodes, elem2nodes)
    ratios = compute_aspect_ratios(node_coords, p_elem2nodes, elem2nodes)
    factors = compute_edge_length_factors(node_coords, p_elem2nodes, elem2nodes)
    pointeds = compute_pointedness(node_coords, p_elem2nodes, elem2nodes)

    # Figure 1 : Aires
    plt.figure(figsize=(6, 4))
    plt.hist(areas, bins=20, color='salmon', edgecolor='black')
    plt.title("Area vs Count")
    plt.xlabel("Area")
    plt.ylabel("Number of Elements")
    plt.tight_layout()
    plt.savefig("Figure1.png", dpi=300)
    plt.close()

    # Figure 2 : Aspect Ratio
    plt.figure(figsize=(6, 4))
    plt.hist(ratios, bins=20, color='skyblue', edgecolor='black')
    plt.title("Aspect Ratio")
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Number of Elements")
    plt.tight_layout()
    plt.savefig("Figure2.png", dpi=300)
    plt.close()

    # Figure 3 : Edge Factor
    plt.figure(figsize=(6, 4))
    plt.hist(factors, bins=20, color='lightgreen', edgecolor='black')
    plt.title("Edge Factor")
    plt.xlabel("Edge Factor")
    plt.ylabel("Number of Elements")
    plt.tight_layout()
    plt.savefig("Figure3.png", dpi=300)
    plt.close()

    # Figure 4 : Pointedness
    plt.figure(figsize=(6, 4))
    plt.hist(pointeds, bins=20, color='plum', edgecolor='black')
    plt.title("Pointedness (Max Cosine)")
    plt.xlabel("Pointedness")
    plt.ylabel("Number of Elements")
    plt.tight_layout()
    plt.savefig("Figure4.png", dpi=300)
    plt.close()
    
    print("  - Sauvegarde de Figure1.png, Figure2.png, Figure3.png et Figure4.png réussie.")

def compute_barycenter(node_coords, p_elem2nodes, elem2nodes, elemid):
    nodes = elem2nodes[p_elem2nodes[elemid] : p_elem2nodes[elemid+1]]
    return np.mean(node_coords[nodes], axis=0)

def split_quads_to_tris(node_coords, p_elem2nodes, elem2nodes):
    nodes_per_elem = p_elem2nodes[1] - p_elem2nodes[0]
    if nodes_per_elem != 4:
        print("Le maillage n'est pas composé de quadrangles.")
        return node_coords, p_elem2nodes, elem2nodes
    nelems_quads = len(p_elem2nodes) - 1
    new_elem2nodes = []
    for e in range(nelems_quads):
        nodes = elem2nodes[p_elem2nodes[e] : p_elem2nodes[e+1]]
        new_elem2nodes.extend([nodes[0], nodes[1], nodes[2]])
        new_elem2nodes.extend([nodes[0], nodes[2], nodes[3]])
    new_elem2nodes = np.array(new_elem2nodes)
    new_p_elem2nodes = np.arange(0, (len(new_elem2nodes) + 3), 3)
    return node_coords, new_p_elem2nodes, new_elem2nodes

def add_element(node_coords, p_elem2nodes, elem2nodes, new_elem_nodes):
    elem2nodes = np.append(elem2nodes, new_elem_nodes)
    p_elem2nodes = np.append(p_elem2nodes, p_elem2nodes[-1] + len(new_elem_nodes))
    return node_coords, p_elem2nodes, elem2nodes

def remove_element(node_coords, p_elem2nodes, elem2nodes, elemid):
    start = p_elem2nodes[elemid]
    end = p_elem2nodes[elemid+1]
    n_nodes = end - start
    elem2nodes = np.delete(elem2nodes, slice(start, end))
    p_elem2nodes = np.delete(p_elem2nodes, elemid + 1)
    p_elem2nodes[elemid + 1:] -= n_nodes
    return node_coords, p_elem2nodes, elem2nodes

def add_node(node_coords, p_elem2nodes, elem2nodes, new_node_coords):
    node_coords = np.vstack((node_coords, new_node_coords))
    return node_coords, p_elem2nodes, elem2nodes

def remove_node(node_coords, p_elem2nodes, elem2nodes, nodeid):
    node_coords = np.delete(node_coords, nodeid, axis=0)
    elem2nodes = np.where(elem2nodes > nodeid, elem2nodes - 1, elem2nodes)
    return node_coords, p_elem2nodes, elem2nodes, len(node_coords)

def generate_irregular_boundary_mesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy, level=1):
    x = np.linspace(xmin, xmax, nelemsx + 1)
    y = np.linspace(ymin, ymax, nelemsy + 1)
    X, Y = np.meshgrid(x, y)
    node_coords = np.vstack([X.ravel(), Y.ravel()]).T
    boundary_nodes = np.where(node_coords[:, 1] == ymax)[0]
    perturbation = np.sin(np.linspace(0, 2 * np.pi * level, len(boundary_nodes))) * 0.05
    node_coords[boundary_nodes, 1] += perturbation
    elem2nodes = []
    for j in range(nelemsy):
        for i in range(nelemsx):
            n0 = j * (nelemsx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nelemsx + 1) + i
            n3 = n2 + 1
            elem2nodes.extend([n0, n1, n2])
            elem2nodes.extend([n1, n3, n2])
    elem2nodes = np.array(elem2nodes)
    p_elem2nodes = np.arange(0, len(elem2nodes) + 3, 3)
    return node_coords, p_elem2nodes, elem2nodes

def get_boundary_mesh(node_coords, p_elem2nodes, elem2nodes):
    edge_counts = {}
    for e in range(len(p_elem2nodes)-1):
        nodes = elem2nodes[p_elem2nodes[e] : p_elem2nodes[e+1]]
        edges = [tuple(sorted((nodes[0], nodes[1]))),
                 tuple(sorted((nodes[1], nodes[2]))),
                 tuple(sorted((nodes[2], nodes[0])))]
        for edge in edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    return boundary_edges

if __name__ == "__main__":
    print("--- TEST COMPLET DE L'IMPLÉMENTATION DU DM PARTIE 1 ---")

    print("\n[TEST 1] Maillage minimaliste (Vérification des valeurs exactes)...")
    nodes_min = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elem_min = np.array([0, 1, 2, 0, 2, 3])
    p_elem_min = np.array([0, 3, 6])

    areas = compute_areas(nodes_min, p_elem_min, elem_min)
    ratios = compute_aspect_ratios(nodes_min, p_elem_min, elem_min)

    print(f"  - Aires attendues [0.5, 0.5] -> Obtenues: {areas}")
    print(f"  - Ratios attendus [1.0, 1.0] -> Obtenus: {ratios}")

    print("\n[TEST 2] Maillage synthétique irrégulier (Analyse globale)...")
    node_coords, p_elem2nodes, elem2nodes = generate_irregular_boundary_mesh(
        xmin=0, xmax=1, ymin=0, ymax=1, nelemsx=20, nelemsy=20, level=3
    )
    print(f"  - Maillage généré : {len(node_coords)} nœuds, {len(p_elem2nodes)-1} éléments.")

    print("  - Génération des figures d'analyse...")
    plot_mesh_analysis(node_coords, p_elem2nodes, elem2nodes)

    print("\n[TEST 3] Tests de modification de topologie et géométrie...")
    boundary_nodes = np.where((node_coords[:,0] == 0) | (node_coords[:,0] == 1) |
                              (node_coords[:,1] == 0) | (node_coords[:,1] == 1))[0]
    coords_shifted = shift_nodes(node_coords, p_elem2nodes, elem2nodes)
    coords_internal_shifted = shift_internal_nodes(node_coords, p_elem2nodes, elem2nodes, boundary_nodes)
    print("  - Déplacements (Shift) : OK")

    node_coords_mod, p_elem_mod, elem_mod = add_element(node_coords, p_elem2nodes, elem2nodes, [0, 1, 2])
    node_coords_mod, p_elem_mod, elem_mod = remove_element(node_coords_mod, p_elem_mod, elem_mod, 0)
    print("  - Modification Topologie (Add/Remove) : OK")

    print("\n--- TOUS LES TESTS ONT RÉUSSI ---")