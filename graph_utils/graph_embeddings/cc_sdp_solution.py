from graph_utils.signed_graph import SignedGraph, read_signed_graph

import numpy as np
import cvxpy as cvx

def solve_sdp(graph: SignedGraph, vtx_to_idx: dict, idx_to_vtx: dict):
    n = graph.G_plus.number_of_nodes()

    C = np.zeros((n, n))
    for i, j in graph.G_plus.edges:
        C[vtx_to_idx[i], vtx_to_idx[j]] = 1
        C[vtx_to_idx[j], vtx_to_idx[i]] = 1

    for i, j in graph.G_minus.edges:
        C[vtx_to_idx[i], vtx_to_idx[j]] = -1
        C[vtx_to_idx[j], vtx_to_idx[i]] = -1
        
    X = cvx.Variable((n, n), PSD=True)
    obj = 0.5 * cvx.trace(C @ X)
    constr = [cvx.diag(X) == 1]
    constr.append(X >= 0)

    problem = cvx.Problem(cvx.Maximize(obj), constraints=constr)

    print("Solving SDP")

    # INFO: Can choose different solver here. Need license for Mosek.
    problem.solve(solver=cvx.MOSEK, mosek_params={'MSK_IPAR_NUM_THREADS': 12}, verbose=True)

    print("SDP Solved")

    # Reconstruct X values
    u, s, v = np.linalg.svd(X.value)
    U = u @ np.diag(np.sqrt(s))

    q1 = np.random.randn(n)
    q1 = q1 / np.linalg.norm(q1)
    
    q2 = np.random.randn(n)
    q2 = q2 / np.linalg.norm(q2)
    
    signs1 = np.sign(U @ q1)
    signs2 = np.sign(U @ q2)

    clusters = {}
    for i in range(n):
        region = (int(signs1[i] < 0), int(signs2[i] < 0))
        if region not in clusters:
            clusters[region] = []
        clusters[region].append(i)
    
    clusters = [nodes for nodes in clusters.values() if nodes]
    return clusters



if __name__ == "__main__":

    #file = "data/cycle2.txt"
    file = "data/soc-sign-bitcoinotc.csv"


    graph = read_signed_graph(file)

    vtx_to_idx = {}
    idx_to_vtx = {}

    for i, vtx in enumerate(graph.G_plus.nodes):
        vtx_to_idx[vtx] = i
        idx_to_vtx[i] = vtx

    clusters = solve_sdp(graph, vtx_to_idx, idx_to_vtx)

    print(clusters)

    total_violations = 0

    for i, j in graph.G_minus.edges:
        for cluster in clusters:
            if vtx_to_idx[i] in cluster and vtx_to_idx[j] in cluster:
                total_violations += 1
                break
    
    for i, j in graph.G_plus.edges:
        for cluster in clusters:
            if vtx_to_idx[i] in cluster and vtx_to_idx[j] not in cluster:
                total_violations += 1
                break

    print("Disagreement: ", total_violations)
    
    # Calculate Agreement

    agreement = graph.G_plus.number_of_edges() + graph.G_minus.number_of_edges() - total_violations

    print("Agreement: ", agreement)