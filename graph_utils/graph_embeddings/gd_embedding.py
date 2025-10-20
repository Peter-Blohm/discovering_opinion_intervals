import numpy as np

from graph_utils.signed_graph import SignedGraph


# ----------------------------
# Helper functions for smooth approximations
# ----------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def smooth_max(a, b, eps=1e-6):
    # a smooth approximation to max(a,b) using the average plus half the (smoothed) absolute difference
    return 0.5 * (a + b) + 0.5 * np.sqrt((a - b)**2 + eps)

def smooth_min(a, b, eps=1e-6):
    # smooth approximation to min(a,b)
    return 0.5 * (a + b) - 0.5 * np.sqrt((a - b)**2 + eps)

# ----------------------------
# Loss function and its gradients
# ----------------------------

def compute_loss(s, t, pos_edges, neg_edges, k=10.0, eps=1e-6):
    """
    s, t are numpy arrays of shape (n,)
    pos_edges and neg_edges are lists of tuples (i, j) where i,j are indices into s and t.
    k is a scaling parameter for the sigmoid.
    """
    loss = 0.0
    # loss for positive edges (want intervals to overlap, i.e. delta > 0)
    for (i, j) in pos_edges:
        max_s = smooth_max(s[i], s[j], eps)
        min_t = smooth_min(t[i], t[j], eps)
        delta = min_t - max_s
        loss += (sigmoid(-k * delta))**2

    # loss for negative edges (want intervals to be disjoint, i.e. delta < 0)
    for (i, j) in neg_edges:
        max_s = smooth_max(s[i], s[j], eps)
        min_t = smooth_min(t[i], t[j], eps)
        delta = min_t - max_s
        loss += (sigmoid(k * delta))**2

    # loss for each vertex to ensure s < t
    for i in range(len(s)):
        loss += (sigmoid(k*(s[i]-t[i])))**2

    return loss

def compute_gradients(s, t, pos_edges, neg_edges, k=10.0, eps=1e-6):
    """
    Computes gradients d(loss)/d(s) and d(loss)/d(t) and returns two arrays of shape (n,)
    using analytical gradients based on our smooth approximations.
    """
    n = len(s)
    grad_s = np.zeros(n)
    grad_t = np.zeros(n)

    # helper: derivative of the squared sigmoid loss
    # For a term L = (sigma(u))^2, dL/du = 2 sigma(u) * sigma(u)*(1-sigma(u)) * (du/d(variable))
    # Note: For our positive edge terms, u = -k*delta; for negative, u = k*delta.

    # Process positive edges
    for (i, j) in pos_edges:
        # Compute smooth functions and their derivatives
        # For s's: we need derivative of smooth_max(s_i,s_j)
        diff_s = s[i] - s[j]
        A = np.sqrt(diff_s**2 + eps)
        # derivative of smooth_max(s[i], s[j]) wrt s[i]
        d_max_si = 0.5 * (1 + diff_s / A)
        d_max_sj = 0.5 * (1 - diff_s / A)

        # For t's: derivative of smooth_min(t[i], t[j])
        diff_t = t[i] - t[j]
        B = np.sqrt(diff_t**2 + eps)
        d_min_ti = 0.5 * (1 - diff_t / B)
        d_min_tj = 0.5 * (1 + diff_t / B)

        max_s = smooth_max(s[i], s[j], eps)
        min_t = smooth_min(t[i], t[j], eps)
        delta = min_t - max_s

        u = -k * delta
        sigma_u = sigmoid(u)
        factor = -2 * k * (sigma_u**2) * (1 - sigma_u)  # d/d(delta) of (sigma(-k delta))^2

        # chain rule contributions:
        grad_s[i] += factor * (-d_max_si)
        grad_s[j] += factor * (-d_max_sj)
        grad_t[i] += factor * (d_min_ti)
        grad_t[j] += factor * (d_min_tj)

    # Process negative edges
    for (i, j) in neg_edges:
        diff_s = s[i] - s[j]
        A = np.sqrt(diff_s**2 + eps)
        d_max_si = 0.5 * (1 + diff_s / A)
        d_max_sj = 0.5 * (1 - diff_s / A)

        diff_t = t[i] - t[j]
        B = np.sqrt(diff_t**2 + eps)
        d_min_ti = 0.5 * (1 - diff_t / B)
        d_min_tj = 0.5 * (1 + diff_t / B)

        max_s = smooth_max(s[i], s[j], eps)
        min_t = smooth_min(t[i], t[j], eps)
        delta = min_t - max_s

        u = k * delta
        sigma_u = sigmoid(u)
        factor = 2 * k * (sigma_u**2) * (1 - sigma_u)  # d/d(delta) of (sigma(k delta))^2

        grad_s[i] += factor * (-d_max_si)
        grad_s[j] += factor * (-d_max_sj)
        grad_t[i] += factor * (d_min_ti)
        grad_t[j] += factor * (d_min_tj)

    # Process interval validity for each vertex: we want s[i] < t[i]
    for i in range(n):
        u = k*(s[i]-t[i])
        sigma_u = sigmoid(u)
        factor = 2 * k * (sigma_u**2) * (1 - sigma_u)
        grad_s[i] += factor  # derivative of (s[i]-t[i]) wrt s[i] is 1
        grad_t[i] += -factor  # derivative wrt t[i] is -1

    return grad_s, grad_t

# ----------------------------
# Gradient Descent Optimization
# ----------------------------

def optimize_via_gd(permutation: np.ndarray, graph: SignedGraph, lr=0.01, iterations=500, k=10.0):
    """
    Optimize the interval embedding via gradient descent.
    - permutation: an array of vertex labels in the initial order.
    - graph: the SignedGraph object.

    The vertices are assumed to be numbered (or labeled) and we create a mapping
    from vertex label to index.
    """
    # Build mapping: use the order given in permutation
    vertices = list(permutation)
    n = len(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}

    print(v_to_idx)

    # Initialize intervals as unit intervals ordered by permutation.
    # We use s[i] = i, t[i] = i + 1.
    s = np.array([10 for i in range(n)], dtype=float)
    # add some gaussian noise to s
    s += np.random.uniform(-5, 5, n)
    t = s + 10.0 + np.random.uniform(-5, 5, n)


    # Plot the GD–optimized intervals
    #plot_combined_graph_and_intervals(graph, s, t, "test.py", draw_edges="existing", show=True)

    # Build edge lists in terms of indices (using G_plus for positive edges and G_minus for negative edges)
    pos_edges = []
    for (u, v) in graph.G_plus.edges:
        if u in v_to_idx and v in v_to_idx:
            i, j = v_to_idx[u], v_to_idx[v]
            # store edges with the smaller index first (order does not matter)
            if i > j:
                i, j = j, i
            pos_edges.append((i, j))

    neg_edges = []
    for (u, v) in graph.G_minus.edges:
        if u in v_to_idx and v in v_to_idx:
            i, j = v_to_idx[u], v_to_idx[v]
            if i > j:
                i, j = j, i
            neg_edges.append((i, j))

    # Run gradient descent
    losses = []
    for it in range(iterations):
        loss = compute_loss(s, t, pos_edges, neg_edges, k=k)
        losses.append(loss)

        grad_s, grad_t = compute_gradients(s, t, pos_edges, neg_edges, k=k)

        # Update steps
        s -= lr * grad_s
        t -= lr * grad_t

        # (Optionally: project back to ensure s < t. Here our penalty already encourages that.)
        if it % 5 == 0:
            print(f"Iteration {it}, Loss = {loss:.4f}")

    print(f"Final loss after {iterations} iterations: {losses[-1]:.4f}")
    return s.tolist(), t.tolist()

# ----------------------------
# (Visualization and Gurobi-based functions remain unchanged)
# ----------------------------

def count_violations(gd_start, gd_end, graph, permutation):
    """
    Counts violations for the interval embedding.
    - gd_start, gd_end: lists (or arrays) of optimized start and end positions.
    - graph: the SignedGraph object containing G_plus and G_minus.
    - permutation: list/array of vertex labels in the same order as gd_start, gd_end.

    Returns a tuple: (pos_edge_violations, neg_edge_violations, vertex_violations, total_violations)
    """
    # Create a mapping from vertex label to index (assuming permutation order)
    v_to_idx = {v: i for i, v in enumerate(permutation)}

    pos_violations = 0
    neg_violations = 0
    vertex_violations = 0

    # Count vertex interval violations: every vertex should have gd_start < gd_end.
    for i in range(len(gd_start)):
        if gd_start[i] >= gd_end[i]:
            vertex_violations += 1

    # Count violations for positive edges: intervals should overlap.
    # Two intervals [s_i, t_i] and [s_j, t_j] overlap if and only if s_i < t_j and s_j < t_i.
    for (u, v) in graph.G_plus.edges:
        if u in v_to_idx and v in v_to_idx:
            i = v_to_idx[u]
            j = v_to_idx[v]
            if not (gd_start[i] <= gd_end[j] and gd_start[j] <= gd_end[i]):
                pos_violations += 1

    # Count violations for negative edges: intervals should not overlap.
    # That is, they violate if they do overlap.
    for (u, v) in graph.G_minus.edges:
        if u in v_to_idx and v in v_to_idx:
            i = v_to_idx[u]
            j = v_to_idx[v]
            if (gd_start[i] <= gd_end[j] and gd_start[j] <= gd_end[i]):
                neg_violations += 1

    total_violations = vertex_violations + pos_violations + neg_violations
    return pos_violations, neg_violations, vertex_violations, total_violations

def count_primitive_violations(permutation, graph):
    """
    Constructs primitive (unit) intervals from the permutation and
    returns the number of violations.

    The primitive intervals are defined as:
      s[i] = i, t[i] = i + 1,
    for vertices in the order given by the permutation.
    """
    n = len(permutation)
    primitive_start = [float(i) for i in range(n)]
    primitive_end = [float(i) + 1.0 for i in range(n)]
    return count_violations(primitive_start, primitive_end, graph, permutation)

