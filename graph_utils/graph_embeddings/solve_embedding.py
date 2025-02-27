import gurobipy as gp
from gurobipy import GRB

from graph_utils.signed_graph import SignedGraph, read_signed_graph

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

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

class DraggableNode:
    def __init__(self, graph, pos, node_colors, ax, draw_edges="existing"):
        self.graph = graph
        self.draw_edges = draw_edges
        self.pos = pos
        self.node_colors = node_colors
        self.ax = ax
        self.selected_node = None
        self.press = None

        # Connect events for dragging nodes
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        for node, (x, y) in self.pos.items():
            dx = x - event.xdata
            dy = y - event.ydata
            distance = np.hypot(dx, dy)
            if distance < 0.05:
                self.selected_node = node
                self.press = (x, y), (event.xdata, event.ydata)
                return

    def on_release(self, event):
        self.selected_node = None
        self.press = None
        self.redraw()

    def on_motion(self, event):
        if self.selected_node is None or event.inaxes != self.ax or self.press is None:
            return

        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.pos[self.selected_node] = (x0 + dx, y0 + dy)
        self.redraw()

    def redraw(self, draw_edges="all"):
        self.ax.clear()

        # Draw nodes
        nx.draw_networkx_nodes(self.graph.G_plus, self.pos, node_color=[self.node_colors[node] for node in self.graph.G_plus.nodes], ax=self.ax)

        # Draw edges for G_plus (green) and G_minus (red)
        nx.draw_networkx_edges(self.graph.G_plus, self.pos, edge_color="green", ax=self.ax, label="Positive Edges")

        if self.draw_edges != "missing":
            nx.draw_networkx_edges(self.graph.G_minus, self.pos, edge_color="red", ax=self.ax, label="Negative Edges")

        # Draw all possible edges not in G_plus or G_minus in blue
        if self.draw_edges != "existing":
            all_possible_edges = set(nx.complete_graph(self.graph.G_plus.nodes).edges)
            existing_edges = set(self.graph.G_plus.edges).union(self.graph.G_minus.edges)
            missing_edges = all_possible_edges - existing_edges
            nx.draw_networkx_edges(self.graph.G_plus, self.pos, edgelist=missing_edges, edge_color="blue", ax=self.ax, style="dashed", label="Missing Edges")

        # Draw labels and title
        nx.draw_networkx_labels(self.graph.G_plus, self.pos, ax=self.ax)
        self.ax.set_title("Interactive Signed Graph - Drag Nodes to Reposition")
        plt.draw()


def plot_combined_graph_and_intervals(graph, start_times, end_times, target_file_path, draw_edges, pos=None, show=False):
    # Generate a color map for nodes
    num_nodes = len(graph.G_plus.nodes)
    colors = cm.rainbow(np.linspace(0, 1, num_nodes))  # Generate distinct colors
    node_colors = {node: colors[i] for i, node in enumerate(graph.G_plus.nodes)}
    # Plot the initial graph on the top subplot (ax1)
    combined_graph = nx.Graph()
    combined_graph.add_nodes_from(graph.G_plus.nodes)
    combined_graph.add_edges_from(graph.G_plus.edges)
    combined_graph.add_edges_from(graph.G_minus.edges)

    # Compute layout based on the combined graph
    if pos is None:
        pos = nx.kamada_kawai_layout(combined_graph)

    # Define the layout for subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Initialize draggable nodes
    draggable = DraggableNode(graph, pos, node_colors, ax1, draw_edges=draw_edges)

    draggable.redraw()  # Initial draw


    # Draw the interval embedding on the bottom subplot
    for i, (s, t) in enumerate(zip(start_times, end_times)):
        node = list(graph.G_plus.nodes)[i]
        color = node_colors[node]
        ax2.plot([s, t], [i, i], marker='o', markersize=5, linewidth=2, color=color, label=f'Node {node}')

    ax2.set_xlabel("Time")
    ax2.set_title("Interval Representation")
    ax2.set_yticks([])
    ax2.invert_yaxis()

    plt.tight_layout()

    if show:
        plt.show()

    # Save the plot to a file
    plt.savefig(target_file_path)

    plt.close()


def build_constraint_model(model: gp.Model, graph: SignedGraph, disjoint_aux: dict, hard_negative_edges: bool = False):
    V = graph.G_plus.nodes
    E_plus = [(min(i,j),max(i,j)) for (i,j) in graph.G_plus.edges]
    E_minus = [(min(i,j),max(i,j)) for (i,j) in graph.G_minus.edges]
    # large M constant, to deactivate constraints (bad model but simple)
    M = 2*len(V)
    eps = 1

    # variables
    s = model.addVars(V, vtype=GRB.CONTINUOUS, lb=0, ub=M, name="start")
    t = model.addVars(V, vtype=GRB.CONTINUOUS, lb=0, ub=M, name="end")
    # x = model.addVars(((i, j) for i in V for j in V if i != j), vtype=GRB.BINARY, name="overlap")
    # we say these variables count towards the objective value with obj=1
    z_miss = model.addVars(E_plus, vtype=GRB.BINARY, name="penalty_miss", obj=1)
    z_extra = model.addVars(E_minus, vtype=GRB.BINARY, name="penalty_extra", obj=1)
    # for dealing with the disjunction in non-overlap constraints
    if disjoint_aux is None:
        disjoint_aux = model.addVars(E_minus, vtype=GRB.BINARY, name="disjoint_left_aux")

    # s and t define (non-empty, could be changed) intervals
    model.addConstrs((s[i] + eps <= t[i] for i in V), name=f"interval")

    # Additional constraint: ensure that the s values are pairwise disjoint.
    # For every pair (i,j) with i < j, enforce either s[i] + eps <= s[j] or s[j] + eps <= s[i].
    d = model.addVars(((i, j) for i in V for j in V if i < j), vtype=GRB.BINARY, name="disjoint_s")
    model.addConstrs((s[i] + eps <= s[j] + M * (1 - d[i, j]) for (i, j) in d.keys()), name="s_order1")
    model.addConstrs((s[j] + eps <= s[i] + M * d[i, j] for (i, j) in d.keys()), name="s_order2")

    # plus edges should overlap, otherwise set z_miss to 1

    model.addConstrs((s[i] <= t[j] + M * z_miss[i, j] for (i, j) in z_miss.keys()), name="plus_edge_overlap_1")
    model.addConstrs((s[j] <= t[i] + M * z_miss[i, j] for (i, j) in z_miss.keys()), name="plus_edge_overlap_2")

    # minus edges should either be left- or right-disjoint, otherwise set z_extra to 1
    model.addConstrs((t[i] + eps <= s[j] + M * (z_extra[i, j] + 1 - disjoint_aux[i, j]) for (i, j) in z_extra.keys()), name=f"nonoverlap_left")
    model.addConstrs((t[j] + eps <= s[i] + M * (z_extra[i, j] + disjoint_aux[i, j]) for (i, j) in z_extra.keys()), name=f"nonoverlap_right")

    #tightening constraints:
    # model.addConstrs((s[i]))
    # disjoint_aux(i,j) = 0 => j links von i. 
    # disjoint_aux(i,j) = 1 => i links von j.


def get_cycle_positions(cycle):
    """
    Generates positions for nodes in a cycle.
    """
    num_nodes = len(cycle)
    angle_step = 2 * np.pi / num_nodes
    positions = {}
    for i, node in enumerate(cycle):
        angle = i * angle_step
        positions[node] = (np.cos(angle), np.sin(angle))
    return positions


def check_embeddability(file: str, disjoint_aux: dict):

    graph = read_signed_graph(file)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 1)
        env.setParam("TimeLimit", 180)
        env.setParam("SoftMemLimit", 16)  # GB (I think)
        env.setParam("Threads", 11)  # TODO: this we need to play with at some point
        env.start()
        with gp.Model("some_model_name", env=env) as model:
            try:
                build_constraint_model(model, graph, disjoint_aux)

                model.optimize()

                # Retrieve solution
                if model.status == GRB.OPTIMAL:
                    starts = [var.X for var in model.getVars() if "start" in var.VarName]
                    ends = [var.X for var in model.getVars() if "end" in var.VarName]
                    overlaps = [(var.VarName, var.X) for var in model.getVars() if "overlap" in var.VarName]
                    miss = [(var.VarName, var.X) for var in model.getVars() if "miss" in var.VarName]
                    extra = [(var.VarName, var.X) for var in model.getVars() if "extra" in var.VarName]
                    
                    print(f"Optimal solution found with objective: {model.ObjVal}")

                    return model.ObjVal, starts, ends
                else:
                    starts = [var.X for var in model.getVars() if "start" in var.VarName]
                    ends = [var.X for var in model.getVars() if "end" in var.VarName]
                    overlaps = [(var.VarName, var.X) for var in model.getVars() if "overlap" in var.VarName]
                    miss = [(var.VarName, var.X) for var in model.getVars() if "miss" in var.VarName]
                    extra = [(var.VarName, var.X) for var in model.getVars() if "extra" in var.VarName]
                    
                    print(f"Inoptimal solution found with objective: {model.ObjVal}")

                    return model.ObjVal, starts, ends

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")
            except AttributeError as attr_err:
                print(f"Encountered an attribute error: {attr_err}")
    
    return False, None, None

def generate_disjoint_aux_vars(permutation: np.ndarray):
    """
    Generates auxiliary variables for the disjointness constraints.
    """
    disjoint_aux = {}
    for i_idx, i in enumerate(permutation):
        for j_idx, j in enumerate(permutation):
            if i == j:
                disjoint_aux[int(i), int(j)] = 0
            else:
                disjoint_aux[int(i), int(j)] = 0 if i_idx < j_idx else 1
    return disjoint_aux

# ----------------------------
# Main function
# ----------------------------

if __name__ == "__main__":
    okay_dir = "data/test_good"
    bad_dir = "data/test_bad"

    os.makedirs(okay_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    #file = "graph_0.txt"
    file = "data/signed_graphs_500_nodes/graph_2.txt"

    graph = read_signed_graph(file)

    #embeddable, start, end = check_embeddability(file, None)
    #permutation = sorted(range(1,len(start)+1), key=lambda i: start[i-1])

    embeddable = False

    # Generate a random permutation
    permutation = np.random.permutation(len(graph.G_plus.nodes))

    if embeddable:
        target_file_path = os.path.join(okay_dir, os.path.basename(file)).replace(".txt", ".png")
        draw_edges = "existing"
    else:
        target_file_path = os.path.join(bad_dir, os.path.basename(file)).replace(".txt", ".png")
        draw_edges = "existing"

    pos = get_cycle_positions(graph.G_plus.nodes) 

    # Plot the initial intervals
    #plot_combined_graph_and_intervals(graph, start, end, target_file_path, pos=pos, draw_edges=draw_edges, show=True)
        
    #print(start)
    print(permutation)

    # Count and print violations for the primitive interval construction.
    prim_pos_v, prim_neg_v, prim_vertex_v, prim_total_v = count_primitive_violations(permutation, graph)
    print(f"Primitive Violations: Total = {prim_total_v}, "
          f"Positive edges = {prim_pos_v}, Negative edges = {prim_neg_v}, Vertex = {prim_vertex_v}")


    # Now optimize using gradient descent
    gd_start, gd_end = optimize_via_gd(np.array(permutation), graph, lr=0.1, iterations=100, k=1)

    pos_v, neg_v, vertex_v, total_v = count_violations(gd_start, gd_end, graph, permutation)
    print(f"Violations: {total_v} total, with {pos_v} positive edge violations, {neg_v} negative edge violations, and {vertex_v} vertex interval violations.")

    # Plot the GD–optimized intervals
    plot_combined_graph_and_intervals(graph, gd_start, gd_end, target_file_path, pos=pos, draw_edges=draw_edges, show=True)
    
