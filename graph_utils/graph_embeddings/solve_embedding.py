import gurobipy as gp
from gurobipy import GRB

from graph_utils.signed_graph import SignedGraph, read_signed_graph

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

class DraggableNode:
    def __init__(self, graph, pos, node_colors, ax):
        self.graph = graph
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

    def redraw(self):
        self.ax.clear()

        # Draw nodes
        nx.draw_networkx_nodes(self.graph.G_plus, self.pos, node_color=[self.node_colors[node] for node in self.graph.G_plus.nodes], ax=self.ax)

        # Draw edges for G_plus (green) and G_minus (red)
        nx.draw_networkx_edges(self.graph.G_plus, self.pos, edge_color="green", ax=self.ax, label="Positive Edges")
        nx.draw_networkx_edges(self.graph.G_minus, self.pos, edge_color="red", ax=self.ax, label="Negative Edges")

        # Draw all possible edges not in G_plus or G_minus in blue
        all_possible_edges = set(nx.complete_graph(self.graph.G_plus.nodes).edges)
        existing_edges = set(self.graph.G_plus.edges).union(self.graph.G_minus.edges)
        missing_edges = all_possible_edges - existing_edges
        nx.draw_networkx_edges(self.graph.G_plus, self.pos, edgelist=missing_edges, edge_color="blue", ax=self.ax, style="dashed", label="Missing Edges")

        # Draw labels and title
        nx.draw_networkx_labels(self.graph.G_plus, self.pos, ax=self.ax)
        self.ax.set_title("Interactive Signed Graph - Drag Nodes to Reposition")
        plt.draw()


def plot_combined_graph_and_intervals(graph, start_times, end_times, target_file_path, pos=None):
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
    draggable = DraggableNode(graph, pos, node_colors, ax1)
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

    # Save the plot to a file
    plt.savefig(target_file_path)

    plt.close()

def build_constraint_model(model: gp.Model, graph: SignedGraph):
    V = graph.G_plus.nodes
    E_plus = graph.G_plus.edges
    E_minus = graph.G_minus.edges
    # large M constant, to deactivate constraints (bad model but simple)
    M = len(V)*2

    # variables
    s = model.addVars(V, vtype=GRB.CONTINUOUS, name="start")
    t = model.addVars(V, vtype=GRB.CONTINUOUS, name="end")
    x = model.addVars([(i, j) for i in V for j in V if i < j], vtype=GRB.BINARY, name="overlap")
    z_miss = model.addVars(E_plus, vtype=GRB.BINARY, name="penalty_miss")
    z_extra = model.addVars(E_minus, vtype=GRB.BINARY, name="penalty_extra")
    # for dealing with the disjunction in non-overlap constraints
    disjoint_aux = model.addVars(E_minus, vtype=GRB.BINARY, name="disjoint_left_aux")

    # 1. interval Definition Constraints
    for i in V:
        model.addConstr(t[i] >= s[i] + 1, name=f"interval_{i}")

    # 2. overlap Constraints
    for (i, j) in E_plus:  # we want xij == 1 if possible
        model.addConstr(x[i, j] + z_miss[i, j] == 1, name=f"overlap_penalty_{i}_{j}")
        # if either si > tj or sj > ti, xij must be set to 0
        model.addConstr(s[i] <= t[j] + M * (1 - x[i, j]), name=f"overlap_1_{i}_{j}")
        model.addConstr(s[j] <= t[i] + M * (1 - x[i, j]), name=f"overlap_2_{i}_{j}")

    # 3. non-overlap Constraints
    for (i, j) in E_minus:  # we want xij ==0 if possible (we can change this to force the solver to respect - edges)
        model.addConstr(x[i, j] - z_extra[i, j] == 0, name=f"nonoverlap_penalty_{i}_{j}")

        # disjoint aux deactivates either constraint, but for both to be disabled (overlap), xij must be 1
        model.addConstr(t[i] + 1 <= s[j] + M * (x[i, j] + 1 - disjoint_aux[i, j]), name=f"nonoverlap_left_{i}_{j}")
        model.addConstr(t[j] + 1 <= s[i] + M * (x[i, j] + disjoint_aux[i, j]), name=f"nonoverlap_right_{i}_{j}")

    # objective: minimize penalties for incorrect overlaps and non-overlaps
    model.setObjective(gp.quicksum(z_miss[i, j] for i, j in E_plus) + gp.quicksum(z_extra[i, j] for i, j in E_minus),
                       GRB.MINIMIZE)


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


def check_embeddability(file: str):

    graph = read_signed_graph(file)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("TimeLimit", 60)
        env.setParam("SoftMemLimit", 16)  # GB (I think)
        env.setParam("Threads", 8)  # TODO: this we need to play with at some point
        env.start()
        with gp.Model("some_model_name", env=env) as model:
            try:
                build_constraint_model(model, graph)

                model.optimize()

                # Retrieve solution
                if model.status == GRB.OPTIMAL:
                    start_times = [var.X for var in model.getVars() if "start" in var.VarName]
                    end_times = [var.X for var in model.getVars() if "end" in var.VarName]
                    overlaps = [(var.VarName, var.X) for var in model.getVars() if "overlap" in var.VarName]
                    miss = [(var.VarName, var.X) for var in model.getVars() if "miss" in var.VarName]
                    extra = [(var.VarName, var.X) for var in model.getVars() if "extra" in var.VarName]
                    
                    print(f"Optimal solution found with objective: {model.ObjVal}")

                    return model.ObjVal == 0, start_times, end_times
                else:
                    print("No feasible solution found.")

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")
            except AttributeError as attr_err:
                print(f"Encountered an attribute error: {attr_err}")
    
    return False, None, None


if __name__ == "__main__":
    okay_dir = "data/test_good"
    bad_dir = "data/test_bad"

    os.makedirs(okay_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    file = "data/cycle.txt"

    check_embeddability(file, okay_dir, bad_dir)