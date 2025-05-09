import gurobipy as gp
from gurobipy import GRB
from numpy.random import permutation
from graph_utils.signed_graph import SignedGraph, read_signed_graph
from graph_utils.signed_graph_kernelization import kernelize_signed_graph

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

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



def check_embeddability(file: str, disjoint_aux: dict):

    graph = read_signed_graph(file)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 1)
        env.setParam("TimeLimit", 18000)
        env.setParam("SoftMemLimit", 30)  # GB (I think )
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
    file = "data/soc-sign-Slashdot090221.txt"

    graph = read_signed_graph(file)

    graphs = kernelize_signed_graph(graph, safe=True)

    embeddable, start, end = check_embeddability(file, None)
    
    #print(embeddable)

    #permutation = sorted(range(1,len(start)+1), key=lambda i: start[i-1])

    embeddable = False

    # Generate a random permutation

    if embeddable:
        target_file_path = os.path.join(okay_dir, os.path.basename(file)).replace(".txt", ".png")
        draw_edges = "missing"
    else:
        target_file_path = os.path.join(bad_dir, os.path.basename(file)).replace(".txt", ".png")
        draw_edges = "missing"
