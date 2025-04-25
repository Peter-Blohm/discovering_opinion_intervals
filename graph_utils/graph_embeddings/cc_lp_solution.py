from graph_utils.signed_graph import SignedGraph, read_signed_graph
from graph_utils.signed_graph_kernelization import kernelize_signed_graph

import numpy as np
import gurobipy as gp
from gurobipy import GRB

def build_constraint_model(model: gp.Model, graph: SignedGraph):
    V = graph.G_plus.nodes
    print(V)
    E_plus = [(min(i,j),max(i,j)) for (i,j) in graph.G_plus.edges]
    E_minus = [(min(i,j),max(i,j)) for (i,j) in graph.G_minus.edges]
    E = E_plus + E_minus + [(j,i) for (i,j) in E_plus] + [(j,i) for (i,j) in E_minus]
    print(len(V),len(E))
    #print(len(list("X[i,k] <= X[i,j] + X[j,k]" 
    #               for (i,j) in E if i < j 
      #             for (i,k) in [(x,y) for (x,y) in E if x == i and (j,y) in E])))
    X = model.addVars(((i,j) for i in V for j in V), vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    print("Variables added.")
    # Constraints
    model.addConstrs((X[i,j] == X[j,i] for i in V for j in V if i < j), name="symmetry")
    print("Constraints added.")
    model.addConstrs((X[i,k] <= X[i,j] + X[j,k] for i in V for j in V for k in V), name="triangle_inequality")
                      # for (i,j) in E for (i2,k) in E if (j,k) in E and i==i2), name="triangle_inequality")
    print("Constraints added.")
    # Objective
    model.setObjective(sum([X[i,j] for i,j in E_plus]) + sum([1 - X[i,j] for i,j in E_minus]), GRB.MAXIMIZE)


def solve_lp(graph: SignedGraph):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 1)
        env.setParam("TimeLimit", 18000)
        env.setParam("SoftMemLimit", 30)
        env.setParam("Threads", 11)
        # Verbose
        env.start()
        with gp.Model("cc_lp_relaxation", env=env) as model:
            try:
                build_constraint_model(model, graph)

                model.optimize()

                # Retrieve solution
                if model.status == GRB.OPTIMAL:
                    x_values = [var.X for var in model.getVars() if "x" in var.VarName]
                    return x_values
                else:
                    return None

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")
            except AttributeError as attr_err:
                print(f"Encountered an attribute error: {attr_err}")
    return None


def find_optimal_radius():
    pass

if __name__ == "__main__":

    #file = "data/cycle2.txt"
    file = "data/soc-sign-bitcoinotc.csv"

    print("bark")
    graph = read_signed_graph(file)
    graph = kernelize_signed_graph(graph, safe=True)[0]
    
    x_values = solve_lp(graph)

    if x_values == None:
        print("No solution found.")
        assert False

    print(x_values)