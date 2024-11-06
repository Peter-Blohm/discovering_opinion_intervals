import gurobipy as gp
from gurobipy import GRB

from graph_utils.signed_graph import SignedGraph, read_signed_graph


def build_constraint_model(model: gp.Model, graph: SignedGraph):
    V = graph.G_plus.nodes
    E_plus = graph.G_plus.edges
    E_minus = graph.G_minus.edges
    # large M constant, to deactivate constraints (bad model but simple)
    M = len(V)

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
    # we could optimize out xij, but it actually might be useful for reading solutions
    model.setObjective(gp.quicksum(z_miss[i, j] for i, j in E_plus) + gp.quicksum(z_extra[i, j] for i, j in E_minus),
                       GRB.MINIMIZE)


if __name__ == "__main__":
    # context handlers take care of handling resources correctly
    data = '../../data/cycle.txt'
    graph = read_signed_graph(data)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("TimeLimit", 60)
        env.setParam("SoftMemLimit", 16)  #GB (i think)
        env.setParam("Threads", 8)  #TODO: this we need to play with at some point
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
                    print("Start Times:", start_times)
                    print("End Times:", end_times)
                    print("Overlaps:", overlaps)
                    print("Penalty for missed overlaps:", miss)
                    print("Penalty for extra overlaps:", extra)
                    print("Deleted edges for opt:", model.ObjVal)
                else:
                    print("No feasible solution found.")



            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")
            except AttributeError:
                print("Encountered an attribute error")
