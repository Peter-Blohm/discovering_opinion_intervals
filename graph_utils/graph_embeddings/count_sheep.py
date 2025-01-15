import gurobipy as gp
from gurobipy import GRB

from graph_utils.signed_graph import SignedGraph, read_signed_graph
from graph_utils.signed_graph_kernelization import kernelize_graph, _find_one_seperated_components

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from tqdm import tqdm

def count_sheep(file: str):

    graph = read_signed_graph(file)
    kernels = kernelize_graph(graph)

    print("Num Nodes in largest kernel: ", len(kernels[0].G_plus.nodes))

    while True:
        new_kernels = []
        for kernel in kernels:
            new_kernels.extend(_find_one_seperated_components(kernel))
        new_kernels = [temp for temp in (kernelize_graph(krnl) for krnl in new_kernels)]
        if len(new_kernels) == len(kernels):
            break
        kernels = new_kernels

    
    print("Num Nodes in largest kernel: ", len(kernels[0].G_plus.nodes))

    graph = kernels[0]

    sheep_count = 0

    print(len(graph.G_plus.nodes))

    adj_plus = list(graph.G_plus.adjacency())
    adj_minus = list(graph.G_minus.adjacency())

    for i, node in tqdm(enumerate(graph.G_plus.nodes)):

        for j, other_node in enumerate(graph.G_plus.nodes):
            if i != j and not graph.has_minus_edge(node, other_node):
                _, adj_dict_i_plus = adj_plus[i]
                _, adj_dict_j_plus = adj_plus[j]
                _, adj_dict_i_minus = adj_minus[i]
                _, adj_dict_j_minus = adj_minus[j]

                adj_dict_i_plus = set(adj_dict_i_plus.keys()) - set([other_node])
                adj_dict_j_plus = set(adj_dict_j_plus.keys()) - set([node])
                adj_dict_i_minus = set(adj_dict_i_minus.keys())
                adj_dict_j_minus = set(adj_dict_j_minus.keys())

                if adj_dict_i_plus.issubset(adj_dict_j_plus) and adj_dict_i_minus.issubset(adj_dict_j_minus):
                    sheep_count += 1
                    break

    return sheep_count
            




if __name__ == "__main__":
    file = "data/soc-sign-bitcoinotc.csv"

    sheep_count = count_sheep(file)

    print("Sheep count: ", sheep_count)


