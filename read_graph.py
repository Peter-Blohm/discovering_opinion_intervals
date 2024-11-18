import re
import numpy as np
import networkx as nx
import os

from graph_utils.signed_graph_heuristics import find_max_ratio_vertex
from graph_utils.signed_graph import SignedGraph, read_signed_graph
from graph_utils.signed_graph_kernelization import kernelize_graph


def read_graph(file):
    G = nx.Graph()

    with open(file, 'r') as file:
        for line in file:
            # Skip comment lines that start with '#'
            if line.startswith('#'):
                continue

            # Split the line into FromNodeId, ToNodeId, and Sign
            parts = re.split(r'[,#;\t]', line.strip())
            if len(parts) >= 3:
                from_node = int(parts[0])
                to_node = int(parts[1])
                sign = np.sign(int(parts[2]))

                # If the edge already exists, prioritize keeping the negative sign
                if G.has_edge(from_node, to_node):
                    # If either existing edge or new edge has a negative sign, set it to -1
                    if G[from_node][to_node]['sign'] == -1 or sign == -1:
                        G[from_node][to_node]['sign'] = -1
                else:
                    # Add the edge with the sign as an attribute
                    G.add_edge(from_node, to_node, sign=sign)
    return G


def chicken_algorithm(G: SignedGraph):
    """
    Greedily pecks at the graph kernels, deleting the "best ratio" vertex at each time-step, producing more kernels
    :param G:
    :return:
    """
    graphs = kernelize_graph(G)
    if len(graphs) == 0:
        return 0
    vio = 0
    for graph in graphs:
        print(graph.G_plus.number_of_nodes())
        ratio_sum = 0
        its = min(1000, graph.G_plus.number_of_nodes()-1)
        for i in range(its):
            vertex, ratio, violations = find_max_ratio_vertex(graph)
            try:
                graph.remove_node(vertex)
            except Exception as e:
                print(e)
                break
            vio += violations
            ratio_sum += ratio
        print(ratio_sum/its)
        vio += chicken_algorithm(graph)

    return vio


if __name__ == "__main__":
    #for file in os.listdir("data"):
    data = f"data/soc-sign-Slashdot090221.txt"

    G = read_signed_graph(data)

    print(f"Name: {data}")
    print(f"Vertices: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Positive Edges: {G.G_plus.number_of_edges()}")
    print(f"Negative edges {G.G_minus.number_of_edges()}")

    # Kernelize the graph
    graphs = kernelize_graph(G, safe=True)
    largest_graph = max(graphs, key=lambda graph: graph.number_of_nodes())

    print(f"Number of vertices in largest connected component after 1 round of error free kernelization: {largest_graph.number_of_nodes()}")
    print(f"Number of positives edges in largest connected component after 1 round of error free kernelization: {largest_graph.G_plus.number_of_edges()}")
    print(f"Number of negative edges in largest connected component after 1 round of error free kernelization: {largest_graph.G_minus.number_of_edges()}")

    print(f"chicken algorithm violated edges:{chicken_algorithm(G)}")