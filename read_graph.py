import re
import numpy as np
import networkx as nx
import os
import h5py

# from graph_utils.graph_embeddings.data.fast_gd_embedding import adjacency_matrix
from graph_utils.signed_graph_heuristics import find_max_ratio_vertex
from graph_utils.signed_graph import SignedGraph, read_signed_graph
from graph_utils.signed_graph_kernelization import kernelize_signed_graph


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
    graphs = kernelize_signed_graph(G)
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


def gaec(G: SignedGraph, k = None):
    """ Takes a signed graph as input and computes a greedy additive edge contraction"""
    # k is the forced number of clusters

    num_vertex = max(G.G_plus.nodes()) + 1
    clusters = -np.ones(num_vertex, dtype=int) #index map

    # edges = [(1,u,v) for (u,v) in G.G_plus.edges() ] + [(-1,u,v) for (u,v) in G.G_minus.edges()]
    edges = np.ones((num_vertex, num_vertex)) * -np.inf
    node_list = list(G.G_plus.nodes())
    print(node_list)
    print(list(G.G_plus.nodes()))
    for v in G.G_plus.nodes():
        for v1 in G.G_plus.nodes():
            edges[v,v1] = 0
            edges[v1,v] = 0
        clusters[v] = v
        edges[v,v] = -np.inf

    for (u,v) in G.G_plus.edges():
        edges[u,v] = 1
        edges[v,u] = 1
        assert clusters[u] != clusters[v]

    for (u,v) in G.G_minus.edges():
        edges[v,u] = -1
        edges[u,v] = -1
        assert clusters[u] != clusters[v]
        # looks weird but takes care of (non-consecutive) indices
    ## take edge, slice clusters and edges
    while (k and len(np.unique(clusters)) > k) or (not k and edges.max()>0):
        weight, (alive_cluster, dead_cluster) = edges.max(), np.unravel_index(edges.argmax(), edges.shape)
        if k < 50:
            print(weight, (alive_cluster, dead_cluster), len(np.unique(clusters)))
        alive_cluster, dead_cluster = min(alive_cluster, dead_cluster), max(alive_cluster, dead_cluster)
        assert weight > -np.inf
        # print((clusters == alive_cluster).sum(),
        #       (clusters == dead_cluster).sum())
        clusters[clusters == dead_cluster] = alive_cluster

        # "If I need to, I'll go through you and absorb your fricking powers" ~ alive_cluster, probably
        edges[alive_cluster,:] += edges[dead_cluster,:]
        edges[:,alive_cluster] += edges[:,dead_cluster]
        edges[dead_cluster,:] = -np.inf
        edges[:,dead_cluster] = -np.inf
        edges[alive_cluster, alive_cluster] = -np.inf

    print(clusters[list(G.G_plus.nodes())])
    obj = 0
    for (u,v) in G.G_plus.edges():
        obj += clusters[u] != clusters[v]

    for (u,v) in G.G_minus.edges():
        obj += clusters[u] == clusters[v]
    print(obj)




if __name__ == "__main__":
    #for file in os.listdir("data"):
    data = f"/home/peter/tu/stefan_project/opinion_formation/data/soc-sign-Slashdot090221.txt"

    G = read_signed_graph(data)

    print(f"Name: {data}")
    print(f"Vertices: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Positive Edges: {G.G_plus.number_of_edges()}")
    print(f"Negative edges {G.G_minus.number_of_edges()}")

    # Kernelize the graph
    graphs = kernelize_signed_graph(G, safe=True)
    largest_graph = max(graphs, key=lambda graph: graph.number_of_nodes())
    G.G_plus = nx.relabel.convert_node_labels_to_integers(G.G_plus, first_label=1, ordering='default')
    G.G_minus = nx.relabel.convert_node_labels_to_integers(G.G_minus, first_label=1, ordering='default')
    gaec(G,5)
    print(f"Number of vertices in largest connected component after 1 round of error free kernelization: {largest_graph.number_of_nodes()}")
    print(f"Number of positives edges in largest connected component after 1 round of error free kernelization: {largest_graph.G_plus.number_of_edges()}")
    print(f"Number of negative edges in largest connected component after 1 round of error free kernelization: {largest_graph.G_minus.number_of_edges()}")

    # print(f"chicken algorithm violated edges:{chicken_algorithm(G)}")
