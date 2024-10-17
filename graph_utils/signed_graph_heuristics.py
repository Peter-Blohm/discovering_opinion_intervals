import networkx as nx

from graph_utils.signed_graph import SignedGraph


def find_max_ratio_vertex(graph: SignedGraph, sign=0) -> (int, float, int):
    """
    returns index of vertex which has the highest imbalance in edge-signs
    :param graph: input graph
    :param sign: 1 or -1 for the corresponding edge type, 0 for both
    :return: tuple - index of the marked vertex, ratio (signed), number of violated edges
    """
    best_vertex = best_ratio = violated_edges = 0

    plus_deg = [degree for node, degree in graph.G_plus.degree()]
    minus_deg = [degree for node, degree in graph.G_minus.degree()]

    for index, node in enumerate([node for node, degree in graph.G_plus.degree()]):
        if (plus_deg[index] + minus_deg[index]) == 0:
            graph.remove_node(node)
            continue
        ratio = plus_deg[index]/(plus_deg[index] + minus_deg[index])

        if ratio > best_ratio and sign != -1:
            best_vertex = node
            best_ratio = ratio
            violated_edges = minus_deg[index]
        if ratio < 1-best_ratio and sign != 1:
            best_vertex = node
            best_ratio = 1-ratio
            violated_edges = plus_deg[index]
    return best_vertex, best_ratio, violated_edges
