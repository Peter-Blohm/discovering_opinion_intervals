import networkx as nx

from graph_utils.signed_graph import SignedGraph


def _find_pos_vertices(graph: SignedGraph):
    # returns a list of vertex ids which have no negative edges
    return [node for node, degree in graph.G_minus.degree() if degree == 0]


def _find_2mix_vertices(graph: SignedGraph):
    # returns a list of vertex ids have exactly one positive and one negative edge
    plus_nodes = [node for node, degree in graph.G_plus.degree() if degree == 1]
    minus_nodes = [node for node, degree in graph.G_minus.degree() if degree == 1]
    return [node for node in plus_nodes if node in minus_nodes]


def _find_plus_connected_components(graph: SignedGraph):
    # takes a graph object and attempts to separate positive clusters, i.e.

    connected_components = list(nx.connected_components(graph.G_plus))

    # List to hold the subgraphs induced by the connected components
    induced_subgraphs = []

    # For each connected component, induce a subgraph with both positive and negative edges
    for component in connected_components:
        # Create the subgraph induced by the component nodes in the original graph
        induced_subgraph = graph.subgraph(component).copy()
        induced_subgraphs.append(induced_subgraph)

    return induced_subgraphs


def _delete_pos_vertices(graph: SignedGraph):
    pos_vertices = _find_pos_vertices(graph)
    new_graph = graph.copy()
    new_graph.remove_nodes_from(pos_vertices)
    return new_graph


def _delete_2mix_vertices(graph: SignedGraph):
    pos_vertices = _find_2mix_vertices(graph)
    new_graph = graph.copy()
    new_graph.remove_nodes_from(pos_vertices)
    return new_graph


def kernelize_graph(graph: SignedGraph, safe=True) -> list[SignedGraph]:
    """
    takes the networkx graph, performs EXACT kernelization methods and returns all subgraph kernels in a list
    :param graph: the input nx.Graph
    :param safe: if True, excludes the 2mix rule, which might not be correct
    :return: list[nx.Graph]
    """
    g_prime = _delete_pos_vertices(graph)
    if not safe:
        g_prime = _delete_2mix_vertices(g_prime)
    graphs = _find_plus_connected_components(g_prime)
    if len(graphs) < 2:
        return graphs
    kernel_graphs = []
    for g in graphs:
        kernel_graphs.extend(kernelize_graph(g))
    return kernel_graphs
