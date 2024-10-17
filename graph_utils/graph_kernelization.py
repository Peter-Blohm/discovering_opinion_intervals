import networkx as nx


def _find_pos_vertices(graph):
    # returns a list of vertex ids which have no negative edges
    # Initialize an empty list to store vertex IDs with no negative edges
    pos_vertices = []

    # Iterate over all nodes in the graph
    for node in graph.nodes:
        has_negative_edge = False

        # Iterate over all neighbors of the node
        for neighbor in graph.neighbors(node):
            # If an edge has a negative sign, set the flag and break
            if graph[node][neighbor]['sign'] == -1:
                has_negative_edge = True
                break

        # If no negative edges were found for this node, add it to the list
        if not has_negative_edge:
            pos_vertices.append(node)

    return pos_vertices


def _find_2mix_vertices(graph):
    # returns a list of vertex ids have exactly one positive and one negative edge
    mix_vertices = []

    # Iterate over all nodes in the graph
    for node in graph.nodes:
        has_one_negative_edge = has_one_positive_edge = False

        # Iterate over all neighbors of the node
        for neighbor in graph.neighbors(node):
            # If an edge has a negative sign, set the flag and break
            if graph[node][neighbor]['sign'] == -1:
                if has_one_negative_edge:
                    has_one_negative_edge = False
                    break
                has_one_negative_edge = True
            if graph[node][neighbor]['sign'] == 1:
                if has_one_positive_edge:
                    has_one_positive_edge = False
                    break
                has_one_positive_edge = True

        if has_one_negative_edge and has_one_positive_edge:
            mix_vertices.append(node)

    return mix_vertices


def _find_plus_connected_components(graph):
    # takes a graph object and attempts to separate positive clusters, i.e.
    # a list of graphs induced by vertex sets which form a connected component.

    # Create a new graph with only positive edges
    positive_graph = nx.Graph()

    # Add only positive edges to the positive_graph
    for u, v, data in graph.edges(data=True):
        if data['sign'] == 1:
            positive_graph.add_edge(u, v)

    # Find connected components in the positive-only graph
    connected_components = list(nx.connected_components(positive_graph))

    # List to hold the subgraphs induced by the connected components
    induced_subgraphs = []

    # For each connected component, induce a subgraph with both positive and negative edges
    for component in connected_components:
        # Create the subgraph induced by the component nodes in the original graph
        induced_subgraph = graph.subgraph(component).copy()
        induced_subgraphs.append(induced_subgraph)

    return induced_subgraphs


def _delete_pos_vertices(graph):
    pos_vertices = _find_pos_vertices(graph)
    new_graph = graph.copy()
    new_graph.remove_nodes_from(pos_vertices)
    return new_graph


def _delete_2mix_vertices(graph):
    pos_vertices = _find_2mix_vertices(graph)
    new_graph = graph.copy()
    new_graph.remove_nodes_from(pos_vertices)
    return new_graph


def kernelize_graph(graph: nx.Graph, safe=True) -> list[nx.Graph]:
    """
    takes the networkx graph, performs EXACT kernelization methods and returns all subgraph kernels in a list
    :param graph: the input nx.Graph
    :param safe: if True, excludes the 2mix rule, which might not be correct
    :return: list[nx.Graph]
    """
    G_prime = _delete_pos_vertices(graph)
    if not safe:
        G_prime = _delete_2mix_vertices(G_prime)
    graphs = _find_plus_connected_components(G_prime)
    if len(graphs) < 2:
        return graphs
    kernel_graphs = []
    for g in graphs:
        kernel_graphs.extend(kernelize_graph(g))
    return kernel_graphs
