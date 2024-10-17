import networkx as nx


def find_mostly_pos_vertices(graph: nx.Graph, threshold = 0.9, sign = 1) -> list[int]:
    """
    returns a list of vertex ids which have no negative edges
    :param graph: input graph of which to count vertices with mostly [sign] edges
    :param threshold: minimum ratio of [sign]/all edges
    :param sign: 1 or -1
    :return: index list of all marked vertices with high enough ratio
    """
    pos_vertices = []

    # Iterate over all nodes in the graph
    for node in graph.nodes:
        pos_neighbors = 0

        # Iterate over all neighbors of the node
        for neighbor in graph.neighbors(node):
            # If an edge has a negative sign, set the flag and break
            pos_neighbors +=graph[node][neighbor]['sign'] == sign

        # If no negative edges were found for this node, add it to the list
        if graph.degree[node] > 0 and pos_neighbors/graph.degree[node] > threshold:
            pos_vertices.append(node)

    return pos_vertices


def delete_mostly_pos_vertices(graph, threshold=0.9, sign=1):

    pos_vertices = find_mostly_pos_vertices(graph, threshold, sign = sign)
    new_graph = graph.copy()
    new_graph.remove_nodes_from(pos_vertices)
    return new_graph


def find_max_ratio_vertex(graph: nx.Graph, sign=0) -> (int, float, int):
    """
    returns index of vertex which has the highest imbalance in edge-signs
    :param graph: input graph
    :param sign: 1 or -1 for the corresponding edge type, 0 for both
    :return: tuple - index of the marked vertex, ratio (signed), number of violated edges
    """
    best_vertex = best_ratio = violated_edges = 0

    for node in graph.nodes:
        pos_neighbors = 0

        for neighbor in graph.neighbors(node):
            pos_neighbors += graph[node][neighbor]['sign'] == 1

        ratio = pos_neighbors/graph.degree[node]

        if ratio > abs(best_ratio) and sign != -1:
            best_vertex = node
            best_ratio = ratio
            violated_edges = graph.degree[node] - pos_neighbors
        if ratio < abs(best_ratio) and sign != 1:
            best_vertex = node
            best_ratio = ratio
            violated_edges = pos_neighbors
    return best_vertex, best_ratio, violated_edges
