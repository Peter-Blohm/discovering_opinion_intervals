import random
import networkx as nx

from signed_graph import SignedGraph, read_signed_graph, save_graph_to_file


def _combined_graph(graph: SignedGraph) -> nx.Graph:
    combined = nx.Graph()
    combined.add_nodes_from(graph.G_plus.nodes())
    combined.add_nodes_from(graph.G_minus.nodes())
    combined.add_edges_from(graph.G_plus.edges())
    combined.add_edges_from(graph.G_minus.edges())
    return combined


def _induced_subgraphs(graph: SignedGraph, node_sets) -> list[SignedGraph]:
    return [graph.subgraph(nodes).copy() for nodes in node_sets]


def remove_vertices_without_negative_edges(graph: SignedGraph) -> SignedGraph:
    reduced = graph.copy()
    no_negative = [v for v, deg in reduced.G_minus.degree() if deg == 0]
    reduced.remove_nodes_from(no_negative)
    return reduced


def split_positive_components(graph: SignedGraph) -> list[SignedGraph] | None:
    components = list(nx.connected_components(graph.G_plus))
    if len(components) <= 1:
        return None
    return _induced_subgraphs(graph, components)


def split_at_vertex_separator(graph: SignedGraph) -> list[SignedGraph] | None:
    combined = _combined_graph(graph)
    components = list(nx.biconnected_components(combined))
    if len(components) <= 1:
        return None
    return _induced_subgraphs(graph, components)


def split_at_positive_edge_separator(graph: SignedGraph) -> list[SignedGraph] | None:
    combined = _combined_graph(graph)
    positive = {frozenset(e) for e in graph.G_plus.edges()}

    cut = _find_positive_edge_cut(combined, positive)
    if cut is None:
        return None

    combined.remove_edges_from(cut)
    parts = list(nx.connected_components(combined))
    return _induced_subgraphs(graph, parts)


def _disconnects(combined: nx.Graph, edges) -> bool:
    """Return True iff removing ``edges`` raises the number of components."""
    before = nx.number_connected_components(combined)
    residual = combined.copy()
    residual.remove_edges_from(edges)
    return nx.number_connected_components(residual) > before


# Refer to two_edge_cut_algorithm.pdf
def _edge_signatures(combined: nx.Graph) -> dict:
    rng = random.Random(0)
    disc: dict = {}
    parent: dict = {}
    order: list = []
    delta: dict = {}
    sig: dict = {}
    timer = 0

    for root in combined.nodes():
        if root in disc:
            continue
        disc[root] = timer
        timer += 1
        order.append(root)
        delta[root] = 0
        stack = [(root, None, iter(combined[root]))]
        while stack:
            u, pu, neighbours = stack[-1]
            for w in neighbours:
                if w not in disc:  # tree edge u -> w
                    disc[w] = timer
                    timer += 1
                    order.append(w)
                    parent[w] = u
                    delta[w] = 0
                    stack.append((w, u, iter(combined[w])))
                    break
                if w != pu and disc[w] < disc[u]:  # back edge u -> ancestor w
                    label = rng.getrandbits(64)
                    delta[u] ^= label
                    delta[w] ^= label
                    sig[frozenset((u, w))] = label
            else:  # neighbours exhausted -> done with u
                stack.pop()

    subxor = dict(delta)
    for node in reversed(order):
        p = parent.get(node)
        if p is not None:
            sig[frozenset((p, node))] = subxor[node]
            subxor[p] ^= subxor[node]
    return sig


def _find_positive_edge_cut(combined: nx.Graph, positive: set):
    # for u, v in nx.bridges(combined):
    #     if frozenset((u, v)) in positive:
    #         return [(u, v)]

    sig = _edge_signatures(combined)
    groups: dict = {}
    for edge in positive:
        signature = sig.get(edge)
        if signature:  # skip bridges (signature 0), already handled above
            groups.setdefault(signature, []).append(edge)

    for edges in groups.values():
        if len(edges) >= 2:
            cut = [tuple(edges[0]), tuple(edges[1])]
            if _disconnects(combined, cut):
                return cut
    return None

def kernelise_graph(
    graph: SignedGraph,
    *,
    rule_i: bool = True,
    rule_ii: bool = True,
    rule_iii: bool = True,
    rule_iv: bool = True,
) -> list[SignedGraph]:

    kernels: list[SignedGraph] = []
    queue: list[SignedGraph] = [graph.copy()]

    while queue:
        g = queue.pop()

        if rule_i:
            g = remove_vertices_without_negative_edges(g)
        if g.G_minus.number_of_edges() == 0:
            continue  # nothing left to violate

        if rule_ii:
            parts = split_positive_components(g)
            if parts is not None:
                queue.extend(parts)
                continue

        if rule_iii:
            parts = split_at_vertex_separator(g)
            if parts is not None:
                queue.extend(parts)
                continue

        if rule_iv:
            parts = split_at_positive_edge_separator(g)
            if parts is not None:
                queue.extend(parts)
                continue

        kernels.append(g)  # no rule applies -> this is a kernel

    return kernels


# def _find_singly_pos_connected_vertices(graph: SignedGraph) -> list:
#     """Vertices with exactly one positive and no negative neighbours."""
#     candidates = []
#     for node, plus_degree in graph.G_plus.degree():
#         if plus_degree == 1 and graph.G_minus.degree(node) == 0:
#             candidates.append(node)
#     return candidates


# def kernelize_for_fixed_intervals(graph: SignedGraph) -> SignedGraph:
#     """Repeatedly remove vertices with one positive and no negative neighbours.

#     :param graph: the input :class:`SignedGraph`
#     :return: the reduced :class:`SignedGraph`
#     """
#     result = graph.copy()
#     while True:
#         to_remove = _find_singly_pos_connected_vertices(result)
#         if not to_remove:
#             break
#         result.remove_nodes_from(to_remove)
#     return result


def find_max_ratio_vertex(graph: SignedGraph) -> tuple:
    best_vertex, best_ratio, best_violations = None, 0.0, 0
    for node in graph.G_plus.nodes():
        plus_degree = graph.G_plus.degree(node)
        minus_degree = graph.G_minus.degree(node)
        total = plus_degree + minus_degree
        if total == 0:
            continue
        ratio = max(plus_degree, minus_degree) / total
        if ratio > best_ratio:
            best_vertex = node
            best_ratio = ratio
            best_violations = min(plus_degree, minus_degree)
    return best_vertex, best_ratio, best_violations


def chicken_algorithm(graph: SignedGraph, alpha: float = 0.0) -> tuple[int, list[SignedGraph]]:
    stack = kernelise_graph(graph)
    total_violations = 0
    remaining: list[SignedGraph] = []

    while stack:
        g = stack.pop()
        vertex, ratio, violations = find_max_ratio_vertex(g)
        if vertex is None:
            continue
        if ratio >= alpha:
            g.remove_node(vertex)
            total_violations += violations
            stack.extend(kernelise_graph(g))
        else:
            remaining.append(g)

    return total_violations, remaining


if __name__ == "__main__":
    import os

    file = "Datasets/wikisigned-k2.txt"
    graph = read_signed_graph(file)

    kernels = kernelise_graph(graph)
    print(f"Number of kernels: {len(kernels)}")

    for kernel in kernels:
        print(
            f"Kernel: {kernel.number_of_nodes()} vertices, "
            f"{kernel.G_plus.number_of_edges()} positive edges, "
            f"{kernel.G_minus.number_of_edges()} negative edges"
        )

    if kernels:
        largest = max(kernels, key=lambda g: g.number_of_nodes())
        print(
            "Largest kernel: "
            f"{largest.number_of_nodes()} vertices, "
            f"{largest.G_plus.number_of_edges()} positive edges, "
            f"{largest.G_minus.number_of_edges()} negative edges"
        )

        name = os.path.splitext(os.path.basename(file))[0] + "_kernel"
        save_graph_to_file(largest, name, "data")
