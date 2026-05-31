import heapq
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
    rule_iii: bool = False,
    rule_iv: bool = False,
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


def _smaller_positive_pieces(plus: dict, alive: set, sources: list) -> list[list]:
    """Decremental-connectivity helper for rule (ii) during pecking.

    A vertex has just been deleted from the positive graph; ``sources`` are its
    former (still-alive) positive neighbours, which together touch every piece
    the vertex's positive component may have split into. Returns the node lists
    of all resulting pieces **except the largest** (which is left untouched).

    A lockstep BFS grows one frontier per source and merges frontiers that
    meet; it stops as soon as a single component is still expanding. That last
    component is the largest and is never fully explored, so the work is bounded
    by the total size of the returned (smaller) pieces -- the "small-to-large"
    trick that keeps the whole greedy near-linear.
    """
    seen = set()
    srcs = []
    for s in sources:
        if s in alive and s not in seen:
            seen.add(s)
            srcs.append(s)
    if len(srcs) <= 1:
        return []  # at most one piece is touched -> no split possible

    k = len(srcs)
    parent = list(range(k))  # union-find over source indices

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    claimed = {}            # node -> index of the source that first reached it
    queue = [None] * k      # per-source BFS stack
    live = set()            # source indices with a non-empty queue
    for i, s in enumerate(srcs):
        if s in claimed:
            union(i, claimed[s])
            queue[i] = []
        else:
            claimed[s] = i
            queue[i] = [s]
            live.add(i)

    while len({find(i) for i in live}) > 1:
        for i in list(live):
            q = queue[i]
            if not q:
                live.discard(i)
                continue
            x = q.pop()
            for y in plus[x]:
                if y in claimed:
                    union(i, claimed[y])
                else:
                    claimed[y] = i
                    q.append(y)
            if not q:
                live.discard(i)

    active = {find(i) for i in live}
    if active:
        largest = next(iter(active))  # still expanding -> the largest piece
    else:
        sizes: dict = {}
        for node, i in claimed.items():
            r = find(i)
            sizes[r] = sizes.get(r, 0) + 1
        largest = max(sizes, key=sizes.get)

    pieces: dict = {}
    for node, i in claimed.items():
        r = find(i)
        if r != largest:
            pieces.setdefault(r, []).append(node)
    return list(pieces.values())


def _greedy_peck(kernel: SignedGraph, alpha: float) -> tuple[int, SignedGraph | None]:
    """Greedily peck a single kernel; the efficient core of the chicken algorithm.

    Selects the most sign-imbalanced vertex with a lazy max-heap keyed by
    ``max(deg^+, deg^-) / deg`` and maintains the signed degrees incrementally,
    so each step touches only the removed vertex's neighbourhood instead of
    rescanning every vertex.

    The exact reductions are folded in incrementally rather than via a full
    re-kernelization:

      * Rule (i): a vertex whose negative degree reaches 0 is removed for free.
      * Rule (ii): the positive-component labels are maintained under deletion
        (:func:`_smaller_positive_pieces`); whenever a removal splits a positive
        component, the negative edges that now cross between pieces are dropped
        (satisfiable for free). This is applied at **every** step, exactly as a
        per-step re-kernelization would, but in time bounded by the size of the
        smaller split-off pieces. Rules (iii)/(iv) never change a degree and are
        not needed here.

    :return: ``(violations, remaining)`` where ``remaining`` is the induced
        signed subgraph of the surviving vertices, or ``None`` if none survive
    """
    plus = {n: set(kernel.G_plus.neighbors(n)) for n in kernel.G_plus.nodes()}
    minus = {n: set(kernel.G_minus.neighbors(n)) for n in kernel.G_minus.nodes()}
    nodes = set(plus) | set(minus)
    for n in nodes:
        plus.setdefault(n, set())
        minus.setdefault(n, set())
    dplus = {n: len(plus[n]) for n in nodes}
    dminus = {n: len(minus[n]) for n in nodes}
    alive = set(nodes)

    # Positive-component labels (a kernel has a connected positive graph, so it
    # starts as a single component; deletions may split it).
    comp = {n: 0 for n in nodes}
    next_comp = 1

    def ratio(n):
        total = dplus[n] + dminus[n]
        return max(dplus[n], dminus[n]) / total if total else 0.0

    free = [n for n in nodes if dminus[n] == 0]
    heap = [(-ratio(n), n, dplus[n], dminus[n]) for n in nodes if dminus[n] > 0]
    heapq.heapify(heap)

    def push(w):
        if dminus[w] == 0:
            free.append(w)
        else:
            heapq.heappush(heap, (-ratio(w), w, dplus[w], dminus[w]))

    def remove_vertex(u):
        nonlocal next_comp
        pos = [w for w in plus[u] if w in alive]
        neg = [w for w in minus[u] if w in alive]
        for w in neg:
            minus[w].discard(u)
            dminus[w] -= 1
        for w in pos:
            plus[w].discard(u)
            dplus[w] -= 1
        alive.discard(u)
        for w in pos:
            if w in alive:
                push(w)
        for w in neg:
            if w in alive:
                push(w)
        # Rule (ii): a removal can split a positive component only if the vertex
        # had at least two positive neighbours.
        if len(pos) >= 2:
            pieces = _smaller_positive_pieces(plus, alive, pos)
            for piece in pieces:
                cid = next_comp
                next_comp += 1
                for node in piece:
                    comp[node] = cid
            for piece in pieces:
                for a in piece:
                    for b in tuple(minus[a]):
                        if b in alive and comp[a] != comp[b]:  # now cross-component
                            minus[a].discard(b)
                            minus[b].discard(a)
                            dminus[a] -= 1
                            dminus[b] -= 1
                            if a in alive:
                                push(a)
                            if b in alive:
                                push(b)

    violations = 0
    while True:
        while free:
            u = free.pop()
            if u in alive and dminus[u] == 0:
                remove_vertex(u)  # rule (i): no negative edges, free
        v = None
        while heap:
            neg_ratio, cand, snap_p, snap_m = heap[0]
            if cand not in alive or dplus[cand] != snap_p or dminus[cand] != snap_m:
                heapq.heappop(heap)  # stale entry
                continue
            v = cand
            break
        if v is None or ratio(v) < alpha:
            break
        heapq.heappop(heap)
        violations += min(dplus[v], dminus[v])
        remove_vertex(v)

    if not alive:
        return violations, None
    g_plus = nx.Graph()
    g_minus = nx.Graph()
    g_plus.add_nodes_from(alive)
    g_minus.add_nodes_from(alive)
    for u in alive:
        for w in plus[u]:
            if w in alive and u < w:
                g_plus.add_edge(u, w)
        for w in minus[u]:
            if w in alive and u < w:
                g_minus.add_edge(u, w)
    return violations, SignedGraph(g_plus, g_minus)


def chicken_algorithm(graph: SignedGraph, alpha: float = 0.0) -> tuple[int, list[SignedGraph]]:
    total_violations = 0
    remaining: list[SignedGraph] = []
    for kernel in kernelise_graph(graph):
        violations, leftover = _greedy_peck(kernel, alpha)
        total_violations += violations
        if leftover is not None:
            print("What is going on?")
            remaining.extend(kernelise_graph(leftover))
    return total_violations, remaining


if __name__ == "__main__":
    import os

    file = "Datasets/slashdot.txt"
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

    total_violations, remaining = chicken_algorithm(graph, alpha=0.5)
    print(f"Chicken Algorithm Total violations: {total_violations}")
    print(f"Remaining kernels after pecking: {len(remaining)}")
