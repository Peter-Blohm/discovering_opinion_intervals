import re

import networkx as nx
import numpy as np


class SignedGraph:

    def __init__(self, G_plus, G_minus):
        self.G_plus = G_plus
        self.G_minus = G_minus

    def number_of_nodes(self):
        return self.G_plus.number_of_nodes()

    def number_of_edges(self):
        return self.G_plus.number_of_edges() + self.G_minus.number_of_edges()

    def has_edge(self, from_node, to_node):
        return self.G_plus.has_edge(from_node, to_node) or self.G_minus.has_edge(from_node, to_node)

    def has_plus_edge(self, from_node, to_node):
        self.G_plus.has_edge(from_node, to_node)

    def add_plus_edge(self, from_node, to_node):
        if self.G_minus.has_edge(from_node, to_node):
            return
        self.G_minus.add_node(from_node)
        self.G_minus.add_node(to_node)
        self.G_plus.add_edge(from_node, to_node)

    def has_minus_edge(self, from_node, to_node):
        self.G_minus.has_edge(from_node, to_node)

    def add_minus_edge(self, from_node, to_node):
        if self.G_plus.has_edge(from_node, to_node):
            self.G_plus.remove_edge(from_node, to_node)
        self.G_plus.add_node(from_node)
        self.G_plus.add_node(to_node)
        self.G_minus.add_edge(from_node, to_node)

    def remove_node(self, n):
        self.G_plus.remove_node(n)
        self.G_minus.remove_node(n)

    def remove_nodes_from(self, nodes):
        self.G_plus.remove_nodes_from(nodes)
        self.G_minus.remove_nodes_from(nodes)

    def remove_edge(self, u, v):
        if self.G_plus.has_edge(u, v):
            self.G_plus.remove_edge(u, v)
        self.G_minus.remove_edge(u, v)

    def subgraph(self, nodes):
        return SignedGraph(self.G_plus.subgraph(nodes), self.G_minus.subgraph(nodes))

    def copy(self):
        return SignedGraph(self.G_plus.copy(), self.G_minus.copy())
    
    def is_embeddable(self):
        """
        Check if the signed graph is embeddable based on the specified condition.
        The positive edges must form a single cycle.
        """
        # This thing only works for positive cycles

        cycle = list(self.G_plus.nodes())
        n = len(cycle)

        # Precompute all possible missing edges (negative edges)
        all_possible_edges = set(
            (min(u, v), max(u, v)) for u in cycle for v in cycle if u != v
        )
        positive_edge_set = set(
            (min(u, v), max(u, v)) for u, v in self.G_plus.edges()
        )
        negative_edge_set = set(
            (min(u, v), max(u, v)) for u, v in self.G_minus.edges()
        )
        missing_edges = all_possible_edges - positive_edge_set - negative_edge_set
        missing_edges = list(missing_edges)

        # The number of missing edges should be exactly n - 3
        required_missing_edges = n - 3
        if len(missing_edges) < required_missing_edges:
            #print(f"Not enough missing edges. Required: {required_missing_edges}, Available: {len(missing_edges)}")
            return False

        # Attempt to find a valid placement for any starting vertex
        for start_idx in range(n):
            ordered_cycle = [cycle[(start_idx + i) % n] for i in range(n)]

            if (min(ordered_cycle[-1], ordered_cycle[1]), max(ordered_cycle[-1], ordered_cycle[1])) not in missing_edges:
                continue

            if self._backtrack_missing_edges(ordered_cycle, missing_edges, required_missing_edges, current_missing=[(min(ordered_cycle[-1], ordered_cycle[1]), max(ordered_cycle[-1], ordered_cycle[1]))]):
                return True
        return False

    def _backtrack_missing_edges(self, ordered_cycle, missing_edges, edges_to_place, current_missing=None, left_ptr=1, right_ptr=-1):
        """
        Recursive backtracking function to place missing edges in a ladder-like pattern.

        Parameters:
        - ordered_cycle: List of vertices ordered starting from a specific start vertex.
        - missing_edges: List of all possible missing edges (as tuples).
        - edges_to_place: Number of missing edges to place.
        - current_missing: Currently placed missing edges.
        - left_ptr: Current left pointer index.
        - right_ptr: Current right pointer index.

        Returns:
        - True if a valid placement is found, False otherwise.
        """
        if len(current_missing) == edges_to_place:
            return True

        if left_ptr >= right_ptr % len(ordered_cycle):
            return False

        options = []

        # Calculate indices with wrap-around
        left_vertex = ordered_cycle[left_ptr]
        right_vertex = ordered_cycle[right_ptr]
        next_right_vertex = ordered_cycle[(right_ptr - 1) % len(ordered_cycle)]
        next_left_vertex = ordered_cycle[(left_ptr + 1) % len(ordered_cycle)]

        option1 = (min(left_vertex, next_right_vertex), max(left_vertex, next_right_vertex))
        option2 = (min(right_vertex, next_left_vertex), max(right_vertex, next_left_vertex))

        # Check if the options are available in missing_edges and not already placed
        if option1 in missing_edges and option1 not in current_missing:
            options.append(("left_to_right", option1))
        if option2 in missing_edges and option2 not in current_missing:
            options.append(("right_to_left", option2))

        for option_type, edge in options:
            current_missing.append(edge)
            if option_type == "left_to_right":
                # Move left pointer forward
                success = self._backtrack_missing_edges(
                    ordered_cycle,
                    missing_edges,
                    edges_to_place,
                    current_missing,
                    left_ptr,
                    right_ptr-1,
                )
            elif option_type == "right_to_left":
                # Move right pointer backward
                success = self._backtrack_missing_edges(
                    ordered_cycle,
                    missing_edges,
                    edges_to_place,
                    current_missing,
                    left_ptr+1,
                    right_ptr,
                )
            else:
                print("wth is going on")
                success = False

            if success:
                return True

            # Backtrack
            current_missing.pop()

        # If no options lead to a solution
        return False, left_ptr, right_ptr


def read_signed_graph(file):
    G = SignedGraph(nx.Graph(), nx.Graph())

    # Open the file and read the content
    with open(file, 'r') as file:
        for line in file:
            # Skip comment lines that start with '#'
            if line.startswith('#'):
                continue

            # Split the line into FromNodeId, ToNodeId, and Sign
            parts = re.split(r'[,#;\t ]+', line.strip())
            if len(parts) >= 3:
                from_node = int(parts[0])
                to_node = int(parts[1])
                sign = np.sign(int(parts[2]))
                if from_node == to_node:
                    continue
                from_node, to_node = (from_node, to_node) if from_node < to_node else (to_node, from_node)
                # If either existing edge or new edge has a negative sign, set it to -1
                if G.has_minus_edge(from_node, to_node) or sign == -1:
                    G.add_minus_edge(from_node, to_node)
                else:
                    # Add the edge with the sign as an attribute
                    G.add_plus_edge(from_node, to_node)
    return G
