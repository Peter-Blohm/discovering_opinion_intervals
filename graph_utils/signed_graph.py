import re

import networkx as nx
import numpy as np


class SignedGraph:
    def __init__(self, G_plus: nx.Graph, G_minus: nx.Graph):
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
    
def read_signed_graph(file: str) -> SignedGraph:
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

def read_weighted_graph(file: str) -> nx.Graph:
    G = nx.Graph()

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
                weight = float(parts[2])
                if from_node == to_node:
                    continue
                from_node, to_node = (from_node, to_node) if from_node < to_node else (to_node, from_node)
                if G.has_edge(from_node, to_node):
                    G[from_node][to_node]['weight'] += weight
                else:
                    G.add_edge(from_node, to_node, weight=weight)
    return G