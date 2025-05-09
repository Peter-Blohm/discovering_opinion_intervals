import re
import numpy as np
import networkx as nx

from graph_utils.signed_graph_heuristics import find_max_ratio_vertex
from graph_utils.signed_graph import SignedGraph, read_weighted_graph, save_graph_to_file, transform_weighted_graph_to_signed_graph
from graph_utils.signed_graph_kernelization import kernelize_signed_graph

def read_graph(file):
    G = nx.Graph()

    with open(file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue

            parts = re.split(r'[,#;\t]', line.strip())
            if len(parts) >= 3:
                from_node = int(parts[0])
                to_node = int(parts[1])
                sign = np.sign(int(parts[2]))

                if G.has_edge(from_node, to_node):
                    if G[from_node][to_node]['sign'] == -1 or sign == -1:
                        G[from_node][to_node]['sign'] = -1
                else:
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


if __name__ == "__main__":
    #for file in os.listdir("data"):
    data = "data/out.wikiconflict"

    print(f"Converting {data} to signed graph")
    G_weighted = read_weighted_graph(data)
    G = transform_weighted_graph_to_signed_graph(G_weighted)


    # print(f"Name: {data}")
    print(f"Pre Vertices: {len(G_weighted.nodes())}")
    # print(f"Pre Edges: {len(G_weighted.edges())}")
    # print(f"Pre Positive Edges: {sum(1 for u, v, data in G_weighted.edges(data=True) if data['weight'] > 0)}")
    # print(f"Pre Negative Edges: {sum(1 for u, v, data in G_weighted.edges(data=True) if data['weight'] < 0)}")

    # print(f"Vertices: {len(set(G.G_plus.nodes()).union(set(G.G_minus.nodes())))}")
    # print(f"Max Node: {max(G.G_plus.nodes())}")
    # # Count positive, negative, and zero edges
    positive_edges = sum(1 for u, v, data in G.G_plus.edges(data=True))# if data['weight'] > 0)
    negative_edges = sum(1 for u, v, data in G.G_minus.edges(data=True))# if data['weight'] < 0)
    
    print(f"Total Edges: {positive_edges + negative_edges}")
    print(f"Positive Edges: {positive_edges}")
    print(f"Negative Edges: {negative_edges}")

    output_dir = "data"
    name = "wikiconflict_processed"

    save_graph_to_file(G, name, output_dir)

    # G = read_signed_graph(data)

    # print(f"Name: {data}")
    # print(f"Vertices: {G.number_of_nodes()}")
    # print(f"Edges: {G.number_of_edges()}")
    # print(f"Positive Edges: {G.G_plus.number_of_edges()}")
    # print(f"Negative edges {G.G_minus.number_of_edges()}")

    # # Kernelize the graph
    # graphs = kernelize_signed_graph(G, safe=True)
    # largest_graph = max(graphs, key=lambda graph: graph.number_of_nodes())
    # G.G_plus = nx.relabel.convert_node_labels_to_integers(G.G_plus, first_label=1, ordering='default')
    # G.G_minus = nx.relabel.convert_node_labels_to_integers(G.G_minus, first_label=1, ordering='default')
    # print(f"Number of vertices in largest connected component after 1 round of error free kernelization: {largest_graph.number_of_nodes()}")
    # print(f"Number of positives edges in largest connected component after 1 round of error free kernelization: {largest_graph.G_plus.number_of_edges()}")
    # print(f"Number of negative edges in largest connected component after 1 round of error free kernelization: {largest_graph.G_minus.number_of_edges()}")

    # print(f"chicken algorithm violated edges:{chicken_algorithm(G)}")
