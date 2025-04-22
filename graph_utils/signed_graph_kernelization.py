import networkx as nx
import os
from graph_utils.signed_graph import SignedGraph, read_signed_graph

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

#region Old code

# def _find_one_seperated_components(graph: SignedGraph):
#     # Find graph components that are separated by a single vertex
#     # i.e. a vertex whose removal would disconnect the graph
#     # both on the positive and negative edges
#     # returns a list of subgraphs

#     components = [graph]
#     new_components = []

#     while True:
#         print("Len Components", len(components))
#         if len(components) == 0:
#             break
#         signed_graph = components.pop()
#         print("Number of nodes", signed_graph.number_of_nodes())

#         combined_graph = nx.Graph()
#         combined_graph.add_edges_from(signed_graph.G_plus.edges())
#         combined_graph.add_edges_from(signed_graph.G_minus.edges())

#         split = False

#         for node in combined_graph.nodes():

#             temp_combined_graph = combined_graph.copy()
#             temp_combined_graph.remove_node(node)        
#             # Create a combined graph of positive and negative edges
            
#             if not nx.is_connected(temp_combined_graph):
#                 # Get the connected components as sets of nodes
#                 conn_components = list(nx.connected_components(temp_combined_graph))
                
#                 # Convert each component to a SignedGraph and add to results
#                 for component_nodes in conn_components:
#                     component = signed_graph.subgraph(component_nodes)
#                     components.append(kernelize_graph(component))
#                     split = True
#                 break
        
#         if not split:
#             new_components.append(signed_graph)
        
#     return new_components

def _find_one_seperated_components(graph: SignedGraph):
    # Find graph components that are separated by a single vertex
    # i.e. a vertex whose removal would disconnect the graph
    # both on the positive and negative edges
    # returns a list of subgraphs

    new_components = []

    signed_graph = graph
    print("Number of nodes", signed_graph.number_of_nodes())

    combined_graph = nx.Graph()
    combined_graph.add_edges_from(signed_graph.G_plus.edges())
    combined_graph.add_edges_from(signed_graph.G_minus.edges())

    split = False

    for node in combined_graph.nodes():
        edges = list(combined_graph.edges(node))
        print(node)

        # remove all the edges
        combined_graph.remove_edges_from(edges)

        connected_components = list(nx.connected_components(combined_graph))

        print("Num connected components", len(connected_components))

        # Create a combined graph of positive and negative edges
        if len(connected_components) >= 3:
            # Get the connected components as sets of nodes
            conn_components = connected_components
            
            # Convert each component to a SignedGraph and add to results
            for component_nodes in conn_components:
                component = signed_graph.subgraph(component_nodes)
                new_components.append(component)
                split = True
            break

        # add the edges back
        combined_graph.add_edges_from(edges)
    
    if not split:
        new_components.append(signed_graph)
        
    return new_components

#endregion

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

    if not safe:
        graphs_prime = []
        for g in graphs:
            combined_graph = nx.compose(g.G_plus, g.G_minus)
            two_edge_connected_components = nx.k_edge_subgraphs(combined_graph, 2)
            for component in two_edge_connected_components:
                subgraph = graph.subgraph(component).copy()
                graphs_prime.append(subgraph)
        graphs = graphs_prime
        
    if len(graphs) < 2:
        return graphs
    kernel_graphs = []
    for g in graphs:
        kernel_graphs.extend(kernelize_graph(g))
    return kernel_graphs

def _find_singly_pos_connected_vertices(graph: SignedGraph):
    """
    Find vertices with exactly one positive neighbor and no negative neighbors.
    :param graph: the input SignedGraph
    :return: list of node IDs
    """
    candidates = []

    for node, plus_degree in graph.G_plus.degree():
        if plus_degree == 1:  # Exactly one positive neighbor
            minus_degree = graph.G_minus.degree(node) if node in graph.G_minus else 0
            if minus_degree == 0:  # No negative neighbors
                candidates.append(node)
    return candidates


def kernelize_for_fixed_intervals(graph: SignedGraph) -> SignedGraph:
    """
    Repeatedly removes vertices that have exactly one positive neighbor and no negative neighbors.
    
    :param graph: the input SignedGraph
    :return: kernelized SignedGraph
    """
    result_graph = graph.copy()
    
    while True:
        to_remove = _find_singly_pos_connected_vertices(result_graph)
        if not to_remove:
            break
        result_graph.remove_nodes_from(to_remove)
    
    return result_graph

def save_graph_to_file(edges, name, output_dir):
    """
    Saves a single valid graph to a text file in the specified format.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{name}.txt")
    with open(filename, "w") as f:
        f.write("# FromNodeId\tToNodeId\tSign\n")
        for u, v, sign in edges:
            f.write(f"{u}\t{v}\t{sign}\n")
    return filename

if __name__ == "__main__":
    file = "data/wiki_L.txt"

    graph = read_signed_graph(file)

    kernel_graph = kernelize_for_fixed_intervals(graph)

    edges = [(u, v, 1) for u, v in kernel_graph.G_plus.edges()] + \
            [(u, v, -1) for u, v in kernel_graph.G_minus.edges()]
    
    name = os.path.splitext(os.path.basename(file))[0] + "_kernel"
    output_dir = "data"

    save_graph_to_file(edges, name, output_dir)
