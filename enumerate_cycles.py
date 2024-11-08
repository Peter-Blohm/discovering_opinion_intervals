import networkx as nx
import itertools
import os
import argparse
import random
from graph_utils.graph_embeddings.solve_embedding import check_embeddability

def is_valid_graph(nodes, edges):
    """
    Check if the graph is valid by:
    - Ensuring positive edges form a single cycle.
    - Verifying no cut exists with only positive edges across any partition.
    """
    G = nx.Graph()
    for u, v, sign in edges:
        G.add_edge(u, v, sign=sign)

    # Check positive cycle condition
    positive_edges = [(u, v) for u, v, sign in edges if sign == 1]
    G_pos = nx.Graph()
    G_pos.add_edges_from(positive_edges)

    # Check if the positive edges form a single cycle
    if not nx.is_connected(G_pos) or len(G_pos.edges()) != nodes:
        return False

    # Check no cut condition
    negative_edges = [(u, v) for u, v, sign in edges if sign == -1]
    G_neg = nx.Graph()
    G_neg.add_edges_from(negative_edges)

    # check if each vertex is in a negative edge
    for node in G.nodes():
        if not G_neg.has_node(node):
            return False

    # Check all cuts
    for cut_size in range(1, nodes // 2 + 1):
        for subset in itertools.combinations(range(nodes), cut_size):
            other_set = set(range(nodes)) - set(subset)
            illegal = True
            for edge in negative_edges:
                if edge[0] in subset and edge[1] in other_set:
                    illegal = False
            if illegal:
                return True
    return False

def save_graph_to_file(edges, nodes, count, output_dir):
    """
    Saves a single valid graph to a text file in the specified format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"graph_{count}.txt")
    with open(filename, "w") as f:
        f.write("# FromNodeId\tToNodeId\tSign\n")
        for u, v, sign in edges:
            f.write(f"{u}\t{v}\t{sign}\n")

    return filename

def generate_and_save_signed_graphs(nodes, target_count=20):
    """
    Generates signed graphs for the specified number of nodes and saves each
    valid graph to a file immediately after validation, up to the target count.
    """
    # Generate positive edges as a cycle
    positive_edges = [(i, (i + 1) % nodes, 1) for i in range(nodes)]
    remaining_edges = [(i, j, -1) for i in range(nodes) for j in range(i + 1, nodes) if (i, j) not in [(u, v) for u, v, _ in positive_edges] + [(v, u) for u, v, _ in positive_edges]]

    graph_dir = f"data/signed_graphs_{nodes}_nodes"
    ok_dir_path = f"data/signed_graphs_{nodes}_nodes_ok"
    bad_dir_path = f"data/signed_graphs_{nodes}_nodes_bad"

    os.makedirs(ok_dir_path, exist_ok=True)
    os.makedirs(bad_dir_path, exist_ok=True)

    valid_graph_count = 0

    # Convert combinations to a list and shuffle it for random ordering
    neg_edge_combinations = list(itertools.combinations(remaining_edges, nodes))
    random.shuffle(neg_edge_combinations)

    for neg_edges in neg_edge_combinations:
        edges = positive_edges + list(neg_edges)
        if is_valid_graph(nodes, edges):
            valid_graph_count += 1
            file_path = save_graph_to_file(edges, nodes, valid_graph_count, graph_dir)
            check_embeddability(file_path, ok_dir_path, bad_dir_path)
            
            if valid_graph_count >= target_count:
                print(f"Generated {target_count} valid graphs for {nodes} nodes.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save signed graphs.")
    parser.add_argument("--num_nodes", type=int, default=8, help="Number of nodes in the graph.")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of valid graphs to generate.")
    args = parser.parse_args()

    generate_and_save_signed_graphs(args.num_nodes, target_count=args.num_examples)
