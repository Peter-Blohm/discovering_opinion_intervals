import os
import argparse
import random
import networkx as nx
from graph_utils.graph_embeddings.solve_embedding import check_embeddability, get_cycle_positions, plot_combined_graph_and_intervals
from graph_utils.signed_graph import read_signed_graph

def save_graph_to_file(edges, count, output_dir, tmp=False):
    """
    Saves a single valid graph to a text file in the specified format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not tmp:
        filename = os.path.join(output_dir, f"graph_{count}.txt")
    else:
        filename = os.path.join(output_dir, f"tmp_graph_{count}.txt")
    with open(filename, "w") as f:
        f.write("# FromNodeId\tToNodeId\tSign\n")
        for u, v, sign in edges:
            f.write(f"{u}\t{v}\t{sign}\n")
    return filename

def generate_and_save_signed_graphs(num_nodes, target_count=20):
    """
    Generates signed graphs for the specified number of nodes and saves each
    valid graph to a file immediately after validation, up to the target count.
    """
    # Generate positive edges as a cycle
    positive_edges = [(i, (i + 1) % num_nodes, 1) for i in range(num_nodes)]
    remaining_edges = [(i, j, -1) for i in range(num_nodes) for j in range(i+1, num_nodes) if (i, j) not in ([(u, v) for u, v, _ in positive_edges] + [(v, u) for u, v, _ in positive_edges])]

    graph_dir = f"data/signed_graphs_{num_nodes}_nodes"
    ok_dir_path = f"data/signed_graphs_{num_nodes}_nodes_ok"
    bad_dir_path = f"data/signed_graphs_{num_nodes}_nodes_bad"

    os.makedirs(ok_dir_path, exist_ok=True)
    os.makedirs(bad_dir_path, exist_ok=True)


    for idx in range(target_count):
        random.shuffle(remaining_edges)
        # take a random subset of the remaining edges of random size at least num_nodes
        num_of_neg_edges = random.randint(num_nodes, len(remaining_edges))
        neg_edges = remaining_edges[:num_of_neg_edges]

        edges = positive_edges + list(neg_edges)
        file_path = save_graph_to_file(edges, idx, graph_dir)

        embeddable, _, _ = check_embeddability(file_path)

        if embeddable:
            for edge in remaining_edges[num_of_neg_edges:]:
                tmp_path = save_graph_to_file(edges + [edge], idx, graph_dir, tmp=True)
                tmp_embeddable, _, _ = check_embeddability(tmp_path)
                if tmp_embeddable:
                    edges.append(edge)
                os.remove(tmp_path)

            file_path = save_graph_to_file(edges, idx, graph_dir)
            _, final_start, final_end = check_embeddability(file_path)

            graph = read_signed_graph(file_path)
            pos = get_cycle_positions(graph.G_plus.nodes) 

            target_file_path = os.path.join(ok_dir_path, os.path.basename(file_path)).replace(".txt", ".png")

            plot_combined_graph_and_intervals(graph, final_start, final_end, target_file_path, pos=pos, draw_edges="missing")

            G_choral = nx.Graph()

            for u, v, sign in edges:
                if sign == 1:
                    G_choral.add_edge(u, v, color="green")
            
            for edge in remaining_edges[num_of_neg_edges:]:
                if edge in edges:
                    continue
                u, v, _ = edge
                G_choral.add_edge(u, v, color="blue")

            # Check chordality
            is_chordal = nx.is_chordal(G_choral)
            print(f"Graph is chordal matches embeddability: {is_chordal}")
    
        else:
            for neg_edge in neg_edges:
                edges.remove(neg_edge)
                tmp_path = save_graph_to_file(edges, idx, graph_dir, tmp=True)
                tmp_embeddable, _, _ = check_embeddability(tmp_path)
                if tmp_embeddable:
                    edges.append(neg_edge)
                os.remove(tmp_path)
    
            file_path = save_graph_to_file(edges, idx, graph_dir)
            _, final_start, final_end = check_embeddability(file_path)
    
            graph = read_signed_graph(file_path)
            pos = get_cycle_positions(graph.G_plus.nodes)
    
            target_file_path = os.path.join(bad_dir_path, os.path.basename(file_path)).replace(".txt", ".png")
    
            plot_combined_graph_and_intervals(graph, final_start, final_end, target_file_path, pos=pos, draw_edges="existing")

            G_choral = nx.Graph()

            for u, v, sign in edges:
                if sign == 1:
                    G_choral.add_edge(u, v, color="green")
            
            for edge in remaining_edges[num_of_neg_edges:]:
                if edge in edges:
                    continue
                u, v, _ = edge
                G_choral.add_edge(u, v, color="blue")

            # Check chordality
            is_chordal = nx.is_chordal(G_choral)
            print(f"Graph is chordal matches embeddability: {not is_chordal}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save signed graphs.")
    parser.add_argument("--num_nodes", type=int, default=9, help="Number of nodes in the graph.")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of valid graphs to generate.")
    args = parser.parse_args()

    generate_and_save_signed_graphs(args.num_nodes, target_count=args.num_examples)
