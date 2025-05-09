import json
import argparse
from signed_graph import read_signed_graph, SignedGraph, read_weighted_graph
import networkx as nx


def write_signed_graph_to_json(signed_graph: SignedGraph, output_file: str):
    edges = []

    # +1 weight for positive edges
    for u, v in signed_graph.G_plus.edges():
        if u > v:
            u, v = v, u
        edges.append({
            "source": u,
            "target": v,
            "weight": 1
        })

    # -1 weight for negative edges
    for u, v in signed_graph.G_minus.edges():
        if u > v:
            u, v = v, u
        edges.append({
            "source": u,
            "target": v,
            "weight": -1
        })

    data = {
        "edges": edges
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def write_weighted_graph_to_json(weighted_graph: nx.Graph, output_file: str):
    edges = []

    for u, v, data in weighted_graph.edges(data=True):
        if u > v:
            u, v = v, u
        if data['weight'] != 0:
            edges.append({
                "source": u,
                "target": v,
                "weight": data['weight']
            })

    data = {
        "edges": edges
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert graph data to JSON format')
    parser.add_argument('--type', choices=['signed', 'weighted', 'both'], default='both',
                        help='Type of conversion to perform (signed, weighted, or both)')
    parser.add_argument('--data', default="data/wiki_L_edges.txt",
                        help='Path to input data file')
    parser.add_argument('--output', 
                        help='Base name for output file(s) (without extension)')
    
    args = parser.parse_args()
    
    data_path = args.data
    
    if args.output:
        output_base = args.output
    else:
        import os
        output_base = os.path.splitext(os.path.basename(data_path))[0]
    
    if args.type == 'signed' or args.type == 'both':
        G_signed = read_signed_graph(data_path)
        signed_output = f"{output_base}_signed.json"
        write_signed_graph_to_json(G_signed, signed_output)
        print(f"Signed graph written to {signed_output}")
    
    if args.type == 'weighted' or args.type == 'both':
        G_weighted = read_weighted_graph(data_path)
        print(f"Number of nodes in weighted graph: {G_weighted.number_of_nodes()}")
        print(f"Number of edges in weighted graph: {G_weighted.number_of_edges()}")
        weighted_output = f"{output_base}_weighted.json"
        write_weighted_graph_to_json(G_weighted, weighted_output)
        print(f"Weighted graph written to {weighted_output}")

