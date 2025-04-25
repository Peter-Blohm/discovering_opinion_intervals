import networkx as nx
import argparse
import os

from graph_utils.signed_graph import read_signed_graph, SignedGraph, read_weighted_graph
from graph_utils.signed_graph_kernelization import kernelize_graph

def write_signed_graph_to_metis(G: SignedGraph, output_file: str):
    # Get all unique nodes from G_plus and sort them
    nodes = G.G_plus.nodes()
    node_mapping = {i: i for i in nodes}
    startFromZero = False

    if min(nodes) == 0:
        print("Remapping node IDs: zero-based to one-based")
        node_mapping = {i: i + 1 for i in nodes}
        startFromZero = True
    
    m_plus = G.G_plus.number_of_edges()
    m_minus = G.G_minus.number_of_edges()
    total_edges = m_plus + m_minus

    with open(output_file, 'w') as f:
        # Write the header line: n m f (f=1 for edge weights)
        f.write(f"{node_mapping[max(nodes)]} {total_edges} 11\n")

        # Iterate through each node in sorted order to maintain 1-based index mapping
        for x in range(0 if startFromZero else 1, max(nodes)+1):
            # Collect all neighbors from G_plus and G_minus
            if x in nodes:
                adj_plus = list(G.G_plus.neighbors(x))
                adj_minus = list(G.G_minus.neighbors(x))

                adj_list = []
                # Add positive edges
                for v in adj_plus:
                    adj_list.append((node_mapping[v], 1))
                # Add negative edges
                for v in adj_minus:
                    adj_list.append((node_mapping[v], -1))

                # Sort the adjacency list by the neighbor's mapped ID
                adj_list.sort(key=lambda x: x[0])

                # Prepare the line parts: start with node weight (1), then pairs of neighbor and sign
                line_parts = ['1']  # Node weight is 1
                for v, sign in adj_list:
                    line_parts.append(str(v))
                    line_parts.append(str(sign))

                # Join the parts into a string and write to file
                line = ' '.join(line_parts) + '\n'
                f.write(line)
            else:
                f.write('1\n')

def write_weighted_graph_to_metis(G: nx.Graph, output_file: str):
    # Get all unique nodes from G
    nodes = list(G.nodes())
    node_mapping = {i: i for i in nodes}
    startFromZero = False

    if min(nodes) == 0:
        print("Remapping node IDs: zero-based to one-based")
        node_mapping = {i: i + 1 for i in nodes}
        startFromZero = True
    
    total_edges = G.number_of_edges()

    with open(output_file, 'w') as f:
        # Write the header line: n m f (f=1 for edge weights)
        f.write(f"{node_mapping[max(nodes)]} {total_edges} 11\n")

        # Iterate through each node in sorted order
        for x in range(0 if startFromZero else 1, max(nodes)+1):
            if x in nodes:
                adj_list = []
                # Get all neighbors with their weights
                for v in G.neighbors(x):
                    weight = int(G[x][v]['weight'])
                    if weight != 0:
                        # Map the neighbor to its new ID
                        adj_list.append((node_mapping[v], weight))

                # Sort the adjacency list by the neighbor's mapped ID
                adj_list.sort(key=lambda x: x[0])

                # Prepare the line parts: start with node weight (1), then pairs of neighbor and weight
                line_parts = ['1']  # Node weight is 1
                for v, weight in adj_list:
                    line_parts.append(str(v))
                    line_parts.append(str(weight))

                # Join the parts into a string and write to file
                line = ' '.join(line_parts) + '\n'
                f.write(line)
            else:
                f.write('1\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert graph to METIS/Chaco format')
    parser.add_argument('--type', choices=['signed', 'weighted', 'both'], default='signed',
                        help='Type of conversion to perform (signed, weighted, or both)')
    parser.add_argument('--data', default="data/wiki_S.txt",
                        help='Path to input data file')
    parser.add_argument('--output', 
                        help='Base name for output file(s) (without extension)')
    
    args = parser.parse_args()
    
    data_path = args.data
    
    # Determine output filename base (without extension)
    if args.output:
        output_base = args.output
    else:
        # Extract filename from the input path
        output_base = os.path.splitext(os.path.basename(data_path))[0]
    
    if args.type == 'signed' or args.type == 'both':
        G_signed = read_signed_graph(data_path)
        signed_output = f"{output_base}_signed.graph"
        write_signed_graph_to_metis(G_signed, signed_output)
        print(f"Signed graph written to {signed_output}")
    
    if args.type == 'weighted' or args.type == 'both':
        G_weighted = read_weighted_graph(data_path)
        weighted_output = f"{output_base}_weighted.graph"
        write_weighted_graph_to_metis(G_weighted, weighted_output)
        print(f"Weighted graph written to {weighted_output}")

