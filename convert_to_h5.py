import h5py
import numpy as np
import argparse
import os

from graph_utils.signed_graph import read_signed_graph, SignedGraph, read_weighted_graph
import networkx as nx


def write_signed_graph_to_hdf5(graph: SignedGraph, output_file: str, graph_name="graph", multiple_edges_enabled=False):
    """
    Writes a signed graph to an HDF5 file in the expected format.

    :param graph: The SignedGraph object to write
    :param output_file: Name of the HDF5 file
    :param graph_name: Name of the graph group in the HDF5 file
    :param multiple_edges_enabled: Whether multiple edges are allowed (default: False)
    """
    pos_edges = list(graph.G_plus.edges)
    neg_edges = list(graph.G_minus.edges)
    edges = np.array(pos_edges + neg_edges)
    edge_values = np.array([1] * len(pos_edges) + [-1] * len(neg_edges), dtype=np.float64)
    
    all_nodes = set(graph.G_plus.nodes()) | set(graph.G_minus.nodes())
    max_node_id = max(all_nodes) if all_nodes else 0

    with h5py.File(output_file, 'w') as f:
        group = f.create_group(graph_name)

        # Store graph attributes
        group.create_dataset('multiple-edges-enabled', data=np.uint8(multiple_edges_enabled), shape=())

        # Convert edges to a NumPy array and store
        group.create_dataset('edges', data=edges, shape=(len(edges), 2), dtype=np.uint64)

        # Store edge weights separately
        group.create_dataset('edge-values', data=edge_values, shape=(len(edges)), dtype=np.float64)
        # Also add edge values at the root level as needed by some tools
        f.create_dataset('edge-values', data=edge_values, shape=(len(edges)), dtype=np.float64)
        
        # Create the graph id
        group.create_dataset('graph-type-id', data=np.intc(10000), shape=(), dtype=np.intc)

        # Create number of vertices, and the rest
        group.create_dataset('number-of-vertices', data=np.uint64(max_node_id + 1), shape=(), dtype=np.uint64)
        group.create_dataset('number-of-edges', data=np.uint64(len(edges)), shape=(), dtype=np.uint64)


def write_weighted_graph_to_hdf5(graph: nx.Graph, output_file: str, graph_name="graph", multiple_edges_enabled=False):
    """
    Writes a weighted graph to an HDF5 file in the expected format.

    :param graph: The NetworkX graph object to write
    :param output_file: Name of the HDF5 file
    :param graph_name: Name of the graph group in the HDF5 file
    :param multiple_edges_enabled: Whether multiple edges are allowed (default: False)
    """
    # Extract edges with weights
    edge_list = []
    edge_values = []
    
    for u, v, data in graph.edges(data=True):
        if data['weight'] != 0:
            edge_list.append((u, v))
            edge_values.append(data['weight'])
    
    edges = np.array(edge_list)
    edge_values = np.array(edge_values, dtype=np.float64)
    
    max_node_id = max(graph.nodes()) if graph.nodes else 0

    with h5py.File(output_file, 'w') as f:
        group = f.create_group(graph_name)

        # Store graph attributes
        group.create_dataset('multiple-edges-enabled', data=np.uint8(multiple_edges_enabled), shape=())

        # Convert edges to a NumPy array and store
        group.create_dataset('edges', data=edges, shape=(len(edges), 2), dtype=np.uint64)

        # Store edge weights separately
        group.create_dataset('edge-values', data=edge_values, shape=(len(edges)), dtype=np.float64)
        # Also add edge values at the root level as needed by some tools
        f.create_dataset('edge-values', data=edge_values, shape=(len(edges)), dtype=np.float64)
        
        # Create the graph id
        group.create_dataset('graph-type-id', data=np.intc(10000), shape=(), dtype=np.intc)

        # Create number of vertices, and the rest
        group.create_dataset('number-of-vertices', data=np.uint64(max_node_id + 1), shape=(), dtype=np.uint64)
        group.create_dataset('number-of-edges', data=np.uint64(len(edges)), shape=(), dtype=np.uint64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert graph data to HDF5 format')
    parser.add_argument('--type', choices=['signed', 'weighted', 'both'], default='signed',
                        help='Type of conversion to perform (signed, weighted, or both)')
    parser.add_argument('--data', default="data/wiki_S.txt",
                        help='Path to input data file')
    parser.add_argument('--output', 
                        help='Base name for output file(s) (without extension)')
    parser.add_argument('--graph-name', default="graph",
                        help='Name of the graph group in the HDF5 file')
    
    args = parser.parse_args()
    
    data_path = args.data
    graph_name = args.graph_name
    
    # Determine output filename base (without extension)
    if args.output:
        output_base = args.output
    else:
        # Extract filename from the input path
        output_base = os.path.splitext(os.path.basename(data_path))[0]
    
    if args.type == 'signed' or args.type == 'both':
        G_signed = read_signed_graph(data_path)
        signed_output = f"{output_base}_signed.h5"
        write_signed_graph_to_hdf5(G_signed, signed_output, graph_name)
        print(f"Signed graph written to {signed_output}")
    
    if args.type == 'weighted' or args.type == 'both':
        G_weighted = read_weighted_graph(data_path)
        weighted_output = f"{output_base}_weighted.h5"
        write_weighted_graph_to_hdf5(G_weighted, weighted_output, graph_name)
        print(f"Weighted graph written to {weighted_output}")

