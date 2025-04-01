import h5py
import numpy as np

from graph_utils.signed_graph import read_signed_graph
from graph_utils.signed_graph_kernelization import kernelize_graph


def write_signed_graph_to_hdf5(file_name, graph_name, G, multiple_edges_enabled=False):
    """
    Writes a signed graph to an HDF5 file in the expected format.

    :param file_name: Name of the HDF5 file.
    :param graph_name: Name of the graph group in the HDF5 file.
    :param edges: List of tuples (source, target, weight) representing edges.
    :param num_vertices: Total number of vertices in the graph.
    :param multiple_edges_enabled: Whether multiple edges are allowed (default: False).
    """
    pos_edges = list(G.G_plus.edges)
    neg_edges = list(G.G_minus.edges)
    edges = np.array(pos_edges + neg_edges)
    print(max(G.G_plus.nodes()))
    edge_values = np.array([1] * len(pos_edges) + [-1] * len(neg_edges), dtype=np.float64)

    with h5py.File(file_name, 'w') as f:
        group = f.create_group(graph_name)

        # Store graph attributes
        group.create_dataset('multiple-edges-enabled', data =np.uint8(multiple_edges_enabled), shape=())

        # Convert edges to a NumPy array and store
        group.create_dataset('edges', data=edges, shape=(len(edges), 2), dtype=np.uint64)

        # Store edge weights separately
        group.create_dataset('edge-values', data=edge_values, shape=(len(edges)), dtype=np.float64)
        # group2 = f.create_group("edge-values")
        f.create_dataset('edge-values', data=edge_values, shape=(len(edges)), dtype=np.float64)
        # Create the graph id
        graph_type_id = np.array([10000], dtype=np.intc)  # Ensure it's a 1-element dataset
        group.create_dataset('graph-type-id', data=np.intc(10000), shape=(), dtype=np.intc)

        # Create number of vertices, and the rest
        group.create_dataset('number-of-vertices', data=np.uint64(max(G.G_plus.nodes())+1), shape=(), dtype=np.uint64)
        group.create_dataset('number-of-edges', data=np.uint64(len(edges)), shape=(), dtype=np.uint64)


if __name__ == "__main__":
    #for file in os.listdir("data"):
    data = f"data/wiki_S.txt"

    G = read_signed_graph(data)

    write_signed_graph_to_hdf5("wiki_S.h5", "graph", G)

