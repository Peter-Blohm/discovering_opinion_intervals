import h5py
import numpy as np
from graph_utils.signed_graph import read_signed_graph

# Replace this with the path to your .h5 file
file_path_h5 = "bitcoin_solution.h5"
file_path_txt = "data/soc-sign-bitcoinotc.csv"

def print_hdf5_structure(item, prefix=''):
    if isinstance(item, h5py.Group):
        for key in item.keys():
            print(f"{prefix}/{key}")
            print_hdf5_structure(item[key], prefix + '/' + key)


with h5py.File(file_path_h5, 'r') as f:
    # print("Structure of the file:")
    # print_hdf5_structure(f)
    
    # Assuming "graph" is one of the groups at the top level
    graph_group = f["graph"]

    # Read edges array
    edges = graph_group["edges"][:]
    # print("Edges data shape:", edges.shape)
    # print("Edges data type:", edges.dtype)
    edges_array = np.array(edges)
    # print("First few edges:", edges_array[:5])

    # Let's read the "labels" dataset

    labels = f["labels"][:]
    # print("Labels data shape:", labels.shape)
    # print("Labels data type:", labels.dtype)
    # Do something with the labels, for example convert to a NumPy array
    labels_array = np.array(labels)
    # print("First few labels:", labels_array)

    
graph = read_signed_graph(file_path_txt)

total_violations = 0

for i, j in graph.G_minus.edges:
    if labels_array[i] == labels_array[j]:
        total_violations += 1

for i, j in graph.G_plus.edges:
    if labels_array[i] != labels_array[j]:
        total_violations += 1

print("Disagreement: ", total_violations)

# Calculate Agreement

agreement = graph.G_plus.number_of_edges() + graph.G_minus.number_of_edges() - total_violations

print("Agreement: ", agreement)