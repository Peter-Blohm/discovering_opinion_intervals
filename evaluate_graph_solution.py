import numpy as np
from graph_utils.signed_graph import read_signed_graph

file_path_clustering = "wiki_S.graph.clustering"
file_path_txt = "data/wiki_S.txt"

def read_numbers_from_file(file_path):
    with open(file_path, 'r') as file:
        numbers = [int(line.strip()) for line in file]
    return numbers

# Example usage
labels_array = read_numbers_from_file(file_path_clustering)

print(len(labels_array))

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