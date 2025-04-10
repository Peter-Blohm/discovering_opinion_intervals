import json
import numpy as np
from graph_utils.signed_graph import read_signed_graph

file_path_json = "heuristics/wiki_S_solution.json"
file_path_txt = "data/wiki_S.txt"

def read_labels_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Convert keys to int (just in case) and ensure proper index access
    labels = {}
    for key, val in data.items():
        labels[int(key)] = val
    return labels

labels_dict = read_labels_from_json(file_path_json)
graph = read_signed_graph(file_path_txt)

total_violations = 0

for i, j in graph.G_minus.edges:
    if labels_dict.get(i) == labels_dict.get(j):
        total_violations += 1

for i, j in graph.G_plus.edges:
    if labels_dict.get(i) != labels_dict.get(j):
        total_violations += 1

print("Disagreement:", total_violations)

total_edges = graph.G_plus.number_of_edges() + graph.G_minus.number_of_edges()
agreement = total_edges - total_violations

print("Agreement:", agreement)
