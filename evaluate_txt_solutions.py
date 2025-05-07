from graph_utils.signed_graph import read_signed_graph, read_weighted_graph, transform_weighted_graph_to_signed_graph

def read_clustering_file(file_path):
    """Reads the clustering solution file where each line represents a cluster's nodes."""
    labels = {}
    with open(file_path, 'r') as f:
        for cluster_idx, line in enumerate(f):
            # Split the line into node IDs (handles tabs and whitespace)
            nodes = list(map(int, line.strip().split()))
            for node in nodes:
                if node in labels:
                    raise ValueError(f"Node {node} appears in multiple clusters.")
                labels[node] = cluster_idx
    # Create labels array assuming nodes are 0-based and contiguous
    n_nodes = len(labels)
    labels_array = [0] * n_nodes
    for node, cluster in labels.items():
        labels_array[node] = cluster
    return labels_array

# Paths to files
file_path_clustering = "../RAMA/slashdot_solution.txt"
file_path_txt = "data/soc-sign-Slashdot090221.txt"

# Read the clustering solution
labels_array = read_clustering_file(file_path_clustering)

# Read the signed graph
G_weighted = read_weighted_graph(file_path_txt)
graph = transform_weighted_graph_to_signed_graph(G_weighted)

# Calculate edge violations
total_violations = 0

# Check negative edges (should be in different clusters)
for i, j in graph.G_minus.edges():
    if labels_array[i] == labels_array[j]:
        total_violations += 1

# Check positive edges (should be in same cluster)
for i, j in graph.G_plus.edges():
    if labels_array[i] != labels_array[j]:
        total_violations += 1

# Output results
print("Number of nodes:", len(labels_array))
print("Total violations (Disagreement):", total_violations)
agreement = graph.G_plus.number_of_edges() + graph.G_minus.number_of_edges() - total_violations
print("Agreement:", agreement)