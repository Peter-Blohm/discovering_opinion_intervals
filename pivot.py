import networkx as nx
import random

from graph_utils.signed_graph import read_signed_graph, SignedGraph
from graph_utils.signed_graph_kernelization import kernelize_graph


def pivot_clustering(graph):
    """
    Perform pivot-based graph clustering.
    
    :param graph: A NetworkX graph
    :return: A dictionary mapping nodes to cluster labels
    """
    unclustered_nodes = set(graph.nodes())
    clusters = {}
    cluster_id = 0
    
    while unclustered_nodes:
        # Select a random pivot
        pivot = random.choice(list(unclustered_nodes))
        
        # Form a cluster with the pivot and its neighbors
        cluster = {pivot} | set(graph.neighbors(pivot))
        clusters.update({node: cluster_id for node in cluster})
        
        # Remove clustered nodes from the unclustered set
        unclustered_nodes -= cluster
        cluster_id += 1

    return clusters

if __name__ == "__main__":
    #for file in os.listdir("data"):
    data = f"data/soc-sign-bitcoinotc.csv"

    G = read_signed_graph(data)

    disagreements = []

    for i in range(20):

        clustering_result = pivot_clustering(G.G_plus)

        # Print cluster assignments
        # for node, cluster in clustering_result.items():
        #     print(f"Node {node} -> Cluster {cluster}")
        
        total_violations = 0

        for i, j in G.G_minus.edges:
            if clustering_result[i] == clustering_result[j]:
                total_violations += 1

        for i, j in G.G_plus.edges:
            if clustering_result[i] != clustering_result[j]:
                total_violations += 1

        # print("Disagreement: ", total_violations)

        disagreements.append(total_violations)

        # Calculate Agreement

        # min_disagreement = min(disagreements)

        print("Disagreement: ", total_violations)

        agreement = G.G_plus.number_of_edges() + G.G_minus.number_of_edges() - total_violations

        print("Agreement: ", agreement)
