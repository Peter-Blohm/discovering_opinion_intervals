import json
from graph_utils.signed_graph import read_signed_graph, SignedGraph


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


if __name__ == "__main__":
    data = f"data/soc-sign-bitcoinotc.csv"

    G = read_signed_graph(data)

    write_signed_graph_to_json(G, "bitcoin.json")

