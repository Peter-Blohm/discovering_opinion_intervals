import networkx as nx

from graph_utils.signed_graph import read_signed_graph, SignedGraph
from graph_utils.signed_graph_kernelization import kernelize_graph

def write_signed_graph_to_metis(G: SignedGraph, output_file: str):
    # Get all unique nodes from G_plus and sort them
    nodes = G.G_plus.nodes()
    node_mapping = {i: i for i in nodes}
    startFromZero = False

    print(min(nodes))

    if min(nodes) == 0:
        print("gotta remap")
        node_mapping = {i: i + 1 for i in nodes}
        startFromZero = True

    print(len(nodes))
    print(min(nodes))
    print(max(nodes))

    print(sorted(list(G.G_plus.neighbors(1))))
    
    m_plus = G.G_plus.number_of_edges()
    m_minus = G.G_minus.number_of_edges()
    total_edges = m_plus + m_minus

    with open(output_file, 'w') as f:
        # Write the header line: n m f (f=1 for edge weights)
        f.write(f"{node_mapping[max(nodes)]} {total_edges} 11\n")

        # Iterate through each node in sorted order to maintain 1-based index mapping
        for x in range(0 if startFromZero else 1, max(nodes)+1):
            # have to iterate differenty
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

                # Prepare the line parts: start with node weight (0), then pairs of neighbor and sign
                line_parts = ['1']  # Node weight is 0
                for v, sign in adj_list:
                    line_parts.append(str(v))
                    line_parts.append(str(sign))

                # Join the parts into a string and write to file
                line = ' '.join(line_parts) + '\n'
                f.write(line)
            else:
                f.write('1\n')

if __name__ == "__main__":
    #for file in os.listdir("data"):
    data = f"data/wiki_S.txt"

    G = read_signed_graph(data)

    write_signed_graph_to_metis(G, "wiki_S.graph")

