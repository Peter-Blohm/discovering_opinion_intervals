import os
import argparse
import time
import hmac
import math
from tqdm import tqdm
import networkx as nx
# Requires networkx>=2.8 for nx.weisfeiler_lehman_graph_hash

from graph_utils.graph_embeddings.solve_embedding import check_embeddability
from graph_utils.signed_graph import read_signed_graph, SignedGraph
from graph_utils.signed_graph_kernelization import kernelize_graph

def save_graph_to_file(edges, count, output_dir):
    """
    Saves a single valid graph to a text file in the specified format.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"graph_{count}.txt")
    with open(filename, "w") as f:
        f.write("# FromNodeId\tToNodeId\tSign\n")
        for u, v, sign in edges:
            f.write(f"{u}\t{v}\t{sign}\n")
    return filename

def feistel_encrypt(num, num_bits, key, rounds=3):
    """
    Encrypt a number using a Feistel cipher for format-preserving encryption.
    num: the number to encrypt (integer)
    num_bits: the number of bits in the number's representation
    key: the encryption key (bytes)
    rounds: number of Feistel rounds
    Returns the encrypted number (integer)
    """
    left_bits = num_bits // 2
    right_bits = num_bits - left_bits

    max_left = (1 << left_bits) - 1
    max_right = (1 << right_bits) - 1

    left = (num >> right_bits) & max_left
    right = num & max_right

    for i in range(rounds):
        # Convert right to bytes, padding with zeros to the required length
        right_bytes = right.to_bytes(math.ceil(right_bits / 8), byteorder='big')
        h = hmac.new(key, right_bytes + bytes([i]), digestmod='sha256')
        digest = h.digest()
        # Convert the digest to an integer modulo (max_left + 1)
        f_result = int.from_bytes(digest, byteorder='big') % (max_left + 1)
        # Update left and right
        new_left = right
        new_right = left ^ f_result
        left, right = new_left, new_right

    # Combine left and right
    encrypted = (left << right_bits) | right
    return encrypted

def generate_all_signed_graphs(num_nodes, output_dir):
    """
    Generates all possible signed graphs for a given number of nodes in a pseudorandom order and saves each graph to a file.
    
    This version uses a two‐step duplicate–pruning:
      1. Build a NetworkX graph based only on the positive edges.
      2. Compute a Weisfeiler–Lehman hash (using the edge attribute 'sign') of that graph.
      3. For graphs with the same hash, run a full isomorphism check (using VF2) to decide if the graph is really isomorphic to one already processed.
    """
    all_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    m = len(all_edges)
    total_combinations = 2 ** math.comb(num_nodes,2)
    key = os.urandom(16)  # Generate a random 128-bit key

    graph_count = 0
    # This dict maps WL hash values to a list of already‐seen graphs (as NetworkX objects)
    canonical_reps = {}

    for i in tqdm(range(total_combinations)):
        # if i%1000 == 0:
        #     print(len(canonical_reps))

        # Generate permuted index using Feistel cipher
        permuted_i = i#feistel_encrypt(i, m, key)

        # Convert to binary string of length m
        binary_str = format(permuted_i, f'0{m}b')
        if len(binary_str) > m:
            binary_str = binary_str[-m:]

        # Generate signs: '1' -> 1, '0' -> -1
        signs = [1 if bit == '1' else -1 for bit in binary_str]

        # Create the full edge list with signs (for saving and embeddability check)
        edges = [(u, v, sign) for (u, v), sign in zip(all_edges, signs)]

        # --- Build a NetworkX graph based on positive edges only ---
        G = SignedGraph(nx.Graph(), nx.Graph())
        for (u, v), sign in zip(all_edges, signs):
            if sign == 1:
                G.add_plus_edge(u, v)
            else:
                G.add_minus_edge(u, v)

        # Compute a hash for the graph (using the edge attribute 'sign')
        graph_hash = nx.weisfeiler_lehman_graph_hash(G.G_plus)

        # Check against previously seen graphs with the same hash.
        skip_graph = False
        if graph_hash in canonical_reps:
            for candidate in canonical_reps[graph_hash]:
                # Use VF2 isomorphism with an edge_match that compares the 'sign' attribute.
                GM = nx.algorithms.isomorphism.GraphMatcher(candidate, G.G_plus)
                if GM.is_isomorphic():
                    skip_graph = True
                    break
            if skip_graph:
                continue
            else:
                canonical_reps[graph_hash].append(G.G_plus)
        else:
            canonical_reps[graph_hash] = [G.G_plus]

        
        kernels = kernelize_graph(G)
        if len(kernels) > 1:
            #print(f"Multiple kernels found.")
            continue

        if len(kernels) < 1:
            #print(f"Graph falls apart.")
            continue
            
        if kernels[0].number_of_nodes() != num_nodes:
            #print(f"Different kernel.")
            continue


        # Save the graph and check embeddability (only for non-duplicate graphs)
        file_path = save_graph_to_file(edges, graph_count, output_dir)
        #error, _, _ = check_embeddability(file_path)
        # if error >= 6:
        #     print(f"Something happened on graph {graph_count}.")
        #    break
        graph_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save all signed graphs (pruning isomorphic duplicates based on positive edges).")
    parser.add_argument("--num_nodes", type=int, required=True, help="Number of nodes in the graph.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated graphs.")
    args = parser.parse_args()

    start_time = time.time()
    generate_all_signed_graphs(args.num_nodes, args.output_dir)
    end_time = time.time()

    print(f"Generated unique signed graphs for {args.num_nodes} nodes in {end_time - start_time:.2f} seconds.")
