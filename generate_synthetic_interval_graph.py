import random
from graph_utils.signed_graph import SignedGraph
import networkx as nx
from convert_to_json import write_signed_graph_to_json
import json
import argparse
import os
import time

def generate_synthetic_graph(intervals, n_per_int, m_frac, p, seed=None):
    random.seed(seed)
    num_intervals = len(intervals)
    total_nodes = num_intervals * n_per_int
    G_plus = nx.Graph()
    G_minus = nx.Graph()
    nodes = list(range(total_nodes))
    G_plus.add_nodes_from(nodes)
    G_minus.add_nodes_from(nodes)
    for u in nodes:
        for v in range(u + 1, total_nodes):
            if random.random() > m_frac:
                continue
            interval_u = intervals[u // n_per_int]
            interval_v = intervals[v // n_per_int]
            overlap = (interval_u['start'] < interval_v['end']) and (interval_v['start'] < interval_u['end'])
            if random.random() < (p if overlap else (1 - p)):
                G_plus.add_edge(u, v)
            else:
                G_minus.add_edge(u, v)
    return SignedGraph(G_plus, G_minus)

def do_intervals_overlap(interval1, interval2):
    # Check if two intervals overlap
    return max(interval1["start"], interval2["start"]) <= min(interval1["end"], interval2["end"])

def calculate_disagreement(graph, intervals, n_per_int):
    total_violations = 0
    
    # Create standard assignment: each node is assigned to the interval it was sampled from
    standard_assignment = {}
    for node in range(len(graph.G_plus.nodes)):
        standard_assignment[node] = node // n_per_int
    
    # Check negative edges - violation if intervals overlap
    for i, j in graph.G_minus.edges:
        i_interval_idx = standard_assignment.get(i)
        j_interval_idx = standard_assignment.get(j)
        
        i_interval = intervals[i_interval_idx]
        j_interval = intervals[j_interval_idx]
        
        # For negative edges: violation if intervals overlap
        if do_intervals_overlap(i_interval, j_interval):
            total_violations += 1

    # Check positive edges - violation if intervals don't overlap
    for i, j in graph.G_plus.edges:
        i_interval_idx = standard_assignment.get(i)
        j_interval_idx = standard_assignment.get(j)
        
        i_interval = intervals[i_interval_idx]
        j_interval = intervals[j_interval_idx]
        
        # For positive edges: violation if intervals don't overlap
        if not do_intervals_overlap(i_interval, j_interval):
            total_violations += 1
            
    return total_violations

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic interval graphs')
    parser.add_argument('--output_dir', type=str, default=".", help='Directory to save the output files')
    # parser.add_argument('--n_per_int', type=int, default=100, help='Number of nodes per interval')
    # parser.add_argument('--m_frac', type=float, default=1.0, help='Fraction of edges to include (0.0 to 1.0)')
    # parser.add_argument('--p', type=float, default=1.0, help='Probability parameter for edge sign (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: random)')
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    p_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    m_frac_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    n_per_int_vals = [100]


    # Use command line arguments with their defaults
    # n_per_int = args.n_per_int
    # m_frac = args.m_frac
    # p = args.p

    for n_per_int in n_per_int_vals:
        for m_frac in m_frac_vals:
            for p in p_vals:

                seed = args.seed if args.seed is not None else int(time.time())
                
                try:
                    with open("intervals.json", "r") as f:
                        intervals_data = json.load(f)
                        intervals = intervals_data["intervals"]
                except FileNotFoundError:
                    assert False, "intervals.json file not found. Please provide a valid file."

                print(f"Generating synthetic graph with parameters:")
                print(f"  - n_per_int: {n_per_int}")
                print(f"  - m_frac: {m_frac}")
                print(f"  - p: {p}")
                print(f"  - seed: {seed}")
                
                signed_graph = generate_synthetic_graph(intervals, n_per_int, m_frac, p, seed)
                
                # Calculate disagreement for standard assignment
                disagreement = calculate_disagreement(signed_graph, intervals, n_per_int)
                print(f"Disagreement for standard assignment: {disagreement}")
                
                # Include disagreement in filename
                output_filename = os.path.join(output_dir, f"synthetic_graph_n-per-int_{n_per_int}_m-frac_{m_frac}_p_{p}_disagreement_{disagreement}.json")
                
                # Save the graph
                write_signed_graph_to_json(signed_graph, output_filename)
                print(f"Graph generated and saved to {output_filename}")
                
                # Also save standard assignment for reference
                standard_assignment = {}
                for node in range(len(signed_graph.G_plus.nodes)):
                    standard_assignment[str(node)] = node // n_per_int
                
                assignment_filename = os.path.join(output_dir, f"standard_assignment_n-per-int_{n_per_int}_m-frac_{m_frac}_p_{p}_disagreement_{disagreement}.json")
                with open(assignment_filename, 'w') as f:
                    json.dump(standard_assignment, f, indent=2)
                print(f"Standard assignment saved to {assignment_filename}")