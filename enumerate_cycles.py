import os
import argparse
import random
from graph_utils.graph_embeddings.solve_embedding import check_embeddability, get_cycle_positions, plot_combined_graph_and_intervals
from graph_utils.signed_graph import read_signed_graph, SignedGraph
import time

def save_graph_to_file(edges, count, output_dir, tmp=False):
    """
    Saves a single valid graph to a text file in the specified format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not tmp:
        filename = os.path.join(output_dir, f"graph_{count}.txt")
    else:
        filename = os.path.join(output_dir, f"tmp_graph_{count}.txt")
    with open(filename, "w") as f:
        f.write("# FromNodeId\tToNodeId\tSign\n")
        for u, v, sign in edges:
            f.write(f"{u}\t{v}\t{sign}\n")
    return filename

def _backtrack_missing_edges_iterative(ordered_cycle, missing_edges, edges_to_place):
    """
    Iterative backtracking function with memoization.
    """
    memo = {}
    stack = []
    initial_state = {
        'left_ptr': 1,
        'right_ptr': -1,
        'edges_to_place': edges_to_place,
        'state': 'new',
    }
    stack.append(initial_state)

    while stack:
        frame = stack[-1]
        left_ptr = frame['left_ptr']
        right_ptr = frame['right_ptr'] % len(ordered_cycle)
        edges_to_place = frame['edges_to_place']
        key = (left_ptr, right_ptr)

        if key in memo:
            if memo[key]:
                return True
            else:
                stack.pop()
                continue

        if edges_to_place == 0:
            memo[key] = True
            return True

        if left_ptr >= right_ptr:
            memo[key] = False
            stack.pop()
            continue

        if frame['state'] == 'new':
            options = []

            left_vertex = ordered_cycle[left_ptr]
            right_vertex = ordered_cycle[right_ptr]
            next_right_vertex = ordered_cycle[(right_ptr - 1) % len(ordered_cycle)]
            next_left_vertex = ordered_cycle[(left_ptr + 1) % len(ordered_cycle)]

            option1 = (min(left_vertex, next_right_vertex), max(left_vertex, next_right_vertex))
            option2 = (min(right_vertex, next_left_vertex), max(right_vertex, next_left_vertex))

            options_list = []
            if option1 in missing_edges:
                options_list.append(('left_to_right', option1))
            if option2 in missing_edges:
                options_list.append(('right_to_left', option2))

            frame['options'] = options_list
            frame['option_index'] = 0
            frame['state'] = 'options'

        options = frame['options']
        option_index = frame['option_index']

        if option_index >= len(options):
            memo[key] = False
            stack.pop()
            continue

        option_type, edge = options[option_index]
        frame['option_index'] += 1

        if option_type == 'left_to_right':
            new_left_ptr = left_ptr
            new_right_ptr = (right_ptr - 1) % len(ordered_cycle)
        elif option_type == 'right_to_left':
            new_left_ptr = (left_ptr + 1) % len(ordered_cycle)
            new_right_ptr = right_ptr

        # Check memoization before pushing new state
        new_key = (new_left_ptr, new_right_ptr)
        if new_key in memo and memo[new_key] == False:
            continue

        new_frame = {
            'left_ptr': new_left_ptr,
            'right_ptr': new_right_ptr,
            'edges_to_place': edges_to_place - 1,
            'state': 'new',
        }
        stack.append(new_frame)

    return False

def check_cycle_subinterval(graph: SignedGraph):
    """
    Check if the signed graph is embeddable based on the specified condition.
    The positive edges must form a single cycle.
    """
    cycle = list(graph.G_plus.nodes())
    n = len(cycle)

    # Precompute all possible missing edges (negative edges)
    all_possible_edges = set(
        (min(u, v), max(u, v)) for u in cycle for v in cycle if u != v
    )
    positive_edge_set = set(
        (min(u, v), max(u, v)) for u, v in graph.G_plus.edges()
    )
    negative_edge_set = set(
        (min(u, v), max(u, v)) for u, v in graph.G_minus.edges()
    )
    missing_edges = all_possible_edges - positive_edge_set - negative_edge_set

    # The number of missing edges should be exactly n - 3
    required_missing_edges = n - 3
    if len(missing_edges) < required_missing_edges:
        return False

    # Attempt to find a valid placement for any starting vertex
    for start_idx in range(n):
        ordered_cycle = [cycle[(start_idx + i) % n] for i in range(n)]

        initial_edge = (min(ordered_cycle[-1], ordered_cycle[1]), max(ordered_cycle[-1], ordered_cycle[1]))
        if initial_edge not in missing_edges:
            continue

        edges_to_place = required_missing_edges - 1
        success = _backtrack_missing_edges_iterative(
            ordered_cycle,
            missing_edges,
            edges_to_place,
        )
        if success:
            return True
    return False

def generate_and_save_signed_graphs(num_nodes, target_count=20):
    """
    Generates signed graphs for the specified number of nodes and saves each
    valid graph to a file immediately after validation, up to the target count.
    """
    # Generate positive edges as a cycle
    positive_edges = [(i, (i + 1) % num_nodes, 1) for i in range(num_nodes)]
    remaining_edges = [(i, j, -1) for i in range(num_nodes) for j in range(i+1, num_nodes) if (i, j) not in ([(u, v) for u, v, _ in positive_edges] + [(v, u) for u, v, _ in positive_edges])]

    graph_dir = f"data/signed_graphs_{num_nodes}_nodes"
    ok_dir_path = f"data/signed_graphs_{num_nodes}_nodes_ok"
    bad_dir_path = f"data/signed_graphs_{num_nodes}_nodes_bad"

    os.makedirs(ok_dir_path, exist_ok=True)
    os.makedirs(bad_dir_path, exist_ok=True)

    random.seed(123)

    for idx in range(target_count):
        random.shuffle(remaining_edges)
        # take a random subset of the remaining edges of random size at least num_nodes
        num_of_neg_edges = random.randint(num_nodes, len(remaining_edges))
        neg_edges = remaining_edges[:num_of_neg_edges]

        edges = positive_edges + list(neg_edges)
        file_path = save_graph_to_file(edges, idx, graph_dir)
        graph = read_signed_graph(file_path)
        
        # start timer
        # start_time = time.time()
        # embeddable, start, end = check_embeddability(file_path)
        # end_time = time.time()
        # print(f"Time taken to check embeddability: {end_time - start_time}")
        print("starting subinterval condition check")
        start_time = time.time()
        subinterval_condition = check_cycle_subinterval(graph)
        end_time = time.time()
        print(f"Time taken to check subinterval condition: {end_time - start_time}")

        #print(f"For graph {idx+1}/{target_count} condition matches embeddability: {embeddable == subinterval_condition}")
        

        if subinterval_condition:
            target_file_path = os.path.join(ok_dir_path, os.path.basename(file_path)).replace(".txt", ".png")
            draw_edges = "missing"
        else:
            target_file_path = os.path.join(bad_dir_path, os.path.basename(file_path)).replace(".txt", ".png")
            draw_edges = "existing"

        pos = get_cycle_positions(graph.G_plus.nodes) 
        plot_combined_graph_and_intervals(graph, [], [], target_file_path, pos=pos, draw_edges=draw_edges)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save signed graphs.")
    parser.add_argument("--num_nodes", type=int, default=9, help="Number of nodes in the graph.")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of valid graphs to generate.")
    args = parser.parse_args()

    generate_and_save_signed_graphs(args.num_nodes, target_count=args.num_examples)
