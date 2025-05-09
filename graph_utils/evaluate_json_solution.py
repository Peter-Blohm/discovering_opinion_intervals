import json
from signed_graph import read_signed_graph


def read_labels_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    labels = {}
    for key, val in data.items():
        labels[int(key)] = val
    return labels

def read_intervals_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["intervals"]

def do_intervals_overlap(interval1, interval2):
    return max(interval1["start"], interval2["start"]) <= min(interval1["end"], interval2["end"])

if __name__ == "__main__":
    
    file_path_json = "slashdot_interval_solution.json"
    interval_path_json = "intervals.json"
    file_path_txt = "data/soc-sign-Slashdot090221.txt"

    labels_dict = read_labels_from_json(file_path_json)

    intervals = read_intervals_from_json(interval_path_json)

    graph = read_signed_graph(file_path_txt)

    total_violations = 0

    for i, j in graph.G_minus.edges:
        i_interval_idx = labels_dict.get(i)
        j_interval_idx = labels_dict.get(j)
        
        if i_interval_idx is None or j_interval_idx is None:
            print("something went wrong")
            continue
        
        i_interval = intervals[i_interval_idx]
        j_interval = intervals[j_interval_idx]
        
        if do_intervals_overlap(i_interval, j_interval):
            total_violations += 1

    for i, j in graph.G_plus.edges:
        i_interval_idx = labels_dict.get(i)
        j_interval_idx = labels_dict.get(j)
        
        if i_interval_idx is None or j_interval_idx is None:
            print("something went wrong")
            continue
        
        i_interval = intervals[i_interval_idx]
        j_interval = intervals[j_interval_idx]
        
        if not do_intervals_overlap(i_interval, j_interval):
            total_violations += 1

    print("Disagreement:", total_violations)

    total_edges = graph.G_plus.number_of_edges() + graph.G_minus.number_of_edges()
    agreement = total_edges - total_violations

    print("Agreement:", agreement)
