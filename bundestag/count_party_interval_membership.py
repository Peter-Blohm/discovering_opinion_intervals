import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from graph_utils.signed_graph import read_signed_graph

def load_id_mapping(mapping_file):
    """Load the ID to person mapping from the CSV file."""
    mapping_df = pd.read_csv(mapping_file)
    id_to_person = dict(zip(mapping_df['ID'], mapping_df['Person']))
    return id_to_person

def load_fraction_data(votes_csv):
    """Load the fraction data from the original votes CSV."""
    votes_df = pd.read_csv(votes_csv)
    # Get unique persons and their fractions
    person_fraction = votes_df[['Bezeichnung', 'Fraktion/Gruppe']].drop_duplicates()
    # Convert to dictionary
    person_to_fraction = dict(zip(person_fraction['Bezeichnung'], person_fraction['Fraktion/Gruppe']))
    return person_to_fraction

def read_labels_from_json(json_path):
    """Read vertex-to-interval assignments from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Convert keys to int and ensure proper index access
    labels = {}
    for key, val in data.items():
        labels[int(key)] = val
    return labels

def read_intervals_from_json(json_path):
    """Read interval definitions from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["intervals"]

def generate_interval_report():
    # File paths
    file_path_json = "heuristics/bundestag2_solution.json"
    interval_path_json = "intervals.json"
    signed_graph_file = "bundestag/graphs/bundestag_signed_graph.txt"
    id_mapping_file = "bundestag/graphs/bundestag_signed_graph_id_mapping.csv" 
    votes_csv_file = "bundestag/all_votes.csv"
    
    # Load data
    id_to_person = load_id_mapping(id_mapping_file)
    person_to_fraction = load_fraction_data(votes_csv_file)
    labels_dict = read_labels_from_json(file_path_json)
    intervals = read_intervals_from_json(interval_path_json)
    
    # Create a dictionary to store which parties are in which intervals
    interval_to_parties = defaultdict(lambda: defaultdict(list))
    
    # Collect party members by interval
    for vertex_id, interval_idx in labels_dict.items():
        if vertex_id in id_to_person:
            person_name = id_to_person[vertex_id]
            if person_name in person_to_fraction:
                party = person_to_fraction[person_name]
                interval_to_parties[interval_idx][party].append(person_name)
    
    # Generate report
    print("===== PARTY INTERVAL ASSIGNMENT REPORT =====")
    print(f"Total intervals: {len(intervals)}")
    print(f"Total assigned vertices: {len(labels_dict)}")
    print("\n")
    
    for interval_idx, interval_data in enumerate(intervals):
        start = interval_data["start"]
        end = interval_data["end"]
        print(f"Interval {interval_idx}: [{start}, {end}]")
        
        if interval_idx in interval_to_parties:
            party_counts = {party: len(members) for party, members in interval_to_parties[interval_idx].items()}
            total_members = sum(party_counts.values())
            print(f"  Total members: {total_members}")
            
            print("  Party breakdown:")
            for party, count in sorted(party_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_members) * 100
                print(f"    {party}: {count} members ({percentage:.1f}%)")
                
        else:
            print("  No members assigned to this interval")
        
        print("\n")
    
    # Generate summary statistics
    print("===== SUMMARY STATISTICS =====")
    party_totals = defaultdict(int)
    for interval_idx in interval_to_parties:
        for party, members in interval_to_parties[interval_idx].items():
            party_totals[party] += len(members)
    
    total_assignments = sum(party_totals.values())
    print(f"Total party member assignments: {total_assignments}")
    
    print("Overall party distribution:")
    for party, count in sorted(party_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_assignments) * 100
        print(f"  {party}: {count} members ({percentage:.1f}%)")

if __name__ == "__main__":
    generate_interval_report()