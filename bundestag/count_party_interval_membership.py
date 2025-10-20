import json
import pandas as pd
from collections import defaultdict

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

def load_bundestag_membership(membership_file):
    """Load the Bundestag membership matrix from file."""
    membership_df = pd.read_csv(membership_file)
    
    # Convert to dictionary format for easier access
    person_to_bundestage = {}
    for _, row in membership_df.iterrows():
        person = row['Name']
        bundestage = {}
        for period in ['17', '18', '19', '20', '21']:
            if period in row and row[period] == 1:
                bundestage[int(period)] = True
        person_to_bundestage[person] = bundestage
    
    return person_to_bundestage

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
    file_path_json = "data/bundestag2_solution_good.json"
    interval_path_json = "intervals.json"
    signed_graph_file = "bundestag/graphs/bundestag_signed_graph.txt"
    id_mapping_file = "bundestag/graphs/bundestag_signed_graph_id_mapping.csv" 
    votes_csv_file = "bundestag/all_votes.csv"
    membership_file = "bundestag/person_bundestag_matrix.csv"
    
    # Load data
    id_to_person = load_id_mapping(id_mapping_file)
    person_to_fraction = load_fraction_data(votes_csv_file)
    labels_dict = read_labels_from_json(file_path_json)
    intervals = read_intervals_from_json(interval_path_json)
    person_to_bundestage = load_bundestag_membership(membership_file)
    
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
    print("===== PARTY INTERVAL ASSIGNMENT REPORT WITH BUNDESTAG MEMBERSHIP =====")
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
            for party, members in sorted(interval_to_parties[interval_idx].items(), key=lambda x: len(x[1]), reverse=True):
                count = len(members)
                percentage = (count / total_members) * 100
                
                bundestag_counts = defaultdict(int)
                for person in members:
                    if person in person_to_bundestage:
                        for period in person_to_bundestage[person]:
                            bundestag_counts[period] += 1
                
                bundestag_info = []
                for period in sorted(bundestag_counts.keys()):
                    period_percentage = (bundestag_counts[period] / count) * 100
                    bundestag_info.append(f"{period_percentage:.0f}% Bundestag {period}")
                
                bundestag_str = ", ".join(bundestag_info)
                print(f"    {party}: {count} members")# ({percentage:.1f}%) ({bundestag_str})")
            
            # New section that lists parties by Bundestag for this interval
            print("\n  Bundestag breakdown:")

            # Step 1: Find all Bundestag periods in this interval
            all_periods = []
            for party, members_list in interval_to_parties[interval_idx].items():
                for person in members_list:
                    if person in person_to_bundestage:
                        # Add all Bundestag periods this person was a member of
                        for period in person_to_bundestage[person]:
                            all_periods.append(period)

            # Get unique Bundestag periods and sort them
            bundestag_periods = sorted(set(all_periods))

            # Step 2: For each Bundestag period, analyze party distribution
            for period in bundestag_periods:
                print(f"    Bundestag {period}:")
                
                # Count how many members from each party were in this Bundestag period
                bundestag_party_counts = defaultdict(int)
                
                # Go through each party and its members in this interval
                for party, members_list in interval_to_parties[interval_idx].items():
                    # Check each person if they were in this Bundestag period
                    for person in members_list:
                        if person in person_to_bundestage and period in person_to_bundestage[person]:
                            # If yes, increment the count for this party
                            bundestag_party_counts[party] += 1
                
                # Calculate total members in this Bundestag period
                total_in_period = sum(bundestag_party_counts.values())
                
                # Print party breakdown for this Bundestag period, sorted by count
                for party, count in sorted(bundestag_party_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_in_period) * 100
                    print(f"      {party}:\t{count} members ({percentage:.1f}%)")
                
        else:
            print("  No members assigned to this interval")
        
        print("\n")
    
    # Generate summary statistics
    print("===== SUMMARY STATISTICS =====")
    party_totals = defaultdict(int)
    party_bundestag = defaultdict(lambda: defaultdict(int))
    
    for interval_idx in interval_to_parties:
        for party, members in interval_to_parties[interval_idx].items():
            party_totals[party] += len(members)
            
            for person in members:
                if person in person_to_bundestage:
                    for period in person_to_bundestage[person]:
                        party_bundestag[party][period] += 1
    
    total_assignments = sum(party_totals.values())
    print(f"Total party member assignments: {total_assignments}")
    
    print("Overall party distribution with Bundestag membership:")
    for party, count in sorted(party_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_assignments) * 100
        
        bundestag_info = []
        for period in sorted(party_bundestag[party].keys()):
            period_percentage = (party_bundestag[party][period] / count) * 100
            bundestag_info.append(f"{period_percentage:.0f}% Bundestag {period}")
        
        bundestag_str = ", ".join(bundestag_info)
        print(f"  {party}: {count} members ({percentage:.1f}%) ({bundestag_str})")

if __name__ == "__main__":
    generate_interval_report()