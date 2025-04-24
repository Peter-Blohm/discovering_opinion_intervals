import pandas as pd
import networkx as nx
import os
from collections import defaultdict, Counter
from graph_utils.signed_graph import SignedGraph, read_signed_graph

def load_id_mapping(mapping_file):
    """
    Load the ID to person mapping from the CSV file.
    Returns a dictionary mapping numeric IDs to person names.
    """
    mapping_df = pd.read_csv(mapping_file)
    id_to_person = dict(zip(mapping_df['ID'], mapping_df['Person']))
    return id_to_person

def load_fraction_data(votes_csv):
    """
    Load the fraction data from the original votes CSV.
    Returns a dictionary mapping person names to their fractions.
    """
    votes_df = pd.read_csv(votes_csv)
    # Get unique persons and their fractions
    person_fraction = votes_df[['Bezeichnung', 'Fraktion/Gruppe']].drop_duplicates()
    # Convert to dictionary
    person_to_fraction = dict(zip(person_fraction['Bezeichnung'], person_fraction['Fraktion/Gruppe']))
    return person_to_fraction

def count_cross_fraction_edges(signed_graph, id_to_person, person_to_fraction):
    """
    Count the number of positive edges between vertices (people) belonging to different fractions.
    
    Args:
        signed_graph: The signed graph object
        id_to_person: Dictionary mapping numeric IDs to person names
        person_to_fraction: Dictionary mapping person names to their fractions
        
    Returns:
        total_cross_edges: Total number of positive edges between different fractions
        fraction_connections: Dictionary with counts of connections between specific fraction pairs
    """
    # Initialize counters
    total_cross_edges = 0
    fraction_connections = defaultdict(int)
    
    # Get all positive edges
    for u_id, v_id in signed_graph.G_plus.edges():
        # Convert IDs to person names if needed
        if id_to_person:
            u_person = id_to_person.get(u_id, str(u_id))
            v_person = id_to_person.get(v_id, str(v_id))
        else:
            u_person = u_id
            v_person = v_id
        
        # Get fractions for both people
        u_fraction = person_to_fraction.get(u_person, "Unknown")
        v_fraction = person_to_fraction.get(v_person, "Unknown")
        
        # Count if they belong to different fractions
        if u_fraction != v_fraction:
            total_cross_edges += 1
            
            # Sort fraction names alphabetically to avoid counting A-B and B-A as different pairs
            fraction_pair = tuple(sorted([u_fraction, v_fraction]))
            fraction_connections[fraction_pair] += 1
    
    return total_cross_edges, fraction_connections

def analyze_fraction_relationships(signed_graph_file, id_mapping_file, votes_csv_file):
    """
    Analyze the relationships between different fractions in the signed graph.
    
    Args:
        signed_graph_file: Path to the signed graph file
        id_mapping_file: Path to the ID mapping file
        votes_csv_file: Path to the original votes CSV file
    """
    # Load the signed graph
    signed_graph = read_signed_graph(signed_graph_file)
    print(f"Loaded signed graph with {signed_graph.number_of_nodes()} nodes")
    print(f"Positive edges: {signed_graph.G_plus.number_of_edges()}")
    print(f"Negative edges: {signed_graph.G_minus.number_of_edges()}")
    
    # Load ID to person mapping
    id_to_person = load_id_mapping(id_mapping_file)
    print(f"Loaded mapping for {len(id_to_person)} persons")
    
    # Load fraction data
    person_to_fraction = load_fraction_data(votes_csv_file)
    print(f"Loaded fraction data for {len(person_to_fraction)} persons")
    
    # Count cross-fraction edges
    total_cross_edges, fraction_connections = count_cross_fraction_edges(
        signed_graph, id_to_person, person_to_fraction
    )
    
    # Calculate percentage of cross-fraction edges
    total_positive_edges = signed_graph.G_plus.number_of_edges()
    percentage = (total_cross_edges / total_positive_edges) * 100 if total_positive_edges > 0 else 0
    
    # Print results
    print(f"\nPositive edges between different fractions: {total_cross_edges} ({percentage:.2f}% of all positive edges)")
    
    print("\nBreakdown of cross-fraction connections:")
    # Sort by number of connections (descending)
    sorted_connections = sorted(fraction_connections.items(), key=lambda x: x[1], reverse=True)
    for fraction_pair, count in sorted_connections:
        print(f"  {fraction_pair[0]} — {fraction_pair[1]}: {count} connections")
    
    # Count number of politicians per fraction
    fraction_counts = Counter(person_to_fraction.values())
    
    print("\nNumber of politicians per fraction:")
    for fraction, count in sorted(fraction_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fraction}: {count} members")
    
    # Analyze within-fraction connections
    within_fraction_edges = defaultdict(int)
    total_within_edges = 0
    
    for u_id, v_id in signed_graph.G_plus.edges():
        if id_to_person:
            u_person = id_to_person.get(u_id, str(u_id))
            v_person = id_to_person.get(v_id, str(v_id))
        else:
            u_person = u_id
            v_person = v_id
        
        u_fraction = person_to_fraction.get(u_person, "Unknown")
        v_fraction = person_to_fraction.get(v_person, "Unknown")
        
        if u_fraction == v_fraction:
            within_fraction_edges[u_fraction] += 1
            total_within_edges += 1
    
    print(f"\nPositive edges within same fraction: {total_within_edges} ({100 - percentage:.2f}% of all positive edges)")
    
    print("\nWithin-fraction connections:")
    for fraction, count in sorted(within_fraction_edges.items(), key=lambda x: x[1], reverse=True):
        fraction_size = fraction_counts[fraction]
        # Calculate maximum possible connections within the fraction: n*(n-1)/2
        max_possible = (fraction_size * (fraction_size - 1)) / 2
        density = (count / max_possible) * 100 if max_possible > 0 else 0
        print(f"  {fraction}: {count} connections (density: {density:.2f}%)")

if __name__ == "__main__":
    # File paths
    signed_graph_file = "bundestag/graphs/bundestag_signed_graph.txt"
    id_mapping_file = "bundestag/graphs/bundestag_signed_graph_id_mapping.csv" 
    votes_csv_file = "bundestag/all_votes.csv"
    
    # Run analysis
    analyze_fraction_relationships(signed_graph_file, id_mapping_file, votes_csv_file)