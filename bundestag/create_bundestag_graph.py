import pandas as pd
import numpy as np
import networkx as nx
import os
from scipy.stats import pearsonr
from graph_utils.signed_graph import SignedGraph

def process_votes_to_matrix(csv_file):
    """
    Process votes from CSV into a matrix format.
    - Converts 'ja' votes to 1 and other votes to -1
    - Creates a matrix where rows are people (Bezeichnung) and columns are votes (filename)
    """
    # Read the CSV file
    votes = pd.read_csv(csv_file)
    
    # Convert votes to numerical values (ja=1, otherwise=-1)
    votes['vote_value'] = votes['janein'].apply(lambda x: 1 if x == 'ja' else -1)
    
    # Create a pivot table with people as rows and votes as columns
    vote_matrix = votes.pivot_table(
        index='Bezeichnung',
        columns='filename',
        values='vote_value',
        fill_value=0  # Fill missing values with 0
    )
    
    return vote_matrix

def create_signed_graph(vote_matrix):
    """
    Create a signed graph where:
    - Vertices are people (Bezeichnung)
    """
    # Create empty positive and negative graphs
    G_plus = nx.Graph()
    G_minus = nx.Graph()
    
    # Add all people as nodes to both graphs
    people = vote_matrix.index.tolist()
    G_plus.add_nodes_from(people)
    G_minus.add_nodes_from(people)
    
    for i, person1 in enumerate(people):
        for j, person2 in enumerate(people):
            if i >= j:  # Skip self-comparisons and duplicates
                continue
                
            # Get voting records for both people
            votes1 = vote_matrix.loc[person1].values
            votes2 = vote_matrix.loc[person2].values
            
            valid_indices = np.logical_and(votes1 != 0, votes2 != 0)
            # if np.sum(valid_indices) >= 3:  # At least 3 common votes for meaningful correlation
            #     corr, _ = pearsonr(
            #         votes1[valid_indices], 
            #         votes2[valid_indices]
            #     )
                
            #     # Add edges based on correlation thresholds
            #     if corr > 0.5:
            #         G_plus.add_edge(person1, person2, weight=corr)
            #     elif corr < -0.5:
            #         G_minus.add_edge(person1, person2, weight=corr)
            if np.sum(valid_indices) >= 3:  # At least 3 common votes for meaningful agreement ratio
                # Calculate agreement ratio instead of agreement ratio
                votes1_valid = votes1[valid_indices]
                votes2_valid = votes2[valid_indices]
                agreement_count = sum(votes1_valid == votes2_valid)
                total_votes = len(votes1_valid)
                agreement_ratio = agreement_count / total_votes
                
                # Add edges based on agreement ratio thresholds
                if agreement_ratio > 0.75:
                    G_plus.add_edge(person1, person2, weight=agreement_ratio)
                elif agreement_ratio < 0.25:
                    G_minus.add_edge(person1, person2, weight=agreement_ratio)
    
    # Create the signed graph using your class
    signed_graph = SignedGraph(G_plus, G_minus)
    
    return signed_graph

def create_person_id_mapping(signed_graph):
    """
    Creates a mapping from person names to numeric IDs.
    Returns:
        - id_to_person: Dictionary mapping numeric IDs to person names
        - person_to_id: Dictionary mapping person names to numeric IDs
    """
    # Get all unique persons from the graph
    all_persons = list(set(list(signed_graph.G_plus.nodes()) + list(signed_graph.G_minus.nodes())))
    all_persons.sort()  # Sort for consistent mapping
    
    # Create mapping dictionaries
    id_to_person = {i: person for i, person in enumerate(all_persons)}
    person_to_id = {person: i for i, person in enumerate(all_persons)}
    
    return id_to_person, person_to_id

def save_graph_to_file(edges, name, output_dir, id_mapping=None):
    """
    Saves a graph to a text file in the specified format.
    If id_mapping is provided, converts person names to IDs before saving.
    
    Args:
        edges: List of (u, v, sign) tuples
        name: Base name for the output file
        output_dir: Directory to save the file
        id_mapping: Optional dictionary mapping person names to numeric IDs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the graph with numeric IDs
    filename = os.path.join(output_dir, f"{name}.txt")
    with open(filename, "w") as f:
        f.write("# FromNodeId\tToNodeId\tSign\n")
        for u, v, sign in edges:
            if id_mapping:
                u_id = id_mapping[u]
                v_id = id_mapping[v]
                f.write(f"{u_id}\t{v_id}\t{sign}\n")
            else:
                f.write(f"{u}\t{v}\t{sign}\n")
    
    # If using ID mapping, also save the mapping for reference
    if id_mapping:
        mapping_filename = os.path.join(output_dir, f"{name}_id_mapping.csv")
        with open(mapping_filename, "w") as f:
            f.write("ID,Person\n")
            for person, person_id in id_mapping.items():
                f.write(f"{person_id},{person}\n")
        
        return filename, mapping_filename
    
    return filename

if __name__ == "__main__":
    csv_file = "bundestag/all_votes.csv"
    
    vote_matrix = process_votes_to_matrix(csv_file)
    print(f"Processed vote matrix with {len(vote_matrix)} people and {vote_matrix.shape[1]} votes")
    
    signed_graph = create_signed_graph(vote_matrix)
    print(f"Created signed graph with {signed_graph.number_of_nodes()} nodes")
    print(f"Positive edges: {signed_graph.G_plus.number_of_edges()}")
    print(f"Negative edges: {signed_graph.G_minus.number_of_edges()}")
    
    # Create ID mapping
    id_to_person, person_to_id = create_person_id_mapping(signed_graph)
    print(f"Created ID mapping for {len(person_to_id)} persons")
    
    # Create list of edges with numeric IDs
    output_dir = "bundestag/graphs"
    edges = [(u, v, 1) for u, v in signed_graph.G_plus.edges()] + \
            [(u, v, -1) for u, v in signed_graph.G_minus.edges()]
    
    print("Number of people with an edge: ")
    print(len(set([u for u, v, sign in edges] + [v for u, v, sign in edges])))

    # Difference between the the people from the vote matrix and the ones from the edge list
    print("Number of people in the vote matrix: ")
    print(len(vote_matrix.index))
    print("Difference: ")
    print(len(vote_matrix.index) - len(set([u for u, v, sign in edges] + [v for u, v, sign in edges])))

    # Find people who are in the vote matrix but don't have any edges
    people_with_edges = set([u for u, v, sign in edges] + [v for u, v, sign in edges])
    people_without_edges = set(vote_matrix.index) - people_with_edges
    
    # Print agreement ratios for people without edges (for sanity checking)
    # print("\nChecking agreement ratios for people without edges:")
    # for person_without_edges in sorted(people_without_edges):
    #     votes_person = vote_matrix.loc[person_without_edges].values
    #     print(f"\n{person_without_edges} agreement ratios:")
        
    #     for other_person in people_with_edges:
    #         votes_other = vote_matrix.loc[other_person].values

    #         # Find where both have valid votes
    #         valid_indices = np.logical_and(votes_person != 0, votes_other != 0)
    #         common_votes = np.sum(valid_indices)
            
    #         if common_votes >= 3:
    #             print(votes_person[valid_indices], votes_other[valid_indices])
    #             # Calculate agreement ratio like in create_signed_graph
    #             votes1_valid = votes_person[valid_indices]
    #             votes2_valid = votes_other[valid_indices]
    #             agreement_count = sum(votes1_valid == votes2_valid)
    #             agreement_ratio = agreement_count / common_votes
    #             print(f"  - With {other_person}: agreement_ratio={agreement_ratio:.3f}, common votes={common_votes}")

    print("\nPeople who voted but don't have agreement_ratios exceeding threshold:")
    for person in sorted(people_without_edges):
        print(f"- {person}")
    print(f"Total: {len(people_without_edges)} people without connections")
    
    # Save graph with ID mapping
    graph_file, mapping_file = save_graph_to_file(edges, "bundestag_signed_graph", output_dir, person_to_id)
    
    print(f"Graph saved to {graph_file}")
    print(f"ID mapping saved to {mapping_file}")