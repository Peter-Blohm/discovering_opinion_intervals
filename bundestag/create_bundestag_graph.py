import pandas as pd
import numpy as np
import networkx as nx
import os
import re
import argparse
from datetime import datetime

BUNDESTAG_PERIODS = {
    17: ("2009-10-27", "2013-10-21"),
    18: ("2013-10-22", "2017-10-22"),
    19: ("2017-10-23", "2021-10-25"),
    20: ("2021-10-26", "2025-03-24"),
    21: ("2025-03-25", "2100-01-01"),
}


def parse_date_from_filename(filename):
    try:

        match = re.match(r"^(\d{8}).*$", filename)
        date_str = match.group(1)
        return datetime.strptime(date_str, "%Y%m%d")
    except (ValueError, IndexError):
        return None


def get_bundestag_for_date(date):
    for period, (start_str, end_str) in BUNDESTAG_PERIODS.items():
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        if start <= date <= end:
            return period
    return None


def filter_votes_by_bundestag(votes_df, selected_bundestage):
    if not selected_bundestage:
        return votes_df

    file_periods = {}
    unique_files = votes_df["filename"].unique()

    for filename in unique_files:
        date = parse_date_from_filename(filename)
        if date:
            period = get_bundestag_for_date(date)
            file_periods[filename] = period

    filtered_files = [
        f for f, period in file_periods.items() if period in selected_bundestage
    ]

    filtered_votes = votes_df[votes_df["filename"].isin(filtered_files)]

    return filtered_votes


def process_votes_to_matrix(votes_df):
    votes_df["vote_value"] = votes_df["janein"].apply(lambda x: 1 if x == "ja" else -1)

    vote_matrix = votes_df.pivot_table(
        index="Bezeichnung", columns="filename", values="vote_value", fill_value=0
    )

    return vote_matrix


def create_signed_graph(vote_matrix, agreement_threshold=0.75):
    G_plus = nx.Graph()
    G_minus = nx.Graph()

    people = vote_matrix.index.tolist()
    G_plus.add_nodes_from(people)
    G_minus.add_nodes_from(people)

    for i, person1 in enumerate(people):
        for j, person2 in enumerate(people):
            if i >= j:
                continue

            votes1 = vote_matrix.loc[person1].values
            votes2 = vote_matrix.loc[person2].values

            valid_indices = np.logical_and(votes1 != 0, votes2 != 0)
            if np.sum(valid_indices) >= 3:
                votes1_valid = votes1[valid_indices]
                votes2_valid = votes2[valid_indices]
                agreement_count = sum(votes1_valid == votes2_valid)
                total_votes = len(votes1_valid)
                agreement_ratio = agreement_count / total_votes

                if agreement_ratio > agreement_threshold:
                    G_plus.add_edge(person1, person2, weight=agreement_ratio)
                elif agreement_ratio < (1 - agreement_threshold):
                    G_minus.add_edge(person1, person2, weight=agreement_ratio)

    signed_graph = {"G_plus": G_plus, "G_minus": G_minus}

    return signed_graph


def create_person_id_mapping(signed_graph):
    all_persons = list(
        set(
            list(signed_graph["G_plus"].nodes()) + list(signed_graph["G_minus"].nodes())
        )
    )
    all_persons.sort()

    id_to_person = {i: person for i, person in enumerate(all_persons)}
    person_to_id = {person: i for i, person in enumerate(all_persons)}

    return id_to_person, person_to_id


def save_graph_to_file(edges, name, output_dir, id_mapping=None):
    os.makedirs(output_dir, exist_ok=True)

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

    if id_mapping:
        mapping_filename = os.path.join(output_dir, f"{name}_id_mapping.csv")
        with open(mapping_filename, "w") as f:
            f.write("ID,Person\n")
            for person, person_id in id_mapping.items():
                f.write(f"{person_id},{person}\n")

        return filename, mapping_filename

    return filename


def validate_bundestage(bundestage):
    valid_periods = list(BUNDESTAG_PERIODS.keys())
    invalid_periods = [p for p in bundestage if p not in valid_periods]

    if invalid_periods:
        raise ValueError(
            f"Invalid Bundestag periods: {invalid_periods}. Valid periods are: {valid_periods}"
        )

    return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create a signed graph from Bundestag voting data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="bundestag/all_votes.csv",
        help="Path to the CSV file with voting data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--bundestage",
        type=int,
        nargs="+",
        default=[],
        help="Filter by specific Bundestag periods (e.g., 17 18 19)",
    )
    parser.add_argument(
        "--agreement-threshold",
        type=float,
        default=0.75,
        help="Threshold for agreement ratio (default: 0.75)",
    )

    args = parser.parse_args()

    if args.bundestage:
        validate_bundestage(args.bundestage)

    return args


def main():
    args = parse_arguments()

    votes_df = pd.read_csv(args.input)

    if args.bundestage:
        bundestag_str = ", ".join(str(p) for p in args.bundestage)
        print(f"Filtering votes for Bundestag periods: {bundestag_str}")
        votes_df = filter_votes_by_bundestag(votes_df, args.bundestage)
        output_name = f"bundestag_signed_graph_periods_{'_'.join(str(p) for p in args.bundestage)}"
    else:
        print("Using all votes (no Bundestag period filter)")
        output_name = "bundestag_signed_graph_all_periods_2"

    vote_matrix = process_votes_to_matrix(votes_df)
    signed_graph = create_signed_graph(vote_matrix, args.agreement_threshold)

    print(f"Created signed graph with {signed_graph['G_plus'].number_of_nodes()} nodes")
    print(f"Positive edges: {signed_graph['G_plus'].number_of_edges()}")
    print(f"Negative edges: {signed_graph['G_minus'].number_of_edges()}")

    id_to_person, person_to_id = create_person_id_mapping(signed_graph)
    edges = [(u, v, 1) for u, v in signed_graph["G_plus"].edges()] + [
        (u, v, -1) for u, v in signed_graph["G_minus"].edges()
    ]

    print(
        f"Number of people with an edge: {len(set([u for u, v, sign in edges] + [v for u, v, sign in edges]))}"
    )

    people_with_edges = set([u for u, v, sign in edges] + [v for u, v, sign in edges])
    people_without_edges = set(vote_matrix.index) - people_with_edges

    print(
        f"\nPeople who voted but don't have connections exceeding threshold: {len(people_without_edges)}"
    )

    graph_file, mapping_file = save_graph_to_file(
        edges, output_name, args.output_dir, person_to_id
    )


if __name__ == "__main__":
    main()
