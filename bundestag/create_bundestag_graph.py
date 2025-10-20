#!/usr/bin/env python3
"""
Create an anonymised signed graph from Bundestag roll-call data.
The output contains *only* integer vertex IDs; no plain-text names are ever written.
"""
import argparse
import os
import re
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd

CUTOFF = "2025-03-25"
BUNDESTAG_PERIODS = {
    17: ("2009-10-27", "2013-10-21"),
    18: ("2013-10-22", "2017-10-22"),
    19: ("2017-10-23", "2021-10-25"),
    20: ("2021-10-26", "2025-03-24"),
    21: ("2025-03-25", "2100-01-01"),
}
NAME_COLS = ["Name","Vorname","Bezeichnung"]


def parse_date_from_filename(filename: str):
    m = re.match(r"^(\d{8})", filename)
    return datetime.strptime(m.group(1), "%Y%m%d") if m else None


def get_bundestag_for_date(date: datetime):
    for period, (start_s, end_s) in BUNDESTAG_PERIODS.items():
        if datetime.strptime(start_s, "%Y-%m-%d") <= date <= datetime.strptime(
            end_s, "%Y-%m-%d"
        ):
            return period
    return None


def filter_votes_by_bundestag(votes: pd.DataFrame, selected_periods):
    if not selected_periods:
        return votes
    file_period = {
        f: get_bundestag_for_date(parse_date_from_filename(f))
        for f in votes["filename"].unique()
    }
    keep = [f for f, p in file_period.items() if p in selected_periods]
    return votes[votes["filename"].isin(keep)]


def process_votes_to_matrix(votes: pd.DataFrame):
    votes["vote_value"] = votes["janein"].map(lambda x: 1 if x == "ja" else -1)
    return votes.pivot_table(
        index="person_id", columns="filename", values="vote_value", fill_value=0
    )


def create_signed_graph(matrix: pd.DataFrame, thr: float = 0.75):
    g_pos, g_neg = nx.Graph(), nx.Graph()
    ids = matrix.index.tolist()
    g_pos.add_nodes_from(ids)
    g_neg.add_nodes_from(ids)

    mat = matrix.values
    for i, u in enumerate(ids):
        for j in range(i + 1, len(ids)):
            v = ids[j]
            v1, v2 = mat[i], mat[j]
            mask = np.logical_and(v1 != 0, v2 != 0)
            if mask.sum() < 3:
                continue
            agree = (v1[mask] == v2[mask]).mean()
            if agree > thr:
                g_pos.add_edge(u, v, weight=agree)
            elif agree < 1 - thr:
                g_neg.add_edge(u, v, weight=agree)
    return {"G_plus": g_pos, "G_minus": g_neg}


def save_edge_list(edges, name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{name}.txt")
    with open(fname, "w") as f:
        f.write("# FromNodeId\tToNodeId\tSign\n")
        for u, v, s in edges:
            f.write(f"{u}\t{v}\t{s}\n")
    return fname


def validate_periods(periods):
    bad = [p for p in periods if p not in BUNDESTAG_PERIODS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Invalid Bundestag periods {bad}. Valid: {list(BUNDESTAG_PERIODS)}"
        )
    return periods


def main():
    ap = argparse.ArgumentParser(description="Build an anonymised signed graph.")
    ap.add_argument(
        "--input",
        default="bundestag/all_votes_name_firstname.csv",
        help="CSV with raw voting data",
    )
    ap.add_argument("--output-dir", default="data", help="Output directory")
    ap.add_argument(
        "--bundestage",
        type=int,
        nargs="+",
        default=[17, 18, 19, 20],
        metavar="N",
        help="Keep only these Bundestag periods",
    )
    ap.add_argument("--agreement-threshold", type=float, default=0.75)
    args = ap.parse_args()

    validate_periods(args.bundestage)

    votes = pd.read_csv(args.input)
    votes = filter_votes_by_bundestag(votes, args.bundestage)

    unique_names = sorted(votes["Bezeichnung"].unique())
    person_to_id = {name: idx for idx, name in enumerate(unique_names)}
    votes["person_id"] = votes["Bezeichnung"].map(person_to_id)
    anon_votes = votes.drop(columns=[c for c in NAME_COLS if c in votes.columns])

    anon_path = os.path.join(args.output_dir, "all_votes_person_id.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    anon_votes.to_csv(anon_path, index=False)
    print(f"Anonymised vote table written to {anon_path}")

    # --------------------------------------------------------------------- #
    #  Graph building
    # --------------------------------------------------------------------- #
    matrix = process_votes_to_matrix(anon_votes)
    sg = create_signed_graph(matrix, args.agreement_threshold)

    edges = [(u, v, 1) for u, v in sg["G_plus"].edges()] + [
        (u, v, -1) for u, v in sg["G_minus"].edges()
    ]
    graph_name = (
        f"bundestag_signed_graph_periods_{'_'.join(map(str, args.bundestage))}"
    )
    edge_file = save_edge_list(edges, graph_name, args.output_dir)

    print(
        f"Signed edge list (|V|={matrix.shape[0]}, "
        f"+|E|={sg['G_plus'].number_of_edges()}, "
        f"-|E|={sg['G_minus'].number_of_edges()}) saved to {edge_file}"
    )


if __name__ == "__main__":
    main()
