"""Evaluate the kernelization rules on the real-world datasets.

Runs :func:`kernelise_graph` on each of the signed-graph datasets used in the
NeurIPS paper and reports, for every dataset, how much the kernelization rules
reduce the instance: the number of kernels produced, the total size of all
kernels combined, and the size of the single largest kernel (the part that
actually still needs to be solved).

Usage::

    python evaluate_kernelization.py                 # all datasets
    python evaluate_kernelization.py bitcoinotc chess # a subset
    python evaluate_kernelization.py --csv out.csv    # also write a CSV table
    python evaluate_kernelization.py --from-csv out.csv --tex table.tex  # table only

"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signed_graph import SignedGraph, read_signed_graph
from signed_graph_kernelization import kernelise_graph

# Dataset file -> display name used in the paper.
DATASETS = {
    "bitcoinotc": "Bitcoin",
    "chess": "Chess",
    "elec": "WikiElec",
    "bundestag": "Bundestag",
    "slashdot": "Slashdot",
    "epinions": "Epinions",
    "wikisigned-k2": "WikiSigned",
    "wikiconflict": "WikiConflict",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Datasets")

# Display name -> LaTeX name macro defined in the paper's commands.tex.
LATEX_NAMES = {
    "Bitcoin": r"\Bitcoin",
    "Chess": r"\Chess",
    "WikiElec": r"\WikiElec",
    "Bundestag": r"\Bundestag",
    "Slashdot": r"\Slashdot",
    "Epinions": r"\Epinions",
    "WikiSigned": r"\WikiSigned",
    "WikiConflict": r"\WikiConflict",
}


def _stats(graph: SignedGraph) -> dict:
    return {
        "V": graph.number_of_nodes(),
        "PE": graph.G_plus.number_of_edges(),
        "NE": graph.G_minus.number_of_edges(),
    }


def evaluate(key: str) -> dict:
    """Load one dataset, kernelize it, and return a row of summary statistics."""
    path = os.path.join(DATA_DIR, f"{key}.txt")
    graph = read_signed_graph(path)
    original = _stats(graph)

    start = time.time()
    kernels = kernelise_graph(graph)
    runtime = time.time() - start

    kernel_V = sum(k.number_of_nodes() for k in kernels)
    kernel_PE = sum(k.G_plus.number_of_edges() for k in kernels)
    kernel_NE = sum(k.G_minus.number_of_edges() for k in kernels)
    largest = max(kernels, key=lambda k: k.number_of_nodes()) if kernels else None
    largest_stats = _stats(largest) if largest is not None else {"V": 0, "PE": 0, "NE": 0}

    vertex_reduction = 1 - kernel_V / original["V"] if original["V"] else 0.0
    edge_reduction = 1 - (kernel_PE + kernel_NE) / (original["PE"] + original["NE"]) if (original["PE"] + original["NE"]) else 0.0

    return {
        "dataset": DATASETS.get(key, key),
        "orig_V": original["V"],
        "orig_PE": original["PE"],
        "orig_NE": original["NE"],
        "num_kernels": len(kernels),
        "kernel_V": kernel_V,
        "kernel_PE": kernel_PE,
        "kernel_NE": kernel_NE,
        "largest_V": largest_stats["V"],
        "largest_PE": largest_stats["PE"],
        "largest_NE": largest_stats["NE"],
        "vertex_reduction": vertex_reduction,
        "edge_reduction": edge_reduction,
        "runtime_s": runtime,
    }


# Cumulative rule sets for the per-rule breakdown: each entry enables one more
# rule than the previous one.
CUMULATIVE_RULES = [
    ("i", dict(rule_i=True, rule_ii=False, rule_iii=False, rule_iv=False)),
    ("+ii", dict(rule_i=True, rule_ii=True, rule_iii=False, rule_iv=False)),
    ("+iii", dict(rule_i=True, rule_ii=True, rule_iii=True, rule_iv=False)),
    ("+iv", dict(rule_i=True, rule_ii=True, rule_iii=True, rule_iv=True)),
]


def evaluate_per_rule(key: str) -> dict:
    """Report the remaining total kernel size as each rule is added in turn.

    Note: the splitting rules (iii, iv) duplicate the separator vertices/edge
    endpoints across the parts they create, so the summed vertex count can grow
    slightly when those rules are enabled even though no edge is ever counted
    twice.
    """
    path = os.path.join(DATA_DIR, f"{key}.txt")
    graph = read_signed_graph(path)
    row = {"dataset": DATASETS.get(key, key), "orig_V": graph.number_of_nodes()}
    for label, flags in CUMULATIVE_RULES:
        kernels = kernelise_graph(graph, **flags)
        row[label] = sum(k.number_of_nodes() for k in kernels)
    return row


def _print_per_rule_table(rows: list[dict]) -> None:
    labels = [label for label, _ in CUMULATIVE_RULES]
    header = f"{'Dataset':<12} {'V':>9} | " + " ".join(f"{l:>9}" for l in labels)
    header += "   (remaining kernel vertices after each rule is added)"
    print(header)
    print("-" * 70)
    for r in rows:
        cells = " ".join(f"{r[l]:>9}" for l in labels)
        print(f"{r['dataset']:<12} {r['orig_V']:>9} | {cells}")


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'Dataset':<12} {'V':>9} {'PE':>9} {'NE':>9} | "
        f"{'#kern':>6} {'kern V':>9} {'kern PE':>9} {'kern NE':>9} | "
        f"{'maxV':>8} {'maxPE':>8} {'maxNE':>8} | {'v. red.':>8} {'e. red.':>8} {'time':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['dataset']:<12} {r['orig_V']:>9} {r['orig_PE']:>9} {r['orig_NE']:>9} | "
            f"{r['num_kernels']:>6} {r['kernel_V']:>9} {r['kernel_PE']:>9} {r['kernel_NE']:>9} | "
            f"{r['largest_V']:>8} {r['largest_PE']:>8} {r['largest_NE']:>8} | "
            f"{r['vertex_reduction'] * 100:>7.1f}% {r['edge_reduction'] * 100:>7.1f}% {r['runtime_s']:>6.1f}s"
        )


def _tex_int(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")

def write_latex_table(rows: list[dict], path: str) -> None:
    header_cells = [
        r"Dataset",
        r"Kernels $|V|$",
        r"Kernels $|E^+|$",
        r"Kernels $|E^-|$",
        r"V. red.",
        r"E. red.",
    ]
    lines = [
        r"\begin{table}[t!bh]",
        r"\centering",
        r"\caption{Effect of the kernelisation rules on the real-world datasets. }",
        r"\label{tab:kernelization}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = [
            LATEX_NAMES.get(r["dataset"], r["dataset"]),
            _tex_int(int(r["kernel_V"])),
            _tex_int(int(r["kernel_PE"])),
            _tex_int(int(r["kernel_NE"])),
            f"{float(r['vertex_reduction']) * 100:.1f}\\%",
            f"{float(r['edge_reduction']) * 100:.1f}\\%",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_latex_per_rule_table(rows: list[dict], path: str) -> None:
    """LaTeX table of the remaining kernel vertices after each rule is added."""
    labels = [label for label, _ in CUMULATIVE_RULES]
    header_cells = ["Dataset", r"$\abs{V}$"] + [f"({l})" for l in labels]
    lines = [
        r"\begin{table}[t!bh]",
        r"\centering",
        r"\caption{Remaining kernel vertices after cumulatively applying each "
        r"kernelisation rule, starting from the original number of vertices $\abs{V}$. "
        r"(i) removes vertices without negative edges, (ii) solves plus-components separately, "
        r"(iii) splits at vertex separators, and (iv) imposes 3-positive-edge-connectivity.}",
        r"\label{tab:kernelization_per_rule}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{l" + "r" * (len(labels) + 1) + "}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = [LATEX_NAMES.get(r["dataset"], r["dataset"]), _tex_int(int(r["orig_V"]))]
        cells += [_tex_int(int(r[l])) for l in labels]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def read_csv_rows(path: str) -> list[dict]:
    """Load previously-saved evaluation rows from a CSV written via ``--csv``."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="dataset keys to evaluate")
    parser.add_argument("--all", action="store_true", help="include the large datasets")
    parser.add_argument("--csv", metavar="FILE", help="write the results to a CSV file")
    parser.add_argument(
        "--tex", metavar="FILE", help="write a booktabs LaTeX table"
    )
    parser.add_argument(
        "--from-csv",
        metavar="FILE",
        help="build the table from a CSV written by a previous --csv run "
        "instead of re-running the experiments",
    )
    parser.add_argument(
        "--per-rule",
        action="store_true",
        help="report the marginal contribution of each rule instead of the summary",
    )
    args = parser.parse_args()

    # Reuse previously-computed results: load the CSV and (re)generate the table
    # without touching the datasets.
    if args.from_csv:
        rows = read_csv_rows(args.from_csv)
        if not args.tex:
            print("[warn] --from-csv only writes a table with --tex", file=sys.stderr)
        else:
            writer = write_latex_per_rule_table if args.per_rule else write_latex_table
            writer(rows, args.tex)
            print(f"Wrote {args.tex}")
        return

    if args.datasets:
        keys = args.datasets
    else:
        keys = list(DATASETS)

    rows = []
    for key in keys:
        if not os.path.exists(os.path.join(DATA_DIR, f"{key}.txt")):
            print(f"[skip] {key}: file not found", file=sys.stderr)
            continue
        print(f"[run ] {key} ...", file=sys.stderr, flush=True)
        rows.append(evaluate_per_rule(key) if args.per_rule else evaluate(key))

    print()
    (_print_per_rule_table if args.per_rule else _print_table)(rows)

    if args.tex and rows:
        writer = write_latex_per_rule_table if args.per_rule else write_latex_table
        writer(rows, args.tex)
        print(f"\nWrote {args.tex}")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
