"""Render a LaTeX table from ``chicken_heuristic/summary.csv``.

Reads the per-seed summary written by ``run_chicken_heuristic.py`` and emits
a booktabs-style ``tabular`` whose rows are datasets and whose columns are
the (alpha, algo) pairs. The cell value is the minimum ``total_violations``
observed across all seeds for that (dataset, alpha, algo) combination. The
row-wise minimum is bolded.

Usage:
    python benchmarking/make_chicken_heuristic_table.py            # print to stdout
    python benchmarking/make_chicken_heuristic_table.py -o file.tex
"""

import argparse
import csv
import os
import sys

DATASETS_ORDER = [
    "Bitcoin", "Chess", "WikiElec", "Bundestag",
    "Slashdot", "Epinions", "WikiSigned", "WikiConflict",
]
DATASET_TEX = {
    "Bitcoin":      r"\Bitcoin",
    "Chess":        r"\Chess",
    "WikiElec":     r"\WikiElec",
    "Bundestag":    r"\Bundestag",
    "Slashdot":     r"\Slashdot",
    "Epinions":     r"\Epinions",
    "WikiSigned":   r"\WikiSigned",
    "WikiConflict": r"\WikiConflict",
}
ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ALGOS = ["gaia", "venus"]

# Original-graph total edge counts (|E^+| + |E^-|), used to render cell values
# as percentages of the input size rather than absolute violations.
TOTAL_EDGES = {
    "Bitcoin":      18281 + 3153,
    "Chess":        19046 + 13604,
    "WikiElec":     78440 + 21915,
    "Bundestag":    320956 + 76541,
    "Slashdot":     380933 + 117599,
    "Epinions":     589888 + 118619,
    "WikiSigned":   628000 + 84337,
    "WikiConflict": 762999 + 1251054,
}

# benchmarking/summary.csv encodes the dataset via the *.json filename it
# was solved on; map back to the display name.
INSTANCE_TO_DATASET = {
    "bitcoinotc_signed.json":    "Bitcoin",
    "chess_signed.json":         "Chess",
    "elec_signed.json":          "WikiElec",
    "bundestag_signed.json":     "Bundestag",
    "slashdot_signed.json":      "Slashdot",
    "epinions_signed.json":      "Epinions",
    "wikisigned-k2_signed.json": "WikiSigned",
    "wikiconflict_signed.json":  "WikiConflict",
}
CONFIG_TO_ALGO = {
    "config_gaia_chunks_10.json":  "gaia",
    "config_venus_chunks_10.json": "venus",
}
BASELINE_STRUCT = "intervals8.json"
BASELINE_SEED_COUNT = 50

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "benchmarking", "chicken_heuristic", "summary.csv")
DEFAULT_BASELINE_CSV = os.path.join(ROOT, "benchmarking", "summary.csv")


def fmt_pct(n, total):
    """Render ``n / total`` as a percentage with two decimals."""
    if n is None or total is None or total == 0:
        return "--"
    return f"{100 * n / total:.2f}"


def collect_min(csv_path):
    """(dataset, alpha, algo) -> min(total_violations) across seeds.

    Also returns the count of seeds per key for sanity reporting.
    """
    best = {}
    counts = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            d = row["dataset"]
            a = round(float(row["alpha"]), 2)
            algo = row["algo"]
            tv = int(row["total_violations"])
            key = (d, a, algo)
            if key not in best or tv < best[key]:
                best[key] = tv
            counts[key] = counts.get(key, 0) + 1
    return best, counts


def collect_baseline(csv_path):
    """(dataset, algo) -> min(best) across seeds, restricted to intervals8 +
    the 10-chunk GAIA/Venus configs in benchmarking/summary.csv.

    Note: ``run_benchmark.sh`` writes the rows as
    ``instance, CONFIG, STRUCT, seed, ...`` but the file header is labelled
    ``instance, struct, config, seed, ...`` (swapped). We follow the data
    layout, not the misleading header.
    """
    best = {}
    counts = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # discard mislabeled header
        # Column order in the actual rows: instance, config, struct, seed,
        # edge_weight, current, best_batch, best, ...
        IDX_INSTANCE, IDX_CONFIG, IDX_STRUCT, IDX_BEST = 0, 1, 2, 7
        for row in reader:
            if not row:
                continue
            instance = row[IDX_INSTANCE].strip()
            config = row[IDX_CONFIG].strip()
            struct = row[IDX_STRUCT].strip()
            if struct != BASELINE_STRUCT:
                continue
            if instance not in INSTANCE_TO_DATASET:
                continue
            if config not in CONFIG_TO_ALGO:
                continue
            d = INSTANCE_TO_DATASET[instance]
            algo = CONFIG_TO_ALGO[config]
            try:
                value = int(row[IDX_BEST].strip())
            except (ValueError, IndexError):
                continue
            key = (d, algo)
            if key not in best or value < best[key]:
                best[key] = value
            counts[key] = counts.get(key, 0) + 1
    return best, counts


def render(best, baseline, datasets):
    n_alphas = len(ALPHAS)
    n_blocks = 1 + n_alphas  # baseline + per-alpha
    lines = []
    lines.append(r"\begin{table}[t!bh]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Best fraction of violated edges per dataset, taken as the "
        r"minimum over 50 seeds and expressed as a percentage of the original "
        r"$|E^+| + |E^-|$. The \emph{no kernel.} column applies GAIA/Venus "
        r"directly to the input graph; the $\alpha$ columns apply the chicken "
        r"algorithm with that threshold first and then solve the residual "
        r"kernels with GAIA/Venus. \emph{G} = GAIA (no annealing), \emph{V} = "
        r"Venus (annealed); all runs use the 8-interval structure with "
        r"10 reassignment chunks. \textbf{Bold}: best per row.}")
    lines.append(r"\label{tab:chicken-heuristic}")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    col_spec = "l" + "rr" * n_blocks
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header = ["Dataset", r"\multicolumn{2}{c}{no kernel.}"] + [
        rf"\multicolumn{{2}}{{c}}{{$\alpha\!=\!{a:.1f}$}}" for a in ALPHAS
    ]
    lines.append(" & ".join(header) + r" \\")
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2*i + 2}-{2*i + 3}}}" for i in range(n_blocks)
    )
    lines.append(cmidrules)
    lines.append(" & " + " & ".join(["G & V"] * n_blocks) + r" \\")
    lines.append(r"\midrule")

    for d in datasets:
        total = TOTAL_EDGES.get(d)
        row_vals = [baseline.get((d, algo)) for algo in ALGOS]
        row_vals += [
            best.get((d, round(a, 2), algo))
            for a in ALPHAS for algo in ALGOS
        ]
        if not any(v is not None for v in row_vals):
            continue
        row_min = min(v for v in row_vals if v is not None)
        cells = [DATASET_TEX.get(d, d)]
        for v in row_vals:
            rendered = fmt_pct(v, total)
            if v is not None and v == row_min:
                cells.append(rf"\textbf{{{rendered}}}")
            else:
                cells.append(rendered)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help=f"Chicken+heuristic summary CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--baseline-csv", default=DEFAULT_BASELINE_CSV,
                    help=f"Vanilla heuristic summary CSV "
                         f"(default: {DEFAULT_BASELINE_CSV})")
    ap.add_argument("-o", "--output", default="-",
                    help="Output .tex file (default: stdout)")
    args = ap.parse_args()

    best, counts = collect_min(args.csv)
    baseline, baseline_counts = collect_baseline(args.baseline_csv)

    # Coverage / sanity reports to stderr.
    expected = {(d, round(a, 2), algo)
                for d in DATASETS_ORDER for a in ALPHAS for algo in ALGOS}
    missing = expected - best.keys()
    if missing:
        print(f"[warn] {len(missing)} (dataset, alpha, algo) combo(s) missing "
              "from the chicken+heuristic CSV; rendering '--' for those.",
              file=sys.stderr)
    seed_counts = sorted(set(counts.values()))
    print(f"[info] chicken+heuristic seed counts per combo: {seed_counts}",
          file=sys.stderr)

    expected_baseline = {(d, algo) for d in DATASETS_ORDER for algo in ALGOS}
    missing_b = expected_baseline - baseline.keys()
    if missing_b:
        print(f"[warn] {len(missing_b)} baseline (dataset, algo) combo(s) "
              "missing from the vanilla CSV; rendering '--' for those.",
              file=sys.stderr)
    baseline_seed_counts = sorted(set(baseline_counts.values()))
    print(f"[info] baseline seed counts per combo: {baseline_seed_counts}",
          file=sys.stderr)
    if baseline_seed_counts and baseline_seed_counts != [BASELINE_SEED_COUNT]:
        print(f"[warn] expected exactly {BASELINE_SEED_COUNT} baseline seeds "
              f"per combo, got {baseline_seed_counts}.", file=sys.stderr)

    table = render(best, baseline, DATASETS_ORDER)
    if args.output == "-":
        sys.stdout.write(table)
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(table)
        print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
