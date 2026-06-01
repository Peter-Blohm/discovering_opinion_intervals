"""Trace the chicken-algorithm trajectory on each real-world dataset.

For every dataset, kernelise once and then peck with ``alpha = 0``, recording
a snapshot at every "new low" event of the running minimum ratio. Plot the
resulting curve of (alive vertices, total violations) as a function of the
ratio threshold ``alpha``, with twin y-axes.

The curve at threshold ``alpha = r`` is what you would obtain by running
chicken with that ``alpha``: all paid removals with ratio >= r plus their
free cascades have happened; nothing strictly below has.
"""

# TODO: This code was skimmed but not checked in great detail.

import argparse
import csv as _csv
import os
import sys
import time
from bisect import bisect_left

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _abbrev(x, _pos=None):
    """Format axis ticks as 1.2k / 3.4M / 5G (no trailing zeros)."""
    if x == 0:
        return "0"
    ax = abs(x)
    for divisor, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if ax >= divisor:
            v = x / divisor
            return f"{v:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{x:g}"


_FMT = FuncFormatter(_abbrev)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signed_graph import read_signed_graph
from signed_graph_kernelization import _greedy_peck, kernelise_graph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Datasets")
OUT_DIR = os.path.join(ROOT, "benchmarking", "figures")
PAPER_CSV_DIR = "/home/florian/Development/work/6711021c4eac0070fc0ee13e/MLG@ECML2026/data/chicken_traces"
PAPER_FIG_PATH = "/home/florian/Development/work/6711021c4eac0070fc0ee13e/MLG@ECML2026/figures/chicken_trajectories.pdf"

COL_ALIVE = "#9467bd"   # lila
COL_VIOL = "#e66100"    # orange

DATASETS = {
    "Bitcoin":  "bitcoinotc.txt",
    "Chess":    "chess.txt",
    "WikiElec": "elec.txt",
    "Bundestag": "bundestag.txt",
    "Slashdot": "slashdot.txt",
    "Epinions": "epinions.txt",
    "WikiSigned": "wikisigned-k2.txt",
    "WikiConflict": "wikiconflict.txt",
}


def aggregate(kernel_traces, kernel_initials):
    """Combine per-kernel snapshot lists into a single global curve.

    ``kernel_initials`` is a list of ``(v, pos_edges, neg_edges)`` tuples
    giving the starting state of each kernel before pecking. At global
    threshold ``r`` the kernel's contribution is the snapshot with the
    smallest per-kernel threshold ``>= r``; if ``r`` exceeds every recorded
    threshold for a kernel, that kernel contributes its initial state with
    zero violations.
    """
    thresholds = sorted({s[0] for sub in kernel_traces for s in sub}, reverse=True)
    if not thresholds:
        v = sum(init[0] for init in kernel_initials)
        ep = sum(init[1] for init in kernel_initials)
        em = sum(init[2] for init in kernel_initials)
        return [(1.0, v, 0, ep, em)]

    sorted_subs = []
    for sub in kernel_traces:
        ordered = sorted(sub, key=lambda s: -s[0])
        keys_asc = list(reversed([s[0] for s in ordered]))
        states_asc = list(reversed([(s[1], s[2], s[3], s[4]) for s in ordered]))
        sorted_subs.append((keys_asc, states_asc))

    points = []
    for r in thresholds:
        total_v = total_viol = total_ep = total_em = 0
        for (init, (keys_asc, states_asc)) in zip(kernel_initials, sorted_subs):
            idx = bisect_left(keys_asc, r)
            if idx == len(keys_asc):
                total_v += init[0]
                total_ep += init[1]
                total_em += init[2]
            else:
                v, viol, ep, em = states_asc[idx]
                total_v += v
                total_viol += viol
                total_ep += ep
                total_em += em
        points.append((r, total_v, total_viol, total_ep, total_em))
    return points


def write_csv(points, out_path):
    """Write the trajectory as one row per snapshot (descending alpha).

    Columns: alpha, alive, violations, pos_edges, neg_edges, edges
    """
    with open(out_path, "w") as f:
        f.write("alpha,alive,violations,pos_edges,neg_edges,edges\n")
        for r, v, viol, ep, em in points:
            f.write(f"{r:.10g},{v},{viol},{ep},{em},{ep + em}\n")


def plot(name, points, out_path):
    xs = [p[0] for p in points]
    edges = [p[3] + p[4] for p in points]
    viols = [p[2] for p in points]

    fig, ax_e = plt.subplots(figsize=(8, 5))
    ax_v = ax_e.twinx()

    ax_e.step(xs, edges, where="post", color=COL_ALIVE, linewidth=2,
              label="remaining edges")
    ax_v.step(xs, viols, where="post", color=COL_VIOL, linewidth=2,
              label="violations")

    ax_e.set_xlabel(r"ratio threshold $\alpha$")
    ax_e.set_ylabel("remaining edges", color=COL_ALIVE)
    ax_v.set_ylabel("violations", color=COL_VIOL)
    ax_e.tick_params(axis="y", labelcolor=COL_ALIVE)
    ax_v.tick_params(axis="y", labelcolor=COL_VIOL)

    ax_e.invert_xaxis()
    ax_e.yaxis.set_major_formatter(_FMT)
    ax_v.yaxis.set_major_formatter(_FMT)
    ax_e.set_title(f"{name}")
    ax_e.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_grid(per_dataset, out_path, cols=4, rows=2):
    """Render one combined PNG with every dataset on a 4-col x 2-row grid.

    Each subplot uses twin y-axes (lila = remaining edges, orange =
    violations) so the per-dataset magnitudes stay readable.
    """
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3.6, rows * 2.4),
    )
    # Tight outer margins but enough left/right room for the rotated y-labels
    # plus their tick numbers; modest column/row gaps.

    fig.subplots_adjust(
        left=0.05, right=0.95,
        top=0.93, bottom=0.13,
        wspace=0.35, hspace=0.40,
    )
    axes = axes.reshape(rows, cols)

    for idx, (name, points) in enumerate(per_dataset):
        r, c = divmod(idx, cols)
        ax_e = axes[r][c]
        ax_v = ax_e.twinx()

        xs = [p[0] for p in points]
        edges = [p[3] + p[4] for p in points]
        viols = [p[2] for p in points]

        ax_e.step(xs, edges, where="post", color=COL_ALIVE, linewidth=1.6)
        ax_v.step(xs, viols, where="post", color=COL_VIOL, linewidth=1.6)

        ax_e.set_title(name, fontsize=12)
        ax_e.invert_xaxis()
        ax_e.yaxis.set_major_formatter(_FMT)
        ax_v.yaxis.set_major_formatter(_FMT)
        ax_e.tick_params(axis="y", labelcolor=COL_ALIVE, labelsize=10)
        ax_v.tick_params(axis="y", labelcolor=COL_VIOL, labelsize=10)
        ax_e.tick_params(axis="x", labelsize=10)
        ax_e.grid(True, alpha=0.25)

        if r == rows - 1:
            ax_e.set_xlabel(r"ratio threshold $\alpha$", fontsize=11)
        if c == 0:
            ax_e.set_ylabel("remaining edges", color=COL_ALIVE, fontsize=11,
                            labelpad=8)
        if c == cols - 1:
            ax_v.set_ylabel("violations", color=COL_VIOL, fontsize=11,
                            labelpad=8)

    # Hide unused panels (if any).
    for k in range(len(per_dataset), rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].set_visible(False)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def read_csv_points(path):
    """Reconstruct the ``(alpha, alive, viol, pos_edges, neg_edges)`` tuples
    from a CSV previously written by :func:`write_csv`."""
    pts = []
    with open(path) as f:
        for row in _csv.DictReader(f):
            pts.append((
                float(row["alpha"]),
                int(row["alive"]),
                int(row["violations"]),
                int(row["pos_edges"]),
                int(row["neg_edges"]),
            ))
    return pts


def regenerate_from_csv():
    """Rebuild the combined grid PDF from cached per-dataset CSVs.

    No chicken peck, no graph loading. Use this when iterating on plot
    cosmetics.
    """
    os.makedirs(os.path.dirname(PAPER_FIG_PATH), exist_ok=True)
    grid = []
    for name in DATASETS:
        p = os.path.join(PAPER_CSV_DIR, f"{name.lower()}.csv")
        if not os.path.exists(p):
            print(f"[skip] {name}: cached CSV missing at {p}")
            continue
        grid.append((name, read_csv_points(p)))
    if not grid:
        print("no cached CSVs found; run without --from-csv first")
        return
    plot_grid(grid, PAPER_FIG_PATH)
    print(f"wrote combined grid figure to {PAPER_FIG_PATH}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-csv",
        action="store_true",
        help="Skip the chicken peck and rebuild the grid PDF from the cached "
             "per-dataset CSVs under PAPER_CSV_DIR. Useful when iterating on "
             "matplotlib styling.",
    )
    args = ap.parse_args()
    if args.from_csv:
        regenerate_from_csv()
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PAPER_CSV_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PAPER_FIG_PATH), exist_ok=True)
    grid_points = []
    for name, fname in DATASETS.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue

        print(f"=== {name} ===")
        t0 = time.time()
        graph = read_signed_graph(path)
        print(f"  load: {graph.number_of_nodes()} V / "
              f"{graph.G_plus.number_of_edges()}+{graph.G_minus.number_of_edges()} E "
              f"in {time.time() - t0:.1f}s")

        t0 = time.time()
        kernels = kernelise_graph(graph)
        initials = [(k.number_of_nodes(),
                     k.G_plus.number_of_edges(),
                     k.G_minus.number_of_edges()) for k in kernels]
        print(f"  kernelise: {len(kernels)} kernels, "
              f"{sum(i[0] for i in initials)} V / "
              f"{sum(i[1] for i in initials)}+{sum(i[2] for i in initials)} E "
              f"in {time.time() - t0:.1f}s")

        traces = []
        total_violations = 0
        t0 = time.time()
        for kernel in kernels:
            sub = []
            violations, _ = _greedy_peck(kernel, alpha=0.0, trace=sub)
            traces.append(sub)
            total_violations += violations
        print(f"  peck: {total_violations} violations, "
              f"{sum(len(s) for s in traces)} snapshots in {time.time() - t0:.1f}s")

        points = aggregate(traces, initials)
        png = os.path.join(OUT_DIR, f"{name.lower()}_chicken_trace.png")
        plot(name, points, png)
        csv = os.path.join(PAPER_CSV_DIR, f"{name.lower()}.csv")
        write_csv(points, csv)
        grid_points.append((name, points))
        print(f"  wrote {png} and {csv}")

    if grid_points:
        plot_grid(grid_points, PAPER_FIG_PATH)
        print(f"wrote combined grid figure to {PAPER_FIG_PATH}")


if __name__ == "__main__":
    main()
