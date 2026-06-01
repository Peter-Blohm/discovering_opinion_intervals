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

import os
import sys
import time
from bisect import bisect_left

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signed_graph import read_signed_graph
from signed_graph_kernelization import _greedy_peck, kernelise_graph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Datasets")
OUT_DIR = os.path.join(ROOT, "benchmarking", "figures")
PAPER_CSV_DIR = "/home/florian/Development/work/6711021c4eac0070fc0ee13e/MLG@ECML2026/data/chicken_traces"
PAPER_FIG_PATH = "/home/florian/Development/work/6711021c4eac0070fc0ee13e/MLG@ECML2026/figures/chicken_trajectories.png"

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


def aggregate(kernel_traces, kernel_initial_sizes):
    """Combine per-kernel snapshot lists into a single global curve.

    For each kernel ``k`` with snapshot list sorted by descending threshold
    ``r_{k,0} >= r_{k,1} >= ...``, the kernel's state at global threshold
    ``r`` is the snapshot with the smallest ``r_{k,i} >= r`` (or the kernel's
    full initial size with zero violations if ``r > r_{k,0}``). The global
    state at ``r`` is the sum of per-kernel states.
    """
    thresholds = sorted({s[0] for sub in kernel_traces for s in sub}, reverse=True)
    if not thresholds:
        return [(1.0, sum(kernel_initial_sizes), 0)]

    sorted_subs = []
    for sub in kernel_traces:
        ordered = sorted(sub, key=lambda s: -s[0])
        keys_desc = [s[0] for s in ordered]
        keys_asc = list(reversed(keys_desc))
        states_asc = list(reversed([(s[1], s[2]) for s in ordered]))
        sorted_subs.append((keys_asc, states_asc))

    points = []
    for r in thresholds:
        total_alive = 0
        total_viol = 0
        for (initial, (keys_asc, states_asc)) in zip(kernel_initial_sizes, sorted_subs):
            idx = bisect_left(keys_asc, r)
            if idx == len(keys_asc):
                total_alive += initial
            else:
                alive, viol = states_asc[idx]
                total_alive += alive
                total_viol += viol
        points.append((r, total_alive, total_viol))
    return points


def write_csv(points, out_path):
    """Write the trajectory as ``alpha,alive,violations`` (one row per
    snapshot, descending alpha). pgfplots can read this directly."""
    with open(out_path, "w") as f:
        f.write("alpha,alive,violations\n")
        for r, alive, viol in points:
            f.write(f"{r:.10g},{alive},{viol}\n")


def plot(name, points, out_path):
    xs = [p[0] for p in points]
    alives = [p[1] for p in points]
    viols = [p[2] for p in points]

    fig, ax_a = plt.subplots(figsize=(8, 5))
    ax_v = ax_a.twinx()

    line_a, = ax_a.step(xs, alives, where="post", color=COL_ALIVE, linewidth=2,
                        label="remaining vertices")
    line_v, = ax_v.step(xs, viols, where="post", color=COL_VIOL, linewidth=2,
                        label="violations")

    ax_a.set_xlabel(r"ratio threshold $\alpha$")
    ax_a.set_ylabel("remaining vertices", color=COL_ALIVE)
    ax_v.set_ylabel("violations", color=COL_VIOL)
    ax_a.tick_params(axis="y", labelcolor=COL_ALIVE)
    ax_v.tick_params(axis="y", labelcolor=COL_VIOL)

    ax_a.invert_xaxis()
    ax_a.set_title(f"{name}")
    ax_a.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_grid(per_dataset, out_path, cols=4, rows=2):
    """Render one combined PNG with every dataset on a 4-col x 2-row grid.

    Each subplot uses twin y-axes (lila = remaining vertices, orange =
    violations) so the per-dataset magnitudes stay readable.
    """
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3.4, rows * 2.4), constrained_layout=True
    )
    axes = axes.reshape(rows, cols)

    for idx, (name, points) in enumerate(per_dataset):
        r, c = divmod(idx, cols)
        ax_a = axes[r][c]
        ax_v = ax_a.twinx()

        xs = [p[0] for p in points]
        alives = [p[1] for p in points]
        viols = [p[2] for p in points]

        ax_a.step(xs, alives, where="post", color=COL_ALIVE, linewidth=1.6)
        ax_v.step(xs, viols, where="post", color=COL_VIOL, linewidth=1.6)

        ax_a.set_title(name, fontsize=10)
        ax_a.invert_xaxis()
        ax_a.tick_params(axis="y", labelcolor=COL_ALIVE, labelsize=7)
        ax_v.tick_params(axis="y", labelcolor=COL_VIOL, labelsize=7)
        ax_a.tick_params(axis="x", labelsize=7)
        ax_a.grid(True, alpha=0.25)

        if r == rows - 1:
            ax_a.set_xlabel(r"ratio threshold $\alpha$", fontsize=8)
        if c == 0:
            ax_a.set_ylabel("vertices", color=COL_ALIVE, fontsize=8)
        if c == cols - 1:
            ax_v.set_ylabel("violations", color=COL_VIOL, fontsize=8)

    # Hide unused panels (if any).
    for k in range(len(per_dataset), rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].set_visible(False)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
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
        sizes = [k.number_of_nodes() for k in kernels]
        print(f"  kernelise: {len(kernels)} kernels, {sum(sizes)} V total "
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

        points = aggregate(traces, sizes)
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
