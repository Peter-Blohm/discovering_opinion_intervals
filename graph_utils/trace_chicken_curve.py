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


def plot(name, points, out_path):
    xs = [p[0] for p in points]
    alives = [p[1] for p in points]
    viols = [p[2] for p in points]

    fig, ax_v = plt.subplots(figsize=(8, 5))
    ax_a = ax_v.twinx()

    ax_v.step(xs, viols, where="post", color="crimson", linewidth=2, label="violations")
    ax_a.step(xs, alives, where="post", color="steelblue", linewidth=2,
              linestyle="--", label="alive vertices")

    ax_v.set_xlabel(r"ratio threshold $\alpha$ (smallest ratio removed so far)")
    ax_v.set_ylabel("total violations", color="crimson")
    ax_a.set_ylabel("alive vertices", color="steelblue")
    ax_v.tick_params(axis="y", labelcolor="crimson")
    ax_a.tick_params(axis="y", labelcolor="steelblue")

    ax_v.invert_xaxis()
    ax_v.set_title(f"{name}: chicken-algorithm trajectory")
    ax_v.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
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
        out = os.path.join(OUT_DIR, f"{name.lower()}_chicken_trace.png")
        plot(name, points, out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
