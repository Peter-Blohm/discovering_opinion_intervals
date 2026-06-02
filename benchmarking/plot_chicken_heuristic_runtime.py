"""Plot average total runtime (chicken + heuristic) per (dataset, alpha).

Reads ``benchmarking/chicken_heuristic/summary.csv`` and renders one subplot
per dataset (4x2 grid). Each subplot has two lines, GAIA and Venus, of mean
``chicken_ms + heuristic_ms`` over all seeds, plus a tinted +/- one-stdev
band, as a function of the chicken threshold ``alpha`` (inverted x-axis,
matches the trace figure). y-axis is linear and per-subplot so each
dataset's scale is its own.
"""

import csv
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "benchmarking", "chicken_heuristic", "summary.csv")
PAPER_FIG_PATH = (
    "/home/florian/Development/work/6711021c4eac0070fc0ee13e/"
    "MLG@ECML2026/figures/chicken_heuristic_runtime.pdf"
)

DATASETS_ORDER = [
    "Bitcoin", "Chess", "WikiElec", "Bundestag",
    "Slashdot", "Epinions", "WikiSigned", "WikiConflict",
]

# Matches the existing user-defined palette.
COL_GAIA = "#05afb0"   # hellblau
COL_VENUS = "#d20000"  # rot


def _fmt_ms(x, _pos=None):
    """Format milliseconds as ms / s / m. Always carries a unit suffix."""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 60_000:
        return f"{x / 60_000:.1f}".rstrip("0").rstrip(".") + "m"
    if ax >= 1_000:
        return f"{x / 1_000:.1f}".rstrip("0").rstrip(".") + "s"
    return f"{x:.0f}ms"


_FMT = FuncFormatter(_fmt_ms)


def _stats(samples):
    if not samples:
        return None, 0.0
    if len(samples) == 1:
        return float(samples[0]), 0.0
    return statistics.mean(samples), statistics.stdev(samples)


def load(csv_path):
    """(dataset, alpha, algo) -> list of total_ms across seeds."""
    totals = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            d = row["dataset"]
            a = round(float(row["alpha"]), 2)
            algo = row["algo"]
            cm = int(row["chicken_ms"])
            hm = int(row["heuristic_ms"])
            totals[(d, a, algo)].append(cm + hm)
    return totals


def plot_grid(totals, out_path, cols=4, rows=2):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.4))
    fig.subplots_adjust(
        left=0.06, right=0.96,
        top=0.88, bottom=0.13,
        wspace=0.20, hspace=0.45,
    )
    axes = axes.reshape(rows, cols)

    series_specs = (("gaia", COL_GAIA, "GAIA"), ("venus", COL_VENUS, "Venus"))

    for idx, name in enumerate(DATASETS_ORDER):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        for algo, color, label in series_specs:
            alphas = sorted(
                {a for (d, a, al) in totals if d == name and al == algo},
                reverse=True,
            )
            xs, means, lo, hi = [], [], [], []
            for a in alphas:
                samples = totals.get((name, a, algo), [])
                m, s = _stats(samples)
                if m is None:
                    continue
                xs.append(a)
                means.append(m)
                lo.append(max(0.0, m - s))
                hi.append(m + s)
            if xs:
                ax.fill_between(xs, lo, hi, color=color, alpha=0.18,
                                linewidth=0)
                ax.plot(xs, means, marker="o", linewidth=1.7, markersize=4,
                        color=color, label=label)

        ax.set_title(name, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.invert_xaxis()
        ax.yaxis.set_major_formatter(_FMT)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(True, alpha=0.25)

        if r == rows - 1:
            ax.set_xlabel(r"ratio threshold $\alpha$", fontsize=11)
        if c == 0:
            ax.set_ylabel("avg total runtime", fontsize=11, labelpad=8)

    # Hide unused panels.
    for k in range(len(DATASETS_ORDER), rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=12)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    out_path = sys.argv[2] if len(sys.argv) > 2 else PAPER_FIG_PATH
    totals = load(csv_path)
    if not totals:
        sys.exit(f"no rows in {csv_path}")
    plot_grid(totals, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
