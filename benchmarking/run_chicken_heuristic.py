"""Chicken + GAIA/Venus combined benchmark.

For every (dataset, alpha) pair, run the chicken algorithm with that alpha
(alpha=1.0 means "just kernelise", alpha=0.5 means "peck everything"), then
solve every leftover kernel with the Rust GAIC heuristic (GAIA = no
annealing, Venus = annealed) using the 8-interval structure and
10-chunk config files. Report chicken violations + sum of per-kernel
heuristic violations, plus the total.

CSV columns:
    dataset,alpha,algo,seed,chicken_violations,kernel_count,
    heuristic_violations,total_violations,chicken_ms,heuristic_ms
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graph_utils"))

from convert_to_json import write_signed_graph_to_json
from signed_graph import read_signed_graph
from signed_graph_kernelization import chicken_algorithm

DATASETS = {
    "Bitcoin":      "bitcoinotc.txt",
    "Chess":        "chess.txt",
    "WikiElec":     "elec.txt",
    "Bundestag":    "bundestag.txt",
    "Slashdot":     "slashdot.txt",
    "Epinions":     "epinions.txt",
    "WikiSigned":   "wikisigned-k2.txt",
    "WikiConflict": "wikiconflict.txt",
}

ALPHAS_DEFAULT = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ALGOS_DEFAULT = ["gaia", "venus"]

DATA_DIR = os.path.join(ROOT, "Datasets")
RUST_EXE = os.path.join(ROOT, "heuristics", "target", "release", "heuristics")
STRUCT = os.path.join(ROOT, "benchmarking", "structs", "intervals8.json")
CONFIG = {
    "gaia":  os.path.join(ROOT, "benchmarking", "configs", "config_gaia_chunks_10.json"),
    "venus": os.path.join(ROOT, "benchmarking", "configs", "config_venus_chunks_10.json"),
}

DEFAULT_OUT = os.path.join(ROOT, "benchmarking", "chicken_heuristic")


def alpha_tag(alpha: float) -> str:
    return f"a{alpha:.2f}".replace(".", "")


def compute_kernels(name, dataset_path, alpha, kernels_dir):
    """Run chicken with given alpha, persist every leftover kernel as JSON.

    Cached: if the kernels for this (name, alpha) already exist on disk we
    skip the (potentially expensive) chicken run and reuse them.
    """
    cache_meta = os.path.join(kernels_dir, "meta.json")
    if os.path.isfile(cache_meta):
        with open(cache_meta) as f:
            meta = json.load(f)
        kernel_paths = [os.path.join(kernels_dir, p) for p in meta["kernels"]]
        if all(os.path.isfile(p) for p in kernel_paths):
            return meta["chicken_violations"], kernel_paths, meta["chicken_ms"]

    print(f"  [chicken] {name} alpha={alpha} ...", flush=True)
    graph = read_signed_graph(dataset_path)
    t0 = time.time()
    chicken_violations, leftover = chicken_algorithm(graph, alpha=alpha)
    chicken_ms = int((time.time() - t0) * 1000)
    print(f"  [chicken] -> {chicken_violations} violations, "
          f"{len(leftover)} leftover kernel(s) in {chicken_ms} ms", flush=True)

    os.makedirs(kernels_dir, exist_ok=True)
    kernel_paths = []
    for i, k in enumerate(leftover):
        out = os.path.join(kernels_dir, f"kernel_{i:03d}.json")
        write_signed_graph_to_json(k, out)
        kernel_paths.append(out)

    with open(cache_meta, "w") as f:
        json.dump({
            "chicken_violations": chicken_violations,
            "chicken_ms": chicken_ms,
            "kernels": [os.path.basename(p) for p in kernel_paths],
        }, f, indent=2)
    return chicken_violations, kernel_paths, chicken_ms


def parse_best(log_path):
    """Last log line is: edge_weight, current, best_batch, best, ..."""
    with open(log_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise RuntimeError(f"empty log {log_path}")
    fields = [x.strip() for x in lines[-1].split(",")]
    if len(fields) < 4:
        raise RuntimeError(f"unexpected last line in {log_path}: {lines[-1]!r}")
    return int(fields[3])


def run_heuristic_on_kernel(kernel_path, algo, seed, solution_path, log_path):
    cmd = [
        RUST_EXE, kernel_path, STRUCT, CONFIG[algo], solution_path,
        "gaic", "--seed", str(seed),
    ]
    t0 = time.time()
    with open(log_path, "w") as logf:
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.DEVNULL).returncode
    elapsed = int((time.time() - t0) * 1000)
    if rc != 0:
        raise RuntimeError(f"heuristic failed (rc={rc}) on {kernel_path}")
    return parse_best(log_path), elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    ap.add_argument("--alphas", nargs="*", type=float, default=ALPHAS_DEFAULT)
    ap.add_argument("--algos", nargs="*", default=ALGOS_DEFAULT, choices=["gaia", "venus"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(1, 51)))
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--summary", default=None,
                    help="CSV summary file (default: <out-dir>/summary.csv)")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 1) - 2),
                    help="Concurrent heuristic processes (default: CPUs - 1). "
                         "Each (algo, seed) combo within a (dataset, alpha) "
                         "bucket runs as one worker.")
    args = ap.parse_args()

    if not os.path.isfile(RUST_EXE):
        sys.exit(f"Rust heuristic binary not found at {RUST_EXE}. "
                 "Build it with `cd heuristics && cargo build --release`.")

    summary = args.summary or os.path.join(args.out_dir, "summary.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(summary) or ".", exist_ok=True)

    new_file = True #not os.path.isfile(summary)
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "dataset", "alpha", "algo", "seed",
                "chicken_violations", "kernel_count",
                "heuristic_violations", "total_violations",
                "chicken_ms", "heuristic_ms",
            ])

        csv_lock = threading.Lock()
        print(f"[pool] {args.workers} workers", flush=True)

        def process_combo(name, alpha, kernel_paths, cv, chicken_ms, algo, seed):
            sol_dir = os.path.join(args.out_dir, "solutions",
                                   name.lower(), alpha_tag(alpha),
                                   algo, f"seed{seed}")
            log_dir = os.path.join(args.out_dir, "logs",
                                   name.lower(), alpha_tag(alpha),
                                   algo, f"seed{seed}")
            os.makedirs(sol_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            hv_sum = 0
            ms_sum = 0
            for i, kp in enumerate(kernel_paths):
                sol = os.path.join(sol_dir, f"kernel_{i:03d}.json")
                log = os.path.join(log_dir, f"kernel_{i:03d}.log")
                hv, ms = run_heuristic_on_kernel(kp, algo, seed, sol, log)
                hv_sum += hv
                ms_sum += ms

            total = cv + hv_sum
            with csv_lock:
                w.writerow([
                    name, f"{alpha:.2f}", algo, seed,
                    cv, len(kernel_paths), hv_sum, total,
                    chicken_ms, ms_sum,
                ])
                f.flush()
            print(f"  >> {name} a={alpha} {algo} seed{seed}: "
                  f"chicken={cv} + heur={hv_sum} = {total} ({ms_sum} ms)",
                  flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for name in args.datasets:
                if name not in DATASETS:
                    print(f"[skip] unknown dataset {name}", flush=True)
                    continue
                dpath = os.path.join(DATA_DIR, DATASETS[name])
                if not os.path.isfile(dpath):
                    print(f"[skip] {name}: {dpath} not found", flush=True)
                    continue

                for alpha in args.alphas:
                    kernels_dir = os.path.join(
                        args.out_dir, "kernels", name.lower(), alpha_tag(alpha))
                    cv, kernel_paths, chicken_ms = compute_kernels(
                        name, dpath, alpha, kernels_dir)

                    futures = [
                        pool.submit(process_combo, name, alpha, kernel_paths,
                                    cv, chicken_ms, algo, seed)
                        for algo in args.algos for seed in args.seeds
                    ]
                    for fut in as_completed(futures):
                        fut.result()  # surface exceptions

    print(f"summary written to {summary}", flush=True)


if __name__ == "__main__":
    main()
