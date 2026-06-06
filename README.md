# Code for "On the Best Interval Approximation Problem"

This repository contains the implementation and evaluation code for the paper **"On the Best Interval Approximation Problem"**.

The paper builds on our previous work, **"Discovering Opinion Intervals from Conflicts in Signed Graphs"** (NeurIPS 2025). New code on this branch:

- **`graph_utils/signed_graph_kernelization.py`** — the four exact pruning rules and the relaxed-pruning heuristic.
- **`graph_utils/evaluate_kernelization.py`** — measures how much the exact pruning rules reduce each real-world dataset (paper RQ1, Table 1).
- **`graph_utils/trace_chicken_curve.py`** — traces the relaxed-pruning trajectory (remaining size vs. introduced violations) as a function of the approximation threshold `alpha` (paper RQ2, relaxed-pruning trajectory figure).
- **`benchmarking/run_chicken_heuristic.py`** — runs relaxed pruning at varying `alpha` and then solves the residual graphs with the `GAIA`/`VENUS` heuristics, recording violations and runtimes (paper RQ3).

## Installation Instructions

### Rust Installation

The pruning rules and relaxed-pruning heuristic are pure Python and need no
compilation. Rust is only required for the downstream `GAIA`/`VENUS` solvers
(used in RQ3). To install Rust, follow the instructions on
[https://rustup.rs/](https://rustup.rs/).

### Conda Environment Setup 

Our utility functions, including the bundestag graph mining code and code for graph conversion and creation, are implemented in Python.  
To manage the dependencies, we use Conda. For setting up the Conda environment, please follow these steps:

1. Ensure that [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed on your system.
2. Navigate to the project directory where the `environment.yml` file is located.
3. Run the following command to create the Conda environment:

```bash
conda env create -f environment.yml
```

### Gurobi Installation

In `graph_utils/solve_embedding.py`, we include a Mixed-Integer Programming (MIP) formulation for the problem.
This MIP program can solve the problem optimally for small graphs.
To run the code, gurobi needs to be installed.  

The python package `gurobipy` is included in the python environment.
However, to run the code, you also need a Gurobi license. (A free academic license can be obtained on [https://www.gurobi.com/](https://www.gurobi.com/).)
To set up the license locally, follow the steps on this website:  

[https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer).

## Repository Structure

The repository is organized as follows:

- **`heuristics/`**: Contains the Rust implementation of the GAIA/VENUS heuristics (from our previous work), used as the downstream solver.
  - `src/`: Source code for the heuristics
  - `data/`: Sample input files for algorithm configurations and interval structures
- **`data/`**: Contains the bundestag dataset and is used as a working directory for other datasets.
- **`graph_utils/`**: Python utilities for working with signed graphs, including:
  - `convert_to_json.py`: Script to convert signed graphs from general txt format to JSON format required by the Rust implementation of our algorithms
  - `generate_synthetic_interval_graph.py`: Script to generate synthetic signed graphs from a given interval structure 
  - `signed_graph.py`: Class for signed graphs
  - `solve_embedding.py`: MIP formulation for optimal solutions on small graphs (requires Gurobi)
  - `signed_graph_kernelization.py`: The pruning rules (exact *kernelization* via `kernelise_graph`) and the relaxed-pruning heuristic (`chicken_algorithm`) for BIA
  - `chicken_algorithm.md`: Write-up of the relaxed-pruning ("chicken") algorithm and its incremental implementation
  - `evaluate_kernelization.py`: Measures the size reduction of the exact pruning rules on each dataset (RQ1)
  - `trace_chicken_curve.py`: Traces the relaxed-pruning size–violation trajectory over the approximation threshold `alpha` (RQ2)
- **`bundestag/`**: Contains code and files related to the scraping and generation of the 'Bundestag' dataset
  - `scrape.py`: Script to scrape the voting data from the Bundestag website
  - `generate_bundestag_graph.py`: Script to generate the signed graph from the voting data obtained by `scrape.py` based on co-voting behavior
- **`benchmarking`**: Contains code for benchmarking the heuristics
  - `run_benchmark.sh`: Bash script to benchmark all heuristics on all interval structures on all datasets for multiple seeds
  - `summary.csv`: Summary of the benchmark results - Each line represents the final output of a single run of a specific heuristic configuration on a single dataset with a single seed and a single interval structure
  - `configs/`: Contains the configuration files for the heuristics
  - `structs/`: Contains the interval structures used for benchmarking
  - `run_chicken_heuristic.py`: Runs relaxed pruning at varying thresholds `alpha`, then solves the residual graphs with `GAIA`/`VENUS` (RQ3)
  - `make_chicken_heuristic_table.py`: Renders the LaTeX results table from the RQ3 benchmark summary
  - `plot_chicken_heuristic_runtime.py`: Plots total relaxed-pruning + solver runtime per dataset against `alpha` (RQ3)

**Bundestag Data Copyright Notice**

We note that the voting data as official works are not subject to copyright.
The legal notice of the Bundestag website is available here: [https://web.archive.org/web/20250422230733/https://www.bundestag.de/impressum](https://web.archive.org/web/20250422230733/https://www.bundestag.de/impressum)


## Usage

The pruning rules and the relaxed-pruning heuristic live in
`graph_utils/signed_graph_kernelization.py` and apply to any `SignedGraph`. Run
the snippets below with `graph_utils/` on your Python path (e.g. `PYTHONPATH=graph_utils`).

**Exact pruning** — `kernelise_graph` decomposes the instance into independent kernels without losing optimality:

```python
from signed_graph import read_signed_graph
from signed_graph_kernelization import kernelise_graph, chicken_algorithm

graph = read_signed_graph("Datasets/bitcoinotc.txt")
kernels = kernelise_graph(graph, rule_iii=True, rule_iv=True)  # rules i, ii on by default
```

**Relaxed pruning** — `chicken_algorithm(graph, alpha)` greedily removes sign-imbalanced vertices, returning the violation count and any residual kernels. `alpha=1.0` is exact pruning (no violations); `alpha=0.5` peels the graph down to a full heuristic solution; values in between trade size reduction for violations:

```python
violations, residual_kernels = chicken_algorithm(graph, alpha=0.8)
```

**Downstream solver** — residual kernels can be solved with the Rust `GAIA`/`VENUS` heuristics from our previous work. Build once with `cd heuristics && cargo build --release`, then:

```bash
./heuristics/target/release/heuristics <instance.json> <intervals.json> <config.json> <output.json> gaic --seed <seed>
```

e.g. `... data/bundestag_signed.json benchmarking/structs/intervals8.json benchmarking/configs/config_venus_chunks_10.json data/out.json gaic --seed 42`. The full "relaxed pruning → GAIA/VENUS" pipeline is automated by `benchmarking/run_chicken_heuristic.py` (see [Reproducing Paper Results](#reproducing-paper-results)).

### Utilities

**Convert Graph to JSON:**  
The Rust solvers consume signed graphs in JSON. To convert a graph from the txt
format used by the [SNAP](https://snap.stanford.edu/data/) and
[KONECT](https://konect.cc/networks/) repositories:

```bash
python graph_utils/convert_to_json.py --type <graph_type> --data <input_file> --output <output_base_name>
```

Where `<graph_type>` is `weighted` (to preserve weighted edges) or `signed` (otherwise).

**Generate Synthetic Graphs:**  
To generate synthetic signed graphs from a given interval structure:

```bash
python graph_utils/generate_synthetic_interval_graph.py --intervals_file <interval_structure_file> --output_dir <output_directory>
```

Further parameters can be set in the script itself.

## Reproducing Paper Results

**(Prepare Datasets:)**  
Our novel bundestag dataset is available in the `data/` directory.
Further instances used in the paper can be dowloaded from the from [SNAP](https://snap.stanford.edu/data/) or [KONECT](https://konect.cc/networks/) network repositories.
The downloaded graph files can then converted to be used by our heuristic algorithms via `graph_utils/convert_to_json.py`.

**RQ1 — Size reduction from the exact pruning rules (Table 1):**

```bash
python graph_utils/evaluate_kernelization.py --csv kernelization.csv --tex table.tex
```

**RQ2 — Relaxed-pruning trajectory (size vs. violations over `alpha`):**

```bash
python graph_utils/trace_chicken_curve.py
```

**RQ3 — Relaxed pruning combined with the `GAIA`/`VENUS` solvers (results table and runtime figure):**

```bash
python benchmarking/run_chicken_heuristic.py

python benchmarking/make_chicken_heuristic_table.py -o table_rp.tex
python benchmarking/plot_chicken_heuristic_runtime.py
```