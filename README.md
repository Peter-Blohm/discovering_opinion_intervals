# Code for the paper "Discovering Opinion Intervals from Conflicts in Signed Graphs"

This repository contains the implementation and evaluation code for the paper "Discovering Opinion Intervals from Conflicts in Signed Graphs".

## Installation Instructions

### Rust Installation

Our main heuristic are implemented in the Rust programming language.  
To install Rust, follow the instructions on [https://rustup.rs/](https://rustup.rs/).

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

- **`heuristics/`**: Contains the Rust implementation of our main heuristics for the opinion interval discovery problem.
  - `src/`: Source code for the heuristics
  - `data/`: Sample input files for algorithm configurations and interval structures
- **`data/`**: Contains the bundestag dataset and is used as a working directory for other datasets.
- **`graph_utils/`**: Python utilities for working with signed graphs, including:
  - `convert_to_json.py`: Script to convert signed graphs from general txt format to JSON format required by the Rust implementation of our algorithms
  - `generate_synthetic_interval_graph.py`: Script to generate synthetic signed graphs from a given interval structure 
  - `signed_graph.py`: Class for signed graphs
  - `solve_embedding.py`: MIP formulation for optimal solutions on small graphs (requires Gurobi)
  - `signed_graph_kernelization.py`: Tools for kernelization of signed graphs w.r.t. the opinion interval discovery problem
- **`bundestag/`**: Contains code and files related to the scraping and generation of the 'Bundestag' dataset
  - `all_votes.csv`: File containing all voting data in the German Bundestag
  - `scrape.py`: Script to scrape the voting data from the Bundestag website
  - `generate_bundestag_graph.py`: Script to generate the signed graph from the voting data based on co-voting behavior
- **`benchmarking`**: Contains code for benchmarking the heuristics
  - `run_benchmark.sh`: Bash script to benchmark all heuristics on all interval structures on all datasets for multiple seeds
  - `summary.csv`: Summary of the benchmark results - Each line represents the final output of a single run of a specific heuristic configuration on a single dataset with a single seed and a single interval structure
  - `configs/`: Contains the configuration files for the heuristics
  - `structs/`: Contains the interval structures used for benchmarking
  
## Usage

### Heuristic Algorithms

To run the heuristic algorithms, the Rust code needs to be compiled first.
To compile the Rust code, navigate to the `heuristics/` directory and run the following command:

```bash
cargo build --release
```

This will create an executable file in the `target/release/` directory.
To run the heuristic algorithms, the following command can be used from the root directory of the repository:

```bash
./heuristics/target/release/heuristics <instance_file> <interval_structure_file> <config_file> <output_file> gaic --seed <seed>
```

Where:
- `<instance_file>`: Path to the input file containing the signed graph
- `<interval_structure_file>`: Path to the file containing the interval structure
- `<config_file>`: Path to the configuration file for the heuristic
- `<output_file>`: Path to the output file where the results will be saved

For example, to run the heuristic on the bundestag dataset with a 8 consequtively overlapping intervals and a simulated annealing configuration, you can use the following command:

```bash
./heuristics/target/release/heuristics data/bundestag_signed.json benchmarking/structs/intervals8.json benchmarking/configs/config_venus_chunks_10.json data/bundestag_signed_solution.json gaic --seed 42
```

### Utilities

**Convert Graph to Json:**  
To convert a signed graph from the general txt format available on the [SNAP](https://snap.stanford.edu/data/) and [KONECT](https://konect.cc/networks/) websites to the JSON format required by the Rust implementation, you can use the following command:

```bash
python graph_utils/convert_to_json.py --type <graph_type> --data <input_file> --output <output_base_name>
```

Where `<graph_type>` is the type of the graph (If the graph contains weighted edges that should be preserved, use `weighted` as the type. Otherwise, use `signed`).

**Generate Synthetic Graphs:**  
To generate synthetic signed graphs from a given interval structure, you can use the following command:

```bash
python graph_utils/generate_synthetic_interval_graph.py --intervals_file <interval_structure_file> --output_dir <output_directory>
```

Further parameters can be set in the script itself.

## Reproducing Paper Results

**(Prepare Datasets:)**  
Our novel bundestag dataset is available in the `data/` directory.
Further instances used in the paper can be dowloaded from the from [SNAP](https://snap.stanford.edu/data/) or [KONECT](https://konect.cc/networks/) network repositories.
The downloaded graph files can then converted to be used by our heuristic algorithms via `graph_utils/convert_to_json.py`.

**Run Benchmarking:**  
To run the benchmarking of all heuristics on all datasets and interval structures, you can use the provided bash script from the root directory of the repository:

```bash
./benchmarking/run_benchmark.sh > benchmarking/summary2.csv &
```