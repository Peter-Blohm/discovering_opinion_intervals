# Code for the paper "Discovering Opinion Intervals from Conflicts in Signed Graphs"

This repository contains the implementation and evaluation code for the paper "Discovering Opinion Intervals from Conflicts in Signed Graphs".

## Installation Instructions

### Install Rust

Our main heuristic are implemented in the Rust programming language.  
To install Rust, follow the instructions on [https://rustup.rs/](https://rustup.rs/).

### Setup Conda Environment

Our utility functions, including the bundestag graph mining code and code for graph conversion and creation, are implemented in Python.  
To manage the dependencies, we use Conda. For setting up the Conda environment, please follow these steps:

1. Ensure that [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed on your system.
2. Navigate to the project directory where the `environment.yml` file is located.
3. Run the following command to create the Conda environment:

```bash
conda env create -f environment.yml
```

### Install Gurobi

In `graph_utils/solve_embedding.py`, we include a Mixed-Integer Programming (MIP) formulation for the problem.
This MIP program can solve the problem optimally for small graphs.
To run the code, gurobi needs to be installed.  

The python package `gurobipy` is included in the python environment.
However, to run the code, you also need a Gurobi license. (A free academic license can be obtained on [https://www.gurobi.com/](https://www.gurobi.com/).)
To set up the license locally, follow the steps on this website:  

[https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer).


## Repository Structure

## Usage