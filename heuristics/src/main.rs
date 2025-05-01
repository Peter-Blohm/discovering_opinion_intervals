use std::{env, fs, path::Path};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use serde_json::json;

mod data_types;
mod algorithms;
mod gaic;

use data_types::{SignedGraph, IntervalStructure};
use algorithms::{greedy_additive_edge_contraction, cc_compute_violations, cc_local_search, brute_force_interval_structure};
use gaic::{greedy_absolute_interval_contraction};
use crate::data_types::UsefulSignedGraph;
use crate::gaic::GaicConfig;
// TODO: Handle case where graph is not connected and few clusters are desired

fn main() {
    //parse cmd args
    let (graph,
        interval_structure,
        config,
        output_filename,
        algorithm,
        run_cc_local_search,
        seed) = parse_args();


    let signed_graph = UsefulSignedGraph::new(&graph);
    let num_clusters = interval_structure.intervals.len();


    if algorithm == "gaic" {

        let node_labels = greedy_absolute_interval_contraction(&signed_graph, &interval_structure, &config, seed as u64);
        let mut label_count:Vec<usize> = vec![0;interval_structure.intervals.len()];
        for &label in &node_labels {
            label_count[label] +=1;
        }
        let mut result = serde_json::Map::new();

        for (internal_id, &cluster) in node_labels.iter().enumerate() {
            let node_id = signed_graph.vertex_id(internal_id);
            result.insert(node_id.to_string(), json!(cluster));
        }
        let json_output = serde_json::to_string_pretty(&result).expect("Failed to serialize JSON");
        let mut file = File::create(output_filename).expect("Failed to create output file");
        file.write_all(json_output.as_bytes()).expect("Failed to write to file");

    } else if algorithm == "gaec" {
        let start_time = Instant::now();

        // GAEC
        let node_labels = greedy_additive_edge_contraction(signed_graph.num_vertices, &signed_graph.edges, num_clusters);

        // Postprocessing
        let elapsed = start_time.elapsed();
        println!("Running time: {:.2?}", elapsed);

        let mut result = serde_json::Map::new();

        for (internal_id, &cluster) in node_labels.iter().enumerate() {
            let node_id = signed_graph.vertex_id(internal_id);
            result.insert(node_id.to_string(), json!(cluster));
        }
        let json_output = serde_json::to_string_pretty(&result).expect("Failed to serialize JSON");
        let mut file = File::create(output_filename).expect("Failed to create output file");
        file.write_all(json_output.as_bytes()).expect("Failed to write to file");

        // Count violations
        let violations = cc_compute_violations(&signed_graph.edges, &node_labels);
        println!("Violations: {}", violations);

        // Find set of unique clusters
        let mut unique_clusters: Vec<usize> = node_labels.iter().cloned().collect();
        unique_clusters.sort();
        unique_clusters.dedup();
        
        println!("Found {} unique clusters", unique_clusters.len());

        // Brute force interval structure assignment
        let (_, interval_violations) = brute_force_interval_structure(&signed_graph.edges, &node_labels, &interval_structure);
        println!("Interval violations: {}", interval_violations);
        
        if run_cc_local_search {
            // Local search
            let local_search_node_labels = cc_local_search(&signed_graph.edges, &node_labels);
        
            // Count violations after local search
            let local_search_violations = cc_compute_violations(&signed_graph.edges, &local_search_node_labels);
            println!("Local search violations: {}", local_search_violations);
        }
    }
    else {
        println!("Unknown algorithm: {}. Use 'gaec' or 'gaic'.", algorithm);
        std::process::exit(1);
    }
}


fn parse_args() -> (SignedGraph, IntervalStructure, GaicConfig, String, String, bool, usize) {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        println!("Usage: {} <graph_file> <interval_file> <config_file> <output_file> <algorithm> [opts]", args[0]);
        println!("Algorithms: gaec (Greedy Additive Edge Contraction), gaic (Greedy Absolute Interval Contraction)");
        println!("Options for gaic: --seed <num> (default: 5)");
        std::process::exit(1);
    }

    let graph_filename = Path::new(&args[1]);
    let graph_json_data = fs::read_to_string(graph_filename)
        .unwrap_or_else(|_| panic!("Failed to read file: {}", graph_filename.display()));
    let graph: SignedGraph = serde_json::from_str(&graph_json_data)
        .unwrap_or_else(|_| panic!("Failed to parse graph JSON from {}", graph_filename.display()));

    let interval_filename = Path::new(&args[2]);
    let interval_json_data = fs::read_to_string(interval_filename)
        .unwrap_or_else(|_| panic!("Failed to read file: {}", interval_filename.display()));
    let interval_structure: IntervalStructure = serde_json::from_str(&interval_json_data)
        .unwrap_or_else(|_| panic!("Failed to parse interval JSON from {}", interval_filename.display()));

    let config_filename = Path::new(&args[3]);
    let config_json_data = fs::read_to_string(config_filename)
        .unwrap_or_else(|_| panic!("Failed to read file: {}", config_filename.display()));
    let config_structure: GaicConfig = serde_json::from_str(&config_json_data)
        .unwrap_or_else(|_| panic!("Failed to parse config JSON from {}", config_filename.display()));


    let output_filename = args[4].clone();
    let algorithm = args[5].to_lowercase();

    let run_cc_local_search = false; // Could also be made a command line argument

    let runs = args.iter().position(|arg| arg == "--seed")
        .and_then(|pos| args.get(pos + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5);
    (graph, interval_structure,config_structure, output_filename, algorithm, run_cc_local_search, runs)
}