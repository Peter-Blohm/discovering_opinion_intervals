use std::{env, fs, path::Path};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use serde_json::json;

mod data_types;
mod algorithms;

use data_types::{SignedGraph, IntervalStructure, Instance};
use algorithms::{greedy_additive_edge_contraction, cc_compute_violations, cc_local_search};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        panic!("Usage: {} <graph_file> <interval_file> <target_clusters> <output_file>", args[0]);
    }

    let graph_filename = Path::new(&args[1]);
    let graph_json_data = fs::read_to_string(graph_filename)
        .unwrap_or_else(|_| panic!("Failed to read file: {}", graph_filename.display()));

    let graph: SignedGraph = serde_json::from_str(&graph_json_data)
        .unwrap_or_else(|_| panic!("Failed to parse JSON from {}", graph_filename.display()));


    let interval_filename = Path::new(&args[2]);
    let interval_json_data = fs::read_to_string(interval_filename)
        .unwrap_or_else(|_| panic!("Failed to read file: {}", interval_filename.display()));

    let interval_structure: IntervalStructure = serde_json::from_str(&interval_json_data)
        .unwrap_or_else(|_| panic!("Failed to parse interval JSON from {}", interval_filename.display()));

    // let instance = Instance {
    //     graph,
    //     interval_structure,
    // };

    let edges: Vec<_> = graph.edges.iter()
        .map(|e| (e.source as usize, e.target as usize, e.weight))
        .collect();

    let max_vertex = edges.iter()
        .flat_map(|&(a, b, _)| [a, b])
        .max()
        .unwrap_or(0);
    let num_vertices = max_vertex + 1;

    let target_clusters = args[3].parse::<usize>().expect("Invalid target_clusters");

    let start_time = Instant::now();

    let node_labels = greedy_additive_edge_contraction(num_vertices, &edges, target_clusters);

    let elapsed = start_time.elapsed();
    println!("Running time: {:.2?}", elapsed);

    let mut result = serde_json::Map::new();
    for (node_id, &cluster) in node_labels.iter().enumerate() {
        result.insert(node_id.to_string(), json!(cluster));
    }

    let output_filename = &args[4];
    let json_output = serde_json::to_string_pretty(&result).expect("Failed to serialize JSON");

    let mut file = File::create(output_filename).expect("Failed to create output file");
    file.write_all(json_output.as_bytes()).expect("Failed to write to file");

    let violations = cc_compute_violations(&graph, &node_labels);
    println!("Violations: {}", violations);

    let local_search_node_labels = cc_local_search(&graph, &node_labels);

    let local_search_violations = cc_compute_violations(&graph, &local_search_node_labels);
    println!("Local search violations: {}", local_search_violations);
}