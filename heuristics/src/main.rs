use std::{env, fs, path::Path};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use serde_json::json;

mod data_types;
mod algorithms;

use data_types::{SignedGraph, IntervalStructure, SignedEdge};
use algorithms::{greedy_additive_edge_contraction, cc_compute_violations, cc_local_search, brute_force_interval_structure};

// TODO: Handle case where graph is not connected and few clusters are desired

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        panic!("Usage: {} <graph_file> <interval_file> <output_file>", args[0]);
    }

    let run_cc_local_search = false;


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

    let edges: Vec<SignedEdge> = graph.edges.iter()
        .map(|edge| {
            let source = edge.source;
            let target = edge.target;
            let weight = edge.weight;
            SignedEdge { source, target, weight }
        })
        .collect();

    let vertices: std::collections::HashSet<usize> = edges.iter()
        .flat_map(|edge| [edge.source, edge.target])
        .collect();
    let num_vertices = vertices.len();

    // Remap vertices to a contiguous range
    let mut vertex_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut remapped_edges = Vec::new();
    for (new_id, &old_id) in vertices.iter().enumerate() {
        vertex_map.insert(old_id, new_id);
    }
    for edge in &edges {
        let new_a = *vertex_map.get(&edge.source).unwrap();
        let new_b = *vertex_map.get(&edge.target).unwrap();
        remapped_edges.push(SignedEdge { source: new_a, target: new_b, weight: edge.weight });
    }
    let edges = remapped_edges;


    let target_clusters = interval_structure.intervals.len();
    println!("Target clusters from interval structure: {}", target_clusters);

    let start_time = Instant::now();

    // GAEC
    let node_labels = greedy_additive_edge_contraction(num_vertices, &edges, target_clusters);

    // Postprocessing
    let elapsed = start_time.elapsed();
    println!("Running time: {:.2?}", elapsed);

    let mut result = serde_json::Map::new();

    // Remap from internal indices back to original node IDs
    let mut reverse_vertex_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (&orig_id, &new_id) in &vertex_map {
        reverse_vertex_map.insert(new_id, orig_id);
    }
    
    for (internal_id, &cluster) in node_labels.iter().enumerate() {
        let node_id = reverse_vertex_map.get(&internal_id).unwrap_or(&internal_id);
        result.insert(node_id.to_string(), json!(cluster));
    }

    let output_filename = &args[3];
    let json_output = serde_json::to_string_pretty(&result).expect("Failed to serialize JSON");

    let mut file = File::create(output_filename).expect("Failed to create output file");
    file.write_all(json_output.as_bytes()).expect("Failed to write to file");

    // Count violations
    let violations = cc_compute_violations(&edges, &node_labels);
    println!("Violations: {}", violations);


    // Find set of unique clusters
    let mut unique_clusters: Vec<usize> = node_labels.iter().cloned().collect();
    unique_clusters.sort();
    unique_clusters.dedup();
    
    println!("Found {} unique clusters", unique_clusters.len());

    // Brute force interval structure assignment
    let (_, interval_violations) = brute_force_interval_structure(&edges, &node_labels, &interval_structure);

    println!("Interval violations: {}", interval_violations);
    
    if run_cc_local_search {
        // Local search
        let local_search_node_labels = cc_local_search(&edges, &node_labels);
    
        // Count violations after local search
        let local_search_violations = cc_compute_violations(&edges, &local_search_node_labels);
        println!("Local search violations: {}", local_search_violations);

    }
}