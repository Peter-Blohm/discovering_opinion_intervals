use std::{env, fs, path::Path};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use serde_json::json;

mod data_types;
mod algorithms;
mod gaic;

use data_types::{SignedGraph, IntervalStructure};
use algorithms::{greedy_additive_edge_contraction, cc_compute_violations, cc_local_search, brute_force_interval_structure, compute_interval_violations};
use gaic::{greedy_absolute_interval_contraction};
use crate::algorithms::compute_satisfied_bad_cycles;
use crate::data_types::UsefulSignedGraph;
// TODO: Handle case where graph is not connected and few clusters are desired

fn main() {
    //parse cmd args
    let (graph,
        mut interval_structure,
        output_filename,
        algorithm,
        run_cc_local_search,
        runs) = parse_args();


    let signed_graph = UsefulSignedGraph::new(&graph);
    let num_clusters = interval_structure.intervals.len();
    println!("Target clusters from interval structure: {}", num_clusters);

    // let f: fn()
    // TODO: switch case for function call

    if algorithm == "gaic" {

        println!("GAIC runs: {}", runs);

        let start_time = Instant::now();
        // let node_labels = greedy_absolute_interval_contraction(signed_graph.num_vertices, &signed_graph.edges, &interval_structure, 100, runs);
        let node_labels = greedy_absolute_interval_contraction(&signed_graph, &interval_structure, 100, runs);
        let elapsed = start_time.elapsed();
        println!("Running time: {:.2?}", elapsed);
        let mut label_count:Vec<usize> = vec![0;interval_structure.intervals.len()];
        for &label in &node_labels {
            label_count[label] +=1;
        }
        println!("Label count {:?}", label_count);
        
       


        // Count interval violations
        println!("Counting violations for interval structure assignment...");

        // Create a dummy cluster to interval map (1:1 mapping)
        let mut cluster_to_interval_map = std::collections::HashMap::new();
        for i in 0..num_clusters {
            cluster_to_interval_map.insert(i, i);
        }

        // Use the imported function to compute violations
        let violation_count = compute_interval_violations(
            &signed_graph.edges,
            &node_labels,
            &interval_structure,
            &cluster_to_interval_map
        );

        println!("Interval violations: {}", violation_count);
        let triangles = compute_satisfied_bad_cycles(
            &signed_graph.edges,
            &node_labels,
            &interval_structure,
            3
        );
        println!("Triangle inequality violations: {} out of {}", triangles, &signed_graph.edges.len());
        let mut result = serde_json::Map::new();

        for (internal_id, &cluster) in node_labels.iter().enumerate() {
            let node_id = signed_graph.vertex_id(internal_id);
            result.insert(node_id.to_string(), json!(cluster));
        }
        let json_output = serde_json::to_string_pretty(&result).expect("Failed to serialize JSON");
        let mut file = File::create(output_filename).expect("Failed to create output file");
        file.write_all(json_output.as_bytes()).expect("Failed to write to file");

    } else if algorithm == "gaic++" {

        println!("GAIC++ runs: {}", runs);

        let start_time = Instant::now();
        let mut interval_structure = IntervalStructure::full_structure(4);
        // let node_labels = greedy_absolute_interval_contraction(signed_graph.num_vertices, &signed_graph.edges, &interval_structure, 100, runs);
        let node_labels = greedy_absolute_interval_contraction(&signed_graph, &interval_structure, 100, 100);
        let mut label_count:Vec<usize> = vec![0;interval_structure.intervals.len()];
        for &label in &node_labels {
            label_count[label] +=1;
        }
        println!("Label count {:?}", label_count);
        let elapsed = start_time.elapsed();
        println!("Running time: {:.2?}", elapsed);

        let mut min_interval = 0;
        let mut num_min_interval = label_count[0];
        for (idx, interval) in label_count.into_iter().enumerate() {
            if interval < num_min_interval {
                min_interval = idx;
                num_min_interval = interval;
            }
        }
        interval_structure.delete_interval(min_interval);
        let start_time = Instant::now();
        // let node_labels = greedy_absolute_interval_contraction(signed_graph.num_vertices, &signed_graph.edges, &interval_structure, 100, runs);
        let node_labels = greedy_absolute_interval_contraction(&signed_graph, &interval_structure, 100, 100);
        let elapsed = start_time.elapsed();
        println!("Running time: {:.2?}", elapsed);
        let mut label_count:Vec<usize> = vec![0;interval_structure.intervals.len()];
        for &label in &node_labels {
            label_count[label] +=1;
        }
        println!("Label count {:?}", label_count);
        let mut result = serde_json::Map::new();

        for (internal_id, &cluster) in node_labels.iter().enumerate() {
            let node_id = signed_graph.vertex_id(internal_id);
            result.insert(node_id.to_string(), json!(cluster));
        }
        let json_output = serde_json::to_string_pretty(&result).expect("Failed to serialize JSON");
        let mut file = File::create(output_filename).expect("Failed to create output file");
        file.write_all(json_output.as_bytes()).expect("Failed to write to file");

    }
    else if algorithm == "gaec" {
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


fn parse_args() -> (SignedGraph, IntervalStructure, String, String, bool, usize) {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        println!("Usage: {} <graph_file> <interval_file> <output_file> <algorithm> [opts]", args[0]);
        println!("Algorithms: gaec (Greedy Additive Edge Contraction), gaic (Greedy Absolute Interval Contraction)");
        println!("Options for gaic: --runs <num> (default: 5)");
        std::process::exit(1);
    }

    let graph_filename = Path::new(&args[1]);
    let graph_json_data = fs::read_to_string(graph_filename)
        .unwrap_or_else(|_| panic!("Failed to read file: {}", graph_filename.display()));

    let interval_filename = Path::new(&args[2]);
    let interval_json_data = fs::read_to_string(interval_filename)
        .unwrap_or_else(|_| panic!("Failed to read file: {}", interval_filename.display()));
    let graph: SignedGraph = serde_json::from_str(&graph_json_data)
        .unwrap_or_else(|_| panic!("Failed to parse JSON from {}", graph_filename.display()));

    let interval_structure: IntervalStructure = serde_json::from_str(&interval_json_data)
        .unwrap_or_else(|_| panic!("Failed to parse interval JSON from {}", interval_filename.display()));

    let output_filename = args[3].clone();
    let algorithm = args[4].to_lowercase();

    let run_cc_local_search = false; // Could also be made a command line argument

    let runs = args.iter().position(|arg| arg == "--runs")
        .and_then(|pos| args.get(pos + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5);
    (graph, interval_structure, output_filename, algorithm, run_cc_local_search, runs)
}