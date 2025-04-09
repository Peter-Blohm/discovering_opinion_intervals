use serde::Deserialize;
use std::{env, fs, path::Path};

#[derive(Deserialize, Debug)]
struct SignedEdge {
    source: u32,
    target: u32,
    weight: i32,
}

#[derive(Deserialize, Debug)]
struct SignedGraph {
    edges: Vec<SignedEdge>,
}

#[derive(Deserialize, Debug)]
struct Interval {
    start: f32,
    end: f32,
}

#[derive(Deserialize, Debug)]
struct IntervalStructure {
    intervals: Vec<Interval>,
}

#[derive(Debug)]
struct Instance {
    graph: SignedGraph,
    interval_structure: IntervalStructure,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        panic!("Usage: {} <graph_file> <interval_file>", args[0]);
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

    let instance = Instance {
        graph,
        interval_structure,
    };

    println!("First few edges:");
    for (i, edge) in instance.graph.edges.iter().take(5).enumerate() {
        println!("Edge {}: source: {}, target: {}, weight: {}", 
            i + 1, edge.source, edge.target, edge.weight);
    }

    println!("\nLast few edges:");
    let edges = &instance.graph.edges;
    let start_index = edges.len().saturating_sub(5);
    for (i, edge) in edges.iter().enumerate().skip(start_index) {
        println!("Edge {}: source: {}, target: {}, weight: {}", 
            i + 1, edge.source, edge.target, edge.weight);
    }

    println!("\nInterval structure:");
    for (i, interval) in instance.interval_structure.intervals.iter().enumerate() {
        println!("Interval {}: {:.1}-{:.1}", 
            i + 1, interval.start, interval.end);
    }

}