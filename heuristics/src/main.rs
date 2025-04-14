use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use serde::Deserialize;
use std::{env, fs, path::Path};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use rand::Rng;  

#[derive(Deserialize, Debug)]
struct SignedEdge {
    source: usize,
    target: usize,
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

struct DynamicGraph {
    vertices: Vec<HashMap<usize, i32>>,
}

impl DynamicGraph {
    fn new(n: usize) -> Self {
        let vertices = vec![HashMap::new(); n];
        DynamicGraph { vertices }
    }

    fn edge_exists(&self, a: usize, b: usize) -> bool {
        self.vertices[a].contains_key(&b)
    }

    fn get_adjacent_vertices(&self, v: usize) -> &HashMap<usize, i32> {
        &self.vertices[v]
    }

    fn get_edge_weight(&self, a: usize, b: usize) -> i32 {
        *self.vertices[a].get(&b).unwrap_or(&0)
    }

    fn remove_vertex(&mut self, v: usize) {
        let neighbors: Vec<usize> = self.vertices[v].keys().cloned().collect();
        for &u in &neighbors {
            self.vertices[u].remove(&v);
        }
        self.vertices[v].clear();
    }

    fn update_edge_weight(&mut self, a: usize, b: usize, w: i32) {
        *self.vertices[a].entry(b).or_insert(0) += w;
        *self.vertices[b].entry(a).or_insert(0) += w;
    }
}

/// Represents an edge in the dynamic graph used specifically for the 
/// Greedy Additive Edge Contraction (GAEC) algorithm. This structure
/// is designed to store information about an edge, including its 
/// endpoints, weight, and edition number, which is used to track 
/// updates to the edge during the contraction process.
///
/// # Fields
/// - `a`: The first vertex of the edge.
/// - `b`: The second vertex of the edge.
/// - `edition`: The edition number of the edge, which tracks how many 
///   times the edge has been updated during the algorithm's execution.
/// - `weight`: The weight of the edge, which determines its priority 
///   in the contraction process.
///
/// This type implements `Ord` and `PartialOrd` to allow edges to be 
/// prioritized in a max-heap (via `BinaryHeap`), where edges with 
/// higher weights are processed first.
#[derive(Debug, Eq, PartialEq)]
struct DynamicEdge {
    a: usize,
    b: usize,
    edition: usize,
    weight: i32,
}

impl Ord for DynamicEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight.cmp(&other.weight)
    }
}

impl PartialOrd for DynamicEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A Union-Find (Disjoint Set Union) data structure optimized for efficient set merging and connectivity queries.
/// Supports path compression (for fast `find`) and union by rank (to keep trees balanced).
struct Partition {
    /// Parent pointers for each element. Each element points to its parent in the set hierarchy.
    /// The root of a set points to itself (e.g., `parents[i] == i` for roots).
    parents: Vec<usize>,
    /// Rank (approximate depth) of the tree rooted at each element. Used during `merge` to keep trees shallow.
    ranks: Vec<usize>,
    /// Current number of disjoint sets (clusters) in the partition.
    num_sets: usize,
}

impl Partition {
    /// Creates a new Partition with `size` elements, each in its own set.
    ///
    /// # Arguments
    /// * `size` - Number of elements in the initial partition.
    fn new(size: usize) -> Self {
        Partition {
            parents: (0..size).collect(), // Each element is its own parent initially
            ranks: vec![0; size], // All ranks start at 0
            num_sets: size, // Each element is its own set
        }
    }

    /// Finds the root of the set containing `x`, with **path compression**.
    /// Path compression flattens the tree by linking nodes directly to the root during traversal.
    ///
    /// # Arguments
    /// * `x` - The element to find the root for.
    ///
    /// # Returns
    /// The root (representative) of the set containing `x`.
    fn find(&mut self, x: usize) -> usize {
        if self.parents[x] != x {
            // Path compression: make parent of `x` point directly to the root
            self.parents[x] = self.find(self.parents[x]);
        }
        self.parents[x]
    }

    /// Merges the sets containing `x` and `y` using **union by rank**.
    /// If `x` and `y` are already in the same set, this does nothing.
    /// Otherwise, the tree with smaller rank is attached to the root of the tree with larger rank.
    ///
    /// # Arguments
    /// * `x`, `y` - Elements whose sets will be merged.
    fn union(&mut self, x: usize, y: usize) {
        let x_root = self.find(x);
        let y_root = self.find(y);

        if x_root != y_root {
            // Union by rank: attach shorter tree to the root of the taller tree
            if self.ranks[x_root] < self.ranks[y_root] {
                self.parents[x_root] = y_root;
            } else {
                self.parents[y_root] = x_root;
                // If ranks are equal, increment the rank of the new root
                if self.ranks[x_root] == self.ranks[y_root] {
                    self.ranks[x_root] += 1;
                }
            }
            self.num_sets -= 1; // Decrease the number of disjoint sets
        }
    }

    /// Returns the current number of disjoint sets (clusters) in the partition.
    fn num_sets(&self) -> usize {
        self.num_sets
    }
}

fn greedy_additive_edge_contraction(
    num_vertices: usize,
    edges: &[(usize, usize, i32)],
    target_clusters: usize,
) -> Vec<usize> {
    let mut original_graph = DynamicGraph::new(num_vertices);
    let mut edge_editions = vec![HashMap::new(); num_vertices];
    let mut queue = BinaryHeap::new();

    for &(a, b, weight) in edges {
        original_graph.update_edge_weight(a, b, weight);

        let (a_sorted, b_sorted) = if a <= b { (a, b) } else { (b, a) };

        *edge_editions[a].entry(b).or_insert(0) += 1;
        let edition = edge_editions[b].entry(a).or_insert(0);
        *edition += 1;

        queue.push(DynamicEdge {
            a: a_sorted,
            b: b_sorted,
            edition: *edition,
            weight,
        });
    }

    let mut partition = Partition::new(num_vertices);
    let mut current_clusters = num_vertices;

    while let Some(edge) = queue.pop() {
        let current_edition = edge_editions[edge.a].get(&edge.b).copied().unwrap_or(0);
        if !original_graph.edge_exists(edge.a, edge.b) || edge.edition < current_edition {
            continue;
        }

        // println!("Processing edge: ({}, {}) with weight {} and edition {}", edge.a, edge.b, edge.weight, edge.edition);
        if (target_clusters > 0 && current_clusters <= target_clusters && edge.weight < 0)
            || (target_clusters == 0 && edge.weight <= 0)
        {
            break;
        }

        let mut stable_vertex = edge.a;
        let mut merge_vertex = edge.b;

        let stable_degree = original_graph.get_adjacent_vertices(stable_vertex).len();
        let merge_degree = original_graph.get_adjacent_vertices(merge_vertex).len();

        if stable_degree < merge_degree {
            std::mem::swap(&mut stable_vertex, &mut merge_vertex);
        }

        partition.union(stable_vertex, merge_vertex);
        current_clusters -= 1;

        let neighbors: Vec<usize> = original_graph
            .get_adjacent_vertices(merge_vertex)
            .keys()
            .cloned()
            .filter(|&n| n != stable_vertex)
            .collect();

        for neighbor in neighbors {
            let weight = original_graph.get_edge_weight(merge_vertex, neighbor);
            original_graph.update_edge_weight(stable_vertex, neighbor, weight);

            let (a, b) = if stable_vertex <= neighbor {
                (stable_vertex, neighbor)
            } else {
                (neighbor, stable_vertex)
            };

            let new_weight = original_graph.get_edge_weight(a, b);

            *edge_editions[b].entry(a).or_insert(0) += 1;

            let edition = edge_editions[a].entry(b).or_insert(0);
            *edition += 1;

            queue.push(DynamicEdge {
                a,
                b,
                edition: *edition,
                weight: new_weight,
            });
        }

        original_graph.remove_vertex(merge_vertex);
    }

    let mut node_labels = Vec::with_capacity(num_vertices);
    for node in 0..num_vertices {
        node_labels.push(partition.find(node));
    }

    node_labels
}

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
        // println!("{} ", cluster);
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

/// Compute the number of violations based on node labels and the signed graph
///
/// A violation occurs when:
/// - A negative edge connects nodes in the same cluster
/// - A positive edge connects nodes in different clusters
///
/// # Arguments
/// * `graph` - A SignedGraph
/// * `node_labels` - The cluster assignment for each node
///
/// # Returns
/// The number of violations
fn cc_compute_violations(graph: &SignedGraph, node_labels: &[usize]) -> usize {
    let mut violations = 0;

    for edge in &graph.edges {
        let same_cluster = node_labels[edge.source] == node_labels[edge.target];
        
        if (edge.weight < 0 && same_cluster) || (edge.weight > 0 && !same_cluster) {
            violations += 1;
        }
    }

    violations
}

// For local search I supposed ideally recompute violations only by checking neighbors

fn cc_local_search(graph: &SignedGraph, node_labels: &[usize]) -> Vec<usize> {
    let num_vertices = node_labels.len();

    let mut original_graph = DynamicGraph::new(num_vertices);

    for edge in &graph.edges {
        original_graph.update_edge_weight(edge.source, edge.target, edge.weight);
    }

    let t_start = Instant::now();

    let mut best_labels = node_labels.to_vec();
    let mut best_violations = cc_compute_violations(graph, &best_labels);
    
    println!("Starting local search with {} violations", best_violations);
    
    // Find set of unique clusters
    let mut unique_clusters: Vec<usize> = best_labels.iter().cloned().collect();
    unique_clusters.sort();
    unique_clusters.dedup();
    
    println!("Found {} unique clusters", unique_clusters.len());
    
    let mut rng = rand::rng();
    let mut improved = true;
    
    // Create a random permutation of node indices to avoid bias
    let node_indices: Vec<usize> = (0..best_labels.len()).collect();

    // Add iteration tracking variables
    let mut iteration_count = 0;
    let mut last_report_time = Instant::now();
    let mut last_iteration_count = 0;
    let report_interval = std::time::Duration::from_secs(1);

    while improved {
        improved = false;
        
        let mut best_move_node_idx = None;
        let mut best_move_cluster = None;
        let mut best_move_violations = usize::MAX;
        let mut best_move_counter = 0;
        
        for &node_idx in &node_indices {
            let current_cluster = best_labels[node_idx];
            for &cluster in &unique_clusters {
                if cluster == current_cluster {
                    continue; // Skip current assignment
                }
                
                // Temporarily reassign the node
                best_labels[node_idx] = cluster;
                
                // Calculate new violation count
                // Compute violations delta by only checking adjacent vertices
                let mut new_violations_delta: i32 = 0;
                let old_cluster = current_cluster;
                let new_cluster = cluster;
                
                // Check all neighbors of node_idx
                for (&neighbor, &weight) in original_graph.get_adjacent_vertices(node_idx) {
                    let neighbor_cluster = best_labels[neighbor];
                    
                    // Old contribution
                    if (weight < 0 && old_cluster == neighbor_cluster) || 
                       (weight > 0 && old_cluster != neighbor_cluster) {
                        new_violations_delta -= 1; // Remove old violation
                    }
                    
                    // New contribution
                    if (weight < 0 && new_cluster == neighbor_cluster) || 
                       (weight > 0 && new_cluster != neighbor_cluster) {
                        new_violations_delta += 1; // Add new violation
                    }
                }
                
                let new_violations = (best_violations as i32 + new_violations_delta) as usize;
                
                // Reservoir sampling with size 1
                if new_violations < best_move_violations {
                    best_move_violations = new_violations;
                    best_move_cluster = Some(cluster);
                    best_move_node_idx = Some(node_idx);
                    best_move_counter = 1;
                } else if new_violations == best_move_violations {
                    best_move_counter += 1;
                    let prob = 1.0 / (best_move_counter as f64);
                    if rng.random_bool(prob) {
                        best_move_cluster = Some(cluster);
                        best_move_node_idx = Some(node_idx);
                    }
                }
                best_labels[node_idx] = current_cluster;
            }
        }

        if best_move_violations < best_violations {
            best_labels[best_move_node_idx.unwrap()] = best_move_cluster.unwrap();
            best_violations = best_move_violations;
            improved = true;
        }

        // Increment iteration counter for each node-cluster evaluation
        iteration_count += 1;
        
        // Report iterations per second at regular intervals
        let now = Instant::now();
        if now.duration_since(last_report_time) >= report_interval {
            let elapsed = now.duration_since(last_report_time).as_secs_f64();
            let iterations_in_interval = iteration_count - last_iteration_count;
            let iterations_per_second = iterations_in_interval as f64 / elapsed;
            println!("Iterations per second: {:.2} (total: {})", iterations_per_second, iteration_count);
            
            last_report_time = now;
            last_iteration_count = iteration_count;
        }
    }
    
    let elapsed = t_start.elapsed();
    let total_iterations_per_second = iteration_count as f64 / elapsed.as_secs_f64();
    
    println!("Local search completed in {:.2?}", elapsed);
    println!("Total iterations: {}", iteration_count);
    println!("Average iterations per second: {:.2}", total_iterations_per_second);
    
    best_labels
}