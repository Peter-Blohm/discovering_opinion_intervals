use std::collections::{BinaryHeap, HashMap};
use std::time::Instant;
use rand::Rng;

use crate::data_types::{DynamicEdge, DynamicGraph, SignedGraph, Partition};

pub fn greedy_additive_edge_contraction(
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
pub fn cc_compute_violations(graph: &SignedGraph, node_labels: &[usize]) -> usize {
    let mut violations = 0;

    for edge in &graph.edges {
        let same_cluster = node_labels[edge.source] == node_labels[edge.target];
        
        if (edge.weight < 0 && same_cluster) || (edge.weight > 0 && !same_cluster) {
            violations += 1;
        }
    }

    violations
}

pub fn cc_local_search(graph: &SignedGraph, node_labels: &[usize]) -> Vec<usize> {
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