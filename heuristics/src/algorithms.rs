use std::collections::{BinaryHeap, HashMap};
use std::time::Instant;
use rand::Rng;

use crate::data_types::{DynamicEdge, DynamicGraph, Partition, Interval, IntervalStructure, SignedEdge};

pub fn greedy_additive_edge_contraction(
    num_vertices: usize,
    edges: &Vec<SignedEdge>,
    target_clusters: usize,
) -> Vec<usize> {
    let mut original_graph = DynamicGraph::new(num_vertices);
    let mut edge_editions = vec![HashMap::new(); num_vertices];
    let mut queue = BinaryHeap::new();

    for edge in edges {
        original_graph.update_edge_weight(edge.source, edge.target, edge.weight);

        let (a_sorted, b_sorted) = if edge.source <= edge.target { (edge.source, edge.target) } else { (edge.target, edge.source) };

        *edge_editions[edge.source].entry(edge.target).or_insert(0) += 1;
        let edition = edge_editions[edge.target].entry(edge.source).or_insert(0);
        *edition += 1;

        queue.push(DynamicEdge {
            a: a_sorted,
            b: b_sorted,
            edition: *edition,
            weight: edge.weight,
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
/// * `edges` - A list of SignedEdges
/// * `node_labels` - The cluster assignment for each node
///
/// # Returns
/// The number of violations
pub fn cc_compute_violations(edges: &Vec<SignedEdge>, node_labels: &[usize]) -> usize {
    let mut violations = 0;

    for edge in edges {
        let same_cluster = node_labels[edge.source] == node_labels[edge.target];
        
        if (edge.weight < 0 && same_cluster) || (edge.weight > 0 && !same_cluster) {
            violations += 1;
        }
    }

    violations
}

pub fn cc_local_search(edges: &Vec<SignedEdge>, node_labels: &[usize]) -> Vec<usize> {
    let num_vertices = node_labels.len();

    let mut original_graph = DynamicGraph::new(num_vertices);

    for edge in edges {
        original_graph.update_edge_weight(edge.source, edge.target, edge.weight);
    }

    let t_start = Instant::now();

    let mut best_labels = node_labels.to_vec();
    let mut best_violations = cc_compute_violations(edges, &best_labels);
    
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


/// Check if two intervals overlap
fn intervals_overlap(interval1: &Interval, interval2: &Interval) -> bool {
    !(interval1.end <= interval2.start || interval2.end <= interval1.start)
}

/// Compute the number of violations for a specific mapping of clusters to intervals
fn compute_interval_violations(
    edges: &Vec<SignedEdge>,
    node_labels: &[usize],
    interval_structure: &IntervalStructure,
    cluster_to_interval_map: &HashMap<usize, usize>
) -> usize {
    let mut violations = 0;

    for edge in edges {
        let source_cluster = node_labels[edge.source];
        let target_cluster = node_labels[edge.target];
        
        // For edges within the same cluster
        if source_cluster == target_cluster {
            // For negative edges within the same cluster, it's always a violation
            // as they'll be assigned to the same interval
            if edge.weight < 0 {
                violations += 1;
            }
            continue;
        }
        
        let source_interval_idx = *cluster_to_interval_map.get(&source_cluster).unwrap_or(&0);
        let target_interval_idx = *cluster_to_interval_map.get(&target_cluster).unwrap_or(&0);
        
        let source_interval = &interval_structure.intervals[source_interval_idx];
        let target_interval = &interval_structure.intervals[target_interval_idx];
        
        let intervals_overlap = intervals_overlap(source_interval, target_interval);
        
        // Violation if:
        // - Positive edge and intervals don't overlap
        // - Negative edge and intervals overlap
        if (edge.weight > 0 && !intervals_overlap) || (edge.weight < 0 && intervals_overlap) {
            violations += 1;
        }
    }

    violations
}

/// Generate all permutations recursively
fn generate_permutations(
    current: &mut Vec<usize>,
    remaining: &mut Vec<usize>,
    permutations: &mut Vec<Vec<usize>>
) {
    if remaining.is_empty() {
        permutations.push(current.clone());
        return;
    }
    
    let n = remaining.len();
    for i in 0..n {
        let element = remaining.remove(i);
        current.push(element);
        
        generate_permutations(current, remaining, permutations);
        
        // Backtrack
        current.pop();
        remaining.insert(i, element);
    }
}

/// Brute force all possible assignments from clusters to intervals to minimize violations
pub fn brute_force_interval_structure(
    edges: &Vec<SignedEdge>,
    node_labels: &[usize],
    interval_structure: &IntervalStructure
) -> (HashMap<usize, usize>, usize) {
    // Find unique clusters - these can be arbitrary numbers
    let mut unique_clusters: Vec<usize> = node_labels.iter().cloned().collect();
    unique_clusters.sort();
    unique_clusters.dedup();
    
    // println!("Unique clusters: {:?}", unique_clusters);
    
    let num_clusters = unique_clusters.len();
    let num_intervals = interval_structure.intervals.len();
    
    println!("Brute-forcing {} clusters to {} intervals", num_clusters, num_intervals);
    
    // If number of clusters > number of intervals, we can't map each cluster to a unique interval
    if num_clusters != num_intervals {
        panic!("Error: Number of clusters ({}) and number of intervals ({}) don't match.", 
               num_clusters, num_intervals);
    }
    
    // Generate all possible interval indices to assign
    let interval_indices: Vec<usize> = (0..num_intervals).collect();
    
    let mut current_permutation: Vec<usize> = Vec::new();
    let mut permutations: Vec<Vec<usize>> = Vec::new();
    
    // We only need permutations of length equal to the number of clusters
    let mut remaining_indices = interval_indices.clone();
    generate_permutations(&mut current_permutation, &mut remaining_indices, &mut permutations);
    
    let total_permutations = permutations.len();
    println!("Testing {} permutations", total_permutations);
    
    let start_time = Instant::now();
    let mut best_violations = usize::MAX;
    let mut best_mapping = HashMap::new();
    
    // For progress reporting
    let report_every = std::cmp::max(1, total_permutations / 100);
    
    // Try each permutation
    for (i, perm) in permutations.iter().enumerate() {
        // Create a mapping from each unique cluster ID to an interval index
        let mut cluster_to_interval = HashMap::new();
        for (j, &cluster) in unique_clusters.iter().enumerate() {
            if j < perm.len() {
                cluster_to_interval.insert(cluster, perm[j]);
            }
        }
        
        let violations = compute_interval_violations(
            edges, 
            node_labels, 
            interval_structure, 
            &cluster_to_interval
        );
        
        if violations < best_violations {
            best_violations = violations;
            best_mapping = cluster_to_interval.clone();
        }
        
        // Report progress
        if i % report_every == 0 {
            let progress = (i as f64 / total_permutations as f64) * 100.0;
            println!("Progress: {:.1}% ({}/{})", progress, i, total_permutations);
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("Brute force completed in {:.2?}", elapsed);
    println!("Best violation count: {}", best_violations);
    
    (best_mapping, best_violations)
}