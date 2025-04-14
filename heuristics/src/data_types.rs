use serde::Deserialize;
use std::collections::HashMap;
use std::cmp::Ordering;

#[derive(Deserialize, Debug)]
pub struct SignedEdge {
    pub source: usize,
    pub target: usize,
    pub weight: i32,
}

#[derive(Deserialize, Debug)]
pub struct SignedGraph {
    pub edges: Vec<SignedEdge>,
}

#[derive(Deserialize, Debug)]
pub struct Interval {
    pub start: f32,
    pub end: f32,
}

#[derive(Deserialize, Debug)]
pub struct IntervalStructure {
    pub intervals: Vec<Interval>,
}

#[derive(Debug)]
pub struct Instance {
    pub graph: SignedGraph,
    pub interval_structure: IntervalStructure,
}

pub struct DynamicGraph {
    pub vertices: Vec<HashMap<usize, i32>>,
}

impl DynamicGraph {
    pub fn new(n: usize) -> Self {
        let vertices = vec![HashMap::new(); n];
        DynamicGraph { vertices }
    }

    pub fn edge_exists(&self, a: usize, b: usize) -> bool {
        self.vertices[a].contains_key(&b)
    }

    pub fn get_adjacent_vertices(&self, v: usize) -> &HashMap<usize, i32> {
        &self.vertices[v]
    }

    pub fn get_edge_weight(&self, a: usize, b: usize) -> i32 {
        *self.vertices[a].get(&b).unwrap_or(&0)
    }

    pub fn remove_vertex(&mut self, v: usize) {
        let neighbors: Vec<usize> = self.vertices[v].keys().cloned().collect();
        for &u in &neighbors {
            self.vertices[u].remove(&v);
        }
        self.vertices[v].clear();
    }

    pub fn update_edge_weight(&mut self, a: usize, b: usize, w: i32) {
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
pub struct DynamicEdge {
    pub a: usize,
    pub b: usize,
    pub edition: usize,
    pub weight: i32,
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
pub struct Partition {
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
    pub fn new(size: usize) -> Self {
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
    pub fn find(&mut self, x: usize) -> usize {
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
    pub fn union(&mut self, x: usize, y: usize) {
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
    pub fn num_sets(&self) -> usize {
        self.num_sets
    }
}