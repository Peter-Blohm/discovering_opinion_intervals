use fxhash::FxHasher;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

type FxMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
type FxSet<T> = std::collections::HashSet<T, BuildHasherDefault<FxHasher>>;

#[derive(Deserialize, Debug)]
pub struct SignedEdge {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
}

#[derive(Deserialize, Debug)]
pub struct SignedGraph {
    pub edges: Vec<SignedEdge>,
}

#[derive(Deserialize, Debug)]
pub struct UsefulSignedGraph {
    pub num_vertices: usize,
    pub edges: Vec<SignedEdge>,
    pub inverse_map: FxMap<usize, usize>,
}

impl UsefulSignedGraph {
    pub fn new(graph: &SignedGraph) -> UsefulSignedGraph {
        let vertices: FxSet<usize> = graph
            .edges
            .iter()
            .flat_map(|edge| [edge.source, edge.target])
            .collect();
        let num_vertices = vertices.len();
        let mut vertex_map: FxMap<usize, usize> = FxMap::default();
        let mut inverse_map: FxMap<usize, usize> = FxMap::default();
        for (new_id, &old_id) in vertices.iter().enumerate() {
            vertex_map.insert(old_id, new_id);
            inverse_map.insert(new_id, old_id);
        }
        let mut edges = Vec::new();
        for edge in &graph.edges {
            let new_a = *vertex_map.get(&edge.source).unwrap();
            let new_b = *vertex_map.get(&edge.target).unwrap();
            edges.push(SignedEdge {
                source: new_a,
                target: new_b,
                weight: edge.weight,
            });
        }
        UsefulSignedGraph {
            num_vertices,
            edges,
            inverse_map,
        }
    }

    pub fn vertex_id(&self, normalized_id: usize) -> usize {
        *self
            .inverse_map
            .get(&normalized_id)
            .unwrap_or_else(|| panic!("Vertex {} not found", normalized_id))
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Interval {
    pub start: f32,
    pub end: f32,
}

#[derive(Deserialize, Debug)]
pub struct IntervalStructure {
    pub intervals: Vec<Interval>,
}

impl IntervalStructure {
    pub(crate) fn intervals_overlap(&self, interval_index1: usize, interval_index2: usize) -> bool {
        let in1 = &self.intervals[interval_index1];
        let in2 = &self.intervals[interval_index2];
        (in2.start <= in1.end) && (in1.start <= in2.end)
    }
}

pub struct DynamicGraph {
    //adjacency lists
    pub vertices: Vec<HashMap<usize, f64>>,
}

impl DynamicGraph {
    pub fn new(n: usize) -> Self {
        let vertices = vec![HashMap::new(); n];
        DynamicGraph { vertices }
    }

    pub fn edge_exists(&self, a: usize, b: usize) -> bool {
        self.vertices[a].contains_key(&b)
    }

    pub fn get_adjacent_vertices(&self, v: usize) -> &HashMap<usize, f64> {
        &self.vertices[v]
    }

    pub fn get_edge_weight(&self, a: usize, b: usize) -> f64 {
        *self.vertices[a].get(&b).unwrap_or(&0.0)
    }

    pub fn remove_vertex(&mut self, v: usize) {
        let neighbors: Vec<usize> = self.vertices[v].keys().cloned().collect();
        for &u in &neighbors {
            self.vertices[u].remove(&v);
        }
        self.vertices[v].clear();
    }

    pub fn update_edge_weight(&mut self, a: usize, b: usize, w: f64) {
        *self.vertices[a].entry(b).or_insert(0.0) += w;
        *self.vertices[b].entry(a).or_insert(0.0) += w;
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
#[derive(Debug, PartialEq)]
pub struct DynamicEdge {
    pub a: usize,
    pub b: usize,
    pub edition: usize,
    pub weight: f32,
}

impl Eq for DynamicEdge {
    fn assert_receiver_is_total_eq(&self) {}
}

impl PartialOrd for DynamicEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.weight.partial_cmp(&other.weight)
    }
}

impl Ord for DynamicEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight
            .partial_cmp(&other.weight)
            .unwrap_or(Ordering::Equal)
    }
}

/// A Union-Find (Disjoint Set Union) data structure optimized for efficient set merging and connectivity queries.
/// Supports path compression (for fast `find`) and union by rank (to keep trees balanced).
pub struct Partition {
    parents: Vec<usize>,
    ranks: Vec<usize>,
}

impl Partition {
    pub fn new(size: usize) -> Self {
        Partition {
            parents: (0..size).collect(),
            ranks: vec![0; size],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parents[x] != x {
            self.parents[x] = self.find(self.parents[x]);
        }
        self.parents[x]
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let x_root = self.find(x);
        let y_root = self.find(y);

        if x_root != y_root {
            if self.ranks[x_root] < self.ranks[y_root] {
                self.parents[x_root] = y_root;
            } else {
                self.parents[y_root] = x_root;
                if self.ranks[x_root] == self.ranks[y_root] {
                    self.ranks[x_root] += 1;
                }
            }
        }
    }
}
