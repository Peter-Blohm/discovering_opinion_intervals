use crate::data_types::{IntervalStructure, SignedEdge, UsefulSignedGraph};
use priority_queue::PriorityQueue;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
use smallvec::SmallVec;
use std::time::{Duration, Instant};
use rand::rngs::StdRng;
use serde::Deserialize;
use fxhash::FxHasher;
use std::hash::BuildHasherDefault;
type FxSet<T> = std::collections::HashSet<T, BuildHasherDefault<FxHasher>>;
type FxPQ<K, P> = PriorityQueue<K, P, BuildHasherDefault<FxHasher>>;

#[derive(Clone)]
pub struct SignedNeighbourhood {
    //adjacency lists
    pub positive_neighbors: FxSet<usize>,
    pub negative_neighbors: FxSet<usize>,
}

impl SignedNeighbourhood {
    pub fn new() -> Self {
        SignedNeighbourhood {
            positive_neighbors: FxSet::default(),
            negative_neighbors: FxSet::default(),
        }
    }
}

pub struct SignedAdjacencyList {
    pub adj_graph: Vec<SignedNeighbourhood>,
}
impl SignedAdjacencyList {
    pub fn new(edges: &Vec<SignedEdge>, num_vertices:usize) -> Self {
        let mut adj_graph: Vec<SignedNeighbourhood> = (0..num_vertices)
            .into_iter()
            .map(|_| SignedNeighbourhood::new())
            .collect();
        for edge in edges {
            if edge.weight == 1 {
                adj_graph[edge.source].positive_neighbors.insert(edge.target);
                adj_graph[edge.target].positive_neighbors.insert(edge.source);
            } else {
                adj_graph[edge.source].negative_neighbors.insert(edge.target);
                adj_graph[edge.target].negative_neighbors.insert(edge.source);
            }
        }
        // no longer mutable
        SignedAdjacencyList { adj_graph }
    }


}

#[derive(Clone)]
pub struct PriorityVertex {
    cluster_affinities: Vec<usize>,
    favorite_cluster: usize,
    runner_up: usize,
    max_affinity: usize,
    priority: usize,
    cluster_priorities: Vec<usize>,
}

impl PriorityVertex {
    pub fn new(priority: usize, k: usize, local_rng: &mut StdRng) -> Self {
        let cluster_affinities = vec![0; k];
        let mut perm: Vec<usize> = (1..=k).collect();
        perm.shuffle(local_rng);
        PriorityVertex {
            cluster_affinities,
            favorite_cluster: 0,
            runner_up: 0,
            max_affinity: 0,
            priority,
            cluster_priorities: perm,
        }
    }

    pub fn reprioritise(&mut self, local_rng: &mut StdRng) {
        let mut perm: Vec<usize> = (0..self.cluster_priorities.len()).collect();
        perm.shuffle(local_rng);
        self.cluster_priorities = perm;
    }
    pub fn increase_affinity(&mut self, cluster: usize) -> bool {
        // returns whether its max_affinity changed
        self.cluster_affinities[cluster] += 1;
        if (
            self.cluster_affinities[cluster],
            self.cluster_priorities[cluster],
        ) > (
            self.max_affinity,
            self.cluster_priorities[self.favorite_cluster],
        ) {
            if cluster != self.favorite_cluster {
                self.runner_up = self.favorite_cluster;
                self.favorite_cluster = cluster;
            }
            self.max_affinity = self.cluster_affinities[cluster];
            return true;
        }
        false
    }

    pub fn decrease_affinity_unsafe(&mut self, cluster: usize) -> bool {
        self.cluster_affinities[cluster] -= 1;
        if self.cluster_affinities[self.favorite_cluster] <= self.cluster_affinities[self.runner_up]
        {
            return true;
        }
        // cluster affinities of favorite cluster larger than all others
        if cluster == self.favorite_cluster {
            self.max_affinity -= 1;
        }
        false
    }
    pub fn update_favorites(&mut self) {
        let mut best_idx = self.favorite_cluster;
        let mut best_aff = self.cluster_affinities[self.favorite_cluster];
        let mut best_pri = self.cluster_priorities[self.favorite_cluster];

        for i in 0..self.cluster_affinities.len() {
            let a = self.cluster_affinities[i];
            let p = self.cluster_priorities[i];
            if (a > best_aff) || (a == best_aff && p > best_pri) {
                best_aff = a;
                best_pri = p;
                self.runner_up = best_idx;
                best_idx = i;
            }
        }
        self.favorite_cluster = best_idx;
        self.max_affinity = best_aff;
    }

    pub fn get_sort_priority(&self) -> (usize,  usize) {
        (self.max_affinity, self.priority)
    }
    pub fn get_favorite_cluster(&self, temperature: f64, local_rng: &mut StdRng) -> usize {
        if temperature == 0.0 { // no simulated annealing
            return self.favorite_cluster
        }
        debug_assert!(temperature.is_sign_positive());
        let inv_t = 1.0 / temperature.max(1e-12);
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for (idx, &aff) in self.cluster_affinities.iter().enumerate() {
            let u: f64 = local_rng.random_range(0.0..1.0);
            let g = -(-u.ln()).ln();            // Gumbel(0,1)
            let val = aff as f64 * inv_t + g;          // aff/T  +  Gumbel
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        best_idx
    }
}

#[derive(Deserialize,Clone)]
pub struct GaicConfig {
    pub num_epochs: usize,
    pub timeout_s: u64,
    pub early_stopping_epochs: usize,
    pub num_reassignment_chunks: usize,

    pub initial_temperature: f64,
    pub temperature_decay_factor: f64,
    pub temperature_decay_epochs: usize,
}


pub fn greedy_absolute_interval_contraction(
    signed_graph: &UsefulSignedGraph,
    interval_structure: &IntervalStructure,
    config: &GaicConfig,
    random_seed: u64,
) -> Vec<usize> {
    let mut local_rng = StdRng::seed_from_u64(random_seed);
    // setup ////////////////////////////////////
    let num_vertices = signed_graph.num_vertices;
    let edges = &signed_graph.edges;
    let num_intervals = interval_structure.intervals.len();
    let adj_graph = SignedAdjacencyList::new(&signed_graph.edges,num_vertices);

    let mut temp = config.initial_temperature;


    let mut perm: Vec<usize> = (0..num_vertices).collect();
    perm.shuffle(&mut local_rng);
    let start_time = Instant::now();
    let mut priority_vertices: Vec<PriorityVertex> = perm
        .into_iter()
        .map(|priority| PriorityVertex::new(priority, num_intervals,&mut local_rng))
        .collect();

    // run ////////////////////////////////////
    let mut assigned = vec![num_intervals; num_vertices];
    let mut agreement = assign(
        &*(0..num_vertices).collect::<Vec<usize>>(),
        &mut assigned,
        interval_structure,
        num_intervals,
        &mut priority_vertices,
        &adj_graph.adj_graph,
        temp,
        &mut local_rng
    );

    // println!("Running time: {:.2?}", start_time.elapsed());
    // println!("One run done");
    println!("current, best_batch, best, epochs_since_restart, current_temp, runtime");
    println!(
        "{}, {}, {}, {}, {}, {:?}",
        edges.len()-agreement,
        edges.len()-agreement,
        edges.len()-agreement,
        0,
        temp,
        Instant::now() - start_time
    );
    let mut best_assigment = assigned.clone();
    let mut best_agreement = agreement.clone();
    let mut last_improvement: usize = 0;
    let mut epoch_solution: usize= 0;
    let timeout =  Duration::from_millis(config.timeout_s * 1000);
    //////////////////// MAIN REASSIGNMENT LOOP ///////////////////////////////
    for epoch in 1..config.num_epochs{
        if (Instant::now() - start_time > timeout) || (last_improvement > config.early_stopping_epochs) {
            break
        }

        let mut indices: Vec<usize> = (0..num_vertices).collect();
        if epoch > 1 { indices.shuffle(&mut local_rng); }

        let chunk_size = (num_vertices + config.num_reassignment_chunks -1 ) / config.num_reassignment_chunks;
        // if last_improvement < 1500 {config.num_reassignment_chunks} else {temp =0.0002+temp;epoch_solution=0;println!("reset");2};

        for chunk in indices.chunks(chunk_size).take(config.num_reassignment_chunks) {
            //new priorities
            let lost_agreement = unassign(chunk, &mut assigned, interval_structure, 
                                          num_intervals, &mut priority_vertices, &adj_graph.adj_graph, &mut local_rng);

            let won_agreement =    assign(chunk, &mut assigned, interval_structure, 
                                          num_intervals, &mut priority_vertices, &adj_graph.adj_graph, temp, &mut local_rng);

            //overwrite the best if there was an improvement
            if agreement + won_agreement - lost_agreement > best_agreement {
                best_assigment = assigned.clone();
                best_agreement =  agreement + won_agreement - lost_agreement;
            }
            agreement = agreement + won_agreement - lost_agreement;
            if agreement > epoch_solution {
                epoch_solution = agreement;
                last_improvement = 0;
            }
        }
        println!(
            "{}, {}, {}, {}, {}, {:?}",
            edges.len()-agreement,
            edges.len()-epoch_solution,
            edges.len()-best_agreement,
            last_improvement,
            temp,
            Instant::now() - start_time
        );
        last_improvement+=1;

        if epoch % config.temperature_decay_epochs == 0 { temp *= config.temperature_decay_factor; }
    }
    best_assigment
}

fn unassign(idx: &[usize],
            assigned: &mut Vec<usize>,
            interval_structure: &IntervalStructure,
            num_intervals: usize,
            priority_vertices: &mut Vec<PriorityVertex>,
            adj_graph: &Vec<SignedNeighbourhood>,
            local_rng: &mut StdRng

) -> usize {
    let mut perm: Vec<usize> = (0..idx.len()).collect();
    perm.shuffle(local_rng);

    let mut updates: SmallVec<usize, 32> = SmallVec::new();
    let mut removed_agreement = 0;
    for (idx, &vertex_id) in idx.iter().enumerate() {
        priority_vertices[vertex_id].priority = perm[idx];
        let fav = assigned[vertex_id];
        let agg = priority_vertices[vertex_id].cluster_affinities[fav];
        removed_agreement += agg;
        priority_vertices[vertex_id].reprioritise(local_rng);
        debug_assert!(fav < num_intervals);
        for cluster in 0..num_intervals {
            let neighbors = if interval_structure.intervals_overlap(cluster, fav) {
                &adj_graph[vertex_id].positive_neighbors
            } else {
                &adj_graph[vertex_id].negative_neighbors
            };
            for &n in neighbors {
                if {
                    let neighbor = &mut priority_vertices[n];
                    neighbor.decrease_affinity_unsafe(cluster)
                } {
                    // remember vertex for changing queue priority later
                    updates.push(n);
                }
            }
        }
        // this is the expensive part
        for n in updates.drain(..) {
            priority_vertices[n].update_favorites();
        }
        // agreement-=agg;
        assigned[vertex_id] = num_intervals;
    }
    removed_agreement
}

fn assign(
    idx: &[usize],
    assigned: &mut Vec<usize>,
    interval_structure: &IntervalStructure,
    num_intervals: usize,
    priority_vertices: &mut Vec<PriorityVertex>,
    adj_graph: &Vec<SignedNeighbourhood>,
    temp: f64,
    local_rng: &mut StdRng
) -> usize {
    let mut agreement: usize = 0;
    let mut pq: FxPQ<usize, (usize, usize)>=
        FxPQ::from_iter(
        idx.into_iter()
            .map(|&i| (i, priority_vertices[i].get_sort_priority())),
    );
    while let Some((id, _)) = pq.pop() {
        let fav = priority_vertices[id].get_favorite_cluster(
            temp,
            local_rng
        );
        assigned[id] = fav;
        agreement += priority_vertices[id].cluster_affinities[fav];

        let mut updates: SmallVec<(usize, (usize, usize)), 32> = SmallVec::new();

        for cluster in 0..num_intervals {
            let neighbors = if interval_structure.intervals_overlap(cluster, fav) {
                &adj_graph[id].positive_neighbors
            } else {
                &adj_graph[id].negative_neighbors
            };
            for &n in neighbors {
                if {
                    let neighbor = &mut priority_vertices[n];
                    neighbor.increase_affinity(cluster)
                } {
                    // remember vertex for changing queue priority later
                    updates.push((n, priority_vertices[n].get_sort_priority()));
                }
            }
        }
        // this is the expensive part
        for (n, prio) in updates.drain(..) {
            pq.change_priority(&n, prio);
        }
    }
    agreement
}
