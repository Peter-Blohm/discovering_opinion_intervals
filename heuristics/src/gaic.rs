use crate::data_types::{IntervalStructure, UsefulSignedGraph};
use priority_queue::PriorityQueue;
use rand::prelude::SliceRandom;
use rand::{rng, RngCore};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::time::Instant;

#[derive(Clone)]
pub struct SignedNeighbourhood {
    //adjacency lists
    pub positive_neighbors: Vec<usize>,
    pub negative_neighbors: Vec<usize>,
}

impl SignedNeighbourhood {
    pub fn new() -> Self {
        SignedNeighbourhood {
            positive_neighbors: Vec::new(),
            negative_neighbors: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct PriorityVertex {
    cluster_affinities: Vec<usize>,
    pub favorite_cluster: usize,
    runner_up: usize,
    max_affinity: usize,
    priority: usize,
    cluster_priorities: Vec<usize>,
}

impl PriorityVertex {
    pub fn new(priority: usize, k: usize) -> Self {
        let cluster_affinities = vec![0; k];
        let mut perm: Vec<usize> = (1..=k).collect();
        perm.shuffle(&mut rng());
        PriorityVertex {
            cluster_affinities,
            favorite_cluster: 0,
            runner_up: 0,
            max_affinity: 0,
            priority,
            cluster_priorities: perm,
        }
    }

    pub fn reprioritise(&mut self) {
        let mut perm: Vec<usize> = (1..=self.cluster_priorities.len()).collect();
        perm.shuffle(&mut rng());
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
                best_idx = i;
            }
        }

        self.favorite_cluster = best_idx;
        self.max_affinity = best_aff;
    }

    pub fn get_sort_priority(&self) -> (usize, usize) {
        (self.max_affinity, self.priority)
    }
}
impl PartialEq for PriorityVertex {
    fn eq(&self, other: &Self) -> bool {
        self.max_affinity == other.max_affinity && self.priority == other.priority
    }
}
impl Eq for PriorityVertex {}

impl Ord for PriorityVertex {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.max_affinity, self.priority).cmp(&(other.max_affinity, other.priority))
    }
}

impl PartialOrd for PriorityVertex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn greedy_absolute_interval_contraction(
    signed_graph: &UsefulSignedGraph,
    interval_structure: &IntervalStructure,
    num_batches: usize, // number of groups
    num_runs: usize,
) -> Vec<usize> {
    // setup ////////////////////////////////////
    let num_vertices = signed_graph.num_vertices;
    let edges = &signed_graph.edges;
    let num_intervals = interval_structure.intervals.len();

    let mut perm: Vec<usize> = (1..=num_vertices).collect();
    perm.shuffle(&mut rng());
    let start_time = Instant::now();
    let mut priority_vertices: Vec<PriorityVertex> = perm
        .into_iter()
        .map(|priority| PriorityVertex::new(priority, num_intervals))
        .collect();

    let mut adj_graph: Vec<SignedNeighbourhood> = (1..=num_vertices)
        .into_iter()
        .map(|_| SignedNeighbourhood::new())
        .collect();
    for edge in edges {
        if edge.weight == 1 {
            adj_graph[edge.source].positive_neighbors.push(edge.target);
            adj_graph[edge.target].positive_neighbors.push(edge.source);
        } else {
            adj_graph[edge.source].negative_neighbors.push(edge.target);
            adj_graph[edge.target].negative_neighbors.push(edge.source);
        }
    }
    // no longer mutable
    let adj_graph = adj_graph;

    // run ////////////////////////////////////

    println!("One run started");
    let (mut assigned, mut agreement) = assign(
        &*(0..num_vertices).collect::<Vec<usize>>(),
        &vec![num_intervals; num_vertices],
        interval_structure,
        num_intervals,
        &mut priority_vertices,
        &adj_graph,
    );

    println!("Running time: {:.2?}", start_time.elapsed());
    println!("One run done");
    println!(
        "agreement after first run {},{},{}",
        edges.len(),
        agreement,
        edges.len() - agreement
    );
    let mut best_assigment = assigned.clone();
    let mut best_agreement = agreement.clone();
    let mut last_improvement: usize = 0;
    let mut epoch_solution: usize= 0;
    for _epoch in 1..num_runs {
        // 1) build & shuffle the full index list
        let mut indices: Vec<usize> = (0..num_vertices).collect();
        indices.shuffle(&mut rng());
        // println!("Test");
        let chunk_size = (num_vertices + num_batches - 1) / if {last_improvement < 500} {num_batches} else {last_improvement=0;epoch_solution=0;println!("reset");1};

        // 3) slice into at most `num_batches` chunks and call `assign`
        for chunk in indices.chunks(chunk_size).take(num_batches) {
            //new priorities
            let mut perm: Vec<usize> = (1..=chunk.len()).collect();
            perm.shuffle(&mut rng());
            let mut updates: SmallVec<usize, 32> = SmallVec::new();
            let mut counter = 0;
            for (idx, &vertex_id) in chunk.iter().enumerate() {
                priority_vertices[vertex_id].priority = perm[idx];
                let fav = assigned[vertex_id];
                let agg = priority_vertices[vertex_id].cluster_affinities[fav];
                counter += agg;
                priority_vertices[vertex_id].reprioritise();
                // println!("agg {}",agg);
                assert!(fav < num_intervals);
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
                // println!("number of affected vertices: {}", updates.len());
                // println!("{:?}", updates);
                for n in updates.drain(..) {
                    priority_vertices[n].update_favorites();
                }
                // agreement-=agg;
                assigned[vertex_id] = num_intervals;
            }
            //unassign

            //reassign
            // assign(chunk);
            let (assigned2, partial_agreement) = assign(
                chunk,
                &assigned,
                interval_structure,
                num_intervals,
                &mut priority_vertices,
                &adj_graph,
            );
            
            if agreement + partial_agreement - counter > best_agreement {
                best_assigment = assigned2.clone();
                best_agreement =  agreement + partial_agreement - counter;
                // last_improvement = 0;
            }
            assigned = assigned2;
            agreement = agreement + partial_agreement - counter;
            if agreement > epoch_solution {
                epoch_solution = agreement;
                last_improvement = 0;
            }
        }
        last_improvement+=1;
        println!(
            "Agreement: {},{},{},{}",
            edges.len()-agreement,
            edges.len()-epoch_solution,
            edges.len()-best_agreement,
            last_improvement
        );
    }
    // println!("{:?}", assigned);
    return best_assigment;
}

fn assign(
    idx: &[usize],
    assigned: &Vec<usize>,
    interval_structure: &IntervalStructure,
    num_intervals: usize,
    mut priority_vertices: &mut Vec<PriorityVertex>,
    adj_graph: &Vec<SignedNeighbourhood>,
) -> (Vec<usize>, usize) {
    let mut agreement: usize = 0;
    let mut assigned = assigned.clone();
    // let mut p_vertices = priority_vertices.clone();

    let mut pq: PriorityQueue<usize, (usize, usize)> = PriorityQueue::from_iter(
        idx.into_iter()
            .map(|&i| (i, priority_vertices[i].get_sort_priority())),
    );
    while let Some((id, _)) = pq.pop() {
        let fav = priority_vertices[id].favorite_cluster;
        assigned[id] = fav;
        agreement += priority_vertices[id].max_affinity;

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
    (assigned, agreement)
}
