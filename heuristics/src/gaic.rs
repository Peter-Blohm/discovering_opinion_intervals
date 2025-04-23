use crate::data_types::{IntervalStructure, UsefulSignedGraph};
use priority_queue::PriorityQueue;
use rand::prelude::SliceRandom;
use rand::{rng, Rng};
use smallvec::SmallVec;
use std::collections::HashSet;
use std::time::Instant;

#[derive(Clone)]
pub struct SignedNeighbourhood {
    //adjacency lists
    pub positive_neighbors: HashSet<usize>,
    pub negative_neighbors: HashSet<usize>,
}

impl SignedNeighbourhood {
    pub fn new() -> Self {
        SignedNeighbourhood {
            positive_neighbors: HashSet::new(),
            negative_neighbors: HashSet::new(),
        }
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
    pub num_neighbors: usize,
    max_count: usize,
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
            num_neighbors: 0,
            max_count : k,
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
        ) == (
            self.max_affinity,
        ) {
            self.max_count += 1;
        }
        if (
            self.cluster_affinities[cluster],
            self.cluster_priorities[cluster],
        ) > (
            self.max_affinity,
            self.cluster_priorities[self.favorite_cluster],
        ) {
            if cluster != self.favorite_cluster {
                self.max_count = 1;
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
            if (a == best_aff) {
                self.max_count += 1;
            }
            if (a > best_aff) || (a == best_aff && p > best_pri) {
                best_aff = a;
                best_pri = p;
                self.runner_up = best_idx;
                best_idx = i;
                self.max_count = 1;
            }

        }

        self.favorite_cluster = best_idx;
        self.max_affinity = best_aff;
    }

    pub fn get_sort_priority(&self) -> (usize, usize, usize) {
        (self.max_affinity, self.max_count, self.priority)
    }
    pub fn get_favorite_cluster_simulated_annealing(&self, temperature: f64) -> usize {
        // returns the favorite cluster, but with a small chance of returning a random one
        let mut rng = rng();
        let temps: Vec<f64> = self.cluster_affinities
            .iter()
            .map(|&a| ((a as f64)/(self.num_neighbors as f64)/(temperature+0.001)).exp())
            .collect();
        let sum_temp: f64 = temps.iter().sum();
        let random_value = rng.gen_range(0.0..1.0);
        let mut cumulative_sum = 0.0003;
        let mut cluster = 0;
        for (i, &temp) in temps.iter().enumerate() {
            
            if random_value > cumulative_sum {
                
                cluster = i;
            }
            cumulative_sum += temp/sum_temp;
        }
        //println!("Random value: {}, cluster: {}", i, cumulative_sum);
        cluster
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

    
    let mut adj_graph: Vec<SignedNeighbourhood> = (1..=num_vertices)
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
    let adj_graph = adj_graph;

    let mut perm: Vec<usize> = (0..num_vertices).collect();
    perm.shuffle(&mut rng());
    let start_time = Instant::now();
    let mut priority_vertices: Vec<PriorityVertex> = perm
        .into_iter()
        .map(|priority| PriorityVertex::new(priority, num_intervals))
        .collect();

    for id in 0..num_vertices {
        priority_vertices[id].num_neighbors = adj_graph[id].positive_neighbors.len() + adj_graph[id].negative_neighbors.len();
    }
    // run ////////////////////////////////////
    let mut assigned = vec![num_intervals; num_vertices];
    println!("One run started");
    let mut agreement = assign(
        &*(0..num_vertices).collect::<Vec<usize>>(),
        &mut assigned,
        interval_structure,
        num_intervals,
        &mut priority_vertices,
        &adj_graph,
        0.05
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
    let mut temp = 0.05;
    for _epoch in 1..num_runs {
        // 1) build & shuffle the full index list
        let mut indices: Vec<usize> = (0..num_vertices).collect();
        if _epoch > 1{
            indices.shuffle(&mut rng());
        }
        // println!("Test");
        let chunk_size = (num_vertices + num_batches -1 ) / if last_improvement < 100 {num_batches} else {temp *= 5.0;epoch_solution=0;println!("reset");2};

        // 3) slice into at most `num_batches` chunks and call `assign`
        for chunk in indices.chunks(chunk_size).take(num_batches) { // +_epoch*2
            
            //new priorities
            let lost_agreement = unassign(chunk, &mut assigned, interval_structure, 
                                          num_intervals, &mut priority_vertices, &adj_graph);
            
            let won_agreement =    assign(chunk, &mut assigned, interval_structure, 
                                          num_intervals, &mut priority_vertices, &adj_graph, temp);

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
            "Agreement: {},{},{},{},{}",
            edges.len()-agreement,
            edges.len()-epoch_solution,
            edges.len()-best_agreement,
            last_improvement,
            temp
        );
        last_improvement+=1;

        if _epoch % 10 == 0 {
            temp *= 0.9;
        }
    }
    // println!("{:?}", assigned);
    best_assigment
}

fn unassign(idx: &[usize],
            assigned: &mut Vec<usize>,
            interval_structure: &IntervalStructure,
            num_intervals: usize,
            priority_vertices: &mut Vec<PriorityVertex>,
            adj_graph: &Vec<SignedNeighbourhood>) -> usize {
    let mut perm: Vec<usize> = (1..=idx.len()).collect();
    perm.shuffle(&mut rng());
    let mut updates: SmallVec<usize, 32> = SmallVec::new();
    let mut removed_agreement = 0;
    for (idx, &vertex_id) in idx.iter().enumerate() {
        priority_vertices[vertex_id].priority = perm[idx];
        let fav = assigned[vertex_id];
        let agg = priority_vertices[vertex_id].cluster_affinities[fav];
        removed_agreement += agg;
        priority_vertices[vertex_id].reprioritise();
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
) -> usize {
    let mut agreement: usize = 0;
    let mut perm: Vec<usize> = (0..assigned.len()).collect();
    perm.shuffle(&mut rng());
    let mut pq: PriorityQueue<usize, (usize, usize, usize)> = PriorityQueue::from_iter(
        idx.into_iter()
            .map(|&i| (i, priority_vertices[i].get_sort_priority())),
    );
    while let Some((id, _)) = pq.pop() {
        let fav = priority_vertices[id].get_favorite_cluster_simulated_annealing(
            temp,
        );
        assigned[id] = fav;
        agreement += priority_vertices[id].cluster_affinities[fav];

        let mut updates: SmallVec<(usize, (usize, usize, usize)), 32> = SmallVec::new();

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
