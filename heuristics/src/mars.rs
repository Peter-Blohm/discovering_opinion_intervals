use std::collections::{HashMap, HashSet};
use crate::data_types::{IntervalStructure, SignedEdge, UsefulSignedGraph};
use priority_queue::PriorityQueue;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use fxhash::FxHasher;
use std::hash::BuildHasherDefault;
use std::ptr::NonNull;
use rand::prelude::SliceRandom;
use crate::gaic::SignedAdjacencyList;

// type FxSet<T> = std::collections::HashSet<T, BuildHasherDefault<FxHasher>>;
type FxMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
type FxPQ<K, P> = PriorityQueue<K, P, BuildHasherDefault<FxHasher>>;


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[derive(Hash)]
pub enum IntervalType {
    Start,
    End,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct VertexInterval {
    pub id: usize,
    pub itype: IntervalType,
}

/// A node that *lives inside* the linked list.
struct Node {
    id: usize,
    itype: IntervalType,
    prev: Option<NonNull<Node>>,
    next: Option<NonNull<Node>>,
    /// monotonically increasing position key
    index: u128,
}

/// Safety: `Node` never implements `Drop`, so a raw pointer
/// to it is always valid while the owning `Box<Node>` is alive.
unsafe impl Send for Node {}
unsafe impl Sync for Node {}

/// A doubly-linked list with *O(1)* splice and a positional
/// map for *O(1)* node lookup.
pub struct OrderedList {
    head: Option<NonNull<Node>>,
    tail: Option<NonNull<Node>>,
    len: usize,
    next_index: u128,               // keeps “2, 4, 6, …” for initial build
    positions: FxMap<(usize, IntervalType), NonNull<Node>>,
    /// Owns the actual memory — pointers inside `positions`
    /// are always into `arena`.
    arena: Vec<Box<Node>>,
}

impl OrderedList {
    /* ---------- construction ------------------------------------------------ */

    #[inline]
    pub fn new() -> Self {
        Self {
            head: None,
            tail: None,
            len: 0,
            next_index: 0,
            positions: FxMap::default(),
            arena: Vec::new(),
        }
    }

    /// Internal helper: allocate a node, push it at the back,
    /// return its raw pointer.
    #[inline]
    fn push_back_node(&mut self, id: usize, itype: IntervalType) -> NonNull<Node> {
        let index = self.next_index;
        self.next_index += 2; // leave a hole for later insertions
        let mut boxed = Box::new(Node {
            id,
            itype,
            prev: self.tail,
            next: None,
            index,
        });
        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) };
        // put the box back in the arena so we still own it
        unsafe { self.arena.push(Box::from_raw(ptr.as_ptr())) };

        match self.tail {
            Some(mut tail_ptr) => unsafe { tail_ptr.as_mut().next = Some(ptr) },
            None => self.head = Some(ptr),
        }
        self.tail = Some(ptr);
        self.len += 1;
        self.positions.insert((id, itype), ptr);
        ptr
    }

    /// Build a full tour from an assignment.
    /// Keeps the same public signature you already had.
    pub fn build_from_assignment(&mut self, assignment: &[usize], k: usize) {
        // 1st pass: collect per cluster
        let mut starts: Vec<Vec<(usize, IntervalType)>> = vec![vec![]; k];
        let mut ends:   Vec<Vec<(usize, IntervalType)>> = vec![vec![]; k];

        for (id, &cluster) in assignment.iter().enumerate() {
            starts[cluster].push((id, IntervalType::Start));
            ends[cluster].push  ((id, IntervalType::End  ));
        }

        // 2nd pass: actually create nodes in the requested order
        // TODO only works for intervals now
        // c0s,c1s,c0e,c2s,c1e
        for (id, ty) in &starts[0] { self.push_back_node(*id, *ty); }
        for c in 1..k {
            for (id, ty) in &starts[c] { self.push_back_node(*id, *ty); }
            for (id, ty) in &ends  [c-1] { self.push_back_node(*id, *ty); }
        }
        for (id, ty) in &ends  [k-1] { self.push_back_node(*id, *ty); }
    }

    /* ---------- O(d log d) ordered-neighbourhood ---------------------------- */

    pub fn ordered_neighborhood(
        &self,
        values: &FxMap<usize, f64>,
    ) -> Vec<(usize, f64, IntervalType)> {
        let mut tmp: Vec<(u128, usize, f64, IntervalType)> = Vec::with_capacity(values.len() * 2);

        for (&id, &score) in values {
            for &itype in [IntervalType::Start, IntervalType::End].iter() {
                if let Some(&ptr) = self.positions.get(&(id, itype)) {
                    // SAFETY: `ptr` is always valid while `self` is alive
                    unsafe { tmp.push((ptr.as_ref().index, id, score, itype)); }
                }
            }
        }

        tmp.sort_unstable_by_key(|&(idx, ..)| idx);
        tmp.into_iter()
            .map(|(_, id, score, itype)| (id, score, itype))
            .collect()
    }

    /* ---------- O(1) reinsertion ------------------------------------------- */

    /// Detach `node` from the list (but keep it alive in the arena).
    #[inline]

    unsafe fn detach(&mut self, mut node: NonNull<Node>) {
        // 1. read neighbours WITHOUT creating a mutable ref
        let prev = node.as_ref().prev;
        let next = node.as_ref().next;

        // 2. patch 'prev'
        if let Some(mut p) = prev {
            p.as_mut().next = next;
        } else {
            self.head = next;
        }

        // 3. patch 'next'
        if let Some(mut n) = next {
            n.as_mut().prev = prev;
        } else {
            self.tail = prev;
        }

        // 4. finally clear our own links – only *now* take a &mut Node
        {
            let n = node.as_mut();
            n.prev = None;
            n.next = None;
        }
    }

    unsafe fn insert_after(&mut self,
                           mut anchor: NonNull<Node>,
                           mut node: NonNull<Node>) -> u128
    {
        let next = anchor.as_ref().next;          // immutable borrow

        // wire new node ----------------------------------------------------------
        {
            let n = node.as_mut();                // first and only &mut to 'node'
            n.prev = Some(anchor);
            n.next = next;
        }

        // anchor becomes mutable only AFTER the previous borrow has ended
        anchor.as_mut().next = Some(node);

        if let Some(mut nxt) = next {
            nxt.as_mut().prev = Some(node);
        } else {
            self.tail = Some(node);
        }

        // -----------------------------------------------------------------------
        // index calculation – immutable refs are fine here
        let left  = anchor.as_ref().index;
        let right = next.map(|p| unsafe { p.as_ref().index });
        let idx = match right {
            Some(r) if r > left + 1 => (left + r) / 2,
            _                       => left + 1,
        };
        node.as_mut().index = idx;   // take a *fresh* mutable ref
        idx
    }

    fn renumber(&mut self) {
        let mut cur = self.head;
        let mut idx: u128 = 0;
        while let Some(mut ptr) = cur {
            unsafe {
                ptr.as_mut().index = idx; // one &mut, dropped at end of loop body
                cur = ptr.as_ref().next;  // only & (immutable) for the read
            }
            idx += 2;
        }
        self.next_index = idx;
    }
    /// Public reinsert API – *O(1)* except when a global renumber happens.
    ///
    /// *If `best_neighbors.0 == best_neighbors.1` the new order is
    /// `best → start → end → …` otherwise `… best0 → start … best1 → end …`*
    pub fn reinsert_after(
        &mut self,
        vertex_id: usize,
        best_neighbors: (VertexInterval, VertexInterval),
    ) {
        // look up our own nodes ------------------------------------------------
        let start_ptr = *self.positions.get(&(vertex_id, IntervalType::Start))
            .expect("vertex must be present");
        let end_ptr   = *self.positions.get(&(vertex_id, IntervalType::End))
            .expect("vertex must be present");

        // look up the anchors --------------------------------------------------
        let anchor_start =
            *self.positions.get(&(best_neighbors.0.id, best_neighbors.0.itype))
                .expect("start anchor must exist");
        let anchor_end =
            *self.positions.get(&(best_neighbors.1.id, best_neighbors.1.itype))
                .expect("end anchor must exist");

        unsafe {
            // detach the pair (constant time)
            self.detach(start_ptr);
            self.detach(end_ptr);

            if anchor_start == anchor_end {
                // special case: start → end after the *same* anchor
                self.insert_after(anchor_start, end_ptr);    // anchor … end
                let idx = self.insert_after(anchor_start, start_ptr); // anchor … start end
                if idx == end_ptr.as_ref().index {
                    self.renumber(); // fell into the gap, rare
                }
            } else {
                // general case
                self.insert_after(anchor_start, start_ptr);
                self.insert_after(anchor_end,   end_ptr);
            }

            // extremely unlikely we ran out of integer space on either side
            if start_ptr.as_ref().index + 1 == end_ptr.as_ref().index {
                self.renumber();
            }
        }
    }
}

/* ---------- optional helpers for debugging / testing ---------------------- */

impl OrderedList {
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Dump the current order as `[(id,S/E), …]` – handy in unit tests.
    pub fn to_vec(&self) -> Vec<(usize, IntervalType)> {
        let mut v = Vec::with_capacity(self.len);
        let mut cur = self.head;
        while let Some(p) = cur {
            unsafe {
                v.push((p.as_ref().id, p.as_ref().itype));
                cur = p.as_ref().next;
            }
        }
        v
    }
}

pub fn greedy_vertex_ordering_from_assignment(signed_graph: &UsefulSignedGraph, assignment: &[usize], num_clusters: usize, random_seed: u64) -> Vec<(usize,IntervalType)> {
    let mut local_rng = StdRng::seed_from_u64(random_seed);
    // setup ////////////////////////////////////
    let num_vertices = signed_graph.num_vertices;
    // let edges = &signed_graph.edges;
    let adj_graph = SignedAdjacencyList::new(&signed_graph.edges,num_vertices);
    let mut vertex_ordering = OrderedList::new();
    vertex_ordering.build_from_assignment(assignment, num_clusters);
    println!("{}, {}",ordering_agreement(&adj_graph,&vertex_ordering), signed_graph.edges.len());
    for epoch in 0..50 {
        let mut perm: Vec<usize> = (0..num_vertices).collect();
        perm.shuffle(&mut local_rng);    
        for vertex in perm {
            let mut start_agreement:f64 = 0.0;
            let mut max_agreement:f64 = 0.0;
            let mut best_neighbors: (VertexInterval,VertexInterval) = (
                VertexInterval {id: 0,itype: IntervalType::Start}, 
                VertexInterval {id: 0,itype: IntervalType::Start});
            let neighbors = vertex_ordering.ordered_neighborhood(&adj_graph.adj_graph[vertex].weighted_neighbors);
            //we are always after the neighbors, unless we did not like any
            for (i,&(start_neighbor,weight,itype)) in neighbors.iter().enumerate() {
                start_agreement += weight * if itype==IntervalType::Start { 0.0 } else { -1.0 };
                let mut agreement:f64 = start_agreement;
                for &(end_neighbor,end_weight,end_itype) in neighbors.iter().skip(i) {
                    agreement += end_weight * if end_itype==IntervalType::Start { 1.0 } else { 0.0 };
                    if  agreement > max_agreement {
                        max_agreement = agreement;
                        best_neighbors = (
                            VertexInterval {id: start_neighbor,itype},
                            VertexInterval {id: end_neighbor,itype:end_itype});
                    }
                }
            }
            vertex_ordering.reinsert_after(vertex, best_neighbors);
            // println!("vertex: {}, new disagreement: {}", vertex, signed_graph.edges.len() as f64 - ordering_agreement(&adj_graph,&vertex_ordering));
                            
        }
        //TODO calc and print agreement
        println!("{}",ordering_agreement(&adj_graph,&vertex_ordering));
    }
    vertex_ordering.to_vec()
    
}


fn ordering_agreement(graph: &SignedAdjacencyList, vertex_ordering :&OrderedList) -> f64 {
    let mut active_vertices: HashSet<usize> = HashSet::new();
    let mut dead_vertices: HashSet<usize> = HashSet::new();
    let mut agreement = 0.0;
    assert_eq!(graph.adj_graph.len()*2,vertex_ordering.len());
    for (id,itype) in vertex_ordering.to_vec() {
        if itype == IntervalType::Start {
            active_vertices.insert(id);
            for (&neighbor_id,&weight) in &graph.adj_graph[id].weighted_neighbors {
                if  active_vertices.contains(&neighbor_id) && weight > 0.0 {
                    agreement += weight;
                }
            }
        } else {
            active_vertices.remove(&id);
            for (&neighbor_id,&weight) in &graph.adj_graph[id].weighted_neighbors {
                if !active_vertices.contains(&neighbor_id) && !dead_vertices.contains(&neighbor_id) && weight < 0.0 {
                    agreement -= weight;
                }
            }
            dead_vertices.insert(id);
        }
    }
    agreement
}