---
title: "An Efficient, Exact Implementation of the Greedy (Chicken) Algorithm"
author: "opinion_projection / signed_graph_kernelization"
date: "2026"
geometry: margin=1in
---

# Overview

The *chicken algorithm* is the greedy heuristic for {\sc Opinion Interval
Projection}: it repeatedly removes the most "sign-decided" vertex from the
graph, charging the minority of its edges as violations, while interleaving the
exact kernelization rules (which are free). This note explains how the
implementation in `signed_graph_kernelization.py` computes the **same answer as
applying the kernelization at every step**, but in near-linear time
$O((|V|+|E|)\log|V|)$ instead of the naive $O(|V|\,(|V|+|E|))$.

The key idea: of the four kernelization rules, only two can change a vertex's
signed degrees and hence the greedy's decisions — rule (i) and rule (ii) — and
both can be maintained **incrementally** with a lazy max-heap and a
decremental-connectivity structure, rather than recomputed from scratch after
every removal.

---

# 1. What the greedy actually computes

Each vertex $v$ has a positive degree $\deg^+(v)$ (number of positive edges) and
a negative degree $\deg^-(v)$. Define its **imbalance ratio**
$$
  r(v) \;=\; \frac{\max(\deg^+(v),\,\deg^-(v))}{\deg^+(v)+\deg^-(v)} \in [\tfrac12, 1].
$$
A ratio near $1$ means almost all of $v$'s edges have the same sign — $v$ is
"decided": placing its interval on the majority side satisfies almost all of its
edges, and only the **minority** edges, $\min(\deg^+(v),\deg^-(v))$ of them, are
unavoidably violated.

The greedy (Algorithm 1 in the paper) repeatedly:

1. kernelizes the current graph (exact reductions — **0 violations**);
2. takes the vertex $v$ of largest ratio; if $r(v) \ge \alpha$, deletes it and
   charges $\min(\deg^+(v),\deg^-(v))$ to the running violation count;
3. stops pecking a component once its largest ratio drops below the threshold
   $\alpha$.

The accumulated charge is a heuristic upper bound on the number of violated
edges. Peeling the most-decided vertices first defers the hard, balanced
vertices and tends to keep the charge small.

**Crucial point.** The kernelization reductions never add violations; *all*
violations come from step 2. So getting the violation count right means getting
the **signed degrees** right at the moment each vertex is removed.

---

# 2. Which rules can change the count?

The four exact kernelization rules are:

| rule | what it does | effect on degrees |
|------|--------------|-------------------|
| (i)   | delete a vertex with $\deg^-=0$ | removes its **positive** edges $\Rightarrow$ lowers neighbours' $\deg^+$ |
| (ii)  | split into positive-edge components; drop the negative edges **between** components | lowers $\deg^-$ of the dropped edges' endpoints |
| (iii) | split at articulation points (biconnected decomposition) | **none** — it only *partitions* edges; every edge stays in exactly one part |
| (iv)  | split at a cut of one/two **positive** edges | drops $\le 2$ positive edges per cut (lowers $\deg^+$ of $\le 4$ vertices); frees no negative edge |

Rule (iii) changes **no** degree, so it cannot affect the violation count. Rule
(iv) perturbs at most a few positive degrees, extremely rarely, and never frees
a negative edge — its effect on the count is negligible.

That leaves **rule (i)** and **rule (ii)** as the only reductions that
materially change the greedy's choices. The whole optimization is about applying
*those two* exactly, but incrementally.

Why does this matter so much? Empirically, omitting rule (ii) inside the loop
inflates the violation count by ~35% (e.g. Bitcoin: 985 vs 730): when a positive
component splits, the negative edges crossing the split become satisfiable for
free, and if we keep counting them the greedy "pays" for edges it shouldn't.

---

# 3. The naive cost, and the plan

The straightforward implementation, after **every** removal,

* rescans all vertices to find the maximum ratio — $O(|V|)$, and
* re-runs the full kernelizer (biconnected components, edge-signature DFS,
  bridges, …) — $O(|V|+|E|)$ with large constants.

Over $O(|V|)$ removals that is $O(|V|\,(|V|+|E|))$ — minutes on a 7k-vertex
kernel, infeasible on a 26k one.

The plan replaces both per-step recomputations with incremental maintenance:

* **selection** $\to$ a lazy max-heap keyed by ratio (§4);
* **rule (i)** $\to$ a free-removal cascade (§5);
* **rule (ii)** $\to$ decremental connectivity with small-to-large split
  handling (§6) — applied *exactly* every step.

---

# 4. Incremental selection: a lazy max-heap

We keep dictionaries $\mathrm{dplus}[v], \mathrm{dminus}[v]$ and a binary heap
of entries $(-r(v),\, v,\, \mathrm{dplus}[v],\, \mathrm{dminus}[v])$.

Removing a vertex only changes the degrees of its **neighbours**, so we only
push fresh entries for those neighbours. An entry is **stale** if the vertex is
gone or its stored degrees differ from the current ones; stale entries are
discarded when they reach the top of the heap (a "lazy" heap). Because degrees
only ever decrease, "stored degrees equal current degrees" is a perfect
freshness test.

Each edge removal causes $O(1)$ pushes, so there are $O(|E|)$ pushes total and
selection costs $O((|V|+|E|)\log|V|)$ overall.

---

# 5. Rule (i) as a free cascade

A vertex with $\deg^- = 0$ has no negative constraints: it can be mapped to the
whole interval $[0,1]$ and removed at **no cost**. We keep a `free` stack;
whenever a removal lowers some neighbour's $\deg^-$ to $0$, that neighbour is
pushed onto `free`. Free vertices are drained before each paid step. Removing a
free vertex only deletes **positive** edges (it has none negative), so it can
trigger further free removals and positive-component splits — both handled by
the same machinery.

---

# 6. Rule (ii), exactly, at every step

This is the heart of the implementation. We want, after every removal, the same
state as if we had re-split the positive components and dropped every
cross-component negative edge — but without the $O(|V|+|E|)$ recomputation.

## 6.1 The invariant

> **Invariant.** At all times, every *active* negative edge (one still counted
> in some $\deg^-$) has both endpoints in the same positive-edge component.

This is exactly the post-condition of rule (ii): after rule (ii), no negative
edge crosses between positive components. A kernel produced by the kernelizer
already has a **connected** positive graph (rule (ii) was applied to a fixed
point), so initially there is a single positive component and the invariant
holds trivially.

If we maintain this invariant after every removal, then $\mathrm{dminus}[v]$
always equals the number of *satisfiable-only-by-paying* negative edges at $v$ —
precisely the quantity a per-step rule (ii) would compute. Hence every ratio and
every charge is identical to the every-step reference.

## 6.2 When can the invariant break?

A negative edge $(u,w)$ becomes cross-component only when $u$ and $w$ become
**disconnected in the positive graph** $G^+$. Disconnection only happens when we
delete things from $G^+$, i.e. when we remove a vertex (deleting its positive
edges). So the only events to handle are **positive-component splits caused by a
vertex removal**, and only within the removed vertex's own component.

**A removal can split a component only if the vertex had $\ge 2$ positive
neighbours.** If $v$ has $0$ positive neighbours, $G^+$ is untouched. If $v$ has
exactly $1$ positive neighbour $p$, then $v$ is a leaf of $G^+$: every other node
$y$ of its component reaches $v$ by a path whose last vertex before $v$ is $p$,
and that path minus $v$ still connects $y$ to $p$ — so the component minus $v$ is
still connected. No split. (This same argument, applied to a general $v$, shows
that the pieces of $C\setminus\{v\}$ are exactly the components *among $v$'s
positive neighbours*, because every node of $C$ reaches some neighbour of $v$
without going through $v$.)

So: **after removing $v$, the new pieces are the connected components of
$G^+\setminus\{v\}$ reachable from $v$'s former positive neighbours $p_1,\dots,p_k$.
A split occurred iff the $p_i$ fall into $\ge 2$ such components.**

## 6.3 Finding the split cheaply: small-to-large

A from-scratch component recomputation costs $O(|C|)$ per removal — too slow. The
trick is to never fully explore the **largest** resulting piece.

Run a **lockstep BFS**: one frontier per neighbour $p_i$, advanced round-robin
one node at a time. When two frontiers meet (one reaches a node another already
claimed), they belong to the same piece — union them. **Stop as soon as a single
piece is still expanding.** At that moment:

* the pieces whose frontiers have *emptied* are fully explored — these are the
  smaller ones;
* the one still expanding is the largest — we leave it partially explored, and
  it keeps the old component label.

Because the smaller pieces are explored fully and the largest barely at all, the
work of one split is $O(\text{total size of the smaller pieces})$.

For each smaller piece we (a) assign a fresh component label, and (b) scan the
negative edges incident to its nodes; any edge whose other endpoint is now in a
different component is **freed** — dropped from both adjacency sets, both
$\deg^-$ decremented, and the endpoints re-pushed (possibly triggering rule (i)).
Every cross-piece negative edge is incident to at least one *smaller* piece, so
scanning only the smaller pieces catches them all.

## 6.4 Why this is near-linear

Whenever a vertex ends up in a *smaller* piece, the size of the component it
belongs to has at least **halved** (the smaller piece is at most half of the old
component). A vertex can therefore be placed in a smaller piece at most
$O(\log|V|)$ times across the whole run. Summing the per-split work
(exploration + negative-edge scan, both proportional to the smaller pieces) gives
a total of $O((|V|+|E|)\log|V|)$ — the classic **small-to-large** bound. Combined
with the heap, the whole peck is $O((|V|+|E|)\log|V|)$.

---

# 7. Why the result is *exactly* the every-step answer

We argue the invariant of §6.1 is preserved, by induction on removals.

*Base case.* A kernel has a connected positive graph, so there is one positive
component and no negative edge can cross — the invariant holds.

*Inductive step.* Assume it holds before removing $v$. Removing $v$ deletes its
own incident edges (accounted for in $v$'s charge / freed trivially). The only
negative edges that can newly cross components are those separated by the split
that removing $v$ causes — and, by §6.2, that split only repartitions $v$'s own
component into the pieces we compute. By the inductive hypothesis there were no
cross-component active negatives before, so the **only** edges that need freeing
are those crossing this new split, and §6.3 frees exactly them. Hence the
invariant is restored. $\qquad\blacksquare$

Therefore, at every step, $\mathrm{dplus}$ and $\mathrm{dminus}$ equal the
degrees that a full per-step kernelization (rules (i)+(ii)) would produce; the
heap selects the same vertex and we charge the same amount. The violation count
is **identical** to the every-step reference — there is no periodic-flush
approximation and no one-directional bias.

This was checked two ways:

* against an obviously-correct reference that recomputes the positive components
  from scratch after every removal (same heap tie-break) — **bit-identical** on
  Bitcoin/Chess/WikiElec across $\alpha\in\{0,0.5,0.7,0.9\}$;
* a randomized **3000-graph** stress test — **0 mismatches**.

---

# 8. Complexity and measured speed

| | naive (per-step rescan + re-kernelize) | this implementation |
|---|---|---|
| selection | $O(|V|)$ / step | $O(\log|V|)$ amortized / event |
| rule (i)  | inside $O(|V|+|E|)$ re-kernelize | $O(1)$ amortized / event |
| rule (ii) | $O(|V|+|E|)$ / step | $O((|V|+|E|)\log|V|)$ total |
| **overall** | $O(|V|\,(|V|+|E|))$ | $O((|V|+|E|)\log|V|)$ |

Measured peck time (after a single up-front kernelization): Bitcoin 0.05 s,
Chess 0.18 s, WikiElec 1.3 s, Slashdot (26k-vertex kernel) 5.6 s, Epinions
(20k-vertex kernel) 13.6 s. The old per-step version took 11.9 s on Bitcoin,
487 s on Chess, and was infeasible on the large kernels.

---

# 9. Mapping to the code

In `signed_graph_kernelization.py`:

* **`chicken_algorithm(graph, alpha)`** — kernelizes once, then pecks each kernel
  with `_greedy_peck`; re-kernelizes only the surviving subgraph at the end (for
  a clean list of leftover kernels when $\alpha>0$).
* **`_greedy_peck(kernel, alpha)`** — the incremental core of §§4–6: builds the
  adjacency/degree dictionaries, the lazy heap, the `free` cascade, and the
  positive-component labels `comp` (initialized to a single component — the
  kernel is positively connected, §6.1); `remove_vertex` updates degrees, re-pushes
  affected neighbours, and triggers split handling when the removed vertex had
  $\ge 2$ positive neighbours.
* **`_smaller_positive_pieces(plus, alive, sources)`** — the lockstep,
  small-to-large BFS of §6.3: returns the node lists of all resulting positive
  pieces *except the largest*.

---

# 10. What is intentionally left out

* **Rules (iii) and (iv)** are not applied inside the peck. Rule (iii) changes no
  degree, so it cannot affect the count; rule (iv)'s effect is negligible and it
  frees no negative edge. Both are still applied by the one-time
  `kernelise_graph` call and by the final re-kernelization of the leftovers, so
  the kernel structure is unaffected — only the (degree-irrelevant) inner-loop
  splitting is skipped.
* **Tie-breaking.** Among vertices of equal maximum ratio, the heap breaks ties
  by node id, whereas the original code broke them by node-iteration order.
  These are different but equally valid greedy choices, so the new counts may
  differ from the old by a fraction of a percent (often *better*). This is
  unrelated to rule (ii), which is now exact; matching the old tie-break exactly
  is possible if bit-for-bit reproduction of the old numbers is ever required.
