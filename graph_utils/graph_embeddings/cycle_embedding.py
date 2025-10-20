from networkx.classes import is_empty
import numpy as np
from graph_utils.signed_graph import SignedGraph


def decrement(k,n):
    return (k + n - 1) % n


def increment(j,n):
    return (j + 1) % n

def edge(u,v):
    return min(u, v), max(u, v)
# embeddability

def check_cycle_embeddable(graph: SignedGraph) -> bool:
    """
    Takes as input a signed graph that forms a cycle with the positive edges.
    Important! the vertices assume to be ordered such that (i,i+1 mod n) are neighbors

    Then uses a cubic algorithm that checks the existence of a traversal
    :param graph: +cycle graph
    :return: whether the graph is embeddable
    """
    cycle = list(graph.G_plus.nodes())
    n = len(cycle)
    edges = np.zeros((n,n))
    r,c = zip(*graph.G_plus.edges())
    edges[r,c] = 1
    r,c = zip(*graph.G_minus.edges())
    edges[r,c] = -1

    if n*(n-1)/2 - graph.number_of_edges() < n - 3: # TODO check!
        return False
    blacklist_points = np.zeros((n,n))
    for i in range(0,n): #for each vertex, try to find a traversal
        j = increment(i,n)
        k = decrement(i,n)
        if edges[edge(i,k)] == -1: #skip if neighbors have a negative edge
            continue
        backtracking_points = []
        while True: #try to traverse

            #check in which direction a traversal is possible
            j_open = edges[edge(increment(j,n),k)] > -1
            k_open = edges[edge(j,decrement(k,n))] > -1
            if blacklist_points[j,k] != 0:
                j_open, k_open = False, False
            if j == k: #traversal finished
                return True
            if k_open:
                if j_open:
                    backtracking_points.append((increment(j,n),k)) #choice, add point to backtrack to
                k=decrement(k,n)
                continue
            if j_open:
                j=increment(j,n)
                continue
            #neither open, backtrack if possible
            if backtracking_points:
                j,k = backtracking_points.pop()
                blacklist_points[decrement(j,n),k] = 1
                continue
            break
    return False

