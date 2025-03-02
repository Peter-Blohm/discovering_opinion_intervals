import networkx as nx
import numpy as np
import torch

from graph_utils.signed_graph import SignedGraph


def optimize_via_gd_but_like_faster(starts: torch.Tensor, targets: torch.Tensor, graph: SignedGraph,
                    lr = 0.01, iterations = 500, verbose_iterations = 50):
    """
    Optimize the interval embedding via gradient descent.

    :param starts: vector for interval starts of length n
    :param targets: vector for interval ends of length n
    :param graph: signed graph object
    :param lr: learn rate
    :param iterations:
    :param verbose_iterations: interval for printing progress, 0 for no printing
    :return: nothing
    """
    n_neg = graph.G_minus.number_of_nodes()
    starts.requires_grad = True
    targets.requires_grad = True

    optimizer = torch.optim.SGD([starts, targets], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    adjacency = adjacency_matrix(graph)
    adjacency.requires_grad = True
    for i in range(iterations):
        optimizer.zero_grad()
        dists = distance_matrix(starts,targets)
        loss = torch.sum(torch.sigmoid(dists*adjacency)) - n_neg

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            targets.copy_(torch.max(targets, starts + 1e-5)) #hacky projected gradient descent
        if verbose_iterations and i % verbose_iterations == 0:
            print(f"Iteration {i}: Loss = {loss.item()}, Violations = {count_violations(starts, targets, graph)}")

def distance_matrix(x1: torch.Tensor,x2: torch.Tensor) -> torch.Tensor:
    return x1[:,None] - x2[None, :]

def adjacency_matrix(graph: SignedGraph) -> torch.Tensor:
    """
    This function returns a signed upper triangular adjacency matrix of the graph
    :param graph: the SignedGraph object.
    :return: adjacency matrix of the graph.
    """
    return (torch.Tensor(nx.adjacency_matrix(graph.G_plus).toarray()) -
            torch.Tensor(nx.adjacency_matrix(graph.G_minus).toarray()))

def count_violations(starts: torch.Tensor, targets: torch.Tensor, graph: SignedGraph):
    dists = distance_matrix(starts, targets) # starts-targets, should be <0 if negative edge, >0 for positive
    adjacency = adjacency_matrix(graph) #is 1 for plus edge, -1 for neg edge
    # violations = 0
    # for v in graph.G_plus.nodes():
    #     for w in graph.G_plus.neighbors(v):
    #         if starts[v] > targets[w]:
    #             violations += 1
    #     for w in graph.G_minus.neighbors(v):
    #         if targets[v] >= starts[w]:
    #             violations += 1
    # print("violations:" + str(violations-graph.G_minus.number_of_edges()))
    return torch.sum(torch.sign(adjacency * dists) == 1) - graph.G_minus.number_of_edges()


def permutation_initial_solution(graph: SignedGraph, permutation: np.ndarray)-> (torch.Tensor, torch.Tensor):
    """
    generates random interval orders
    :param permutation:
    :param graph:
    :return:
    """
    starts = 2*torch.Tensor(permutation)
    targets = 2*torch.Tensor(permutation)+3
    return starts, targets

def central_initial_solution(graph: SignedGraph, sigma = 1)-> (torch.Tensor, torch.Tensor):
    """
    generates interval bounds that are central and overlap, with some gaussian noise
    :param sigma: variance of the gaussian noise, #TODO do not make it too large, no checks
    :param graph:
    :return: vectors for start and target
    """
    n = graph.number_of_nodes()
    starts = torch.ones(n)*n/2 + torch.randn(n)*sigma*n/2
    targets = torch.ones(n)*n/2*3 + torch.randn(n)*sigma*n/2
    return starts, targets