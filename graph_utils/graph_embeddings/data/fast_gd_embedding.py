import networkx as nx
import numpy as np
import torch

from graph_utils.signed_graph import SignedGraph


def optimize_via_gd_but_like_faster(starts: torch.Tensor, targets: torch.Tensor, graph: SignedGraph,
                    lr = 0.01, iterations = 500, verbose_iterations = 50, **kwargs):
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
    n_neg = graph.G_minus.number_of_edges()
    starts.requires_grad = True
    targets.requires_grad = True

    optimizer = torch.optim.SGD([starts, targets], lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)

    adjacency = adjacency_matrix(graph)
    adjacency.requires_grad = True
    for i in range(iterations):
        optimizer.zero_grad()
        dists = distance_matrix(starts,targets)
        # loss = torch.sum(torch.nn.functional.leaky_relu(dists*adjacency)) - n_neg
        loss = torch.sum(torch.sigmoid(dists*adjacency)) - n_neg

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            starts.copy_(torch.min(starts, targets - 1e-5)) #hacky projected gradient descent

            # # Combine starts and targets for joint normalization
            # all_vals = torch.cat([starts, targets])
            # min_val = all_vals.min()
            # max_val = all_vals.max()

            # # Prevent division by zero if all values are the same
            # if max_val > min_val:
            #     scale = 4.0 / (max_val - min_val)  # 4 = 5 - 1
            #     offset = 1.0 - min_val * scale
            #     starts.copy_(starts * scale + offset)
            #     targets.copy_(targets * scale + offset)
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
    starts = torch.ones(n) + torch.randn(n)*sigma
    targets = torch.ones(n)*3 + torch.randn(n)*sigma
    return starts, targets