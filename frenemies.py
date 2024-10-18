import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import cvxpy as cp
import random
import itertools

def sdp_embedding(friends, enemies, n_vertices, k=2):
    X = cp.Variable((n_vertices, n_vertices), symmetric=True)

    friend_term = sum(X[i, i] + X[j, j] - 2 * X[i, j] for i, j in friends)
    enemy_term = sum(X[i, i] + X[j, j] - 2 * X[i, j] for i, j in enemies)

    objective = cp.Maximize(enemy_term - friend_term)

    constraints = [
        X >> 0,  # X must be positive semidefinite
        cp.diag(X) == 1  # Diagonal entries must be 1 (unit norm constraint)
    ]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status == cp.OPTIMAL:
        _, eigvecs = np.linalg.eigh(X.value)
        if k == 1:
            embedding = eigvecs[:, -1]
        else:
            embedding = eigvecs[:, -2:]
    else:
        print(f"Problem status: {prob.status}")
        return None

    return embedding

def generate_random_edges(n_vertices, n_friends, n_enemies):
    all_edges = [(i, j) for i in range(n_vertices) for j in range(i+1, n_vertices)]
    random.shuffle(all_edges)
    return all_edges[:n_friends], all_edges[n_friends:n_friends+n_enemies]

def visualize_embedding_2d(embedding, friends, enemies):
    plt.figure(figsize=(12, 8))
    
    # Plot vertices
    plt.scatter(embedding[:, 0], embedding[:, 1], c='gray')
    
    # Plot friend edges as lines
    for i, j in friends:
        plt.plot(embedding[[i, j], 0], embedding[[i, j], 1], 'g-', alpha=0.5)

    # Plot enemy edges as lines
    for i, j in enemies:
        plt.plot(embedding[[i, j], 0], embedding[[i, j], 1], 'r-', alpha=0.5)

    plt.title("2D Embedding of Friend-Enemy Graph")
    plt.xlabel("First Eigenvector")
    plt.ylabel("Second Eigenvector")
    plt.show()

def visualize_embedding_1d(embedding, friends, enemies):
    plt.figure(figsize=(12, 4))

    # Plot vertices as points on a horizontal line
    plt.scatter(embedding, np.zeros_like(embedding), c='gray', zorder=2)

    # Plot friend edges as arcs above the line
    for i, j in friends:
        mid = (embedding[i] + embedding[j]) / 2
        height = abs(embedding[i] - embedding[j]) / 2
        arc = Arc([mid, height], width=abs(embedding[i] - embedding[j]), height=height * 2, angle=0, theta1=0, theta2=180, color='g', alpha=0.5, zorder=1)
        plt.gca().add_patch(arc)

    # Plot enemy edges as arcs below the line
    for i, j in enemies:
        mid = (embedding[i] + embedding[j]) / 2
        height = abs(embedding[i] - embedding[j]) / 2
        arc = Arc([mid, -height], width=abs(embedding[i] - embedding[j]), height=height * 2, angle=0, theta1=180, theta2=360, color='r', alpha=0.5, zorder=1)
        plt.gca().add_patch(arc)

    plt.title("1D Embedding of Friend-Enemy Graph")
    plt.xlabel("Embedding (1D)")
    plt.yticks([])
    plt.show()

def evaluate_embedding(embedding, friends, enemies):
    friend_distances = [np.linalg.norm(embedding[i] - embedding[j]) for i, j in friends]
    enemy_distances = [np.linalg.norm(embedding[i] - embedding[j]) for i, j in enemies]
    
    avg_friend_distance = np.mean(friend_distances)
    avg_enemy_distance = np.mean(enemy_distances)
    
    # Initialize counters
    vertices_satisfying_condition = 0
    
    # Dictionary to hold distances for each vertex
    friend_dist_per_vertex = {i: [] for i in range(embedding.shape[0])}
    enemy_dist_per_vertex = {i: [] for i in range(embedding.shape[0])}

    # Populate the distance dictionaries
    for i, j in friends:
        dist = np.linalg.norm(embedding[i] - embedding[j])
        friend_dist_per_vertex[i].append(dist)
        friend_dist_per_vertex[j].append(dist)
    
    for i, j in enemies:
        dist = np.linalg.norm(embedding[i] - embedding[j])
        enemy_dist_per_vertex[i].append(dist)
        enemy_dist_per_vertex[j].append(dist)

    # Check for each vertex if all friends are closer than all enemies
    for i in range(embedding.shape[0]):
        if friend_dist_per_vertex[i] and enemy_dist_per_vertex[i]:  # Ensure vertex has both friends and enemies
            max_friend_dist = max(friend_dist_per_vertex[i])
            min_enemy_dist = min(enemy_dist_per_vertex[i])
            if max_friend_dist < min_enemy_dist:
                vertices_satisfying_condition += 1

    proportion_satisfying_condition = vertices_satisfying_condition / embedding.shape[0]
    
    print(f"Average friend distance: {avg_friend_distance:.4f}")
    print(f"Average enemy distance: {avg_enemy_distance:.4f}")
    print(f"Ratio (enemy/friend): {avg_enemy_distance / avg_friend_distance:.4f}")
    print(f"Number of vertices where all friends are closer than enemies: {vertices_satisfying_condition}")
    print(f"Proportion of such vertices: {proportion_satisfying_condition:.4f}")


def check_condition_for_ordering(ordering, friends, enemies):
    vertices_satisfying_condition = 0
    n = len(ordering)

    for i in range(n):
        vertex = ordering[i]

        # Find the closest friend and enemy on the left side (smaller index)
        closest_friend_left = closest_enemy_left = None
        for left in range(i - 1, -1, -1):
            if (ordering[left], vertex) in friends or (vertex, ordering[left]) in friends:
                closest_friend_left = left
                break
            elif (ordering[left], vertex) in enemies or (vertex, ordering[left]) in enemies:
                closest_enemy_left = left
                break

        # Find the closest friend and enemy on the right side (larger index)
        closest_friend_right = closest_enemy_right = None
        for right in range(i + 1, n):
            if (ordering[right], vertex) in friends or (vertex, ordering[right]) in friends:
                closest_friend_right = right
                break
            elif (ordering[right], vertex) in enemies or (vertex, ordering[right]) in enemies:
                closest_enemy_right = right
                break

        is_valid = True

        # Left side: friend must be closer than any enemy
        if closest_friend_left is not None and closest_enemy_left is not None:
            if closest_friend_left > closest_enemy_left:
                is_valid = False

        # Right side: friend must be closer than any enemy
        if closest_friend_right is not None and closest_enemy_right is not None:
            if closest_friend_right < closest_enemy_right:
                is_valid = False

        if is_valid:
            vertices_satisfying_condition += 1

    return vertices_satisfying_condition

def brute_force_maximum_satisfied_vertices(friends, enemies, n_vertices):
    max_satisfied = 0

    all_orderings = itertools.permutations(range(n_vertices))

    for ordering in all_orderings:
        satisfied = check_condition_for_ordering(ordering, friends, enemies)
        max_satisfied = max(max_satisfied, satisfied)

    return max_satisfied

n_vertices = 10
n_friends = 10
n_enemies = 10

friends, enemies = generate_random_edges(n_vertices, n_friends, n_enemies)

embedding_2d_sdp = sdp_embedding(friends, enemies, n_vertices, k=2)
if embedding_2d_sdp is not None:
    evaluate_embedding(embedding_2d_sdp, friends, enemies)
    visualize_embedding_2d(embedding_2d_sdp, friends, enemies)

embedding_1d_sdp = sdp_embedding(friends, enemies, n_vertices, k=1)
if embedding_1d_sdp is not None:
    evaluate_embedding(embedding_1d_sdp, friends, enemies)  
    visualize_embedding_1d(embedding_1d_sdp, friends, enemies)

max_satisfied = brute_force_maximum_satisfied_vertices(friends, enemies, n_vertices)
print(f"Maximum number of vertices where all friends are closer than enemies: {max_satisfied}")
