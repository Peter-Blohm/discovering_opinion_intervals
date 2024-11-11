import os
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

def load_graph_images(okay_dir, bad_dir):
    """
    Loads the paths of graph images from the 'okay' and 'bad' directories,
    labels them accordingly, and combines them into a single list.
    """
    # Load embeddable graphs
    okay_images = [
        (os.path.join(okay_dir, img), True)
        for img in os.listdir(okay_dir)
        if img.endswith('.png')
    ]
    
    # Load non-embeddable graphs
    bad_images = [
        (os.path.join(bad_dir, img), False)
        for img in os.listdir(bad_dir)
        if img.endswith('.png')
    ]
    
    # Combine and shuffle
    all_images = okay_images[:len(bad_images)] + bad_images
    random.shuffle(all_images)
    
    return all_images

def display_graph(image_path):
    """
    Displays the graph image to the player.
    """
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

def get_player_guess():
    """
    Prompts the player for their guess and validates the input.
    """
    while True:
        guess = input("Is the graph embeddable? (yes/no): ").strip().lower()
        if guess in ['yes', 'no', 'y', 'n']:
            return guess in ['yes', 'y']
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def play_game(okay_dir, bad_dir):
    """
    Main game loop. Presents graphs to the player and updates the score based on their guesses.
    """
    images = load_graph_images(okay_dir, bad_dir)
    score = 0
    total = len(images)
    question_number = 1

    print("\nWelcome to the Graph Embeddability Game!")
    print("You will be shown a series of graphs.")
    print("For each graph, decide whether it is embeddable.")
    print("Type 'yes' if you think it is embeddable, or 'no' if you think it is not.\n")

    for image_path, is_embeddable in images:
        print(f"Graph {question_number} of {total}")
        display_graph(image_path)
        player_guess = get_player_guess()
        
        if player_guess == is_embeddable:
            print("Correct!\n")
            score += 1  # You can adjust the points as needed
        else:
            print("Incorrect.\n")
            score -= 1  # You can adjust the penalty as needed
        
        question_number += 1

    print("Game Over!")
    print(f"Your total score is: {score} out of {total}")
    if score == total:
        print("Excellent job! You got all of them correct.")
    elif score > 0:
        print("Good effort! Keep practicing to improve your score.")
    else:
        print("Don't worry, it was a tough game. Try again to improve your score.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the Graph Embeddability Game.")
    parser.add_argument("--okay_dir", type=str, default="data/signed_graphs_9_nodes_ok", help="Directory containing embeddable graph images.")
    parser.add_argument("--bad_dir", type=str, default="data/signed_graphs_9_nodes_bad", help="Directory containing non-embeddable graph images.")
    args = parser.parse_args()
    
    # Ensure directories exist
    if not os.path.exists(args.okay_dir) or not os.path.exists(args.bad_dir):
        print("Error: One or both of the specified directories do not exist.")
        print("Please ensure that the 'okay' and 'bad' directories exist and contain graph images.")
    else:
        play_game(args.okay_dir, args.bad_dir)
