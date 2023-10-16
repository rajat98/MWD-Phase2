import numpy as np
import pandas as pd
from random import seed
from random import randint
import pickle
from pathlib import Path
import torch
import cv2
import torchvision
from IPython.core.display_functions import display
from torchvision import datasets, models, transforms
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import re, os, pickle


def save_similarity_matrix(similarity_matrix, file_path):
    """Save the image similarity matrix to a CSV file"""
    similarity_matrix_storage = {'image_similarity_matrix': similarity_matrix}
    # Ensure that the directory path exists, creating it if necessary
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(similarity_matrix_storage, file)


dataset = torchvision.datasets.Caltech101('/Users/maryamcheema/Documents/Phase2/caltech-101', download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)


def create_image_similarity_matrix(data):
    """Calculate image-image similarity matrix based on mean vectors
    Return a square matrix representing label similarities"""
    similarity_matrix = np.zeros((len(data), len(data)))

    for i in tqdm(range(len(data))):
        image1_vector = data[i]
        for j in range(i, len(data)):
            image2_vector = data[j]
            similarity_score = cosine_similarity([image1_vector], [image2_vector])[0][0]
            # Fill both upper and lower triangles of the similarity matrix
            similarity_matrix[i, j] = similarity_score
            similarity_matrix[j, i] = similarity_score
    return similarity_matrix


def load_latent_semantics(file_path):
    """Load the latent semantics from the saved file."""
    with open(file_path, 'rb') as file:
        similarity_matrix = pickle.load(file)
    return similarity_matrix


'''n stands for the number of n most similar images '''


def create_similarity_graph(similarity_matrix, n):
    graph = []
    adjacency_matrix = np.zeros((len(similarity_matrix), len(similarity_matrix)))

    for i in range(len(similarity_matrix)):
        # converting to a numpy array
        row_array = similarity_matrix[i]
        similar_images = np.argsort(row_array)[::-1]
        similar_images_scores = row_array[similar_images]

        # getting the image ids aswell as weights of the n_most similar images
        n_similar_images = similar_images[1:n + 1]
        n_similar_scores = similar_images_scores[1:n + 1]
        graph.append([n_similar_images, n_similar_scores])
        adjacency_matrix[i][n_similar_images] = 1
    return graph, adjacency_matrix


def print_graph(graph):
    for i in range(len(graph)):
        vertices = graph[i][0]
        print(f"Node pairs for image {i}")
        for j in range(len(vertices)):
            print(f"({i},{vertices[j]})")


def personalized_pagerank(graph, m, n, num_iterations: int = 100, d: float = 0.85):
    prob_scores = np.ones(len(graph)) / len(graph)
    old_scores = np.ones(len(graph)) / len(graph)
    random_seed_index = randint(0, len(graph))
    prob_scores[random_seed_index] = 1.0

    for _ in range(num_iterations):
        old_scores[:] = prob_scores
        for i in range(len(graph)):

            image_edges = graph[i]

            if image_edges[random_seed_index] == 1:
                prob_scores[i] = (1 - d) + d * prob_scores[random_seed_index]
            else:
                neighbors = np.nonzero(graph[i])[0]
                sum = 0
                for neighbor in neighbors:
                    sum += prob_scores[neighbor] / n
                prob_scores[i] = (1 - d) * sum
    sorted_nodes = np.argsort(prob_scores)[::-1]
    return sorted_nodes


def most_significant_images(label, nodes, objects, m):
    most_sig_img_ids = []
    for i in range(len(nodes)):
        if objects[nodes[i]] == label:
            most_sig_img_ids.append(nodes[i])
        if len(most_sig_img_ids) == m:
            break
    return most_sig_img_ids


def print_menu():
    feature_model_dict = {
        1: "HOG",
        2: "CM",
        3: "L3",
        4: "AvgPool",
        5: "FC",
        6: "RESNET"
    }

    print("Enter the feature model")
    for k, v in feature_model_dict.items():
        print(str(k) + '.', v)
    feature_model = int(input("Feature model selected: "))

    m = int(input("Enter the value of m: "))

    n = int(input("Enter the value of n: "))

    label = int(input("Please enter the label(0-100): "))

    feature_or_latent = int(
        input("Press 1 if you want to use a feature model, press 2 if you want to use a latent space"))
    if feature_or_latent == 1:
        feature_model_similarity_path = f"Outputs/similarity_matrices/{feature_model_dict[feature_model]}_image_similarity.pkl"
        #         print("this is my path ", feature_model_similarity_path)
        similarity_matrix = load_latent_semantics(feature_model_similarity_path)
        return similarity_matrix, m, n, label

    if feature_or_latent == 2:
        k = int(input("Enter the value of k: "))
        method_dict = {1: 'SVD', 2: 'NNMF', 3: 'LDA', 4: 'kmeans'}
        print('Choose dimensionality reduction technique')
        for a, v in method_dict.items():
            print(str(a) + '.', v)
        method = int(input("Choose dimensionality reduction selected: "))
        task = int(input("Please enter the task number(3, 4, 5, 6): "))
        calculate_similarity = input("Does a new similarity matrix need to be computed, type y for yes: ")
        if calculate_similarity == 'y':
            latent_feature_storage_path = f"Outputs/T{task}/{feature_model_dict[feature_model]}/{method_dict[method]}_{k}.pkl"
            semantics = load_latent_semantics(latent_feature_storage_path)
            similarity_matrix = create_image_similarity_matrix(semantics['image_to_latent_features'])
            save_similarity_matrix(similarity_matrix,
                                   f"Outputs/similarity_matrices/T{task}_{feature_model_dict[feature_model]}_{method_dict[method]}_{k}.pkl")
            new_similarity_matrix = load_latent_semantics(
                f"Outputs/similarity_matrices/T{task}_{feature_model_dict[feature_model]}_{method_dict[method]}_{k}.pkl")
            return new_similarity_matrix, m, n, label

        else:
            similarity_matrix = load_latent_semantics(
                f"Outputs/similarity_matrices/T{task}_{feature_model_dict[feature_model]}_{method_dict[method]}_{k}.pkl")
            return similarity_matrix, m, n, label


def driver():
    similarity_matrix, m, n, label = print_menu()
    graph, adjacency_matrix = create_similarity_graph(similarity_matrix['image_similarity_matrix'], n)
    sorted_nodes = personalized_pagerank(adjacency_matrix, m, n)

    df = pd.read_csv('FD_Objects.csv')
    data = df['Labels']
    len(data)
    most_sig_img_ids = most_significant_images(label, sorted_nodes, data, m)

    print(f"Here are the most significant images for the label {label}")
    for i in range(len(most_sig_img_ids)):
        img, label = dataset[most_sig_img_ids[i] * 2]
        new_img = img.resize((100, 100))
        display(new_img)

    print_graph(graph)


if __name__ == "__main__":
    driver()
