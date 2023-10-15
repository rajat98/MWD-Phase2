import numpy as np
import pandas as pd
from random import seed
from random import randint

'''n stands for the number of n most similar images '''


def create_similarity_graph(similarity_matrix, n):
    graph = []
    adjacency_matrix = np.zeros((len(similarity_matrix), len(similarity_matrix)))

    for i in range(len(similarity_matrix)):
        # converting to a numpy array
        row_array = similarity_matrix.iloc[i].values
        similar_images = np.argsort(row_array)[::-1]
        similar_images_scores = row_array[similar_images]

        # getting the image ids aswell as weights of the n_most similar images
        n_similar_images = similar_images[:n]
        n_similar_scores = similar_images_scores[:n]
        # print(n_similar_images, n_similar_scores)
        graph.append([n_similar_images, n_similar_scores])
        adjacency_matrix[i][n_similar_images] = 1
    return graph, adjacency_matrix


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
        if even_data[nodes[i] * 2] == label:
            most_sig_img_ids.append(nodes[i] * 2)
        if len(most_sig_img_ids) == m:
            break
    return most_sig_img_ids


graph, adjacency_matrix = create_similarity_graph(even_rows_df, 10)
sorted_nodes = personalized_pagerank(adjacency_matrix, 5, 10)
most_sig_img_ids = most_significant_images(20, sorted_nodes, even_data, 5)