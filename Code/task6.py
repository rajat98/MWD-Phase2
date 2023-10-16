import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from utilities import feature_model_dict, cosine_similarity
from utilities import svd, lda, nnmf, kmeans, print_image_id_weight_pairs, save_latent_features_to_file, DATABASE, \
    feature_option_to_feature_index_mapping


def parse_string(string):
    values = re.findall(r'-?\d+\.\d+', string)
    np_array = np.array(values, dtype=float)
    np_array = np.round(np_array, 2)
    return np_array


def create_image_similarity_matrix(feature_model):
    collection = DATABASE.feature_descriptors
    images_size = collection.count_documents({})

    index = feature_option_to_feature_index_mapping[feature_model]

    def get_image_vector(image_id):
        image_vector = collection.find({"image_id": image_id}, {index: 1, "_id": 0})
        image_vector = np.array(image_vector[0][index])
        return image_vector

    similarity_matrix = np.zeros((images_size, images_size))

    # Use ThreadPoolExecutor to parallelize the task
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers according to your CPU capabilities
        futures = []
        for i in tqdm(range(images_size)):
            image1_id = 2 * i
            image1_vector = executor.submit(get_image_vector, image1_id)
            for j in range(i, images_size):
                image2_id = 2 * j
                image2_vector = executor.submit(get_image_vector, image2_id)
                futures.append((i, j, image1_vector, image2_vector))

        for i, j, image1_vector, image2_vector in tqdm(futures):
            similarity_score = cosine_similarity(image1_vector.result(), image2_vector.result())
            similarity_matrix[i, j] = similarity_score
            similarity_matrix[j, i] = similarity_score

    return similarity_matrix
def perform_dimensionality_reduction(similarity_matrix, feature_model, method, k):
    if method == 1:
        image_to_latent_features, latent_feature_to_original_feature = svd(k, similarity_matrix)
        print_image_id_weight_pairs(image_to_latent_features)
        save_latent_features_to_file(6, image_to_latent_features, latent_feature_to_original_feature, feature_model,
                                     method, k)
    elif method == 2:
        image_to_latent_features, latent_feature_to_original_feature = nnmf(k, similarity_matrix)
        print(image_to_latent_features)
        print_image_id_weight_pairs(image_to_latent_features)
        save_latent_features_to_file(6, image_to_latent_features, latent_feature_to_original_feature, feature_model,
                                     method, k)
    elif method == 3:
        image_to_latent_features, latent_feature_to_original_feature = lda(k, similarity_matrix)
        print_image_id_weight_pairs(image_to_latent_features)
        save_latent_features_to_file(6, image_to_latent_features, latent_feature_to_original_feature, feature_model,
                                     method, k)
    elif method == 4:
        centroid, latent_features = kmeans(k, similarity_matrix)
        image_to_latent_features = latent_features
        latent_feature_to_original_feature = centroid
        print_image_id_weight_pairs(image_to_latent_features)
        save_latent_features_to_file(6, image_to_latent_features, latent_feature_to_original_feature, feature_model,
                                     method, k)


def save_similarity_matrix(similarity_matrix):
    """Save the image similarity matrix to a CSV file"""
    similarity_matrix_storage = {'image_similarity_matrix': similarity_matrix}
    latent_feature_storage_path = f"../Outputs/T5/image_similarity.pkl"
    # Ensure that the directory path exists, creating it if necessary
    os.makedirs(os.path.dirname(latent_feature_storage_path), exist_ok=True)
    with open(latent_feature_storage_path, 'wb') as file:
        pickle.dump(similarity_matrix_storage, file)


def print_menu():
    print("Enter the feature model")
    for k, v in feature_model_dict.items():
        print(str(k) + '.', v)
    feature_model = int(input("Feature model selected: "))
    k = int(input("Enter the value of k: "))
    method_dict = {1: 'SVD', 2: 'NNMF', 3: 'LDA', 4: 'kmeans'}
    print('Choose dimensionality reduction technique')
    for k, v in method_dict.items():
        print(str(k) + '.', v)
    method = int(input("Choose dimensionality reduction selected: "))
    print('Calculate a image-image similarity matrix again? ')
    cm = input('[y|Y/n|N]:')
    return feature_model, method, k, cm


def driver():
    # feature_model, method, k, cm = print_menu()
    feature_model = 5  # remove later
    method = 4  # remove later
    k = 5  # remove later
    cm = 'y'  # remove later
    feature_model, method, k, cm = print_menu()

    if cm == 'y' or cm == 'Y':
        similarity_matrix = create_image_similarity_matrix(feature_model)
        # Save the similarity matrix
        save_similarity_matrix(similarity_matrix)
    else:
        filepath = f"../Outputs/T6/image_similarity.pkl"
        with open(filepath, 'rb') as file:
            image_similarity_matrix_file = pickle.load(file)
        similarity_matrix = image_similarity_matrix_file["image_similarity_matrix"]

    print(similarity_matrix.shape)
    perform_dimensionality_reduction(similarity_matrix, feature_model, method, k)


if __name__ == "__main__":
    driver()
