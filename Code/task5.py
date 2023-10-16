import os
import pickle
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utilities import feature_model_dict
from utilities import svd, lda, nnmf, kmeans, print_image_id_weight_pairs, save_latent_features_to_file, DATABASE, \
    feature_option_to_feature_index_mapping


def parse_string(string):
    values = re.findall(r'-?\d+\.\d+', string)
    np_array = np.array(values, dtype=float)
    return np_array


def create_label_similarity_matrix(feature_model):
    """Calculate label-label similarity matrix based on mean vectors
    Return a square matrix representing label similarities"""
    collection = DATABASE.feature_descriptors
    labels_size = 101
    similarity_matrix = np.zeros((labels_size, labels_size))
    index = feature_option_to_feature_index_mapping[feature_model]
    for i in tqdm(range(labels_size)):
        label1_data = collection.find({"image_label": i}, {index: 1, "_id": 0})
        label1_data = [input_label_feature[index] for input_label_feature in label1_data]
        label1_data = np.array(label1_data, dtype=float)
        label1_mean_vector = np.mean(np.vstack(label1_data), axis=0)
        for j in range(i, labels_size):
            label2_data = collection.find({"image_label": j}, {index: 1, "_id": 0})
            label2_data = [input_label_feature[index] for input_label_feature in label2_data]
            label2_data = np.array(label2_data, dtype=float)
            label2_mean_vector = np.mean(np.vstack(label2_data), axis=0)
            similarity_score = cosine_similarity([label1_mean_vector], [label2_mean_vector])[0][0]
            # Fill both upper and lower triangles of the similarity matrix
            similarity_matrix[i, j] = similarity_score
            similarity_matrix[j, i] = similarity_score
    return similarity_matrix


def perform_dimensionality_reduction(similarity_matrix, method, k, feature_model):
    if method == 1:
        image_to_latent_features, latent_feature_to_original_feature = svd(k, similarity_matrix)
        print_image_id_weight_pairs(image_to_latent_features)
        save_latent_features_to_file(5, image_to_latent_features, latent_feature_to_original_feature, feature_model,
                                     method, k)
    elif method == 2:
        image_to_latent_features, latent_feature_to_original_feature = nnmf(k, similarity_matrix)
        print(image_to_latent_features)
        print_image_id_weight_pairs(image_to_latent_features)
        save_latent_features_to_file(5, image_to_latent_features, latent_feature_to_original_feature, feature_model,
                                     method, k)
    elif method == 3:
        image_to_latent_features, latent_feature_to_original_feature = lda(k, similarity_matrix)
        print_image_id_weight_pairs(image_to_latent_features)
        save_latent_features_to_file(5, image_to_latent_features, latent_feature_to_original_feature, feature_model,
                                     method, k)
    elif method == 4:
        centroid, latent_features = kmeans(k, similarity_matrix)
        image_to_latent_features = latent_features
        latent_feature_to_original_feature = centroid
        print_image_id_weight_pairs(image_to_latent_features)
        save_latent_features_to_file(5, image_to_latent_features, latent_feature_to_original_feature, feature_model,
                                     method, k)


def save_similarity_matrix(similarity_matrix, feature_model, method):
    """Save the label similarity matrix to a pickle file"""
    similarity_matrix_storage = {'label_similarity_matrix': similarity_matrix}
    latent_feature_storage_path = f"../Outputs/T5/label_similarity.pkl"
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
    method_dict = {1: 'SVD', 2: 'NMF', 3: 'LDA', 4: 'K-Means'}
    print('Choose dimensionality reduction technique')
    for key, v in method_dict.items():
        print(str(key) + '.', v)
    method = int(input("Choose dimensionality reduction selected: "))
    print('Calculate a label-label similarity matrix again? ')
    cm = input('[y|Y/n|N]:')
    return feature_model, method, k, cm


def driver():
    # feature_model, method, k, cm = print_menu()
    feature_model = 5  # remove later
    method = 4  # remove later
    k = 5  # remove later
    cm = 'y'  # remove later

    # df = pd.read_csv('FD_Objects.csv')
    # grouped_data = df.groupby('Labels')
    # labels = list(grouped_data.groups.keys())
    if cm == 'y' or cm == 'Y':
        print('Calculating Similarity Matrix...')
        similarity_matrix = create_label_similarity_matrix(feature_model)
        save_similarity_matrix(similarity_matrix, feature_model, method)

    filepath = f"../Outputs/T5/label_similarity.pkl"

    with open(filepath, 'rb') as file:
        similarity_matrix_file = pickle.load(file)
    similarity_matrix = similarity_matrix_file["label_similarity_matrix"]
    print(similarity_matrix.shape)
    perform_dimensionality_reduction(similarity_matrix, method, k, feature_model)


if __name__ == "__main__":
    driver()
