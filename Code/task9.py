import pickle

import pandas as pd

from utilities import calculate_similarity
from utilities import feature_model_dict
from utilities import task_to_string_map, dim_red_opn_to_string_map, feature_option_to_feature_index_map


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
    label = int(input("Please enter the label(0-100): "))
    task = int(input("Please enter the task number(3, 4, 5, 6): "))
    return feature_model, method, k, label, task


def load_latent_semantics(file_path):
    """Load the latent semantics (W matrix) from the saved file."""
    with open(file_path, 'rb') as file:
        similarity_matrix = pickle.load(file)
    return similarity_matrix


def find_top_k_matching_labels(image_to_latent_features, selected_label_index, number_of_similar):
    """Identify and list k most likely matching labels."""
    # image_to_latent_features = image_to_latent_features.values
    selected_latent_semantics = image_to_latent_features[selected_label_index]
    similarity_scores = []
    for i in range(len(image_to_latent_features)):
        # if i != selected_label_index:
        similarity_score = calculate_similarity(image_to_latent_features[i], selected_latent_semantics)
        similarity_scores.append((i, similarity_score))
    similarity_scores.sort(key=lambda x: x[1], reverse=False)

    return similarity_scores[:number_of_similar]


def driver():
    # feature_model, method, k, label = print_menu()

    feature_option = 5  # remove later
    method = 4  # remove later
    k = 5  # remove later
    selected_label_index = 0  # remove later
    task_number = 3
    number_of_similar = 5

    # Example usage:
    task = task_to_string_map[task_number]
    feature_model = feature_option_to_feature_index_map[feature_option]
    method = dim_red_opn_to_string_map[method]
    latent_feature_storage_path = f"../Outputs/{task}/{feature_model}/{method}_{k}.pkl"
    image_to_latent_features = load_latent_semantics(latent_feature_storage_path)['image_to_latent_features']
    if method == 4:
        image_to_latent_features = pd.DataFrame(image_to_latent_features)
    top_matching_labels = find_top_k_matching_labels(image_to_latent_features, selected_label_index, number_of_similar)
    print(f'Top {number_of_similar} matching labels for label {selected_label_index}:')
    for label_index, similarity_score in top_matching_labels:
        print(f'Label {label_index} (Similarity Score: {round(similarity_score, 2)})')


if __name__ == "__main__":
    driver()
