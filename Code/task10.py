import pickle

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

from utilities import calculate_similarity
from utilities import task_to_string_map, dim_red_opn_to_string_map, feature_option_to_feature_index_map, BASE_DIR, \
    DATABASE

dataset = torchvision.datasets.Caltech101(BASE_DIR)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)


def load_latent_semantics(file_path):
    """Load the latent semantics from the saved file."""
    with open(file_path, 'rb') as file:
        similarity_matrix = pickle.load(file)
    return similarity_matrix


def print_menu():
    feature_model_dict = {
        1: 'ColorMoments',
        2: 'HOG',
        3: 'ResNet_AvgPool_1024',
        4: 'ResNet_Layer3_1024',
        5: 'ResNet_FC_1000'}
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
    reduce_dimension = int(input("Please enter the reduced dimension of latent space: "))
    return feature_model, method, k, label, task, reduce_dimension

def driver():
    # feature_model, method, k, label = print_menu()

    # feature_option = 5  # remove later
    # method_option = 4  # remove later
    # k = 5  # remove later
    # selected_label_index = 0  # remove later
    # task_number = 3
    # number_of_similar = 5

    feature_option, method_option, number_of_similar, selected_label_index, task_number, reduce_dimension = print_menu()

    target_labels = []
    collection = DATABASE.feature_descriptors
    label_to_image_id_list = collection.find({"image_label": selected_label_index}, {"image_id": 1, "_id": 0})
    for h in label_to_image_id_list:
        target_labels.append(h["image_id"])

    top_scores = []

    task = task_to_string_map[task_number]
    feature_model = feature_option_to_feature_index_map[feature_option]
    method = dim_red_opn_to_string_map[method_option]
    latent_feature_storage_path = f"../Outputs/{task}/{feature_model}/{method}_{reduce_dimension}.pkl"
    image_to_latent_features = load_latent_semantics(latent_feature_storage_path)['image_to_latent_features']
    # if (method_option != 4):
    #     image_to_latent_features = image_to_latent_features.values
    target_vectors = np.array([image_to_latent_features[i] for i in target_labels])
    label_mean_vector = np.mean(np.vstack(target_vectors), axis=0)
    for idx, image_latent_vector in enumerate(image_to_latent_features):
        relevance_score = calculate_similarity(label_mean_vector, image_latent_vector)
        top_scores.append((idx, relevance_score))
    top_scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = top_scores[:number_of_similar]

    # Display Top-k Relevant Images
    images = []
    sample_image = None
    sample_label = None

    # Assuming 'dataset' is your dataset, and 'top_scores' contains the top-k relevant images and their scores
    for image_index, similarity_score in top_scores:
        pil_image, l = dataset[image_index]
        images.append(pil_image)
        print(f'Image Index {image_index} | Label {l} | (Similarity Score: {round(similarity_score, 2)})')

    # If sample image is not found in top scores, search the entire dataset
    if sample_image is None and sample_label is None:
        for pil_image, l in dataset:
            if l == selected_label_index:  # Assuming 'label' is the desired label
                sample_image = pil_image
                sample_label = l
                break

    print(f'Actual label is {sample_label}')

    # Plotting
    fig = plt.figure(figsize=(15, 5))

    for i, img in enumerate(images):
        plt.subplot(1, len(images) + 1, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    # Add the sample image
    plt.subplot(1, len(images) + 1, len(images) + 1)
    plt.imshow(sample_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    driver()