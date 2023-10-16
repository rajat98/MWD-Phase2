import os
import pickle

import PIL.Image
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.models as models
from pymongo import MongoClient
from sklearn.decomposition import NMF

from Code.utilities import calculate_euclidian_distance, cosine_similarity, kmeans
from Code.utilities import extract_hog_descriptor, \
    extract_resnet_layer3_1024, extract_color_moment, extract_resnet_avgpool_1024, extract_resnet_fc_1000, \
    get_positive_feature_matrix, lda, svd

ROOT_DIR = '/home/rpaw/MWD/caltech-101/caltech-101/101_ObjectCategories/'
CNN_MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
BASE_DIR = '/home/rpaw/MWD/'

# MongoDB Client Setup
MONGO_CLIENT = MongoClient("mongodb://adminUser:adminPassword@localhost:27017/mwd_db?authSource=admin")
DATABASE = MONGO_CLIENT['mwd_db']


def get_feature_vector_similarity_sorted_pairs(feature_vector_similarity_list, input_image_feature, image_feature,
                                               feature_option):
    # Calculated euclidian distance between input image and iterated image for feature descriptor
    if feature_option in [1, 2]:
        feature_vector_similarity = calculate_euclidian_distance(np.array(input_image_feature),
                                                                 np.array(image_feature))
    else:
        feature_vector_similarity = cosine_similarity(np.array(input_image_feature),
                                                      np.array(image_feature))
    return np.append(feature_vector_similarity_list, feature_vector_similarity)


# Function to calculate user specified features and display input image
def extract_features(image_id, feature_options):
    feature_map = dict()
    dataset = datasets.Caltech101(BASE_DIR)

    if image_id.isnumeric():
        image_id = int(image_id)
        image = dataset[image_id][0]
    else:
        image = PIL.Image.open(image_id)
    image = image.convert('RGB')
    image.show()
    match feature_options:
        case 1:
            feature_map["hog_descriptor"] = extract_hog_descriptor(image)
        case 2:
            feature_map["color_moments"] = extract_color_moment(image)
        case 3:
            feature_map["resnet_layer3_1024"] = extract_resnet_layer3_1024(image)
        case 4:
            feature_map["resnet_avgpool_1024"] = extract_resnet_avgpool_1024(image)
        case 5:
            feature_map["resnet_fc_1000"] = extract_resnet_fc_1000(image)

    return feature_map


# Driver function to take user inputs and find k similar images to the input image
def driver():
    feature_option = int(input("Please pick one of the below options\n"
                               "1. HOG\n"
                               "2. Color Moments\n"
                               "3. Resnet Layer 3\n"
                               "4. Resnet Avgpool\n"
                               "5. Resnet FC"
                               "6. Resnet\n"))
    while feature_option not in list(range(1, 7)):
        print(f"Invalid input: {feature_option}")
        feature_option = int(input("Please pick one of the below options\n"
                                   "1. HOG\n"
                                   "2. Color Moments\n"
                                   "3. Resnet Layer 3\n"
                                   "4. Resnet Avgpool\n"
                                   "5. Resnet FC"
                                   "6. Resnet\n"))

    k = int(input("Select K to find K similar images to given input image\n"))
    while k < 1 or k > 8676:
        print(f"Invalid K value: {k}. Please pick K in range of 1-8676.")
        k = int(input("Select K to find K similar images to given input image\n"))

    dim_red_opn = int(input("Select any of the following dimensionality reduction technique\n"
                            "1. SVD\n"
                            "2. NNMF\n"
                            "3. LDA\n"
                            "4. k-means\n"))
    while dim_red_opn < 1 or dim_red_opn > 4:
        print(f"Invalid dimensionality reduction technique: {dim_red_opn}. Please pick in range of 1-4.")
        dim_red_opn = int(input("Select any of the following dimensionality reduction technique\n"
                                "1. SVD\n"
                                "2. NNMF\n"
                                "3. LDA\n"
                                "4. k-means\n"))
    process_top_k_latent_semantics(feature_option, k, dim_red_opn)


def process_top_k_latent_semantics(feature_option, k, dim_red_opn):
    feature_matrix = get_feature_matrix(feature_option)
    image_to_latent_features = np.array([])
    latent_feature_to_original_feature = np.array([])
    match dim_red_opn:
        case 1:
            image_to_latent_features, latent_feature_to_original_feature = svd(k, feature_matrix)
        case 2:
            feature_matrix = get_positive_feature_matrix(feature_matrix)

            # W, H = nmf_sgd(feature_matrix, k)
            #
            # image_to_latent_features = feature_matrix @ H.T
            # latent_feature_to_original_feature = H

            nmf = NMF(n_components=k)
            nmf.fit(feature_matrix)
            latent_feature_to_original_feature = nmf.components_
            image_to_latent_features = feature_matrix @ latent_feature_to_original_feature.T
        case 3:
            feature_matrix = get_positive_feature_matrix(feature_matrix)
            image_to_latent_features, latent_feature_to_original_feature = lda(k, feature_matrix)
        case 4:
            centroid, latent_features = kmeans(k, feature_matrix)
            image_to_latent_features = latent_features
            latent_feature_to_original_feature = centroid

    print_image_id_weight_pairs(image_to_latent_features, dim_red_opn)
    save_latent_features_to_file(image_to_latent_features, latent_feature_to_original_feature, dim_red_opn, k, feature_option)


def save_latent_features_to_file(image_to_latent_features, latent_feature_to_original_feature, dim_red_opn, k, feature_option):
    latent_features = {'image_to_latent_features': image_to_latent_features,
                       'latent_feature_to_original_feature': latent_feature_to_original_feature}
    dim_red_opn_to_string_map = {
        1: "SVD",
        2: "NNMF",
        3: "LDA",
        4: "kmeans"
    }
    feature_option_to_feature_index_map = {
        1: "HOG",
        2: "CM",
        3: "L3",
        4: "AvgPool",
        5: "FC",
        6: "RESNET"
    }

    latent_feature_storage_path = f"../Outputs/T3/{feature_option_to_feature_index_map[feature_option]}/{dim_red_opn_to_string_map[dim_red_opn]}_{k}.pkl"
    # Ensure that the directory path exists, creating it if necessary
    os.makedirs(os.path.dirname(latent_feature_storage_path), exist_ok=True)
    with open(latent_feature_storage_path, 'wb') as file:
        pickle.dump(latent_features, file)


def print_image_id_weight_pairs(latent_features, dim_red_opn):
    for index, latent_features in enumerate(latent_features, 1):
        image_id = 2 * index
        sorted_indices = np.argsort(-latent_features)
        # sorted_data = latent_features[sorted_indices]
        print(f"image_id: {image_id}")
        for sorted_index in sorted_indices:
            print(f"latent feature: {sorted_index}  latent feature value: {latent_features[sorted_index]}")


def nmf_sgd(X, num_components, num_iterations=1000, learning_rate=0.01, batch_size=10):
    n, m = X.shape
    W = np.random.rand(n, num_components)
    H = np.random.rand(num_components, m)

    for iteration in range(num_iterations):
        # Randomly shuffle the data indices for mini-batch SGD
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            X_batch = X[batch_indices, :]
            W_batch = W[batch_indices, :]
            H_batch = H[:, :]

            X_approx = np.dot(W_batch, H_batch)

            # Compute the gradients using matrix multiplication
            gradient_W = -2 * X_batch.dot(H_batch.T) + 2 * W_batch.dot(H_batch.dot(H_batch.T))
            gradient_H = -2 * W_batch.T.dot(X_batch) + 2 * W_batch.T.dot(W_batch.dot(H_batch))

            # Update W and H using the gradients
            W_batch -= learning_rate * gradient_W
            H_batch -= learning_rate * gradient_H

    return W, H


def get_feature_matrix(feature_option):
    feature_option_to_feature_index_map = {
        1: "hog_descriptor",
        2: "color_moments",
        3: "resnet_layer3_1024",
        4: "resnet_avgpool_1024",
        5: "resnet_fc_1000",
        6: "resnet_fc_1000"
    }
    index = feature_option_to_feature_index_map[feature_option]
    collection = DATABASE.feature_descriptors
    input_image_features = collection.find({}, {index: 1, "_id": 0})
    feature_matrix = []
    for feature in input_image_features:
        if feature_option == 6:
            feature_matrix.append(torch.nn.functional.softmax(torch.tensor(feature[index], dtype=torch.float), dim=0).tolist())
        else:
            feature_matrix.append(feature[index])

    return np.array(feature_matrix)


if __name__ == "__main__":
    # driver()
    process_top_k_latent_semantics(6, 5, 4)
