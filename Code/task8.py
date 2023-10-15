import os
import pickle
import re
from datetime import datetime

import PIL
import numpy as np
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
from matplotlib import pyplot as plt
from pymongo import MongoClient

from Code.utilities import calculate_euclidian_distance, cosine_similarity
from Code.utilities import extract_hog_descriptor, \
    extract_resnet_layer3_1024, extract_color_moment, extract_resnet_avgpool_1024, extract_resnet_fc_1000

ROOT_DIR = '/home/rpaw/MWD/caltech-101/caltech-101/101_ObjectCategories/'
CNN_MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
BASE_DIR = '/home/rpaw/MWD/'

# MongoDB Client Setup
MONGO_CLIENT = MongoClient("mongodb://adminUser:adminPassword@localhost:27017/mwd_db?authSource=admin")
DATABASE = MONGO_CLIENT['mwd_db']


def get_feature_vector_similarity_sorted_pairs(feature_vector_similarity_list, input_image_feature, image_feature):
    # Calculated euclidian distance between input image and iterated image for feature descriptor
    # if feature_option in [1, 2]:
    # feature_vector_similarity = calculate_euclidian_distance(np.array(input_image_feature),
    #                                                              np.array(image_feature))
    # else:
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
        case "hog_descriptor":
            feature_map["hog_descriptor"] = extract_hog_descriptor(image)
        case "color_moments":
            feature_map["color_moments"] = extract_color_moment(image)
        case "resnet_layer3_1024":
            feature_map["resnet_layer3_1024"] = extract_resnet_layer3_1024(image)
        case "resnet_avgpool_1024":
            feature_map["resnet_avgpool_1024"] = extract_resnet_avgpool_1024(image)
        case "resnet_fc_1000":
            feature_map["resnet_fc_1000"] = extract_resnet_fc_1000(image)

    return feature_map



# Function to plot k similar images against input image for all 5 feature models
def plot_result(feature_vector_similarity_sorted_pairs, image_id_list, k, input_label, latent_feature_option):
    dataset = datasets.Caltech101(BASE_DIR)

    # Number of images per row
    images_per_row = k

    # Number of rows needed(1 Original image + 5 Feature models)
    num_rows = 2
    fig, axes = plt.subplots(num_rows, images_per_row + 1, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.5)

    # Load and display the original image
    original_label = f"Input Label: {input_label}"

    if input_label.isnumeric():
        input_image_id = int(input_label)
        original_img = dataset[input_image_id][0]
    else:
        original_img = PIL.Image.open(input_label)

    axes[0, 1].imshow(original_img, cmap="gray")
    axes[0, 1].axis('off')
    axes[0, 0].set_title(original_label, loc='center', pad=10, verticalalignment='center')
    axes[0, 0].axis('off')

    feature_option_to_label_map = {
        1: 'HOG FD',
        2: 'Color Moments FD',
        3: 'Resnet Layer 3 FD',
        4: 'Resnet Avgpool FD',
        5: 'Resnet FC FD',
    }
    feature_option_to_similarity_type_map = {
        1: 'Euclidian Distance',
        2: 'Euclidian Distance',
        3: 'Cosine Similarity',
        4: 'Cosine Similarity',
        5: 'Cosine Similarity',

    }

    for i in range(1):
        axes[i + 1, 0].set_title(latent_feature_option, loc='center', pad=10,
                                 verticalalignment='top')
        axes[i + 1, 0].axis('off')
        for j in range(images_per_row):
            if j < len(feature_vector_similarity_sorted_pairs):
                similarity_score, image_id_index = feature_vector_similarity_sorted_pairs[j]
                image_id = image_id_list[image_id_index]
                img = dataset[image_id][0]
                axes[i + 1, j + 1].imshow(img, cmap="gray")
                axes[i + 1, j + 1].set_title(
                    f'Euclidian Distance: {similarity_score:.2f}', pad=5,
                    verticalalignment='top')
                axes[i + 1, j + 1].axis('off')

    # Removed empty subplot in row 0
    for j in range(0, images_per_row + 1):
        if j in [0, 1]:
            continue
        fig.delaxes(axes[0, j])

    plt.tight_layout()

    # Saved output to output dir
    current_epoch_timestamp = int(datetime.now().timestamp())
    plt.savefig(f"../Outputs/id_{input_label}_k_{k}_ts_{current_epoch_timestamp}.png")

    plt.show()


def get_reduced_features_params(latent_semantic_option):
    latent_semantic_tokens = latent_semantic_option.split("-")
    task = latent_semantic_tokens[0]
    feature_model = latent_semantic_tokens[1]
    reduced_feature = f"{latent_semantic_tokens[-1]}_{latent_semantic_tokens[-2]}.pkl"
    return task, feature_model, reduced_feature


def get_latent_feature_storage_path(task, feature_model, reduced_feature):
    return f"../Outputs/{task}/{feature_model}/{reduced_feature}"


def load_reduced_features(task, feature_model, reduced_feature):
    file_path = get_latent_feature_storage_path(task, feature_model, reduced_feature)
    if not os.path.exists(file_path):
        print(f"The file at '{file_path}' does not exist.\n")
        exit()

    # Load the NumPy arrays from the pickle file
    with open(file_path, 'rb') as file:
        latent_features = pickle.load(file)

    image_to_latent_features = latent_features["image_to_latent_features"]
    latent_feature_to_original_feature = latent_features["latent_feature_to_original_feature"]
    return image_to_latent_features, latent_feature_to_original_feature


def get_kmeans_latent_feature(input_image_features, latent_feature_to_original_feature):
    latent_feature = []
    for centroid in latent_feature_to_original_feature:
        latent_feature.append(calculate_euclidian_distance(input_image_features, centroid))
    return np.array(latent_feature)


def get_input_image_latent_feature(input_image_features, latent_feature_to_original_feature, reduced_feature):
    input_image_features = np.array(input_image_features)
    if reduced_feature.startswith("kmeans"):
        latent_input_image_feature = get_kmeans_latent_feature(input_image_features, latent_feature_to_original_feature)
    else:
        latent_input_image_feature = input_image_features @ latent_feature_to_original_feature.T

    return latent_input_image_feature


def get_label_features(input_label_features, image_id_label_list):
    result = []
    label_feature = []
    for i in range(101):
        label_feature.append([])
    image_id_label_map = dict()
    for image_id_label_tuple in image_id_label_list:
        image_id_label_map[image_id_label_tuple["image_id"]] = image_id_label_tuple["image_label"]

    for index, input_label_feature in enumerate(input_label_features, 0):
        label_feature[image_id_label_map[2*index]].append(input_label_feature)

    for lf in label_feature:
        result.append(np.array(lf).mean(axis=0).tolist())

    return result


def get_k_nearest_neighbours(image_id, k, latent_semantic_option):
    # Loaded datatset
    dataset = datasets.Caltech101(BASE_DIR)
    collection = DATABASE.feature_descriptors

    task, feature_model, reduced_feature = get_reduced_features_params(latent_semantic_option)
    image_to_latent_features, latent_feature_to_original_feature = load_reduced_features(task, feature_model,
                                                                                         reduced_feature)

    feature_model_name_map = {
        "CM": "color_moments",
        "HOG": "hog_descriptor",
        "AvgPool": "resnet_avgpool_1024",
        "L3": "resnet_layer3_1024",
        "FC": "resnet_fc_1000",
        "RESNET": "resnet"
    }

    # Extracted Input Image
    if image_id.isnumeric() and int(image_id) % 2 == 0:
        input_image_features = collection.find_one({"image_id": int(image_id)})
    else:
        input_image_features = extract_features(image_id, feature_model_name_map[feature_model])

    input_image_features = input_image_features[feature_model_name_map[feature_model]]

    input_image_latent_feature = get_input_image_latent_feature(input_image_features,
                                                                latent_feature_to_original_feature, reduced_feature)

    image_superset_latent_features = image_to_latent_features
    image_id_label_map = collection.find({}, {"image_label": 1, "image_id": 1, "_id": 0})
    label_latent_features = get_label_features(image_superset_latent_features, image_id_label_map)

    # Initialized np array for storing similarity measures for each feature model
    feature_vector_similarity_list = np.array([])
    image_id_list = []
    feature_vector_similarity_sorted_pairs = []

    # Iterated over images from superset to compute measures corresponding to each image
    for index, label_latent_feature in enumerate(label_latent_features, 0):
        feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(feature_vector_similarity_list,
                                                                                    input_image_latent_feature,
                                                                                    label_latent_feature)
        # Appended similarity results to the corresponding list
        current_image_id = 2 * index
        image_id_list.append(current_image_id)

    # Sorted and paired index with similarity score
    # if latent_semantic_option in [1, 2]:
    # feature_vector_similarity_sorted_indices = np.argsort(feature_vector_similarity_list)
    # feature_vector_similarity_sorted_elements = np.sort(feature_vector_similarity_list)
    # else:
    feature_vector_similarity_sorted_indices = np.argsort(feature_vector_similarity_list)[::-1]
    feature_vector_similarity_sorted_elements = np.sort(feature_vector_similarity_list)[::-1]

    feature_vector_similarity_sorted_pairs = list(
        zip(feature_vector_similarity_sorted_elements, feature_vector_similarity_sorted_indices))[:k]

    print(f"K similar labels to input image:\n")
    for score, label in feature_vector_similarity_sorted_pairs:
        print(f"label: {label} score: {score}")
    # Plotted results
    # plot_result(feature_vector_similarity_sorted_pairs[:k], image_id_list, k, image_id, latent_semantic_option)


# Driver function to compute features
def driver():
    image_id = input("Enter image ID between 0-8676 or image path.\n")
    latent_semantic_option = input(
        "Please enter latent semantic in format Task-FeatureModel-ReducedDimension-DimensionReductionTechnique\n"
        "Valid Tasks: T3, T4, T5, T6\n"
        "Valid Feature Model: CM, HOG, AvgPool, L3, FC, RESNET\n"
        "Valid Reduced Dimension: 1 - length of feature model\n"
        "Valid Dimension Reduction Technique: SVD, NNMF, LDA, kmeans\n")
    is_valid_feature_option = validate_latent_semantic_option(latent_semantic_option)
    while not is_valid_feature_option:
        print(f"Invalid input: {latent_semantic_option}")
        latent_semantic_option = input(
            "Please enter latent semantic in format Task-FeatureModel-ReducedDimension-DimensionReductionTechnique\n"
            "Valid Tasks: T3, T4, T5, T6\n"
            "Valid Feature Model: CM, HOG, AvgPool, L3, FC, RESNET\n"
            "Valid Reduced Dimension: 1 - length of feature model\n"
            "Valid Dimension Reduction Technique: SVD, NNMF, LDA, kmeans\n")
        is_valid_feature_option = validate_latent_semantic_option(latent_semantic_option)

    k = int(input("Select K to find K similar images to given input image\n"))
    while k < 1 or k > 8676:
        print(f"Invalid K value: {k}. Please pick K in range of 1-8676.")
        k = int(input("Select K to find K similar images to given input image\n"))
    get_k_nearest_neighbours(image_id, k, latent_semantic_option)


def validate_latent_semantic_option(latent_semantic_option):
    # Define regular expressions for each part of the input format
    task_pattern = r"(T3|T4|T5|T6)"
    feature_model_pattern = r"(CM|HOG|AvgPool|L3|FC|RESNET)"
    reduced_dimension_pattern = r"\d+"
    dimension_reduction_technique_pattern = r"(SVD|NNMF|LDA|kmeans)"

    # Combine the patterns into a single regex for the full format
    input_pattern = re.compile(
        f"{task_pattern}-{feature_model_pattern}-{reduced_dimension_pattern}-{dimension_reduction_technique_pattern}"
    )

    # Get user input
    user_input = latent_semantic_option

    # Check if the input matches the format
    if input_pattern.match(user_input):
        return True

    return False


if __name__ == "__main__":
    # driver()
    get_k_nearest_neighbours("0", 5, "T3-FC-5-SVD")
