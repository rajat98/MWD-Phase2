import os
from datetime import datetime

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.models as models
from pymongo import MongoClient

from Code.utilities import calculate_euclidian_distance, cosine_similarity, extract_hog_descriptor, \
    extract_resnet_layer3_1024, extract_color_moment, extract_resnet_avgpool_1024, extract_resnet_fc_1000, \
    feature_option_to_feature_index_map

ROOT_DIR = '/home/rpaw/MWD/caltech-101/caltech-101/101_ObjectCategories/'
CNN_MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
BASE_DIR = '/home/rpaw/MWD/'


# MongoDB Client Setup
MONGO_CLIENT = MongoClient("mongodb://adminUser:adminPassword@localhost:27017/mwd_db?authSource=admin")
DATABASE = MONGO_CLIENT['mwd_db']


# Function to plot k similar images against input image for all 5 feature models
def plot_result(feature_vector_similarity_sorted_pairs, image_id_list, k, input_image_id, feature_option):
    dataset = datasets.Caltech101(BASE_DIR)

    # Number of images per row
    images_per_row = k

    # Number of rows needed(1 Original image + 5 Feature models)
    num_rows = 2
    fig, axes = plt.subplots(num_rows, images_per_row + 1, figsize=(30, 10))
    plt.subplots_adjust(wspace=0.5)

    # Load and display the original image
    original_label = "Input Image"
    if input_image_id.isnumeric():
        input_image_id = int(input_image_id)
        original_img = dataset[input_image_id][0]
    else:
        original_img = PIL.Image.open(input_image_id)

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
        axes[i + 1, 0].set_title(feature_option_to_label_map[feature_option], loc='center', pad=10,
                                 verticalalignment='top')
        axes[i + 1, 0].axis('off')
        for j in range(images_per_row):
            if j < len(feature_vector_similarity_sorted_pairs):
                similarity_score, image_id_index = feature_vector_similarity_sorted_pairs[j]
                image_id = image_id_list[image_id_index]
                img = dataset[image_id][0]
                axes[i + 1, j + 1].imshow(img, cmap="gray")
                axes[i + 1, j + 1].set_title(
                    f'{feature_option_to_similarity_type_map[feature_option]}: {similarity_score:.2f}', pad=5,
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
    output_path = f"../Outputs/T0b/{feature_option_to_feature_index_map[feature_option]}/id_{image_id}_k_{k}_ts_{current_epoch_timestamp}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)

    plt.show()


# Function to calculate K similar images pertaining to the given image for all feature models
def get_k_nearest_neighbours(image_id, k, feature_option):
    # Loaded datatset
    dataset = datasets.Caltech101(BASE_DIR)
    collection = DATABASE.feature_descriptors
    # Extracted Input Image
    if image_id.isnumeric() and int(image_id) % 2 == 0:
        input_image_features = collection.find_one({"image_id": int(image_id)})
    else:
        input_image_features = extract_features(image_id, feature_option)

    # Extracted feature superset from the DB
    image_superset_features = collection.find({})

    # Initialized np array for storing similarity measures for each feature model
    feature_vector_similarity_list = np.array([])
    image_id_list = []
    feature_vector_similarity_sorted_pairs = []

    # Iterated over images from superset to compute measures corresponding to each image
    for image_features in image_superset_features:
        match feature_option:
            case 1:
                # Calculated euclidian distance between input image and iterated image for HOG feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(feature_vector_similarity_list,
                                                                                       input_image_features[
                                                                                           "hog_descriptor"],
                                                                                       image_features["hog_descriptor"],
                                                                                       feature_option)

            case 2:
                # Calculated euclidian distance between input image and iterated image for color moments feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(feature_vector_similarity_list,
                                                                                       input_image_features[
                                                                                           "color_moments"],
                                                                                       image_features["color_moments"],
                                                                                       feature_option)

            case 3:
                # Calculated cosine similarity between input image and iterated image for resnet layer 3 feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(feature_vector_similarity_list,
                                                                                       input_image_features[
                                                                                           "resnet_layer3_1024"],
                                                                                       image_features[
                                                                                           "resnet_layer3_1024"],
                                                                                       feature_option)

            case 4:
                # Calculated cosine similarity between input image and iterated image for resnet avgpool layer feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(feature_vector_similarity_list,
                                                                                       input_image_features[
                                                                                           "resnet_avgpool_1024"],
                                                                                       image_features[
                                                                                           "resnet_avgpool_1024"],
                                                                                       feature_option)
            case 5:
                # Calculated cosine similarity between input image and iterated image for resnet FC layer feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(feature_vector_similarity_list,
                                                                                       input_image_features[
                                                                                           "resnet_fc_1000"],
                                                                                       image_features["resnet_fc_1000"],
                                                                                       feature_option)
        # Appended similarity results to the corresponding list
        image_id_list.append(image_features["image_id"])

    # Sorted and paired index with similarity score
    if feature_option in [1, 2]:
        feature_vector_similarity_sorted_indices = np.argsort(feature_vector_similarity_list)
        feature_vector_similarity_sorted_elements = np.sort(feature_vector_similarity_list)
    else:
        feature_vector_similarity_sorted_indices = np.argsort(feature_vector_similarity_list)[::-1]
        feature_vector_similarity_sorted_elements = np.sort(feature_vector_similarity_list)[::-1]

    feature_vector_similarity_sorted_pairs = list(
        zip(feature_vector_similarity_sorted_elements, feature_vector_similarity_sorted_indices))


    # Plotted results
    plot_result(feature_vector_similarity_sorted_pairs[:k], image_id_list, k, image_id, feature_option)


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
    image_id = input("Enter image ID between 0-8676 or image path.\n")
    feature_option = int(input("Please pick one of the below options\n"
                               "1. HOG\n"
                               "2. Color Moments\n"
                               "3. Resnet Layer 3\n"
                               "4. Resnet Avgpool\n"
                               "5. Resnet FC\n"))
    while feature_option not in list(range(1, 7)):
        print(f"Invalid input: {feature_option}")
        feature_option = int(input("Please pick one of the below options\n"
                                   "1. HOG\n"
                                   "2. Color Moments\n"
                                   "3. Resnet Layer 3\n"
                                   "4. Resnet Avgpool\n"
                                   "5. Resnet FC\n"))

    k = int(input("Select K to find K similar images to given input image\n"))
    while k < 1 or k > 8676:
        print(f"Invalid K value: {k}. Please pick K in range of 1-8676.")
        k = int(input("Select K to find K similar images to given input image\n"))
    get_k_nearest_neighbours(image_id, k, feature_option)


if __name__ == "__main__":
    driver()
    # for input_label in [0, 880, 2500, 5122, 8676]:
    #     for i in range(1,6):
    #         get_k_nearest_neighbours(str(input_label), 10, i)
