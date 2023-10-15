import PIL.Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.models as models
from pymongo import MongoClient

from Code.utilities import calculate_euclidian_distance, cosine_similarity, extract_hog_descriptor, \
    extract_resnet_layer3_1024, extract_color_moment, extract_resnet_avgpool_1024, extract_resnet_fc_1000

ROOT_DIR = '/home/rpaw/MWD/caltech-101/caltech-101/101_ObjectCategories/'
CNN_MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
BASE_DIR = '/home/rpaw/MWD/'

# MongoDB Client Setup
MONGO_CLIENT = MongoClient("mongodb://adminUser:adminPassword@localhost:27017/mwd_db?authSource=admin")
DATABASE = MONGO_CLIENT['mwd_db']


def get_label_features(input_label_features, index):
    result = []
    for i in range(101):
        label_feature = [d[index] for d in input_label_features if d["image_label"] == i]
        n = len(label_feature)
        cumulative_input_label_features = [sum(inner_list) for inner_list in zip(*label_feature)]

        # Divide the sums by the number of inner lists to get the average
        final_label_features = [i / n for i in cumulative_input_label_features]
        result.append(final_label_features)

    return result


def get_k_nearest_neighbours(image_id, k, feature_option):
    # Loaded datatset
    dataset = datasets.Caltech101(BASE_DIR)
    collection = DATABASE.feature_descriptors
    # Extracted Input Image
    feature_option_to_feature_index_map = {
        1: "hog_descriptor",
        2: "color_moments",
        3: "resnet_layer3_1024",
        4: "resnet_avgpool_1024",
        5: "resnet_fc_1000"
    }
    index = feature_option_to_feature_index_map[feature_option]

    # Extracted Input Image
    if image_id.isnumeric() and int(image_id) % 2 == 0:
        input_image_features = collection.find_one({"image_id": int(image_id)})
    else:
        input_image_features = extract_features(image_id, feature_option)

    input_image_features = input_image_features[index]

    input_label_features = collection.find({}, {"image_label": 1, index: 1, "_id": 0})
    input_label_features = get_label_features(list(input_label_features), index)

    # Initialized np array for storing similarity measures for each feature model
    feature_vector_similarity_list = np.array([])
    feature_vector_similarity_sorted_pairs = []

    # Iterated over images from superset to compute measures corresponding to each image
    for image_features in input_label_features:
        match feature_option:
            case 1:
                # Calculated euclidian distance between input image and iterated image for HOG feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(
                    feature_vector_similarity_list,
                    input_image_features,
                    image_features,
                    feature_option)

            case 2:
                # Calculated euclidian distance between input image and iterated image for color moments feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(
                    feature_vector_similarity_list,
                    input_image_features,
                    image_features,
                    feature_option)

            case 3:
                # Calculated cosine similarity between input image and iterated image for resnet layer 3 feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(
                    feature_vector_similarity_list,
                    input_image_features,
                    image_features,
                    feature_option)

            case 4:
                # Calculated cosine similarity between input image and iterated image for resnet avgpool layer feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(
                    feature_vector_similarity_list,
                    input_image_features,
                    image_features,
                    feature_option)
            case 5:
                # Calculated cosine similarity between input image and iterated image for resnet FC layer feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(
                    feature_vector_similarity_list,
                    input_image_features,
                    image_features,
                    feature_option)

    # Sorted and paired index with similarity score
    if feature_option in [1, 2]:
        feature_vector_similarity_sorted_indices = np.argsort(feature_vector_similarity_list)
        feature_vector_similarity_sorted_elements = np.sort(feature_vector_similarity_list)
    else:
        feature_vector_similarity_sorted_indices = np.argsort(feature_vector_similarity_list)[::-1]
        feature_vector_similarity_sorted_elements = np.sort(feature_vector_similarity_list)[::-1]

    feature_vector_similarity_sorted_pairs = list(
        zip(feature_vector_similarity_sorted_elements, feature_vector_similarity_sorted_indices))[:k]

    print(f"K similar labels to input image:\n")
    for score, label in feature_vector_similarity_sorted_pairs:
        print(f"label: {label} score: {score}")


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
