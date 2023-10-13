import re
from datetime import datetime

import PIL
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageOps, Image
from matplotlib import pyplot as plt
from pymongo import MongoClient
from scipy.signal import convolve2d

ROOT_DIR = '/home/rpaw/MWD/caltech-101/caltech-101/101_ObjectCategories/'
CNN_MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
BASE_DIR = '/home/rpaw/MWD/'

# MongoDB Client Setup
MONGO_CLIENT = MongoClient("mongodb://adminUser:adminPassword@localhost:27017/mwd_db?authSource=admin")
DATABASE = MONGO_CLIENT['mwd_db']


# Function to preprocess image before feeding to Resnet50 feature extractor
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    transformed_image = transform(image).unsqueeze(0)
    return transformed_image


# Generic function to extract output of intermediate layers of Resnet 50 model
def extract_feature_vector(image, layer):
    hook_output = []

    def hook_fn(module, input, output):
        hook_output.append(output)

    # Attached a hook to the input layer
    hook_layer = layer.register_forward_hook(hook_fn)

    # Loaded and preprocessed image
    image = preprocess_image(image)

    CNN_MODEL.eval()

    # Forward Passed the image through the model
    with torch.no_grad():
        CNN_MODEL(image)

    hook_layer.remove()

    return hook_output[0].squeeze()


# function to extract color moments from an image
def extract_color_moment(image):
    # Resized image as per specs
    new_size = (300, 100)
    resized_image = image.resize(new_size)
    image_array = np.array(resized_image)

    # Partitioned the image into a 10x10 grid
    num_cols, num_rows = resized_image.size
    grid_rows, grid_cols = 10, 10
    cell_height, cell_width = num_rows // grid_rows, num_cols // grid_cols

    # Initialized lists to store color moments for each channel
    moments_red, moments_green, moments_blue = [], [], []

    # Iterate through each grid cell
    for i in range(grid_rows):
        for j in range(grid_cols):
            # Defined the region of interest (ROI) for the current cell
            cell_start_x, cell_start_y = j * cell_width, i * cell_height
            cell_end_x, cell_end_y = cell_start_x + cell_width, cell_start_y + cell_height
            roi = image_array[cell_start_y:cell_end_y, cell_start_x:cell_end_x]

            # Calculated color moments for each channel (Red, Green, Blue)
            # 0: Red, 1: Green, 2: Blue
            for channel in range(3):
                channel_data = roi[:, :, channel]
                mean = np.mean(channel_data)
                std_dev = np.std(channel_data)
                skewness = np.cbrt(np.mean((channel_data - mean) ** 3))

                # Appended the calculated moments to the respective channel list
                if channel == 0:
                    moments_red.extend([mean, std_dev, skewness])
                elif channel == 1:
                    moments_green.extend([mean, std_dev, skewness])
                elif channel == 2:
                    moments_blue.extend([mean, std_dev, skewness])

    # Combined the color moments for all cells into a unified 10x10x3x3 = 900 dimensional feature descriptor
    feature_descriptor = np.concatenate([moments_red, moments_green, moments_blue])
    return feature_descriptor


# Function to extract histogram of oriented gradiant features
def extract_hog_descriptor(image):
    # Converted image to grayscale
    grayscale_image = ImageOps.grayscale(image)

    # Resized image as per specs
    resized_image = grayscale_image.resize(size=(300, 100))
    image_array = np.array(resized_image)

    # Defined the parameters
    image_height = image_array.shape[0]
    image_width = image_array.shape[1]
    grid_size = 10
    cell_size_x = image_width // grid_size
    cell_size_y = image_height // grid_size
    bin_count = 9
    angle_per_bin = 40

    # Initialized the HOG feature vector
    hog_descriptor = np.zeros((10, 10, 9))

    # Computed gradients (dI/dx and dI/dy) using the Sobel operators
    dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # Transposed for dI/dy
    dy = dx.T

    gradient_x = convolve2d(image_array, dx, mode='same', boundary='wrap')
    gradient_y = convolve2d(image_array, dy, mode='same', boundary='wrap')

    # Computed gradient magnitudes and angles
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_angle = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    # Ensured all angles are positive (0 to 180 degrees)
    gradient_angle[gradient_angle < 0] += 180

    # Created histograms for each cell
    for i in range(10):
        for j in range(10):
            cell_x_start = j * cell_size_x
            cell_x_end = (j + 1) * cell_size_x
            cell_y_start = i * cell_size_y
            cell_y_end = (i + 1) * cell_size_y

            cell_magnitude = gradient_magnitude[cell_y_start:cell_y_end, cell_x_start:cell_x_end]
            cell_angle = gradient_angle[cell_y_start:cell_y_end, cell_x_start:cell_x_end]

            for b in range(bin_count):
                bin_lower_angle = b * angle_per_bin
                bin_upper_angle = (b + 1) * angle_per_bin

                # Found pixels with angles within this bin's range
                bin_mask = np.logical_and(cell_angle >= bin_lower_angle, cell_angle < bin_upper_angle)

                # Summed up magnitudes of pixels in this bin
                hog_descriptor[i, j, b] = np.sum(cell_magnitude[bin_mask])

    # Normalized the HOG descriptor
    hog_descriptor /= np.sum(hog_descriptor)

    # Flattened the 10x10x9 descriptor into a 1D array (900-dimensional feature descriptor)
    hog_feature_vector = hog_descriptor.flatten()
    return hog_feature_vector


# Function to extract Resnet 50 Avgpool layer features
def extract_resnet_avgpool_1024(image):
    layer = CNN_MODEL.avgpool
    avgpool_output = extract_feature_vector(image, layer)
    reduced_feature = avgpool_output.view(-1, 2).mean(dim=1)
    return reduced_feature.numpy()


# Function to extract Resnet 50 layer 3 features
def extract_resnet_layer3_1024(image):
    layer = CNN_MODEL.layer3
    layer3_output = extract_feature_vector(image, layer)
    reduced_feature = layer3_output.view(1024, 1, -1).mean(dim=2).reshape(-1)
    return reduced_feature.numpy()


# Function to extract Resnet 50 Fully Connected layer features
def extract_resnet_fc_1000(image):
    layer = CNN_MODEL.fc
    feature_output = extract_feature_vector(image, layer)
    return feature_output.numpy()


# Function to print feature descriptors
def print_feature_details(features):
    for feature_name, feature in features:
        print(f"Feature Descriptor Name: {feature_name}")
        print(f"Feature Descriptor Shape: {feature.shape}")
        print(f"Feature Descriptor: {feature}")


# Function to calculate user specified features and display input image
def extract_features_and_visualize_image(image_id, feature_options):
    feature_list = []
    dataset = datasets.Caltech101(BASE_DIR)
    image = dataset[image_id][0]
    image = image.convert('RGB')
    image.show()
    match feature_options:
        case 1:
            feature_list.append(("HOG feature", extract_hog_descriptor(image)))
        case 2:
            feature_list.append(("Color Moments feature", extract_color_moment(image)))
        case 3:
            feature_list.append(("Resnet Layer 3 feature", extract_resnet_layer3_1024(image)))
        case 4:
            feature_list.append(("Resnet Avgpool feature", extract_resnet_avgpool_1024(image)))
        case 5:
            feature_list.append(("Resnet FC feature", extract_resnet_fc_1000(image)))
        case 6:
            feature_list.append(("HOG feature", extract_hog_descriptor(image)))
            feature_list.append(("Color Moments feature", extract_color_moment(image)))
            feature_list.append(("Resnet Layer 3 feature", extract_resnet_layer3_1024(image)))
            feature_list.append(("Resnet Avgpool feature", extract_resnet_avgpool_1024(image)))
            feature_list.append(("Resnet FC feature", extract_resnet_fc_1000(image)))

    print_feature_details(feature_list)


def get_k_nearest_neighbours(input_label, k, feature_option):
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
    input_label_features = collection.find({"image_label": input_label}, {index: 1, "_id": 0})
    input_label_features = [input_label_feature[index] for input_label_feature in input_label_features]

    cumulative_input_label_features = [sum(inner_list) for inner_list in zip(*input_label_features)]

    # Divide the sums by the number of inner lists to get the average
    input_label_features = [i / len(input_label_features) for i in cumulative_input_label_features]

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
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(
                    feature_vector_similarity_list,
                    input_label_features,
                    image_features["hog_descriptor"],
                    feature_option)

            case 2:
                # Calculated euclidian distance between input image and iterated image for color moments feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(
                    feature_vector_similarity_list,
                    input_label_features,
                    image_features["color_moments"],
                    feature_option)

            case 3:
                # Calculated cosine similarity between input image and iterated image for resnet layer 3 feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(
                    feature_vector_similarity_list,
                    input_label_features,
                    image_features[
                        "resnet_layer3_1024"],
                    feature_option)

            case 4:
                # Calculated cosine similarity between input image and iterated image for resnet avgpool layer feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(
                    feature_vector_similarity_list,
                    input_label_features,
                    image_features[
                        "resnet_avgpool_1024"],
                    feature_option)
            case 5:
                # Calculated cosine similarity between input image and iterated image for resnet FC layer feature descriptor
                feature_vector_similarity_list = get_feature_vector_similarity_sorted_pairs(
                    feature_vector_similarity_list,
                    input_label_features,
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
    plot_result(feature_vector_similarity_sorted_pairs[:k], image_id_list, k, input_label, feature_option)


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


# Function to calculate cosine similarity between 2 vectors
def cosine_similarity(vector_a, vector_b):
    # Calculated the dot product of the two vectors
    dot_product = np.dot(vector_a, vector_b)

    # Calculated the Euclidean norm (magnitude) of each vector
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # Calculated the cosine similarity
    similarity = dot_product / (norm_a * norm_b)

    return similarity


# Function to calculate euclidian distance between 2 vectors
def calculate_euclidian_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


# Function to plot k similar images against input image for all 5 feature models
def plot_result(feature_vector_similarity_sorted_pairs, image_id_list, k, input_label, feature_option):
    dataset = datasets.Caltech101(BASE_DIR)

    # Number of images per row
    images_per_row = k

    # Number of rows needed(1 Original image + 5 Feature models)
    num_rows = 2
    fig, axes = plt.subplots(num_rows, images_per_row + 1, figsize=(30, 25))
    plt.subplots_adjust(wspace=0.5)

    # Load and display the original image
    original_label = f"Input Label: {input_label}"

    # axes[0, 1].imshow(original_img, cmap="gray")
    # axes[0, 1].axis('off')

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
        if j in [0]:
            continue
        fig.delaxes(axes[0, j])

    plt.tight_layout()

    # Saved output to output dir
    current_epoch_timestamp = int(datetime.now().timestamp())
    plt.savefig(f"../Outputs/id_{input_label}_k_{k}_ts_{current_epoch_timestamp}.png")

    plt.show()


# Driver function to compute features
def driver():
    input_label = int(input("Please enter image label\n"))
    while input_label < 0 or input_label > 100:
        print(f"Invalid label value: {input_label}. Please pick label in range of 0-100\n.")
        input_label = int(input("Please enter image label\n"))
    latent_semantic_option = int(input("Please enter latent semantic in format Task-FeatureModel-ReducedDimension-DimensionReductionTechnique\n"
                               "Valid Tasks: T3, T4, T5, T6\n"
                               "Valid Feature Model: CM, HOG, AvgPool, L3, FC, RESNET\n"
                               "Valid Reduced Dimension: 1 - length of feature model\n"
                               "Valid Dimension Reduction Technique: SVD, NNMF, LDA, kmeans\n"))
    is_valid_feature_option = validate_latent_semantic_option(latent_semantic_option)
    while is_valid_feature_option not in list(range(1, 7)):
        print(f"Invalid input: {latent_semantic_option}")
        latent_semantic_option = int(input(
            "Please enter latent semantic in format Task-FeatureModel-ReducedDimension-DimensionReductionTechnique\n"
            "Valid Tasks: T3, T4, T5, T6\n"
            "Valid Feature Model: CM, HOG, AvgPool, L3, FC, RESNET\n"
            "Valid Reduced Dimension: 1 - length of feature model\n"
            "Valid Dimension Reduction Technique: SVD, NNMF, LDA, kmeans\n"))
        is_valid_feature_option = validate_latent_semantic_option(latent_semantic_option)

    k = int(input("Select K to find K similar images to given input image\n"))
    while k < 1 or k > 8676:
        print(f"Invalid K value: {k}. Please pick K in range of 1-8676.")
        k = int(input("Select K to find K similar images to given input image\n"))
    get_k_nearest_neighbours(input_label, k, latent_semantic_option)


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
    driver()
