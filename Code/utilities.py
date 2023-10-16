import os
import pickle

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from PIL import ImageOps
from pymongo import MongoClient
from scipy.signal import convolve2d
from sklearn.decomposition import LatentDirichletAllocation
from torchvision.transforms import transforms
from tqdm import tqdm

torch.set_grad_enabled(False)

ROOT_DIR = '/home/rpaw/MWD/caltech-101/caltech-101/101_ObjectCategories/'
CNN_MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
BASE_DIR = '/home/rpaw/MWD/'

# MongoDB Client Setup
MONGO_CLIENT = MongoClient("mongodb://adminUser:adminPassword@localhost:27017/mwd_db?authSource=admin")
DATABASE = MONGO_CLIENT['mwd_db']

task_to_string_map = {
    3: "T3",
    4: "T4",
    5: "T5",
    6: "T6"
}
dim_red_opn_to_string_map = {
    1: "SVD",
    2: "NMF",
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
feature_model_dict = {
    1: 'HOG',
    2: 'ColorMoments',
    3: 'ResNet_Layer3_1024',
    4: 'ResNet_AvgPool_1024',
    5: 'ResNet_FC_1000'}

feature_option_to_feature_index_mapping = {
    1: "hog_descriptor",
    2: "color_moments",
    3: "resnet_layer3_1024",
    4: "resnet_avgpool_1024",
    5: "resnet_fc_1000"
}


def get_positive_feature_matrix(feature_matrix):
    min_value = np.min(feature_matrix)
    matrix_positive = feature_matrix + abs(min_value)
    normalized_matrix = (matrix_positive - np.min(matrix_positive)) / (
            np.max(matrix_positive) - np.min(matrix_positive))
    return normalized_matrix


def svd(k, feature_matrix):
    covariance_matrix_1 = np.dot(feature_matrix.T, feature_matrix)
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(covariance_matrix_1)
    ncols1 = np.argsort(eigenvalues_1)[::-1]
    covariance_matrix_2 = np.dot(feature_matrix, feature_matrix.T)
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(covariance_matrix_2)
    ncols2 = np.argsort(eigenvalues_2)[::-1]
    v_transpose = eigenvectors_1[ncols1].T
    u = eigenvectors_2[ncols2]
    sigma = np.diag(np.sqrt(eigenvalues_1)[::-1])
    trucated_u = u[:, :k]
    trucated_sigma = sigma[:k, :k]
    truncated_v_transpose = v_transpose[:k, :]
    image_to_latent_features = feature_matrix @ truncated_v_transpose.T
    latent_feature_to_original_feature = truncated_v_transpose
    # svd = TruncatedSVD(n_components=k)
    # reduced_data = svd.fit_transform(feature_matrix)
    # image_to_latent_features = feature_matrix @ v_transpose.T
    # latent_feature_to_original_feature = v_transpose
    return image_to_latent_features, latent_feature_to_original_feature


def kmeans(k, feature_matrix):
    m, n = feature_matrix.shape
    # Initialize centroids randomly
    # feature_matrix = feature_matrix.to_numpy()
    centroids = feature_matrix[np.random.choice(n, k, replace=False)]
    # Number of iterations
    max_iterations = 100
    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(feature_matrix[:, np.newaxis] - centroids, axis=2), axis=1)
        # Update centroids to the mean of the assigned data points
        new_centroids = np.array([feature_matrix[labels == i].mean(axis=0) for i in range(k)])
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    k, _ = centroids.shape
    expanded_data = feature_matrix[:, np.newaxis, :]
    expanded_centroids = centroids[np.newaxis, :, :]
    latent_feature = np.sqrt(np.sum((expanded_data - expanded_centroids) ** 2, axis=2))
    return centroids, latent_feature


def lda(k, feature_matrix):
    feature_matrix = get_positive_feature_matrix(feature_matrix)
    lda = LatentDirichletAllocation(n_components=k)
    lda.fit(feature_matrix)
    latent_feature_to_original_feature = lda.components_
    image_to_latent_features = feature_matrix @ latent_feature_to_original_feature.T
    return image_to_latent_features, latent_feature_to_original_feature


def nnmf(k, feature_matrix):
    max_iter = 200
    feature_matrix = get_positive_feature_matrix(feature_matrix)
    number_labels = feature_matrix.shape[0]
    W = np.random.uniform(1, 2, (number_labels, k))
    H = np.random.uniform(1, 2, (k, number_labels))
    for n in tqdm(range(max_iter)):
        # Update H
        W_TA = np.dot(W.T, feature_matrix)
        W_TWH = np.dot(np.dot(W.T, W), H) + + 1.0e-10
        H = H * (W_TA / W_TWH)
        # Update W
        AH_T = np.dot(feature_matrix, H.T)
        WHH_T = np.dot(np.dot(W, H), H.T) + 1.0e-10
        W = W * (AH_T / WHH_T)
    image_to_latent_features = feature_matrix @ H.T
    latent_feature_to_original_feature = H
    return image_to_latent_features, latent_feature_to_original_feature


def print_image_id_weight_pairs(latent_features):
    latent_features = pd.DataFrame(latent_features)
    for index, latent_feature in latent_features.iterrows():
        print(f'Original Features: {latent_feature.values}')
        # Get sorted indices in descending order
        sorted_indices = latent_feature.argsort()[::-1]
        # Sort the row in descending order
        sorted_row = latent_feature[sorted_indices]
        print(f'Dominant Features: {sorted_row.values}')
        print(f'Dominant Feature Number: {sorted_indices.values}')
        print('-' * 30)


def save_latent_features_to_file(task_number, image_to_latent_features, latent_feature_to_original_feature,
                                 feature_option, dim_red_opn, k):
    latent_features = {'image_to_latent_features': image_to_latent_features,
                       'latent_feature_to_original_feature': latent_feature_to_original_feature}
    task = task_to_string_map[task_number]
    feature_model = feature_option_to_feature_index_map[feature_option]
    method = dim_red_opn_to_string_map[dim_red_opn]
    latent_feature_storage_path = f"Outputs/{task}/{feature_model}/{method}_{k}.pkl"
    os.makedirs(os.path.dirname(latent_feature_storage_path), exist_ok=True)
    with open(latent_feature_storage_path, 'wb') as file:
        pickle.dump(latent_features, file)


def calculate_similarity(latent_semantics1, latent_semantics2):
    """Calculate the cosine similarity between two latent semantics."""
    # return np.dot(latent_semantics1, latent_semantics2) / (np.linalg.norm(latent_semantics1) * np.linalg.norm(latent_semantics2))
    return np.linalg.norm(latent_semantics1 - latent_semantics2)


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


# Function to extract color moments from an image
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
    image_height = 100
    image_width = 300
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

    gradient_x = convolve2d(image_array, dx, mode='same', boundary='symm')
    gradient_y = convolve2d(image_array, dy, mode='same', boundary='symm')

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


# Function to calculate cosine similarity between 2 vectors
def cosine_similarity(vector_a, vector_b):
    # Calculated the dot product of the two vectors
    dot_product = np.dot(vector_a, vector_b)

    # Calculated the Euclidean norm (magnitude) of each vector
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # Calculated the cosine similarity
    similarity = dot_product / (norm_a * norm_b)
    print("future computed")
    return similarity


# Function to calculate euclidian distance between 2 vectors
def calculate_euclidian_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def get_resnet_feature(feature):
    return torch.nn.functional.softmax(torch.tensor(feature, dtype=torch.float)).tolist()
