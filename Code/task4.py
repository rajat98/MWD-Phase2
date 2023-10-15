import re

import numpy as np
import pandas as pd
from tensorly.decomposition import parafac


########################################################################
# function to parse string to get feature data in csv
########################################################################
def parse_string(string):
    values = re.findall(r'-?\d+\.\d+', string)
    np_array = np.array(values, dtype=float)
    return np_array


########################################################################
# function to create tensor
########################################################################
def create_tensor(model):
    df = pd.read_csv('even_image_features.csv')
    image_data = df[['ImageID', 'Labels', model]]

    # create tensor: [number of Images] x [Feature Length] x [number of Labels]
    num_images = len(image_data)
    feat_len = parse_string(image_data[model][0]).size
    num_labels = 101

    tensor = np.zeros((num_images, feat_len, num_labels))  # empty tensor

    for i in range(num_images):
        image_features = parse_string(image_data[model][i])
        tensor[i, :, image_data['Labels'][i]] = image_features

    return tensor


########################################################################
# Perform CP decomposition and get top k latent semantics
########################################################################
##### Perform CP decomposition and get top k latent semantics #####
def CP_extraction(model, k):
    tensor = create_tensor(model)

    # rank = number of latent semantics
    weights, factors = parafac(tensor, rank=k, normalize_factors=True)

    # Extract top-k latent semantics for each factor matrices
    factor_matrices_semantics = []
    for factor in range(len(factors)):
        semantics = []
        for component in range(k):
            labels = [*range(factors[factor].shape[0])]
            # get weight for current latent semantic and scale by parafac weight
            adjusted_weights = factors[factor][:, component] * weights[component]

            # pair label with weights
            label_weight_pairs = [(labels[i], float(adjusted_weights[i])) for i in range(len(labels))]

            # sort based on weight magnitude
            sorted_pairs = sorted(label_weight_pairs, key=lambda x: abs(x[1]), reverse=True)

            semantics.append(sorted_pairs)

    factor_matrices_semantics.append(semantics)
    return factor_matrices_semantics


########################################################################
# main - performs task4
# asks user for feature selection and value k
# creates csv for top-k latent semantics
########################################################################
def main():
    selection_list = ["CM", "HOG", "AvgPool", "L3", "FC", "RESNET"]

    model = input("Input a feature model from the following list:"
                  + "\n CM, HOG, AvgPool, L3, FC, RESNET")

    k = int(input("Input a value k"))

    # handle error if user inputs invalid feature selection
    if model not in selection_list:
        print("Invalid feature model selection. Please select from the following list:"
              + "\n CM, HOG, AvgPool, L3, FC, RESNET")
        return

    # run task 4, get label-weight pairs
    label_weight_pairs = CP_extraction(model, k)

    # Store the latent semantics in a properly named output file
    # List label-weight pairs, ordered in decreasing order of weights
    latent_semantics = CP_extraction(model, k)

    df1 = pd.DataFrame(latent_semantics[0])
    df2 = pd.DataFrame(latent_semantics[1])
    df3 = pd.DataFrame(latent_semantics[2])

    # Store the latent semantics in a properly named output file
    # List label-weight pairs, ordered in decreasing order of weights
    output_file = "t4_" + model + "_" + str(k) + ".csv"
    with open(output_file, 'w', newline='') as csvfile:
        for df in [df1, df2, df3]:
            df.to_csv(csvfile)
            csvfile.write("\n")

    return


if __name__ == "__main__":
    main()

