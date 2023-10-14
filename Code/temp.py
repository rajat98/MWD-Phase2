import numpy as np

# Input array
input_array = np.array([[[1, 2], [1, 2]], [[2, 2], [2, 2]]])

# Calculate the average along the first axis (axis=0)
averaged_array = np.mean(input_array, axis=1)

# Convert the result to a list if needed
averaged_list = averaged_array.tolist()

print(averaged_list)