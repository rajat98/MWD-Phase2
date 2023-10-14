input_string = "T6-RESNET-5-SVD"

# Split the input string by '-' and get the last element (SVD_5)
parts = input_string.split('-')
svd_value = "_"parts[-2:]

print(svd_value)