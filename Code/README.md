
---

# Image Feature Extraction and Similarity Analysis

This repository contains Python scripts for the performing following tasks:
1. Compute Resnet 50(Layer 3, Avgpool & Fully Connected layer), Color Moments and Histogram of oriented gradients(HOG) features for a given image.
2. Extracting and storing all feature descriptors in MongoDB.
3. Finding K similar images using each feature descriptor paired with some similarity measure individually for the given input image.

The code is designed to work with the Caltech-101 dataset but can be adapted for other image datasets.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Features](#features)

## Prerequisites

Before using this code, ensure you have the following prerequisites installed:

- Python 3
- PyTorch
- torchvision
- NumPy
- SciPy
- Matplotlib
- PIL (Python Imaging Library)
- MongoDB (for storing and retrieving feature descriptors)

## Project Structure                  

The project is organized into three main Python scripts, each serving a specific purpose:

1. **task1.py**: Extracts various image feature descriptors, including Histogram of Oriented Gradients (HOG), Color Moments, and features from a pre-trained ResNet-50 model.
   
2. **task2.py**: Saves extracted feature descriptors to a MongoDB database for efficient retrieval and analysis.
   
3. **task3.py**: Performs similarity analysis on the dataset by calculating the similarity between images using different feature descriptors and visualizes the results.

## Usage

To use this code, follow these steps:

1. Ensure that you have installed all the prerequisites mentioned above.

2. Download the Caltech-101 dataset and place it in the specified `ROOT_DIR`.

3. Run `task1.py` to extract and preprocess feature descriptors for a given image. Modify the `ROOT_DIR` and `BASE_DIR` variables to match your dataset's location.  
Input Format:  
Enter valid `Image ID` to compute all feature descriptor for it.
4. Run `task2.py` to save the extracted feature descriptors to a MongoDB database. Make sure to configure the MongoDB connection in the script.  
Input Format:  
No input required  
5. Run `task3.py` to perform similarity analysis on the dataset. You can specify the image ID and the number of similar images (K) to retrieve for each feature descriptor.  
Input Format:  
Enter valid `Image ID` to compute all feature descriptor for it.  
Enter valid value of `K` to find `K` similar images to the input image.

## Features

This code supports the extraction and analysis of the following image feature descriptors:

- Histogram of Oriented Gradients (HOG)
- Color Moments
- Features from ResNet-50:
  - Resnet Layer 3 features
  - Resnet Avgpool features
  - Resnet Fully Connected (FC) layer features
---

