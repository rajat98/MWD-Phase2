import torch
import torchvision.datasets as datasets
import torchvision.models as models
from pymongo import MongoClient

from Code.utilities import extract_hog_descriptor, \
    extract_resnet_layer3_1024, extract_color_moment, extract_resnet_avgpool_1024, extract_resnet_fc_1000

torch.set_grad_enabled(False)

ROOT_DIR = '/home/rpaw/MWD/caltech-101/caltech-101/101_ObjectCategories/'
CNN_MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
BASE_DIR = '/home/rpaw/MWD/'

# MongoDB Client Setup
MONGO_CLIENT = MongoClient("mongodb://adminUser:adminPassword@localhost:27017/mwd_db?authSource=admin")
DATABASE = MONGO_CLIENT['mwd_db']


# Function to save all 5 feature descriptor for o image
def save_feature_descriptor(document):
    collection = DATABASE.feature_descriptors
    try:
        collection.insert_one(document)
        print("Data inserted successfully")
    except Exception as e:
        print(f"Error inserting data: {e}")


# Function to save all 5 feature descriptor for complete dataset
def save_feature_descriptors():
    dataset = datasets.Caltech101(BASE_DIR)
    collection = DATABASE.even_numbered_feature_space

    for image_id in range(len(dataset)):
        if image_id % 2 == 1:
            continue
        print(f"Image ID: {image_id}")
        image = dataset[image_id][0]
        image = image.convert('RGB')
        image_label = dataset[image_id][1]

        if collection.find_one({"image_id": image_id}) is not None:
            continue

        # Extracted Feature
        features = {"resnet_avgpool_fd": extract_resnet_avgpool_1024(image),
                    "resnet_layer3_fd": extract_resnet_layer3_1024(image),
                    "resnet_fc_fd": extract_resnet_fc_1000(image),
                    "HOG_fd": extract_hog_descriptor(image),
                    "color_moment_fd": extract_color_moment(image)}

        # Created document for feature_descriptor collection
        document = {
            "image_id": image_id,
            "image_label": image_label,
            "color_moments": features["color_moment_fd"].tolist(),
            "hog_descriptor": features["HOG_fd"].tolist(),
            "resnet_avgpool_1024": features["resnet_avgpool_fd"].tolist(),
            "resnet_layer3_1024": features["resnet_layer3_fd"].tolist(),
            "resnet_fc_1000": features["resnet_fc_fd"].tolist()
        }
        save_feature_descriptor(document)


save_feature_descriptors()
