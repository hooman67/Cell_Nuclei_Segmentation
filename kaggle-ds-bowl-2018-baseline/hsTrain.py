import os

import sys
import random
import math
import time

#from hs_config import hs_config
from hs_config_resNet50 import hs_config
from bowl_dataset import BowlDataset
import utils
import hsModel as modellib
from hsModel import log
from glob import glob



# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

########## HS path to whatever .h5 file you want####################
hsSavedWeightPath = os.path.join(MODEL_DIR, "maskRCNN_Resnet50_ep194_score339.h5")
###########################################


# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
model = modellib.MaskRCNN(mode="training", config=hs_config,
                          model_dir=MODEL_DIR)



# Which weights to start with?
init_with = "hs"  # imagenet, coco, last, or hs

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

elif init_with == "hs":
    # hs to manually load the weights
    model.load_weights(hsSavedWeightPath, by_name=True)


    
# Training dataset
dataset_train = BowlDataset()
dataset_train.load_bowl('stage1_train')
dataset_train.prepare()

# # Validation dataset
dataset_val = BowlDataset()
dataset_val.load_bowl('stage1_train')
dataset_val.prepare()



# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
#model.train(dataset_train, dataset_val, 
#            learning_rate=bowl_config.LEARNING_RATE, 
#            epochs=1, 
#            layers='heads')

model.train(dataset_train, dataset_val, 
            learning_rate=hs_config.LEARNING_RATE,
            epochs=400, 
            layers="all")
