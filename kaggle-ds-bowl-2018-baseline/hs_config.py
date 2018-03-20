from config import Config

#HS config classes must override the config.py. bowl_config was wan extension. this is mine.

class HsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "hsConfig"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2 
    '''HS: 
    orig was 2, 16 woked with resNet50 in hsDebugTrain.ipyb but didnt work in shell with 16,12, or 8, or 4.
    With ResNet101, only works with 2, not even with 4.
    '''

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512 # chaging this alone to 128 works I think
    IMAGE_MAX_DIM = 512 # hs chaging this to 128 failed

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600

    STEPS_PER_EPOCH = None

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    USE_MINI_MASK = True

    MAX_GT_INSTANCES = 256

    DETECTION_MAX_INSTANCES = 512

    RESNET_ARCHITECTURE = "resnet101"  # original was "resnet50" i tried "resnet101" works with batchsize = 2.


hs_config = HsConfig()
hs_config.display()
