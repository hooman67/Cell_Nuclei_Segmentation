import math
import numpy as np

class HsConfig():
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2#3 is too much for resnet101 
    '''HS: 
    orig was 2, 16 woked with resNet50 in hsDebugTrain.ipyb but didnt work in shell with 16,12, or 8, or 4.
    With ResNet101, only works with 2, not even with 4.
    '''


    '''
    After training for 2 epochs with RL0.0005 test error actually went up. So im gonna try
    0.0001LR with 0.001WeightDecat.

    I trained for 14 epochs using LR0.001 and WeightDecay0.0007 to achieve a 0.396 score.
    Adter eapoch 24 (actualyl epoch 10 since i was training on top of ep 14), the loss
    wasnt going down anymore. So I'm gonna try with LR0.0005, 

    Before hsTrain would divide the LR by 10, Learning with, after ep 49, I used 0.0001 but this 
    (actually 0.00001) was too low and the algorithm didnt learn after several epochs.. 
    #orig, 0.001

    The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    weights to explode. Likely due to differences in optimzer implementation.
    '''
    LEARNING_RATE = 0.0005 #6: trying 0.0001 again -- 5:0.0005 for some reason this didnt work -- 4:0.001 this was good for me -- 3: 0.0001 -- 2: 0.00001 = TOO LOW -- orig  0.001
    LEARNING_MOMENTUM = 0.9 

    # Weight decay regularization
    #hs: For me 0.0007 peformed much better than 0.0001
    WEIGHT_DECAY = 0.001 #3: 0.001, 2: 0.0007 #orig 0.0001.


    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.8 #orig0.7 i increase since I generate more ground truth ROI per image now  


    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    # Use small images for faster training. Set the limits of the small side
    # the same as the larger side, and that determines the image shape.
    IMAGE_MIN_DIM = 512 #orig512 chaging this alone to 128 works I think
    IMAGE_MAX_DIM = 512 #orig512 chaging this to 128 failed

        # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported


    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64) #orig (8, 16, 32, 64, 128)  # anchor side in pixels


    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600#orig 600, 1000 was too much


    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.

    #hs paper had this at 1000
    STEPS_PER_EPOCH = 1000 #orig None

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5# orig 5



    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE =  (56,56) #orig (56, 56)  # (height, width) of the mini-mask

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 512 #orig 256

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 512



    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]


    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]


    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512

    # Anchor stride
    # If 1 then anchors are created for every cell in the backbone feature map.
    # If 2, then anchors are created for every other cell (skip cells), and so on.
    RPN_ANCHOR_STRIDE = 1

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000#orig2000
    POST_NMS_ROIS_INFERENCE = 1000


    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    
    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])


    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True


    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei


    # Give the configuration a recognizable name
    NAME = "hsConfig"
    RESNET_ARCHITECTURE = "resnet101"  # original was "resnet50" i tried "resnet101" works with batchsize = 2.


    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


hs_config = HsConfig()
hs_config.display()
