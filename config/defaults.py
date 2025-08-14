from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# Name of backbone
_C.MODEL.TYPE = ''
# Model name
_C.MODEL.NAME = ''
_C.MODEL.LAST_LAYER= ''

# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.LOSS_TYPE = ['ce','triplet']
_C.MODEL.CLIP_LOSS_TYPE = 'constant'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False
# Dimension of the attribute list
_C.MODEL.META_DIMS = []
_C.MODEL.CAMERA_XISHU = 3
# ID number of GPU
_C.MODEL.DEVICE_ID = "0"
_C.MODEL.NONBIO_HEAD=False
_C.MODEL.CLIP_DIM=1024
_C.MODEL.EMBEDDING_DIM=0
_C.MODEL.SUBSPACE_DIM=0
_C.MODEL.BOTTEL_NECK=False
_C.MODEL.NORM_FEAT=False


# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1



# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 8
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_HEIGHT = 224
_C.DATA.IMG_WIDTH = 224
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4
# Data root
_C.DATA.ROOT = '/home/c3-0/datasets'
# Number of instances
_C.DATA.NUM_INSTANCES = 2 #8
# Batch size during testing
_C.DATA.TEST_BATCH = 128
# Data sampling strategy
_C.DATA.SAMPLER = 'softmax_triplet'
# Extract data containing attributes during data processing, options: 'True', 'False'
_C.DATA.AUX_INFO = False
# Filename containing attributes
_C.DATA.META_DIR = 'PAR_PETA_105.txt'
_C.DATA.NOBIO_INDEX=['2']
_C.DATA.BIO_INDEX='0'
_C.DATA.TEXT_MODEL='EVA02-CLIP'
_C.DATA.CAPTION_DIR='/home/xi860799/dataset/CogVLM_results/LTCC_ReID2'
_C.DATA.TEXT_MODEL_VERSION='EVA02-CLIP-bigE-14'
_C.DATA.SUMMERY_TEXT=True

_C.DATA.PIXEL_MEAN=[0.485, 0.456, 0.406]
_C.DATA.PIXEL_STD=[0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()

# Random crop prob
_C.AUG.RC_PROB = 0.5
# Random erase prob
_C.AUG.RE_PROB = 0.5
# Random flip prob
_C.AUG.RF_PROB = 0.5


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.WARMUP_LR = 7.8125e-07
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.LARGE_FC_FACTOR = 2.0
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.1
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 60)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 20
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.FREEZE_EPOCH=0

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Path to trained model
_C.TEST.WEIGHT = ""
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Test using images only
_C.TEST.TYPE = 'image_only'
_C.TEST.STATUS = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
_C.OUTPUT_NAME = ""


#video
_C.DATA.SAMPLING_STEP = 64
_C.AUG.SEQ_LEN = 0
# Sampling stride of each input video clip
_C.AUG.SAMPLING_STRIDE = 4
