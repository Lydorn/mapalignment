import os
import sys

sys.path.append("../../utils")
import python_utils


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset online processing
DATA_DIR = python_utils.choose_first_existing_path([
    "/local/shared/epitome-polygon-deep-learning/data",  # Try local node first
    "/home/nigirard/epitome-polygon-deep-learning/data",

    "/workspace/data",  # Try inside Docker image
])
if DATA_DIR is None:
    print("ERROR: Data directory not found!")
    exit()
else:
    print("Using data from {}".format(DATA_DIR))

REFERENCE_PIXEL_SIZE = 0.3  # In meters.
DS_FAC_LIST = [1, 2, 4, 8]
DS_REPEAT_LIST = [1, 4, 16, 64]  # To balance more samples in batches, otherwise there would be too few samples with downsampling_factor=8
IMAGE_DYNAMIC_RANGE = [-1, 1]
DISP_MAP_DYNAMIC_RANGE_FAC = 0.5  # Sets disp_map values in [-0.5, 0.5]
DISP_MAX_ABS_VALUE = 4
TFRECORDS_DIR_LIST = [
    os.path.join(DATA_DIR, "AerialImageDataset/tfrecords.mapalign.multires"),
    os.path.join(DATA_DIR, "bradbury_buildings_roads_height_dataset/tfrecords.mapalign.multires"),
    os.path.join(DATA_DIR, "mapping_challenge_dataset/tfrecords.mapalign.multires"),
]
TFRECORD_FILENAME_FORMAT = "{}.ds_fac_{:02d}.{{:06d}}.tfrecord"  # Dataset fold, downsampling factor, shard index
KEEP_POLY_PROB = 0.1  # Default: 0.1  # Default: 0.5  # Each input misaligned polygon has a 50% change to be kept and 50% to be removed
# KEEP_POLY_PROB = 1  # Default: 0.1  # Default: 0.5  # Each input misaligned polygon has a 50% change to be kept and 50% to be removed
DATA_AUG = True

# Model(s)
INPUT_RES = 220
IMAGE_INPUT_CHANNELS = 3
POLY_MAP_INPUT_CHANNELS = 3  # (0: area, 1: edge, 2: vertex)
IMAGE_FEATURE_BASE_COUNT = 16 * 2
POLY_MAP_FEATURE_BASE_COUNT = 8 * 2
COMMON_FEATURE_BASE_COUNT = 24 * 2
POOL_COUNT = 3  # Number of 2x2 pooling operations (Min: 1). Results in (MODEL_POOL_COUNT + 1) resolution levels.
# POOL_COUNT = 2  # Number of 2x2 pooling operations (Min: 1). Results in (MODEL_POOL_COUNT + 1) resolution levels.
DISP_OUTPUT_CHANNELS = 2  # Displacement map channel count (0: i, 1: j)
ADD_SEG_OUTPUT = True
# ADD_SEG_OUTPUT = False
SEG_OUTPUT_CHANNELS = 4  # Segmentation channel count (0: background, 1: area, 2: edge, 3: vertex)

# Losses
# Implicitly we have DISP_POLYGON_BACKGROUND_COEF = 0.0
DISP_POLYGON_FILL_COEF = 0.1
DISP_POLYGON_OUTLINE_COEF = 1
DISP_POLYGON_VERTEX_COEF = 10

SEG_BACKGROUND_COEF = 0.05
SEG_POLYGON_FILL_COEF = 0.1
SEG_POLYGON_OUTLINE_COEF = 1
SEG_POLYGON_VERTEX_COEF = 10

DISP_LOSS_COEF = 100
SEG_LOSS_COEF = 50
LAPLACIAN_PENALTY_COEF = 0  # Default: 10000  # TODO: experiment again with non-zero values (Now that the Laplacian penalty bug is fixed)

# Each level's prediction has a different loss coefficient that can also be changed over time
# Note: len(LEVEL_LOSS_COEFS_PARAMS) must be equal to MODEL_POOL_COUNT
# Note: There are (MODEL_POOL_COUNT + 1) resolution levels in total but the last level does not have prediction outputs
# to compute a level loss on (it is the bottom of the "U" of the U-Net)
# Note: Values must be floats
LEVEL_LOSS_COEFS_PARAMS = [
    # Level 0, same resolution as input image
    {
        "boundaries": [2500, 5000, 7500],
        "values": [0.50, 0.75, 0.9, 1.0]
    },
    {
        "boundaries": [2500, 5000, 7500],
        "values": [0.35, 0.20, 0.1, 0.0]
    },
    {
        "boundaries": [2500, 5000, 7500],
        "values": [0.15, 0.05, 0.0, 0.0]
    },
]
# LEVEL_LOSS_COEFS_PARAMS = [
#     # Level 0, same resolution as input image
#     {
#         "boundaries": [2500, 5000, 7500],
#         "values": [1.0, 1.0, 1.0, 1.0]
#     },
#     {
#         "boundaries": [2500, 5000, 7500],
#         "values": [0.0, 0.0, 0.0, 0.0]
#     },
#     {
#         "boundaries": [2500, 5000, 7500],
#         "values": [0.0, 0.0, 0.0, 0.0]
#     },
# ]
# LEVEL_LOSS_COEFS_PARAMS = [
#     # Level 0, same resolution as input image
#     {
#         "boundaries": [2500, 5000, 7500],
#         "values": [1.0, 1.0, 1.0, 1.0]
#     },
#     {
#         "boundaries": [2500, 5000, 7500],
#         "values": [0.0, 0.0, 0.0, 0.0]
#     },
# ]
assert len(LEVEL_LOSS_COEFS_PARAMS) == POOL_COUNT, \
    "LEVEL_LOSS_COEFS_PARAMS ({} elements) must have MODEL_RES_LEVELS ({}) elements".format(
        len(LEVEL_LOSS_COEFS_PARAMS), POOL_COUNT)

# Training
PLOT_RESULTS = False  # Is extremely slow when True inside Docker...

BASE_LEARNING_RATE = 1e-4
LEARNING_RATE_PARAMS = {
    "boundaries": [25000],
    "values": [BASE_LEARNING_RATE, 0.5 * BASE_LEARNING_RATE]
}
WEIGHT_DECAY = 1e-4  # Default: 1e-6
DROPOUT_KEEP_PROB = 1.0
MAX_ITER = 100000

TRAIN_SUMMARY_STEP = 50
VAL_SUMMARY_STEP = 250
CHECKPOINT_STEP = 250

# Outputs
MODEL_NAME = "mapalign_mutlires"
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
LOGS_DIRNAME = "logs"
CHECKPOINTS_DIRNAME = "checkpoints"
