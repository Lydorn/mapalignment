import os

# A list of possible root dir depending on where the code is:
ROOT_DIR_LIST = [
    "/local/shared/epitome-polygon-deep-learning",  # Inria cluster node nefgpu23
    "/home/nigirard/epitome-polygon-deep-learning",  # Landsat
    "/workspace",  # Docker (mainly when using Deepsat or personal computer)
]
ROOT_DIR_INDEX = 0
while not os.path.exists(ROOT_DIR_LIST[ROOT_DIR_INDEX]):
    ROOT_DIR_INDEX += 1
ROOT_DIR = ROOT_DIR_LIST[ROOT_DIR_INDEX]
print("ROOT_DIR: {}".format(ROOT_DIR))

# Dataset offline pre-processing
DATASET_DIR = os.path.join(ROOT_DIR, "data/AerialImageDataset")
IMAGES_DIR_LIST = [
    os.path.join(DATASET_DIR, "raw/train/images"),
    os.path.join(DATASET_DIR, "raw/test/images")
]
IMAGE_EXTENSION = "tif"
GT_POLYGONS_DIR_NAME = "gt_polygons"
GT_POLYGONS_EXTENSION = "npy"

TILE_RES = 220  # The maximum patch size will be 220. Adjusted for rotation will be ceil(220*sqrt(2)) = 312.
TILE_STRIDE = 100  # The maximum inner path res will be 100

# If True, generates patches with increased size to account for cropping during the online processing step
DATA_AUG_ROT = True   # data_aug_rot only applies to train

TFRECORDS_DIR = os.path.join(DATASET_DIR, "tfrecords.mapalign.multires")
TFRECORD_FILENAME_FORMAT = "{}.ds_fac_{:02d}.{{:06d}}.tfrecord"

DISP_MAP_COUNT = 1  # Number of displacement applied to polygons to generate the displaced gt map (1 minimum, more for dataset augmentation)
DISP_MODES = 30  # Number of Gaussians mixed up to make the displacement map (Default: 20)
DISP_GAUSS_MU_RANGE = [0, 1]  # Coordinates are normalized to [0, 1] before the function is applied
DISP_GAUSS_SIG_SCALING = [0.0, 0.002]  # Coordinates are normalized to [0, 1] before the function is applied
DISP_MAX_ABS_VALUE = 4  # In pixels in the downsampled resolutions.
# DISP_MAX_ABS_VALUE = 2  # In pixels in the downsampled resolutions.

# Tile generation
DOWNSAMPLING_FACTORS = [1, 2, 4, 8, 16]
# For 16, 5000x5000px images will be rescaled to 312x312px. Which corresponds to the rotation-adjusted tile_res

# Choose min downsampling factor for each city (depends on the quality of alignement)
CITY_MIN_DOWNSAMPLING_FACTOR = {
    "bloomington": 4,
    "bellingham": 4,
    "innsbruck": 2,
    "sfo": 4,
    "tyrol-e": 4,
    "austin": 1,
    "chicago": 1,
    "kitsap": 4,
    "tyrol-w": 2,
    "vienna": 2,
}

# Split data into TRAIN, VAL and TEST
TRAIN_COUNT = 288  # 360*  0.8 = 288
VAL_COUNT = 72  # 360 * 0.2 = 288
TEST_COUNT = 0

IMAGE_INDEX_START = 0
IMAGE_INDEX_END = -1  # -1 Means process all images (other values are used for testing quickly)

# Split large tfrecord into several smaller tfrecords (shards)
RECORDS_PER_SHARD = 100
