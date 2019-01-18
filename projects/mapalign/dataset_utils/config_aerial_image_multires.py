import os


def choose_first_existing_dir(dir_list):
    for dir in dir_list:
        if os.path.exists(dir):
            return dir
    return None

ROOT_DIR = choose_first_existing_dir([
    "/local/shared/epitome-polygon-deep-learning",  # Inria cluster node nefgpu23
    "/home/nigirard/epitome-polygon-deep-learning",  # Landsat
    "/workspace",  # Docker (mainly when using Deepsat or personal computer)
])
print("ROOT_DIR: {}".format(ROOT_DIR))

# Dataset offline pre-processing
DATASET_DIRPATH = os.path.join(ROOT_DIR, "data/AerialImageDataset")
DATASET_RAW_DIRPATH = os.path.join(DATASET_DIRPATH, "raw")
DATASET_OVERWRITE_POLYGON_DIR_NAME = None  # Can be "aligned_noisy_gt_polygons_1"

TILE_RES = 220  # The maximum patch size will be 220. Adjusted for rotation will be ceil(220*sqrt(2)) = 312.
TILE_STRIDE = 100  # The maximum inner path res will be 100

# If True, generates patches with increased size to account for cropping during the online processing step
DATA_AUG_ROT = True  # data_aug_rot only applies to train

TFRECORDS_DIR = os.path.join(DATASET_DIRPATH, "tfrecords.mapalign.multires.aligned_noisy_1")
TFRECORD_FILEPATH_FORMAT = "{}/{}/ds_fac_{:02d}.{{:06d}}.tfrecord"  # Fold, image name, ds_fac, shard number

DISP_MAP_COUNT = 1  # Number of displacement applied to polygons to generate the displaced gt map (1 minimum, more for dataset augmentation)
DISP_MODES = 30  # Number of Gaussians mixed up to make the displacement map (Default: 20)
DISP_GAUSS_MU_RANGE = [0, 1]  # Coordinates are normalized to [0, 1] before the function is applied
DISP_GAUSS_SIG_SCALING = [0.0, 0.002]  # Coordinates are normalized to [0, 1] before the function is applied
DISP_MAX_ABS_VALUE = 4  # In pixels in the downsampled resolutions.
# DISP_MAX_ABS_VALUE = 2  # In pixels in the downsampled resolutions.

# Tile generation
DOWNSAMPLING_FACTORS = [1, 2, 4, 8, 16]
# For 16, 5000x5000px images will be rescaled to 312x312px. Which corresponds to the rotation-adjusted tile_res

TRAIN_IMAGES = [
    {
        "city": "bloomington",
        "numbers": list(range(1, 37)),
        "min_downsampling_factor": 1,  # Default: 4
    },
    {
        "city": "bellingham",
        "numbers": list(range(1, 37)),
        "min_downsampling_factor": 1,  # Default: 4
    },
    {
        "city": "innsbruck",
        "numbers": list(range(1, 37)),
        "min_downsampling_factor": 1,  # Default: 2
    },
    {
        "city": "sfo",
        "numbers": list(range(1, 37)),
        "min_downsampling_factor": 1,  # Default: 4
    },
    {
        "city": "tyrol-e",
        "numbers": list(range(1, 37)),
        "min_downsampling_factor": 1,  # Default: 4
    },
    {
        "city": "austin",
        "numbers": list(range(1, 37)),
        "min_downsampling_factor": 1,  # Default: 1
    },
    {
        "city": "chicago",
        "numbers": list(range(1, 37)),
        "min_downsampling_factor": 1,  # Default: 1
    },
    {
        "city": "kitsap",
        "numbers": list(range(1, 37)),
        "min_downsampling_factor": 1,  # Default: 4
    },
    {
        "city": "tyrol-w",
        "numbers": list(range(1, 37)),
        "min_downsampling_factor": 1,  # Default: 2
    },
    {
        "city": "vienna",
        "numbers": list(range(1, 37)),
        "min_downsampling_factor": 1,  # Default: 2
    },
]
VAL_IMAGES = []
TEST_IMAGES = []

# Split large tfrecord into several smaller tfrecords (shards)
RECORDS_PER_SHARD = 100
