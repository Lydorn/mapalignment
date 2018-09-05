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
DATASET_DIRPATH = os.path.join(ROOT_DIR, "data/mapping_challenge_dataset")
DATASET_RAW_DIRPATH = os.path.join(DATASET_DIRPATH, "raw")

TFRECORDS_DIR = os.path.join(DATASET_DIRPATH, "tfrecords.mapalign.multires")
TFRECORD_FILENAME_FORMAT = "{}.ds_fac_{:02d}.{{:06d}}.tfrecord"

DISP_GLOBAL_SHAPE = (5000, 5000)  # As Aerial Inria Dataset images
DISP_PATCH_RES = 300
DISP_MAP_COUNT = 1  # Number of displacement applied to polygons to generate to the displaced gt map (1 minimum, more for dataset augmentation)
DISP_MODES = 30  # Number of Gaussians mixed up to make the displacement map
DISP_GAUSS_MU_RANGE = [0, 1]  # Coordinates are normalized to [0, 1] before the function is applied
DISP_GAUSS_SIG_SCALING = [0.0, 0.002]  # Coordinates are normalized to [0, 1] before the function is applied
DISP_MAX_ABS_VALUE = 4
# DISP_MAP_MARGIN = 100  # Some polygons are slightly outside the image, this margin allows to take them into account

# Tile generation
REFERENCE_PIXEL_SIZE = 0.3  # In meters.
DOWNSAMPLING_FACTORS = [1, 2, 4, 8]
# The resulting pixel sizes will be equal to [REFERENCE_PIXEL_SIZE/DOWNSAMPLING_FACTOR for DOWNSAMPLING_FACTOR in DOWNSAMPLING_FACTORS]

# Split data into TRAIN and VAL with the already-made split of the data

# Split large tfrecord into several smaller tfrecords (shards)
RECORDS_PER_SHARD = 100
