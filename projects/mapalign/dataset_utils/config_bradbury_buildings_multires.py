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
DATASET_DIRPATH = os.path.join(ROOT_DIR, "data/bradbury_buildings_roads_height_dataset")
DATASET_RAW_DIRPATH = os.path.join(DATASET_DIRPATH, "raw")
DATASET_OVERWRITE_POLYGONS_FILENAME_EXTENSION = None  # Can be "_aligned_noisy_building_polygons_1.npy"

TILE_RES = 220  # The maximum patch size will be 220. Adjusted for rotation will be ceil(220*sqrt(2)) = 312.
TILE_STRIDE = 100  # The maximum inner path res will be 100

# If True, generates patches with increased size to account for cropping during the online processing step
DATA_AUG_ROT = True  # data_aug_rot only applies to train

TFRECORDS_DIR = os.path.join(DATASET_DIRPATH, "tfrecords.mapalign.multires.aligned_noisy_1")
TFRECORD_FILEPATH_FORMAT = "{}/{}/ds_fac_{:02d}.{{:06d}}.tfrecord"  # Fold, image name, ds_fac, shard number

DISP_MAP_COUNT = 1  # Number of displacement applied to polygons to generate to the displaced gt map (1 minimum, more for dataset augmentation)
DISP_MODES = 30  # Number of Gaussians mixed up to make the displacement map (Default: 20)
DISP_GAUSS_MU_RANGE = [0, 1]  # Coordinates are normalized to [0, 1] before the function is applied
DISP_GAUSS_SIG_SCALING = [0.0, 0.002]  # Coordinates are normalized to [0, 1] before the function is applied
DISP_MAX_ABS_VALUE = 4
# DISP_MAP_MARGIN = 100  # Some polygons are slightly outside the image, this margin allows to take them into account

# Tile generation
REFERENCE_PIXEL_SIZE = 0.3  # In meters.
# All images in AerialImage Dataset have a pixel size of 0.3m and
# a lot of this dataset's images have the same pixel size so that is why we choose this pixel size as a reference.
DOWNSAMPLING_FACTORS = [1, 2, 4, 8]
# The resulting pixel sizes will be equal to [REFERENCE_PIXEL_SIZE/DOWNSAMPLING_FACTOR for DOWNSAMPLING_FACTOR in DOWNSAMPLING_FACTORS]

# Split data into TRAIN, VAL and TEST
TRAIN_IMAGES = [
    {
        "city": "Arlington",
        "number": 1,
        "min_downsampling_factor": 1,
    },
    {
        "city": "Arlington",
        "number": 2,
        "min_downsampling_factor": 1,
    },
    {
        "city": "Arlington",
        "number": 3,
        "min_downsampling_factor": 1,
    },

    {
        "city": "Atlanta",
        "number": 1,
        "min_downsampling_factor": 1,
    },
    {
        "city": "Atlanta",
        "number": 2,
        "min_downsampling_factor": 1,
    },
    {
        "city": "Atlanta",
        "number": 3,
        "min_downsampling_factor": 1,
    },

    {
        "city": "Austin",
        "number": 1,
        "min_downsampling_factor": 1,
    },
    {
        "city": "Austin",
        "number": 2,
        "min_downsampling_factor": 1,
    },
    {
        "city": "Austin",
        "number": 3,
        "min_downsampling_factor": 1,
    },

    {
        "city": "DC",
        "number": 1,
        "min_downsampling_factor": 1,
    },
    {
        "city": "DC",
        "number": 2,
        "min_downsampling_factor": 1,
    },

    {
        "city": "NewHaven",
        "number": 1,
        "min_downsampling_factor": 1,
    },
    {
        "city": "NewHaven",
        "number": 2,
        "min_downsampling_factor": 1,
    },

    {
        "city": "NewYork",
        "number": 1,
        "min_downsampling_factor": 1,
    },
    {
        "city": "NewYork",
        "number": 2,
        "min_downsampling_factor": 1,
    },
    {
        "city": "NewYork",
        "number": 3,
        "min_downsampling_factor": 1,
    },

    {
        "city": "Norfolk",
        "number": 1,
        "min_downsampling_factor": 1,
    },
    {
        "city": "Norfolk",
        "number": 2,
        "min_downsampling_factor": 1,
    },
    {
        "city": "Norfolk",
        "number": 3,
        "min_downsampling_factor": 1,
    },

    {
        "city": "SanFrancisco",
        "number": 1,
        "min_downsampling_factor": 1,
    },
    {
        "city": "SanFrancisco",
        "number": 2,
        "min_downsampling_factor": 1,
    },
    {
        "city": "SanFrancisco",
        "number": 3,
        "min_downsampling_factor": 1,
    },

    {
        "city": "Seekonk",
        "number": 1,
        "min_downsampling_factor": 1,
    },
    {
        "city": "Seekonk",
        "number": 2,
        "min_downsampling_factor": 1,
    },
    {
        "city": "Seekonk",
        "number": 3,
        "min_downsampling_factor": 1,
    },

]
VAL_IMAGES = [

]
TEST_IMAGES = [

]

# Split large tfrecord into several smaller tfrecords (shards)
RECORDS_PER_SHARD = 100

### --- Save --- #
# TRAIN_IMAGES = [
#     {
#         "city": "Arlington",
#         "number": 3,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "Atlanta",
#         "number": 1,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "Atlanta",
#         "number": 2,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "Atlanta",
#         "number": 3,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "Austin",
#         "number": 1,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "Austin",
#         "number": 2,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "Austin",
#         "number": 3,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "NewYork",
#         "number": 2,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "SanFrancisco",
#         "number": 1,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "SanFrancisco",
#         "number": 2,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "SanFrancisco",
#         "number": 3,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "Norfolk",
#         "number": 1,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "Norfolk",
#         "number": 2,
#         "min_downsampling_factor": 1,
#     },
#     {
#         "city": "Norfolk",
#         "number": 3,
#         "min_downsampling_factor": 1,
#     },
# ]
