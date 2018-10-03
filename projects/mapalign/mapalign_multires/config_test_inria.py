import os
import numpy as np

import config

# CHANGE to the path of you own dataset:
DATASET_RAW_DIR = os.path.join(config.DATA_DIR, "AerialImageDataset/raw")

# CHANGE to your own way of identifying your images. For example in Inria dataset, images have a city name plus a number.
# The load_gt_data() function of the read.py script takes those 2 parameters as input to identify the image to load.
IMAGES_INFO_LIST = [
    {
        "city": "austin",
        "numbers":  [1, 2, 3, 5, 6, 8, 9, 12, 13, 14, 19, 20, 25, 26, 27, 28, 29, 30, 32, 33, 35],
    },
    {
        "city": "chicago",
        "numbers":  [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    },
]

# Displacement map generation params
DISP_MAP_PARAMS = {
    "disp_map_count": 1,
    "disp_modes": 30,  # Number of Gaussians mixed up to make the displacement map (Default: 20)
    "disp_gauss_mu_range": [0, 1],  # Coordinates are normalized to [0, 1] before the function is applied
    "disp_gauss_sig_scaling": [0.0, 0.002],  # Coordinates are normalized to [0, 1] before the function is applied
    "disp_max_abs_value": 32,
}

# Model
BATCH_SIZE = 12
MODEL_DISP_MAX_ABS_VALUE = 4

THRESHOLDS = np.arange(0, 16.25, 0.25)

OUTPUT_SHAPEFILES = False

# CHANGE to your own output directory for the test results:
OUTPUT_DIR = "test/inria"

DISP_MAPS_DIR = OUTPUT_DIR + ".disp_maps"
