import os
import numpy as np

import config

DATASET_RAW_DIR = os.path.join(config.DATA_DIR, "AerialImageDataset/raw")

IMAGES = [
    {
        "city": "sfo",
        "number": 31,
    },
]
# Displacement map
DISP_MAP_PARAMS = {
    "disp_map_count": 10,
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

OUTPUT_DIR = "test/inria"
DISP_MAPS_DIR = OUTPUT_DIR + ".disp_maps"
