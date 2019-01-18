import os
import numpy as np

import config

DATASET_RAW_DIR = os.path.join(config.DATA_DIR, "bradbury_buildings_roads_height_dataset/raw")

# IMAGES = [
#     {
#         "city": "SanFrancisco",
#         "number": 1,
#     },
#     {
#         "city": "SanFrancisco",
#         "number": 2,
#     },
#     {
#         "city": "SanFrancisco",
#         "number": 3,
#     },
# ]
IMAGES = [
    {
        "city": "Norfolk",
        "number": 1,
    },
    {
        "city": "Norfolk",
        "number": 2,
    },
    # Too few buildings for accuracy measurement:
    # {
    #     "city": "Norfolk",
    #     "number": 3,
    # },
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

OUTPUT_SHAPEFILES = False  # Bradbury images are not geo-localized

OUTPUT_DIR = "test.accv2018/bradbury_buildings"
DISP_MAPS_DIR = OUTPUT_DIR + ".disp_maps"
