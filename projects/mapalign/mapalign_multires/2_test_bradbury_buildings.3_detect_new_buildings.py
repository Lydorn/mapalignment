import sys
import os
import numpy as np

import config
import test
import config_test_bradbury_buildings as config_test

sys.path.append("../../../data/bradbury_buildings_roads_height_dataset")
import read

# --- Params --- #

# Models
DS_FAC_LIST = [
    # 8,
    # 4,
    # 2,
    1,
]  # Must be in descending order
RUN_NAME_LIST = [
    # "ds_fac_8",
    # "ds_fac_4",
    # "ds_fac_2",
    "ds_fac_1",
]
assert len(DS_FAC_LIST) == len(RUN_NAME_LIST), "DS_FAC_LIST and RUN_NAME_LIST should have the same length (and match)"
# Both list should match and be in descending order of downsampling factor.

FILL_THRESHOLD = 0.5
OUTLINE_THRESHOLD = 0.05
SELEM_WIDTH = 3
ITERATIONS = 6

TEST_OUTPUT_DIR = config_test.OUTPUT_DIR + ".seg" + ".ds_fac_1"

# --- --- #


def test_detect_new_buildings(image_info, batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds, test_output_dir):

    # --- Load data --- #
    ori_image, ori_metadata, ori_gt_polygons = read.load_gt_data(config_test.DATASET_RAW_DIR, image_info["city"], image_info["number"])
    image_name = read.IMAGE_NAME_FORMAT.format(city=image_info["city"], number=image_info["number"])

    polygonization_params = {
        "fill_threshold": FILL_THRESHOLD,
        "outline_threshold": OUTLINE_THRESHOLD,
        "selem_width": SELEM_WIDTH,
        "iterations": ITERATIONS,
    }

    test.test_detect_new_buildings(image_name, ori_image, ori_metadata, ori_gt_polygons, batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, polygonization_params, thresholds, test_output_dir, output_shapefiles=config_test.OUTPUT_SHAPEFILES)


def main():
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)

    for image_info in config_test.IMAGES:
        test_detect_new_buildings(image_info, config_test.BATCH_SIZE, DS_FAC_LIST, RUN_NAME_LIST, config_test.MODEL_DISP_MAX_ABS_VALUE, config_test.THRESHOLDS, TEST_OUTPUT_DIR)


if __name__ == '__main__':
    main()
