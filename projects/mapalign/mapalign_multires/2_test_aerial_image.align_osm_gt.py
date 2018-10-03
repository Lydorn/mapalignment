import sys
import os

import test

# CHANGE to you own test config file:
import config_test_inria as config_test

# CHANGE to the path of your own read.py script:
sys.path.append("../../../data/AerialImageDataset")
import read

# --- Params --- #

# Iteratively use these downsampling factors (should be in descending order):
DS_FAC_LIST = [
    8,
    4,
    2,
    1,
]
# Name of the runs to use (in the same order as the DS_FAC_LIST list):
RUN_NAME_LIST = [
    "ds_fac_8",
    "ds_fac_4",
    "ds_fac_2",
    "ds_fac_1",
]
assert len(DS_FAC_LIST) == len(RUN_NAME_LIST), "DS_FAC_LIST and RUN_NAME_LIST should have the same length (and match)"

OUTPUT_DIR = config_test.OUTPUT_DIR + ".align" + ".ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1"

# --- --- #


def test_image(image_info, batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds, test_output_dir):
    # --- Load data --- #
    # CHANGE the arguments of the load_gt_data() function if using your own and it does not take the same arguments:
    ori_image, ori_metadata, ori_disp_polygons = read.load_gt_data(config_test.DATASET_RAW_DIR, image_info["city"], image_info["number"])
    # CHANGE the arguments of the IMAGE_NAME_FORMAT format string if using your own and it does not take the same arguments:
    image_name = read.IMAGE_NAME_FORMAT.format(city=image_info["city"], number=image_info["number"])

    ori_gt_polygons = []

    test.test(ori_image, ori_metadata, ori_gt_polygons, ori_disp_polygons, batch_size, ds_fac_list, run_name_list,
              model_disp_max_abs_value, thresholds, test_output_dir, image_name, output_shapefiles=config_test.OUTPUT_SHAPEFILES)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for image_info in config_test.IMAGES:
        test_image(image_info, config_test.BATCH_SIZE, DS_FAC_LIST,
                   RUN_NAME_LIST, config_test.MODEL_DISP_MAX_ABS_VALUE, config_test.THRESHOLDS, OUTPUT_DIR)


if __name__ == '__main__':
    main()
