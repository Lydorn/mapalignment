import sys
import os
import numpy as np

import test
import config_test_inria as config_test

sys.path.append("../../../data/AerialImageDataset")
import read
# --- Params --- #

# Must be in descending order:
DS_FAC_LIST = [
    8,
    4,
    2,
    1,
]
RUN_NAME_LIST = [
    "ds_fac_8_no_inria",
    "ds_fac_4_no_inria",
    "ds_fac_2_no_inria",
    "ds_fac_1_no_inria",
]
assert len(DS_FAC_LIST) == len(RUN_NAME_LIST), "DS_FAC_LIST and RUN_NAME_LIST should have the same length (and match)"
# Both list should match and be in descending order of downsampling factor.

OUTPUT_DIR = config_test.OUTPUT_DIR + ".align" + ".ds_fac_8_no_inria.ds_fac_4_no_inria.ds_fac_2_no_inria.ds_fac_1_no_inria"
# OUTPUT_DIR = config_test.OUTPUT_DIR + ".align" + ".ds_fac_4_disp_max_16_quicksilver"

# --- --- #


def load_disp_maps(disp_maps_dir, image_info, disp_map_count):
    disp_map_filename_format = "{}.disp_{:02d}.disp_map.npy"

    disp_map_list = []
    for i in range(disp_map_count):
        image_name = read.IMAGE_NAME_FORMAT.format(city=image_info["city"], number=image_info["number"])
        disp_map_filename = disp_map_filename_format.format(image_name, i)
        disp_map_filepath = os.path.join(disp_maps_dir, disp_map_filename)
        disp_map = np.load(disp_map_filepath)
        disp_map_list.append(disp_map)
    disp_maps = np.stack(disp_map_list, axis=0)
    return disp_maps


def test_image(image_info, disp_maps_dir, disp_map_count, disp_max_abs_value, batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds, test_output_dir):
    # --- Load data --- #
    ori_image, ori_metadata, ori_gt_polygons = read.load_gt_data(config_test.DATASET_RAW_DIR, image_info["city"], image_info["number"])
    image_name = read.IMAGE_NAME_FORMAT.format(city=image_info["city"], number=image_info["number"])

    # --- Load disp maps --- #
    disp_maps = load_disp_maps(disp_maps_dir, image_info, disp_map_count)

    test.test_image_with_gt_polygons_and_disp_maps(image_name, ori_image, ori_metadata, ori_gt_polygons, disp_maps, disp_max_abs_value, batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds, test_output_dir, output_shapefiles=config_test.OUTPUT_SHAPEFILES)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for images_info in config_test.IMAGES_INFO_LIST:
        for number in images_info["numbers"]:
            image_info = {
                "city": images_info["city"],
                "number": number,
            }
            test_image(image_info, config_test.DISP_MAPS_DIR, config_test.DISP_MAP_PARAMS["disp_map_count"], config_test.DISP_MAP_PARAMS["disp_max_abs_value"], config_test.BATCH_SIZE, DS_FAC_LIST, RUN_NAME_LIST, config_test.MODEL_DISP_MAX_ABS_VALUE, config_test.THRESHOLDS, OUTPUT_DIR)


if __name__ == '__main__':
    main()
