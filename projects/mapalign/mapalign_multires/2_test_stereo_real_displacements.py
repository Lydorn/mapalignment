import sys
import os
import numpy as np
import itertools

import config
import test

sys.path.append("../../../data/stereo_dataset")
import read

sys.path.append("../../utils")
import geo_utils

# --- Params --- #

DATASET_DIR = os.path.join(config.PROJECT_DIR, "../../../data/stereo_dataset")
FILE_PARAMS = {
    "raw_dataset_dir": os.path.join(DATASET_DIR, "raw"),
    "gt_views": ["ref", "rec"],
    "image_name_suffix": "ortho",
    "image_modes": ["RGB", "NIRRG"],
    "image_extension": "tif",
    "image_format": "{}_{}_{}_{}.{}",  # To be used as IMAGE_FORMAT.format(name, image_name_suffix, gt_views[i], image_modes[j], image_extension)
    "poly_name_capitalize": True,  # If True, the gt name will be capitalised when building the gt filename to load
    "poly_tag": "buildings",
    "poly_extension": "filtered.shp",  # Use filtered shapefiles (no intersecting polygons)
    "poly_format": "{}_{}_{}.{}",  # To be used as IMAGE_FORMAT.format(capitalize(name), POLY_TAG, GT_VIEWS[i], poly_extension)
}

TEST_IMAGES = ["leibnitz"]

# Models
BATCH_SIZE = 6
DS_FAC_LIST = [8, 4, 2]  # Must be in descending order
RUN_NAME_LIST = [
    "ds_fac_8",
    "ds_fac_4",
    "ds_fac_2",
]
assert len(DS_FAC_LIST) == len(RUN_NAME_LIST), "DS_FAC_LIST and RUN_NAME_LIST should have the same length (and match)"
MODEL_DISP_MAX_ABS_VALUE = 4
# Both list should match and be in descending order of downsampling factor.

THRESHOLDS = np.arange(0, 16.25, 0.25)

TEST_OUTPUT_DIR = "test/stereo_dataset_real_displacements.align.ds_fac_8.ds_fac_4.ds_fac_2"

# --- --- #


def test_image(image_name, view_pair, file_params, batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds, test_output_dir):
    # --- Load data --- #
    ori_image, ori_metadata = read.load_image_data(image_name, view_pair[0], file_params)
    ori_gt_polygons, ori_gt_properties_list = read.load_polygon_data(image_name, view_pair[0], file_params)
    ori_disp_polygons, ori_disp_properties_list = read.load_polygon_data(image_name, view_pair[1], file_params)

    # --- Test --- #
    # Add view to the image name (otherwise the result of the last view will overwrite previous ones)
    test_image_name = image_name + "_" + view_pair[0] + "_" + view_pair[1]
    test.test_image_with_gt_and_disp_polygons(test_image_name, ori_image, ori_metadata, ori_gt_polygons, ori_disp_polygons, ori_disp_properties_list, batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds, test_output_dir)


def main():
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)

    view_pairs = itertools.permutations(FILE_PARAMS["gt_views"])

    for image_name in TEST_IMAGES:
        for view_pair in view_pairs:
            test_image(image_name, view_pair, FILE_PARAMS, BATCH_SIZE, DS_FAC_LIST, RUN_NAME_LIST, MODEL_DISP_MAX_ABS_VALUE, THRESHOLDS, TEST_OUTPUT_DIR)


if __name__ == '__main__':
    main()
