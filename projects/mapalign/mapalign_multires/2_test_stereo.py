import sys
import os
import numpy as np

import config
import test

sys.path.append("../../../data/stereo_dataset")
import read

# --- Params --- #

DATASET_DIR = os.path.join(config.PROJECT_DIR, "../../../data/stereo_dataset")
FILE_PARAMS = {
    "raw_dataset_dir": os.path.join(DATASET_DIR, "raw"),
    "gt_views": ["rec", "ref"],
    "image_name_suffix": "ortho",
    "image_modes": ["RGB", "NIRRG"],
    "image_extension": "tif",
    "image_format": "{}_{}_{}_{}.{}",  # To be used as IMAGE_FORMAT.format(name, image_name_suffix, gt_views[i], image_modes[j], image_extension)
    "poly_name_capitalize": True,  # If True, the gt name will be capitalised when building the gt filename to load
    "poly_tag": "buildings",
    "poly_extension": "shp",
    "poly_format": "{}_{}_{}.{}",  # To be used as IMAGE_FORMAT.format(capitalize(name), POLY_TAG, GT_VIEWS[i], IMAGE_EXTENSION)
}

TEST_IMAGES = ["leibnitz"]

# Displacement map
DISP_MAP_PARAMS = {
    "disp_map_count": 1,
    "disp_modes": 30,  # Number of Gaussians mixed up to make the displacement map (Default: 20)
    "disp_gauss_mu_range": [0, 1],  # Coordinates are normalized to [0, 1] before the function is applied
    "disp_gauss_sig_scaling": [0.0, 0.002],  # Coordinates are normalized to [0, 1] before the function is applied
    "disp_max_abs_value": 32,
}


# Models
BATCH_SIZE = 32
DS_FAC_LIST = [8, 4, 2]  # Must be in descending order
# DS_FAC_LIST = [8, 4]
RUN_NAME_LIST = [
    # "ds_fac_16",
    "ds_fac_8_double",
    "ds_fac_4_double",
    "ds_fac_2_double_seg",
    # "ds_fac_1_double_seg",
]
assert len(DS_FAC_LIST) == len(RUN_NAME_LIST), "DS_FAC_LIST and RUN_NAME_LIST should have the same length (and match)"
MODEL_DISP_MAX_ABS_VALUE = 4
# Both list should match and be in descending order of downsampling factor.

THRESHOLDS = np.arange(0, 16.5, 0.5)

TEST_OUTPUT_DIR = "test/stereo_dataset.ds_fac_8_double.ds_fac_4_double.ds_fac_2_double_seg.ds_fac_1_double_seg"

# --- --- #


def test_image(image_name, view, file_params, disp_map_params, batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds, test_output_dir):
    # --- Load data --- #
    ori_image, ori_metadata = read.load_image_data(image_name, view, file_params)
    ori_gt_polygons = read.load_polygon_data(image_name, view, file_params)

    # --- Test --- #
    # Add view to the image name (otherwise the result of the last view will overwrite previous ones)
    test_image_name = image_name + "_" + view
    test.test_image_with_gt_polygons(test_image_name, ori_image, ori_metadata, ori_gt_polygons, disp_map_params, batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds, test_output_dir)


def main():
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)

    if not os.path.exists(TEST_OUTPUT_DIR + ".no_align"):
        os.makedirs(TEST_OUTPUT_DIR + ".no_align")

    for image_name in TEST_IMAGES:
        for view in FILE_PARAMS["gt_views"]:
            test_image(image_name, view, FILE_PARAMS, DISP_MAP_PARAMS, BATCH_SIZE, DS_FAC_LIST, RUN_NAME_LIST, MODEL_DISP_MAX_ABS_VALUE, THRESHOLDS, TEST_OUTPUT_DIR)


if __name__ == '__main__':
    main()
