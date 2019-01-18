import sys
import os
import numpy as np

import config_test_bradbury_buildings as config_test
import test

sys.path.append("../../utils")
import math_utils
import run_utils
import python_utils

sys.path.append("../../../data/bradbury_buildings_roads_height_dataset")
import read

# --- Params --- #

CONFIG_NAME = "config"
TEST_CONFIG_NAME = "config.test.bradbury_buildings"

# --- --- #


def generate_disp_maps(dataset_raw_dir, image_info, disp_map_params, thresholds, output_dir):
    disp_map_filename_format = "{}.disp_{:02d}.disp_map.npy"
    accuracies_filename_format = "{}.disp_{:02d}.accuracy.npy"

    # --- Load data --- #
    ori_image, ori_metadata, ori_gt_polygons = read.load_gt_data(dataset_raw_dir, image_info["city"], image_info["number"])
    image_name = read.IMAGE_NAME_FORMAT.format(city=image_info["city"], number=image_info["number"])
    spatial_shape = ori_image.shape[:2]
    ori_normed_disp_field_maps = math_utils.create_displacement_field_maps(spatial_shape,
                                                                           disp_map_params["disp_map_count"],
                                                                           disp_map_params["disp_modes"],
                                                                           disp_map_params["disp_gauss_mu_range"],
                                                                           disp_map_params["disp_gauss_sig_scaling"])
    disp_polygons_list = test.generate_disp_data(ori_normed_disp_field_maps, ori_gt_polygons, disp_map_params["disp_max_abs_value"])

    # Save disp maps and accuracies individually
    for i, (ori_normed_disp_field_map, disp_polygons) in enumerate(zip(ori_normed_disp_field_maps, disp_polygons_list)):
        disp_map_filename = disp_map_filename_format.format(image_name, i)
        disp_map_filepath = os.path.join(output_dir, disp_map_filename)
        np.save(disp_map_filepath, ori_normed_disp_field_map)

        accuracies_filename = accuracies_filename_format.format(image_name, i)
        accuracies_filepath = os.path.join(output_dir, accuracies_filename)
        integer_thresholds = [threshold for threshold in thresholds if (int(threshold) == threshold)]
        accuracies = test.measure_accuracies(ori_gt_polygons, disp_polygons, integer_thresholds, accuracies_filepath)


def main():
    # load config file
    config = run_utils.load_config(CONFIG_NAME)
    config_test = run_utils.load_config(TEST_CONFIG_NAME)

    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dir is None:
        print("ERROR: Data directory not found!")
        exit()
    else:
        print("Using data from {}".format(data_dir))

    dataset_raw_dirpath = os.path.join(data_dir, config_test["dataset_raw_partial_dirpath"])

    if not os.path.exists(config_test["disp_maps_dir"]):
        os.makedirs(config_test["disp_maps_dir"])

    for images_info in config_test["images_info_list"]:
        for number in images_info["numbers"]:
            image_info = {
                "city": images_info["city"],
                "number": number,
            }
            generate_disp_maps(dataset_raw_dirpath, image_info, config_test["disp_map_params"],
                               config_test["thresholds"], config_test["disp_maps_dir"])


if __name__ == '__main__':
    main()
