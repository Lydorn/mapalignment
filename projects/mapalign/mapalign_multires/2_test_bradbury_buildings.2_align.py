import sys
import os
import numpy as np

import test

sys.path.append("../../../data/bradbury_buildings_roads_height_dataset")
import read as read_bradbury_buildings

sys.path.append("../../utils")
import run_utils
import python_utils

# --- Params --- #

TEST_CONFIG_NAME = "config.test.bradbury_buildings"

KEEP_PROB = 1  # Randomly drop some polygons

# Must be in descending order:
DS_FAC_LIST = [
    8,
    4,
    # 2,
    # 1,
]
RUNS_DIRPATH = "runs.igarss2019"
RUN_NAME_LIST = ["ds_fac_{}".format(ds_fac) for ds_fac in DS_FAC_LIST]

OUTPUT_DIRNAME_EXTENTION = "." + ".".join(RUN_NAME_LIST)


# --- --- #


def drop_items(items, keep_prob):
    random_numbers = np.random.rand(len(items))
    new_items = []
    for item, random_number in zip(items, random_numbers):
        if random_number < keep_prob:
            new_items.append(item)
    return new_items


def load_disp_maps(disp_maps_dir, image_info, disp_map_count):
    disp_map_filename_format = "{}.disp_{:02d}.disp_map.npy"

    disp_map_list = []
    for i in range(disp_map_count):
        image_name = read_bradbury_buildings.IMAGE_NAME_FORMAT.format(city=image_info["city"],
                                                                      number=image_info["number"])
        disp_map_filename = disp_map_filename_format.format(image_name, i)
        disp_map_filepath = os.path.join(disp_maps_dir, disp_map_filename)
        disp_map = np.load(disp_map_filepath)
        disp_map_list.append(disp_map)
    disp_maps = np.stack(disp_map_list, axis=0)
    return disp_maps


def test_image(runs_dirpath, dataset_raw_dirpath, image_info, disp_maps_dir, disp_map_count, disp_max_abs_value,
               batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds, test_output_dir,
               output_shapefiles):
    # --- Load data --- #
    ori_image, ori_metadata, ori_gt_polygons = read_bradbury_buildings.load_gt_data(dataset_raw_dirpath,
                                                                                    image_info["city"],
                                                                                    image_info["number"])
    image_name = read_bradbury_buildings.IMAGE_NAME_FORMAT.format(city=image_info["city"], number=image_info["number"])

    # --- Randomly drop some polygons --- #
    if KEEP_PROB < 1:
        ori_gt_polygons = drop_items(ori_gt_polygons, KEEP_PROB)

    # --- Load disp maps --- #
    disp_maps = load_disp_maps(disp_maps_dir, image_info, disp_map_count)

    test.test_image_with_gt_polygons_and_disp_maps(runs_dirpath, image_name, ori_image, ori_metadata, ori_gt_polygons,
                                                   disp_maps,
                                                   disp_max_abs_value, batch_size, ds_fac_list, run_name_list,
                                                   model_disp_max_abs_value, thresholds, test_output_dir,
                                                   output_shapefiles=output_shapefiles)


def main():
    # load config file
    config_test = run_utils.load_config(TEST_CONFIG_NAME)

    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(config_test["data_dir_candidates"])
    if data_dir is None:
        print("ERROR: Data directory not found!")
        exit()
    else:
        print("Using data from {}".format(data_dir))

    dataset_raw_dirpath = os.path.join(data_dir, config_test["dataset_raw_partial_dirpath"])

    output_dir = config_test["align_dir"] + OUTPUT_DIRNAME_EXTENTION

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for images_info in config_test["images_info_list"]:
        for number in images_info["numbers"]:
            image_info = {
                "city": images_info["city"],
                "number": number,
            }
            test_image(RUNS_DIRPATH, dataset_raw_dirpath, image_info, config_test["disp_maps_dir"],
                       config_test["disp_map_params"]["disp_map_count"],
                       config_test["disp_map_params"]["disp_max_abs_value"], config_test["batch_size"], DS_FAC_LIST,
                       RUN_NAME_LIST, config_test["model_disp_max_abs_value"], config_test["thresholds"], output_dir,
                       config_test["output_shapefiles"])


if __name__ == '__main__':
    main()
