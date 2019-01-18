import sys
import os

import tensorflow as tf
import numpy as np

import test

# CHANGE to the path of your own read.py script:
sys.path.append("../../../data/bradbury_buildings_roads_height_dataset")
import read as read_bradbury_buildings

sys.path.append("../../utils")
import run_utils
import python_utils
import print_utils

# --- Command-line FLAGS --- #

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', None, "Batch size. Generally set as large as the VRAM can handle.")

# Some examples:
# On Quadro M2200, 4GB VRAM: python 2_test_aerial_image.align_gt.py --batch_size=12
# On GTX 1080 Ti, 11GB VRAM: python 2_test_aerial_image.align_gt.py --batch_size=32

# --- --- #

# --- Params --- #

# CHANGE to you own test config file:
TEST_CONFIG_NAME = "config.test.bradbury_buildings.align_gt"

# Must be in descending order:
DS_FAC_LIST = [
    8,
    4,
    2,
    1,
]
RUNS_DIRPATH = "runs.igarss2019"
RUN_NAME_LIST = ["ds_fac_{}_noisy_inria_bradbury_all_1".format(ds_fac) for ds_fac in DS_FAC_LIST]

OUTPUT_DIRNAME_EXTENTION = "." + ".".join(RUN_NAME_LIST)

INPUT_POLYGONS_FILENAME_EXTENSION = "_buildingCoord.csv"  # Set to None to use default gt polygons
ALIGNED_GT_POLYGONS_FILENAME_EXTENSION = "_aligned_noisy_building_polygons_1.npy"

# --- --- #


def test_image(runs_dirpath, dataset_raw_dirpath, image_info, batch_size, ds_fac_list, run_name_list,
               model_disp_max_abs_value, output_dir, output_shapefiles):
    # --- Load data --- #
    # CHANGE the arguments of the load_gt_data() function if using your own and it does not take the same arguments:
    ori_image, ori_metadata, gt_polygons = read_bradbury_buildings.load_gt_data(dataset_raw_dirpath, image_info["city"],
                                                                 image_info["number"])
    if INPUT_POLYGONS_FILENAME_EXTENSION is not None:
        gt_polygons = read_bradbury_buildings.load_polygons(dataset_raw_dirpath, image_info["city"], image_info["number"], INPUT_POLYGONS_FILENAME_EXTENSION)
    else:
        gt_polygons = gt_polygons

    if gt_polygons is not None:
        # CHANGE the arguments of the IMAGE_NAME_FORMAT format string if using your own and it does not take the same arguments:
        image_name = read_bradbury_buildings.IMAGE_NAME_FORMAT.format(city=image_info["city"], number=image_info["number"])
        print_utils.print_info("Processing image {}".format(image_name))

        aligned_gt_polygons = test.test_align_gt(runs_dirpath, ori_image, ori_metadata, gt_polygons, batch_size,
                                                 ds_fac_list, run_name_list,
                                                 model_disp_max_abs_value, output_dir, image_name,
                                                 output_shapefiles=output_shapefiles)

        # Save aligned_gt_polygons in dataset dir:
        aligned_gt_polygons_filepath = read_bradbury_buildings.get_polygons_filepath(dataset_raw_dirpath, image_info["city"], image_info["number"], ALIGNED_GT_POLYGONS_FILENAME_EXTENSION)
        os.makedirs(os.path.dirname(aligned_gt_polygons_filepath), exist_ok=True)
        np.save(aligned_gt_polygons_filepath, aligned_gt_polygons)


def main(_):
    # load config file
    config_test = run_utils.load_config(TEST_CONFIG_NAME)

    # Handle FLAGS
    if FLAGS.batch_size is not None:
        batch_size = FLAGS.batch_size
    else:
        batch_size = config_test["batch_size"]

    print("#--- Used params: ---#")
    print("batch_size: {}".format(FLAGS.batch_size))

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
            test_image(RUNS_DIRPATH, dataset_raw_dirpath, image_info, batch_size, DS_FAC_LIST,
                       RUN_NAME_LIST, config_test["model_disp_max_abs_value"],
                       output_dir, config_test["output_shapefiles"])


if __name__ == '__main__':
    tf.app.run(main=main)
