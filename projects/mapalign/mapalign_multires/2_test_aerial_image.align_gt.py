import sys
import os

import tensorflow as tf
import numpy as np

import test

# CHANGE to the path of your own read.py script:
sys.path.append("../../../data/AerialImageDataset")
import read

sys.path.append("../../utils")
import run_utils
import python_utils

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
TEST_CONFIG_NAME = "config.test.aerial_image.align_gt"

# Must be in descending order:
DS_FAC_LIST = [
    8,
    4,
    2,
    1,
]
RUNS_DIRPATH = "runs.igarss2019"
RUN_NAME_LIST = ["ds_fac_{}_noisy_inria_bradbury_all_2".format(ds_fac) for ds_fac in DS_FAC_LIST]

OUTPUT_DIRNAME_EXTENTION = "." + ".".join(RUN_NAME_LIST)

INPUT_POLYGONS_DIRNAME = "noisy_gt_polygons"  # Set to None to use default gt polygons
ALIGNED_GT_POLYGONS_DIRNAME = "aligned_noisy_gt_polygons_2"
# --- --- #


def test_image(runs_dirpath, dataset_raw_dirpath, image_info, batch_size, ds_fac_list, run_name_list,
               model_disp_max_abs_value, output_dir, output_shapefiles):
    # --- Load data --- #
    # CHANGE the arguments of the load_gt_data() function if using your own and it does not take the same arguments:
    ori_image, ori_metadata, ori_gt_polygons = read.load_gt_data(dataset_raw_dirpath, image_info["city"],
                                                                 image_info["number"])
    if INPUT_POLYGONS_DIRNAME is not None:
        gt_polygons = read.load_polygons(dataset_raw_dirpath, INPUT_POLYGONS_DIRNAME, image_info["city"], image_info["number"])
    else:
        gt_polygons = ori_gt_polygons

    if gt_polygons is not None:
        # CHANGE the arguments of the IMAGE_NAME_FORMAT format string if using your own and it does not take the same arguments:
        image_name = read.IMAGE_NAME_FORMAT.format(city=image_info["city"], number=image_info["number"])

        aligned_gt_polygons = test.test_align_gt(runs_dirpath, ori_image, ori_metadata, gt_polygons, batch_size,
                                                 ds_fac_list, run_name_list,
                                                 model_disp_max_abs_value, output_dir, image_name,
                                                 output_shapefiles=output_shapefiles)

        # Save aligned_gt_polygons in dataset dir:
        aligned_gt_polygons_filepath = read.get_polygons_filepath(dataset_raw_dirpath, ALIGNED_GT_POLYGONS_DIRNAME, image_info["city"], image_info["number"])
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
