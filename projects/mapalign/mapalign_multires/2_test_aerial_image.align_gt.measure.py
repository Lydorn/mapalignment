import sys
import os

import numpy as np

import test

# CHANGE to the path of your own read.py script:
sys.path.append("../../../data/AerialImageDataset")
import read

sys.path.append("../../utils")
import run_utils
import python_utils
import geo_utils
import polygon_utils

# --- Command-line FLAGS --- #

# --- --- #

# --- Params --- #

# CHANGE to you own test config file:
TEST_CONFIG_NAME = "config.test.aerial_image.align_gt"

PERFECT_GT_POLYGONS_DIRNAME = "manually_aligned_gt_polygons"
GT_POLYGONS_DIRNAME_LIST = [
    "gt_polygons",
    "aligned_gt_polygons",
    "aligned_gt_polygons_1",
    "aligned_gt_polygons_2",

    "noisy_gt_polygons",
    "aligned_noisy_gt_polygons",
    "aligned_noisy_gt_polygons_1",
    "aligned_noisy_gt_polygons_2",
]

THRESHOLDS = np.arange(0, 32.25, 0.25)
# --- --- #


def measure_image(dataset_raw_dirpath, image_info, perfect_gt_polygons_dirname, gt_polygons_dirname_list, thresholds, output_dir_stem):
    accuracies_filename_format = "{}.accuracy.npy"
    # --- Load shapefiles --- #
    # CHANGE the arguments of the load_gt_data() function if using your own and it does not take the same arguments:
    image_filepath = read.get_image_filepath(dataset_raw_dirpath, image_info["city"], image_info["number"])
    polygons_filename_format = read.IMAGE_NAME_FORMAT + ".shp"
    perfect_gt_polygons_filepath = read.get_polygons_filepath(dataset_raw_dirpath, perfect_gt_polygons_dirname,
                                                     image_info["city"], image_info["number"],
                                                     overwrite_polygons_filename_format=polygons_filename_format)
    perfect_gt_polygons, _ = geo_utils.get_polygons_from_shapefile(image_filepath, perfect_gt_polygons_filepath)
    if perfect_gt_polygons is None:
        return None
    perfect_gt_polygons = polygon_utils.orient_polygons(perfect_gt_polygons)

    print("len(perfect_gt_polygons) = {}".format(len(perfect_gt_polygons)))

    for gt_polygons_dirname in gt_polygons_dirname_list:
        gt_polygons = read.load_polygons(dataset_raw_dirpath, gt_polygons_dirname, image_info["city"],
                                                 image_info["number"])
        if gt_polygons is None:
            break
        gt_polygons = polygon_utils.orient_polygons(gt_polygons)

        # CHANGE the arguments of the IMAGE_NAME_FORMAT format string if using your own and it does not take the same arguments:
        image_name = read.IMAGE_NAME_FORMAT.format(city=image_info["city"], number=image_info["number"])

        # --- Measure accuracies --- #
        output_dir = output_dir_stem + "." + gt_polygons_dirname
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        accuracies_filename = accuracies_filename_format.format(image_name)

        accuracies_filepath = os.path.join(output_dir, accuracies_filename)
        accuracies = test.measure_accuracies(perfect_gt_polygons, gt_polygons, thresholds, accuracies_filepath)
        print(accuracies)


def main():
    # load config file
    config_test = run_utils.load_config(TEST_CONFIG_NAME)

    # # Handle FLAGS
    # if FLAGS.batch_size is not None:
    #     batch_size = FLAGS.batch_size
    # else:
    #     batch_size = config_test["batch_size"]
    # print("#--- Used params: ---#")
    # print("batch_size: {}".format(FLAGS.batch_size))

    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(config_test["data_dir_candidates"])
    if data_dir is None:
        print("ERROR: Data directory not found!")
        exit()
    else:
        print("Using data from {}".format(data_dir))

    dataset_raw_dirpath = os.path.join(data_dir, config_test["dataset_raw_partial_dirpath"])

    output_dir_stem = config_test["align_dir"]

    for images_info in config_test["images_info_list"]:
        for number in images_info["numbers"]:
            image_info = {
                "city": images_info["city"],
                "number": number,
            }
            measure_image(dataset_raw_dirpath, image_info,
                          PERFECT_GT_POLYGONS_DIRNAME, GT_POLYGONS_DIRNAME_LIST, THRESHOLDS, output_dir_stem)


if __name__ == '__main__':
    main()
