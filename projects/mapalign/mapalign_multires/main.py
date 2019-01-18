#####
#
# Quicky align the OSM data of your images with this script
#
####

import sys
import os
import argparse

import skimage.io
import numpy as np

import test

sys.path.append("../../utils")
import run_utils
import print_utils
import geo_utils

# -- Default script arguments: --- #
CONFIG = "config"
FILEPATH = "geo_images/test_image.tif"
BATCH_SIZE = 12
RUNS_DIRPATH = "runs.igarss2019"  # Best models: runs.igarss2019
# Should be in descending order:
DS_FAC_LIST = [
    8,
    4,
    2,
    1,
]

# --- Params: --- #
RUN_NAME_FORMAT = "ds_fac_{}_inria_bradbury_all_2"  # Best models: ds_fac_{}_inria_bradbury_all_2

# --- --- #


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        default=CONFIG,
        type=str,
        help='Name of the config file, excluding the .json file extension.')
    argparser.add_argument(
        '-f', '--filepath',
        default=FILEPATH,
        type=str,
        help='Filepath to the GeoTIFF image.')
    argparser.add_argument(
        '-b', '--batch_size',
        default=BATCH_SIZE,
        type=int,
        help='Batch size. Generally set as large as the VRAM can handle. Default value can be set in config file.')
    argparser.add_argument(
        '-r', '--runs_dirpath',
        default=RUNS_DIRPATH,
        type=str,
        help='Name of the config file, excluding the .json file extension.')
    argparser.add_argument(
        '-d', '--ds_fac',
        default=DS_FAC_LIST,
        type=int,
        nargs='+',
        help='Downscaling factors. Should be a list of descending integers. Used to retrieve run names')

    args = argparser.parse_args()
    return args


def read_image(filepath):
    image_array = skimage.io.imread(filepath)
    pixelsize = geo_utils.get_pixelsize(filepath)
    image_metadata = {
        "filepath": filepath,
        "pixelsize": pixelsize,
    }
    return image_array, image_metadata


def get_osm_annotations(filepath):
    filename_no_extension = os.path.splitext(filepath)[0]
    npy_filepath = filename_no_extension + ".npy"
    if os.path.exists(npy_filepath):
        print_utils.print_info("Loading OSM building data from disc...")
        gt_polygons = np.load(npy_filepath)
    else:
        print_utils.print_info("Fetching OSM building data from the internet...")
        gt_polygons = geo_utils.get_polygons_from_osm(filepath, tag="building")
        # Save npy to avoid re-fetching:
        np.save(npy_filepath, gt_polygons)
        # Save shapefile for visualisation:
        shp_filepath = filename_no_extension + ".shp"
        geo_utils.save_shapefile_from_polygons(gt_polygons, filepath, shp_filepath)
    return gt_polygons


def save_annotations(image_filepath, polygons):
    filename_no_extension = os.path.splitext(image_filepath)[0]
    npy_filepath = filename_no_extension + ".aligned.npy"
    shp_filepath = filename_no_extension + ".aligned.shp"
    np.save(npy_filepath, polygons)
    geo_utils.save_shapefile_from_polygons(polygons, image_filepath, shp_filepath)


def main():
    working_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Process args --- #
    args = get_args()
    config = run_utils.load_config(args.config)
    if config is None:
        print_utils.print_error(
            "ERROR: cannot continue without a config file. Exiting now...")
        exit()
    print_utils.print_info("Using downscaling factors: {}".format(args.ds_fac))
    run_name_list = [RUN_NAME_FORMAT.format(ds_fac) for ds_fac in args.ds_fac]

    # --- Read image --- #
    print_utils.print_info("Reading image...")
    if os.path.isabs(args.filepath):
        absolute_filepath = args.filepath
    else:
        absolute_filepath = os.path.join(working_dir, args.filepath)
    image, image_metadata = read_image(absolute_filepath)

    # --- Load or fetch OSM building data --- #
    gt_polygons = get_osm_annotations(absolute_filepath)

    print_utils.print_info("Aligned OSM building annotations...")
    aligned_polygons = test.test_align_gt(args.runs_dirpath, image, image_metadata, gt_polygons, args.batch_size,
                                          args.ds_fac, run_name_list, config["disp_max_abs_value"],
                                          output_shapefiles=False)

    print_utils.print_info("Saving aligned OSM building annotations...")
    save_annotations(args.filepath, aligned_polygons)


if __name__ == '__main__':
    main()
