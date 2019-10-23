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
IMAGE = "geo_images/test_image.tif"
SHAPEFILE = None
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
        '-i', '--image',
        default=IMAGE,
        type=str,
        help='Filepath to the GeoTIFF image.')
    argparser.add_argument(
        '-s', '--shapefile',
        default=SHAPEFILE,
        type=str,
        help='Filepath to the shapefile.')
    argparser.add_argument(
        '-b', '--batch_size',
        default=BATCH_SIZE,
        type=int,
        help='Batch size. Generally set as large as the VRAM can handle. Default value can be set in config file.')
    argparser.add_argument(
        '-r', '--runs_dirpath',
        default=RUNS_DIRPATH,
        type=str,
        help='Name of directory where the models can be found.')
    argparser.add_argument(
        '-d', '--ds_fac',
        default=DS_FAC_LIST,
        type=int,
        nargs='+',
        help='Downscaling factors. Should be a list of descending integers. Used to retrieve run names')
    argparser.add_argument(
        '--pixelsize',
        type=float,
        help='Set pixel size (in meters) of the image. Useful when the image does not have this value in its metadata.')

    args = argparser.parse_args()
    return args


def read_image(filepath, pixelsize=None):
    image_array = skimage.io.imread(filepath)
    if pixelsize is None:
        pixelsize = geo_utils.get_pixelsize(filepath)
    assert type(pixelsize) == float, "pixelsize should be float, not {}".format(type(pixelsize))
    if pixelsize < 1e-3:
        print_utils.print_warning("WARNING: pixel size of image is detected to be {}m which seems very small to be correct. "
              "If problems occur specify pixelsize with the pixelsize command-line argument".format(pixelsize))
    image_metadata = {
        "filepath": filepath,
        "pixelsize": pixelsize,
    }
    return image_array, image_metadata


def normalize(image, mu=None, sigma=None):
    if mu is None:
        mu = np.mean(image)
    if sigma is None:
        sigma = np.std(image)
    return (image - mu) / sigma


def get_osm_annotations(filepath):
    filename_no_extension = os.path.splitext(filepath)[0]
    npy_filepath = filename_no_extension + ".npy"
    if os.path.exists(npy_filepath):
        print_utils.print_info("Loading OSM building data from disc...")
        gt_polygons = np.load(npy_filepath, allow_pickle=True)
    else:
        print_utils.print_info("Fetching OSM building data from the internet...")
        gt_polygons = geo_utils.get_polygons_from_osm(filepath, tag="building")
        # Save npy to avoid re-fetching:
        np.save(npy_filepath, gt_polygons)
        # Save shapefile for visualisation:
        shp_filepath = filename_no_extension + ".shp"
        geo_utils.save_shapefile_from_polygons(gt_polygons, filepath, shp_filepath)
    return gt_polygons


def get_shapefile_annotations(image_filepath, shapefile_filepath):
    polygons, _ = geo_utils.get_polygons_from_shapefile(image_filepath, shapefile_filepath)
    return polygons


def save_annotations(image_filepath, polygons):
    filename_no_extension = os.path.splitext(image_filepath)[0]
    npy_filepath = filename_no_extension + ".aligned.npy"
    shp_filepath = filename_no_extension + ".aligned.shp"
    np.save(npy_filepath, polygons)
    geo_utils.save_shapefile_from_polygons(polygons, image_filepath, shp_filepath)


def get_abs_path(filepath):
    working_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(filepath):
        abs_path = filepath
    else:
        abs_path = os.path.join(working_dir, filepath)
    return abs_path


def print_hist(hist):
    print("hist:")
    for (bin, count) in zip(hist[1], hist[0]):
        print("{}: {}".format(bin, count))


def clip_image(image, min, max):
    image = np.maximum(np.minimum(image, max), min)
    return image


def get_min_max(image, std_factor=2):
    mu = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    min = mu - std_factor * std
    max = mu + std_factor * std
    return min, max


def stretch_image(image, min, max, target_min, target_max):
    image = (image - min) / (max - min)
    image = image * (target_max - target_min) + target_min
    return image


def check_polygons_in_image(image, polygons):
    """
    Allows some vertices to be outside the image. Return s true if at least 1 is inside.
    :param image:
    :param polygons:
    :return:
    """
    height = image.shape[0]
    width = image.shape[1]
    min_i = min([polygon[:, 0].min() for polygon in polygons])
    min_j = min([polygon[:, 1].min() for polygon in polygons])
    max_i = max([polygon[:, 0].max() for polygon in polygons])
    max_j = max([polygon[:, 1].max() for polygon in polygons])
    return not (max_i < 0 or height < min_i or max_j < 0 or width < min_j)


def main():
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
    image_filepath = get_abs_path(args.image)
    image, image_metadata = read_image(image_filepath, args.pixelsize)
    image = clip_image(image, 0, 255)

    # hist = np.histogram(image)
    # print_hist(hist)

    im_min, im_max = get_min_max(image, std_factor=3)

    # print("min: {}, max: {}".format(im_min, im_max))

    image = stretch_image(image, im_min, im_max, 0, 255)
    image = clip_image(image, 0, 255)

    # hist = np.histogram(image)
    # print_hist(hist)

    print("Image stats:")
    print("\tShape: {}".format(image.shape))
    print("\tMin: {}".format(image.min()))
    print("\tMax: {}".format(image.max()))

    # --- Read shapefile if it exists --- #
    if args.shapefile is not None:
        shapefile_filepath = get_abs_path(args.shapefile)
        gt_polygons = get_shapefile_annotations(image_filepath, shapefile_filepath)

    else:
        # --- Load or fetch OSM building data --- #
        gt_polygons = get_osm_annotations(image_filepath)

    # --- Print polygon info --- #
    print("Polygons stats:")
    print("\tCount: {}".format(len(gt_polygons)))
    print("\tMin: {}".format(min([polygon.min() for polygon in gt_polygons])))
    print("\tMax: {}".format(max([polygon.max() for polygon in gt_polygons])))

    if not check_polygons_in_image(image, gt_polygons):
        print_utils.print_error("ERROR: polygons are not inside the image. This is most likely due to using the wrong projection when reading the input shapefile. Aborting...")
        exit()

    print_utils.print_info("Aligning building annotations...")
    aligned_polygons = test.test_align_gt(args.runs_dirpath, image, image_metadata, gt_polygons, args.batch_size,
                                          args.ds_fac, run_name_list, config["disp_max_abs_value"],
                                          output_shapefiles=False)

    print_utils.print_info("Saving aligned building annotations...")
    save_annotations(args.image, aligned_polygons)


if __name__ == '__main__':
    main()
