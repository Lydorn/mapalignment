import os.path
import csv
import sys
import numpy as np

import skimage.io

CITY_METADATA_DICT = {
    "bloomington": {
        "fold": "test",
        "pixelsize": 0.5,
    },
    "bellingham": {
        "fold": "test",
        "pixelsize": 0.5,
    },
    "innsbruck": {
        "fold": "test",
        "pixelsize": 0.5,
    },
    "sfo": {
        "fold": "test",
        "pixelsize": 0.5,
    },
    "tyrol-e": {
        "fold": "test",
        "pixelsize": 0.5,
    },
    "austin": {
        "fold": "train",
        "pixelsize": 0.5,
    },
    "chicago": {
        "fold": "train",
        "pixelsize": 0.5,
    },
    "kitsap": {
        "fold": "train",
        "pixelsize": 0.5,
    },
    "tyrol-w": {
        "fold": "train",
        "pixelsize": 0.5,
    },
    "vienna": {
        "fold": "train",
        "pixelsize": 0.5,
    },
}

IMAGE_DIR_NAME = "images"
IMAGE_NAME_FORMAT = "{city}{number}"
IMAGE_FILENAME_FORMAT = IMAGE_NAME_FORMAT + ".tif"  # City name, number
POLYGON_DIR_NAME = "gt_polygons"
POLYGONS_FILENAME_FORMAT = IMAGE_NAME_FORMAT + ".npy"  # City name, number


def get_image_filepath(raw_dirpath, city, number):
    fold = CITY_METADATA_DICT[city]["fold"]
    filename = IMAGE_FILENAME_FORMAT.format(city=city, number=number)
    filepath = os.path.join(raw_dirpath, fold, IMAGE_DIR_NAME, filename)
    return filepath


def get_polygons_filepath(raw_dirpath, city, number):
    fold = CITY_METADATA_DICT[city]["fold"]
    filename = POLYGONS_FILENAME_FORMAT.format(city=city, number=number)
    filepath = os.path.join(raw_dirpath, fold, POLYGON_DIR_NAME, filename)
    return filepath


def load_image(raw_dirpath, city, number):
    filepath = get_image_filepath(raw_dirpath, city, number)
    image_array = skimage.io.imread(filepath)

    # The following is writen this way for future image-specific addition of metadata:
    image_metadata = {
        "filepath": filepath,
        "pixelsize": CITY_METADATA_DICT[city]["pixelsize"]
    }

    return image_array, image_metadata


def load_polygons(raw_dirpath, city, number):
    filepath = get_polygons_filepath(raw_dirpath, city, number)
    try:
        gt_polygons = np.load(filepath)
    except FileNotFoundError:
        print("City {}, number {} does not have gt polygons".format(city, number))
        gt_polygons = None
    return gt_polygons


def load_gt_data(raw_dirpath, city, number):
    # Load image
    image_array, image_metadata = load_image(raw_dirpath, city, number)

    # Load CSV data
    gt_polygons = load_polygons(raw_dirpath, city, number)

    # TODO: remove
    # gt_polygons_filepath = get_polygons_filepath(raw_dirpath, city, number)
    # visualization.save_plot_image_polygons(gt_polygons_filepath + ".polygons.png", image_array, [], gt_polygons, [])
    # TODO end

    return image_array, image_metadata, gt_polygons


def main():
    raw_dirpath = "raw"
    city = "bloomington"
    number = 1
    image_array, image_metadata, gt_polygons = load_gt_data(raw_dirpath, city, number)

    print(image_array.shape)
    print(image_metadata)
    print(gt_polygons)


if __name__ == "__main__":
    main()
