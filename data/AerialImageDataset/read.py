import os.path
import csv
import sys
import numpy as np

import skimage.io

CITY_METADATA_DICT = {
    "bloomington": {
        "fold": "test",
        "pixelsize": 0.3,
        "numbers": list(range(1, 37)),
    },
    "bellingham": {
        "fold": "test",
        "pixelsize": 0.3,
        "numbers": list(range(1, 37)),
    },
    "innsbruck": {
        "fold": "test",
        "pixelsize": 0.3,
        "numbers": list(range(1, 37)),
    },
    "sfo": {
        "fold": "test",
        "pixelsize": 0.3,
        "numbers": list(range(1, 37)),
    },
    "tyrol-e": {
        "fold": "test",
        "pixelsize": 0.3,
        "numbers": list(range(1, 37)),
    },
    "austin": {
        "fold": "train",
        "pixelsize": 0.3,
        "numbers": list(range(1, 37)),
    },
    "chicago": {
        "fold": "train",
        "pixelsize": 0.3,
        "numbers": list(range(1, 37)),
    },
    "kitsap": {
        "fold": "train",
        "pixelsize": 0.3,
        "numbers": list(range(1, 37)),
    },
    "tyrol-w": {
        "fold": "train",
        "pixelsize": 0.3,
        "numbers": list(range(1, 37)),
    },
    "vienna": {
        "fold": "train",
        "pixelsize": 0.3,
        "numbers": list(range(1, 37)),
    },
}

IMAGE_DIR_NAME = "images"
IMAGE_NAME_FORMAT = "{city}{number}"
IMAGE_FILENAME_FORMAT = IMAGE_NAME_FORMAT + ".tif"  # City name, number
POLYGON_DIRNAME = "gt_polygons"
POLYGONS_FILENAME_FORMAT = IMAGE_NAME_FORMAT + ".npy"  # City name, number


def get_tile_info_list():
    tile_info_list = []
    for city, info in CITY_METADATA_DICT.items():
        for number in info["numbers"]:
            image_info = {
                "city": city,
                "number": number,
            }
            tile_info_list.append(image_info)
    return tile_info_list


def get_image_filepath(raw_dirpath, city, number):
    fold = CITY_METADATA_DICT[city]["fold"]
    filename = IMAGE_FILENAME_FORMAT.format(city=city, number=number)
    filepath = os.path.join(raw_dirpath, fold, IMAGE_DIR_NAME, filename)
    return filepath


def get_polygons_filepath(raw_dirpath, polygon_dirname, city, number, overwrite_polygons_filename_format=None):
    if overwrite_polygons_filename_format is None:
        polygons_filename_format = POLYGONS_FILENAME_FORMAT
    else:
        polygons_filename_format = overwrite_polygons_filename_format
    fold = CITY_METADATA_DICT[city]["fold"]
    filename = polygons_filename_format.format(city=city, number=number)
    filepath = os.path.join(raw_dirpath, fold, polygon_dirname, filename)
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


def load_polygons(raw_dirpath, polygon_dirname, city, number):
    filepath = get_polygons_filepath(raw_dirpath, polygon_dirname, city, number)
    try:
        gt_polygons = np.load(filepath)
    except FileNotFoundError:
        print("City {}, number {} does not have gt polygons in directory {}".format(city, number, polygon_dirname))
        gt_polygons = None
    return gt_polygons


def load_gt_data(raw_dirpath, city, number, overwrite_polygon_dir_name=None):
    if overwrite_polygon_dir_name is None:
        polygon_dirname = POLYGON_DIRNAME
    else:
        polygon_dirname = overwrite_polygon_dir_name

    # Load image
    image_array, image_metadata = load_image(raw_dirpath, city, number)

    # Load polygon data
    gt_polygons = load_polygons(raw_dirpath, polygon_dirname, city, number)

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
