import os.path
import sys

import read

FILE_DIRNAME = os.getcwd()
sys.path.append(os.path.join(FILE_DIRNAME, "../../projects/utils"))
import geo_utils

# --- Params --- #

RAW_DIRPATH = os.path.join(FILE_DIRNAME, "raw")

IMAGE_INFO_LIST = [
    {
        "city": "bloomington",
        "numbers": list(range(1, 37)),
    },
    {
        "city": "bellingham",
        "numbers": list(range(1, 37)),
    },
    {
        "city": "innsbruck",
        "numbers": list(range(1, 37)),
    },
    {
        "city": "sfo",
        "numbers": list(range(1, 37)),
    },
    {
        "city": "tyrol-e",
        "numbers": list(range(1, 37)),
    },
    {
        "city": "austin",
        "numbers": list(range(1, 37)),
    },
    {
        "city": "chicago",
        "numbers": list(range(1, 37)),
    },
    {
        "city": "kitsap",
        "numbers": list(range(1, 37)),
    },
    {
        "city": "tyrol-w",
        "numbers": list(range(1, 37)),
    },
    {
        "city": "vienna",
        "numbers": list(range(1, 37)),
    },
]

POLYGON_DIR_NAME = "aligned_gt_polygons_1"
SHAPEFILE_FILENAME_FORMAT = read.IMAGE_NAME_FORMAT + ".shp"  # City name, number

# --- --- #


def convert_npy_to_shp(raw_dirpath, polygon_dirname, city, number, shapefile_filename_format):
    # --- Load data --- #
    # Load polygon data
    image_filepath = read.get_image_filepath(raw_dirpath, city, number)
    polygons = read.load_polygons(raw_dirpath, polygon_dirname, city, number)

    if polygons is not None:
        output_shapefile_filepath = read.get_polygons_filepath(raw_dirpath, polygon_dirname, city, number, overwrite_polygons_filename_format=shapefile_filename_format)
        geo_utils.save_shapefile_from_polygons(polygons, image_filepath, output_shapefile_filepath)


def main():
    print("Converting polygons from {}".format(POLYGON_DIR_NAME))
    for image_info in IMAGE_INFO_LIST:
        for number in image_info["numbers"]:
            print("Converting polygons of city {}, number {}".format(image_info["city"], number))
            convert_npy_to_shp(RAW_DIRPATH, POLYGON_DIR_NAME, image_info["city"], number, SHAPEFILE_FILENAME_FORMAT)


if __name__ == "__main__":
    main()
