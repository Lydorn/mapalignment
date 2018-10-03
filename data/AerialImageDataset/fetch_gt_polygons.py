import sys
import os
import numpy as np

sys.path.append("../../../projects/utils")
import python_utils
import polygon_utils
import geo_utils


# --- Params --- #

DIR_PATH_LIST = ["./raw/train", "./raw/test"]

IMAGE_DIR_NAME = "images"
IMAGE_EXTENSION = "tif"

GT_POLYGONS_DIR_NAME = "gt_polygons"

# --- --- #


def load_gt_polygons(image_filepath):
    gt_polygons = geo_utils.get_polygons_from_osm(image_filepath, tag="building")
    if len(gt_polygons):
        gt_polygons = polygon_utils.polygons_remove_holes(gt_polygons)  # TODO: Remove

        # Remove redundant vertices
        gt_polygons = polygon_utils.simplify_polygons(gt_polygons, tolerance=1)

        return gt_polygons
    return None


def fetch_from_images_in_directory(dir_path):
    print("Fetching for images in {}".format(dir_path))
    gt_polygons_dir_path = os.path.join(dir_path, GT_POLYGONS_DIR_NAME)
    if not os.path.exists(gt_polygons_dir_path):
        os.makedirs(gt_polygons_dir_path)

    images_dir_path = os.path.join(dir_path, IMAGE_DIR_NAME)
    image_filepaths = python_utils.get_filepaths(images_dir_path, IMAGE_EXTENSION)

    for i, image_filepath in enumerate(image_filepaths):
        image_basename = os.path.basename(image_filepath)
        image_name = os.path.splitext(image_basename)[0]
        print("Fetching for image {}. Progress: {}/{}".format(image_name, i+1, len(image_filepaths)))
        gt_polygons_path = os.path.join(gt_polygons_dir_path, "{}.npy".format(image_name))
        if not os.path.exists(gt_polygons_path):
            gt_polygons = load_gt_polygons(image_filepath)
            if gt_polygons is not None:
                np.save(gt_polygons_path, gt_polygons)
            else:
                print("Fetching did not return any polygons. Skip this one.")
        else:
            print("GT polygons data was already fetched, skip this one. (Delete the gt_polygons file to re-fetch)")


def main():

    for dir_path in DIR_PATH_LIST:
        fetch_from_images_in_directory(dir_path)


if __name__ == "__main__":
    main()
