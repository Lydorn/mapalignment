import sys
sys.path.append("../utils")
import visualization

from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os

# --- Params --- #

FOLD_LIST = ["train", "val"]

IMAGES_DIRPATH_FORMAT = "{}/images"  # var: fold
ANNOTATIONS_FILEPATH_FORMAT = "{}/annotation.json"  # var: fold
# ANNOTATIONS_FILEPATH_FORMAT = "{}/annotation-small.json"  # var: fold

PIXELSIZE = 0.3  # This is a guess, as that information is unknown

# --- --- #


def swap_coords(polygon):
    polygon_new = polygon.copy()
    polygon_new[..., 0] = polygon[..., 1]
    polygon_new[..., 1] = polygon[..., 0]
    return polygon_new


class Reader:
    def __init__(self, raw_dirpath, fold):
        assert fold in FOLD_LIST, "Input fold={} should be in FOLD_LIST={}".format(fold, FOLD_LIST)
        self.images_dirpath = os.path.join(raw_dirpath, IMAGES_DIRPATH_FORMAT.format(fold))
        self.annotations_filepath = os.path.join(raw_dirpath, ANNOTATIONS_FILEPATH_FORMAT.format(fold))
        self.coco = COCO(self.annotations_filepath)
        self.category_id_list = self.coco.loadCats(self.coco.getCatIds())
        self.image_id_list = self.coco.getImgIds(catIds=self.coco.getCatIds())

    def load_image(self, image_id):
        img = self.coco.loadImgs(image_id)[0]
        image_filepath = os.path.join(self.images_dirpath, img["file_name"])
        image = io.imread(image_filepath)

        image_metadata = {
            "filepath": image_filepath,
            "pixelsize": PIXELSIZE
        }

        return image, image_metadata

    def load_polygons(self, image_id):
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotation_list = self.coco.loadAnns(annotation_ids)

        polygons_coords_list = []
        for annotation in annotation_list:
            flattened_segmentation_list = annotation["segmentation"]
            flattened_arrays = np.array(flattened_segmentation_list)
            arrays = np.reshape(flattened_arrays, (flattened_arrays.shape[0], -1, 2))
            arrays = swap_coords(arrays)
            array_list = []
            for array in arrays:
                array_list.append(array)
                array_list.append(np.array([[np.nan, np.nan]]))
            concatenated_array = np.concatenate(array_list, axis=0)
            polygons_coords_list.append(concatenated_array)

        return polygons_coords_list

    def load_gt_data(self, image_id):
        # Load image
        image_array, image_metadata = self.load_image(image_id)

        # Load polygon data
        gt_polygons = self.load_polygons(image_id)

        # TODO: remove
        visualization.save_plot_image_polygons("polygons.png", image_array, [], gt_polygons, [])
        # TODO end

        return image_array, image_metadata, gt_polygons


def main():

    raw_dirpath = "raw"
    fold = "train"

    reader = Reader(raw_dirpath, fold)

    image_id = reader.image_id_list[1]

    image_array, image_metadata, gt_polygons = reader.load_gt_data(image_id)


    print(image_array.shape)
    print(image_metadata)
    print(gt_polygons)


if __name__ == "__main__":
    main()
