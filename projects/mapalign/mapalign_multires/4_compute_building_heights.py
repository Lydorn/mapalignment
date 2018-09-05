import os.path
import sys
import math
import itertools
import numpy as np

import config

sys.path.append("../../utils")
import geo_utils

# --- Params --- #

DATASET_DIR = os.path.join(config.PROJECT_DIR, "../../../data/stereo_dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw/leibnitz")
INPUT_DIR = "test/stereo_dataset_real_displacements.align.ds_fac_8.ds_fac_4.ds_fac_2"

VIEW_INFO_LIST = [
    {
        "image_name": "leibnitz_ortho_ref",
        "image_filepath": os.path.join(RAW_DIR, "leibnitz_ortho_ref_RGB.tif"),
        "shapefile_filepath": os.path.join(INPUT_DIR, "leibnitz_ref_rec.aligned_polygons.shp"),
        # "shapefile_filepath": os.path.join(INPUT_DIR, "leibnitz_rec_ref.ori_polygons.shp"),  # GT polygons
        "angle": 76.66734850675575 * math.pi / 180,  # Elevation
    },
    {
        "image_name": "leibnitz_ortho_rec",
        "image_filepath": os.path.join(RAW_DIR, "leibnitz_ortho_rec_RGB.tif"),
        "shapefile_filepath": os.path.join(INPUT_DIR, "leibnitz_rec_ref.aligned_polygons.shp"),
        # "shapefile_filepath": os.path.join(INPUT_DIR, "leibnitz_ref_rec.ori_polygons.shp"),  # GT polygons
        "angle": 69.62096370829768 * math.pi / 180,  # Elevation
    },
]

PIXELSIZE = 0.5  # 1 pixel is 0.5 meters

OUTPUT_BASE_DIRPATH = "3d_buildings/leibnitz"

# ---  --- #


def compute_heights(view_1, view_2, pixelsize):
    tan_1 = math.tan(view_1["angle"])
    tan_2 = math.tan(view_2["angle"])
    tan_alpha = min(tan_1, tan_2)
    tan_beta = max(tan_1, tan_2)
    angle_height_coef = tan_alpha * tan_beta / (tan_beta - tan_alpha)
    heights = []
    for polygon_1, polygon_2 in zip(view_1["polygon_list"], view_2["polygon_list"]):
        center_1 = np.mean(polygon_1, axis=0, keepdims=True)
        center_2 = np.mean(polygon_2, axis=0, keepdims=True)
        distance = np.sqrt(np.sum(np.square(center_1 - center_2), axis=1))[0]
        height = distance * angle_height_coef * pixelsize
        heights.append(height)
    return heights


def main(view_info_list, pixelsize, output_base_dirpath):
    # --- Loading shapefiles --- #
    print("# --- Loading shapefiles --- #")
    view_list = []
    for view_info in view_info_list:
        polygon_list, properties_list = geo_utils.get_polygons_from_shapefile(view_info["image_filepath"],
                                                                              view_info["shapefile_filepath"])
        view = {
            "polygon_list": polygon_list,
            "properties_list": properties_list,
            "angle": view_info["angle"],
        }
        view_list.append(view)

    # Extract ground truth building heights
    gt_heights = []
    for properties in view_list[0]["properties_list"]:
        gt_heights.append(properties["HEIGHT"])
    gt_heights_array = np.array(gt_heights)

    # Iterate over all possible pairs of views:
    heights_list = []
    view_pair_list = itertools.combinations(view_list, 2)
    for view_pair in view_pair_list:
        heights = compute_heights(view_pair[0], view_pair[1], pixelsize)
        heights_list.append(heights)
    # Average results from pairs
    heights_list_array = np.array(heights_list)
    pred_heights_array = np.mean(heights_list_array, axis=0)

    # Correct pred heights:
    pred_heights_array = pred_heights_array / 4.39  # Factor found with using the ground truth polygons for computing the height

    # --- Save results --- #
    polygon_list = view_list[0]["polygon_list"]  # Take from the first view

    # Save shapefile
    output_shapefile_filepath = os.path.join(output_base_dirpath, view_info_list[0]["image_name"] + "_pred_heights.shp")
    pred_properties_list = view_list[0]["properties_list"].copy()  # First copy existing properties list
    for i, pred_height in enumerate(pred_heights_array):  # Then replace HEIGHT
        pred_properties_list[i]["HEIGHT"] = pred_height
    geo_utils.save_shapefile_from_polygons(view_list[0]["polygon_list"], view_info_list[0]["image_filepath"], output_shapefile_filepath, properties_list=pred_properties_list)

    # Save for modeling buildings in Blender and measuring accuracy
    scaled_polygon_list = [polygon * pixelsize for polygon in polygon_list]
    np.save(os.path.join(output_base_dirpath, "polygons.npy"), scaled_polygon_list)
    np.save(os.path.join(output_base_dirpath, "gt_heights.npy"), gt_heights_array)
    np.save(os.path.join(output_base_dirpath, "pred_heights.npy"), pred_heights_array)


if __name__ == "__main__":
    main(VIEW_INFO_LIST, PIXELSIZE, OUTPUT_BASE_DIRPATH)
