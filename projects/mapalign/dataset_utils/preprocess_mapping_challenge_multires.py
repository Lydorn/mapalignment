import sys
import os
import json

import skimage.transform
import skimage.draw
import numpy as np

# from PIL import Image, ImageDraw
# Image.MAX_IMAGE_PIXELS = 200000000

import tensorflow as tf

import config_mapping_challenge_multires as config

sys.path.append("../../../data/mapping_challenge_dataset")
import read

# sys.path.append("../utils")
# import visualization

sys.path.append("../../utils")
import tf_utils
import polygon_utils
import image_utils
# import python_utils
import math_utils
import dataset_utils

# if python_utils.module_exists("matplotlib.pyplot"):
#     import matplotlib.pyplot as plt


def downsample_gt_data(image, metadata, gt_polygons, normed_disp_field_maps, downsampling_factor):
    # First, correct the downsampling_factor so that:
    # A downsampling_factor of 1 results in a final pixel_size equal to config.REFERENCE_PIXEL_SIZE
    # A downsampling_factor of 2 results in a final pixel_size equal to 2 * config.REFERENCE_PIXEL_SIZE
    corrected_downsampling_factor = downsampling_factor * config.REFERENCE_PIXEL_SIZE / metadata["pixelsize"]
    scale = 1 / corrected_downsampling_factor
    downsampled_image = skimage.transform.rescale(image, scale, order=3, preserve_range=True)
    downsampled_image = downsampled_image.astype(image.dtype)
    downsampled_gt_polygons = polygon_utils.rescale_polygon(gt_polygons, scale)
    downsampled_normed_disp_field_maps = np.empty((normed_disp_field_maps.shape[0],
                                                   round(normed_disp_field_maps.shape[1] / corrected_downsampling_factor),
                                                   round(normed_disp_field_maps.shape[2] / corrected_downsampling_factor),
                                                   normed_disp_field_maps.shape[3]))
    for i in range(normed_disp_field_maps.shape[0]):
        downsampled_normed_disp_field_maps[i] = skimage.transform.rescale(normed_disp_field_maps[i],
                                                                          scale, order=3,
                                                                          preserve_range=True)
    return downsampled_image, downsampled_gt_polygons, downsampled_normed_disp_field_maps


def generate_disp_data(normed_disp_field_maps, gt_polygons, disp_max_abs_value, spatial_shape):
    scaled_disp_field_maps = normed_disp_field_maps * disp_max_abs_value
    disp_polygons_list = polygon_utils.apply_displacement_fields_to_polygons(gt_polygons,
                                                                             scaled_disp_field_maps)
    disp_polygon_maps = polygon_utils.draw_polygon_maps(disp_polygons_list, spatial_shape, fill=True,
                                                        edges=True, vertices=True)
    return disp_polygons_list, disp_polygon_maps


def save_patch_to_tfrecord(patch, shard_writer):
    # print(patch["disp_field_maps"].min() / 2147483647, patch["disp_field_maps"].max() / 2147483647)

    # visualization.plot_field_map("disp_field_map", patch["disp_field_maps"][0])

    # Compress image into jpg
    image_raw = image_utils.convert_array_to_jpg_bytes(patch["image"], mode="RGB")
    gt_polygon_map_raw = patch["gt_polygon_map"].tostring()  # TODO: convert to png
    disp_field_maps_raw = patch["disp_field_maps"].tostring()
    disp_polygon_maps_raw = patch[
        "disp_polygon_maps"].tostring()  # TODO: convert to png (have to handle several png images...)

    if patch["gt_polygons"] is not None and patch["disp_polygons"] is not None:
        gt_polygons_raw = patch["gt_polygons"].tostring()
        disp_polygons_raw = patch["disp_polygons"].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'tile_res': tf_utils.int64_feature(patch["tile_res"]),
            'disp_map_count': tf_utils.int64_feature(patch["disp_map_count"]),
            'image': tf_utils.bytes_feature(image_raw),
            'gt_polygon_count': tf_utils.int64_feature(patch["gt_polygons"].shape[0]),
            'gt_polygon_length': tf_utils.int64_feature(patch["gt_polygons"].shape[1]),
            'gt_polygons': tf_utils.bytes_feature(gt_polygons_raw),
            'disp_polygons': tf_utils.bytes_feature(disp_polygons_raw),
            'gt_polygon_map': tf_utils.bytes_feature(gt_polygon_map_raw),
            'disp_field_maps': tf_utils.bytes_feature(disp_field_maps_raw),
            'disp_polygon_maps': tf_utils.bytes_feature(disp_polygon_maps_raw)
        }))
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'tile_res': tf_utils.int64_feature(patch["tile_res"]),
            'disp_map_count': tf_utils.int64_feature(patch["disp_map_count"]),
            'image': tf_utils.bytes_feature(image_raw),
            'gt_polygon_map': tf_utils.bytes_feature(gt_polygon_map_raw),
            'disp_field_maps': tf_utils.bytes_feature(disp_field_maps_raw),
            'disp_polygon_maps': tf_utils.bytes_feature(disp_polygon_maps_raw),
        }))

    shard_writer.write(example.SerializeToString())


def process_image(reader, image_id, downsampling_factors, disp_field_maps_patch_creator, disp_max_abs_value,
                  include_polygons,
                  downsampling_factor_writers):
    """
    Writes to all the writers (one for each resolution) all sample patches extracted from the image_info.

    :param reader:
    :param image_id:
    :param downsampling_factors:
    :param disp_field_maps_patch_creator:
    :param disp_max_abs_value:
    :param include_polygons:
    :param downsampling_factor_writers:
    :return:
    """
    ori_image, ori_metadata, ori_gt_polygons = reader.load_gt_data(image_id)

    ori_gt_polygons = polygon_utils.polygons_remove_holes(ori_gt_polygons)  # TODO: Remove

    # Remove redundant vertices
    ori_gt_polygons = polygon_utils.simplify_polygons(ori_gt_polygons, tolerance=1)

    # visualization.init_figures(["gt_data"], figsize=(60, 40))
    # visualization.plot_example_polygons("gt_data", ori_image, ori_gt_polygons)

    # Get displacement maps
    ori_normed_disp_field_maps = disp_field_maps_patch_creator.get_patch()
    # ori_normed_disp_field_maps = np.zeros((config.DISP_MAP_COUNT, ori_image.shape[0], ori_image.shape[1], 2))  # TODO: remove

    # # TODO: remove
    # np.random.seed(seed=0)
    # colors = np.random.randint(0, 255, size=(len(downsampling_factors), 3), dtype=np.uint8)

    for index, downsampling_factor in enumerate(downsampling_factors):
        # print("downsampling_factor: {}".format(downsampling_factor))
        # Downsample ground-truth
        image, gt_polygons, normed_disp_field_maps = downsample_gt_data(ori_image, ori_metadata, ori_gt_polygons,
                                                                        ori_normed_disp_field_maps, downsampling_factor)

        spatial_shape = image.shape[:2]

        # Random color
        # image = np.tile(colors[index], reps=[image.shape[0], image.shape[1], 1])  # TODO: remove

        # Draw gt polygon map
        gt_polygon_map = polygon_utils.draw_polygon_map(gt_polygons, spatial_shape, fill=True, edges=True,
                                                        vertices=True)

        # Generate final displacement
        disp_polygons_list, disp_polygon_maps = generate_disp_data(normed_disp_field_maps, gt_polygons,
                                                                   disp_max_abs_value, spatial_shape)

        if gt_polygons[0][0][0] == np.nan or gt_polygons[0][0][1] == np.nan:
            print(gt_polygons[0][0])

        if disp_polygons_list[0][0][0][0] == np.nan or disp_polygons_list[0][0][0][1] == np.nan:
            print("disp_polygons_list:")
            print(disp_polygons_list[0][0])

        # Compress data
        gt_polygons = [polygon.astype(np.float16) for polygon in gt_polygons]
        disp_polygons_list = [[polygon.astype(np.float16) for polygon in polygons] for polygons in disp_polygons_list]
        disp_field_maps = normed_disp_field_maps * 32767  # int16 max value = 32767
        disp_field_maps = np.round(disp_field_maps)
        disp_field_maps = disp_field_maps.astype(np.int16)

        if include_polygons:
            gt_polygons, \
            disp_polygons_array = polygon_utils.prepare_polygons_for_tfrecord(gt_polygons, disp_polygons_list)
        else:
            gt_polygons = disp_polygons_array = None

        assert image.shape[0] == image.shape[1], "image should be square otherwise tile_res cannot be defined"
        tile_res = image.shape[0]
        disp_map_count = disp_polygon_maps.shape[0]

        patch = {
            "tile_res": tile_res,
            "disp_map_count": disp_map_count,
            "image": image,
            "gt_polygons": gt_polygons,
            "disp_polygons": disp_polygons_array,
            "gt_polygon_map": gt_polygon_map,
            "disp_field_maps": disp_field_maps,
            "disp_polygon_maps": disp_polygon_maps,
        }

        save_patch_to_tfrecord(patch, downsampling_factor_writers[downsampling_factor])

    return True


def process_dataset(dataset_fold,
                    dataset_raw_dirpath,
                    downsampling_factors,
                    disp_max_abs_value):
    print("Processing images from {}".format(dataset_raw_dirpath))

    # Create shard writers
    shard_writers = {}
    for downsampling_factor in downsampling_factors:
        filename_format = os.path.join(config.TFRECORDS_DIR,
                                       config.TFRECORD_FILENAME_FORMAT.format(dataset_fold, downsampling_factor))
        shard_writer = dataset_utils.TFRecordShardWriter(filename_format, config.RECORDS_PER_SHARD)
        shard_writers[downsampling_factor] = shard_writer

    # Create reader
    reader = read.Reader(dataset_raw_dirpath, dataset_fold)

    # Create DispFieldMapsPatchCreator
    disp_field_maps_patch_creator = math_utils.DispFieldMapsPatchCreator(config.DISP_GLOBAL_SHAPE, config.DISP_PATCH_RES, config.DISP_MAP_COUNT, config.DISP_MODES, config.DISP_GAUSS_MU_RANGE, config.DISP_GAUSS_SIG_SCALING)

    for image_index, image_id in enumerate(reader.image_id_list):
        if (image_index + 1) % 10 == 0:
            print("Processing progression: {}/{}"
                  .format(image_index + 1, len(reader.image_id_list)))

        include_polygons = (dataset_fold == "val" or dataset_fold == "test")

        process_image(reader, image_id,
                      downsampling_factors,
                      disp_field_maps_patch_creator,
                      disp_max_abs_value,
                      include_polygons,
                      shard_writers)

    # Close writers
    for downsampling_factor in downsampling_factors:
        shard_writers[downsampling_factor].close()


def save_metadata(meta_data_filepath, disp_max_abs_value, downsampling_factors):
    data = {
        "disp_max_abs_value": disp_max_abs_value,
        "downsampling_factors": downsampling_factors,
    }
    with open(meta_data_filepath, 'w') as outfile:
        json.dump(data, outfile)


def main():
    # input("Prepare dataset, overwrites previous data. This can take a while (1h), press <Enter> to continue...")

    # Create dataset tfrecords directory of it does not exist
    if not os.path.exists(config.TFRECORDS_DIR):
        os.makedirs(config.TFRECORDS_DIR)

    # Save meta-data
    meta_data_filepath = os.path.join(config.TFRECORDS_DIR, "metadata.txt")
    save_metadata(meta_data_filepath, config.DISP_MAX_ABS_VALUE,
                  config.DOWNSAMPLING_FACTORS)

    process_dataset("train",
                    config.DATASET_RAW_DIRPATH,
                    config.DOWNSAMPLING_FACTORS,
                    config.DISP_MAX_ABS_VALUE)
    process_dataset("val",
                    config.DATASET_RAW_DIRPATH,
                    config.DOWNSAMPLING_FACTORS,
                    config.DISP_MAX_ABS_VALUE)


if __name__ == "__main__":
    main()
