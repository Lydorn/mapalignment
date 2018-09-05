import sys
import os
import math
import json
import random
import re

import skimage.transform
import numpy as np

import tensorflow as tf

import config_aerial_image_multires as config

sys.path.append("../utils")
import visualization

sys.path.append("../../utils")
import tf_utils
import polygon_utils
import image_utils
import python_utils
import math_utils
import dataset_utils

if python_utils.module_exists("matplotlib.pyplot"):
    import matplotlib.pyplot as plt


def load_gt_data(image_filepath, gt_polygons_filepath):
    if not os.path.exists(image_filepath):
        print("{} does not exist".format(image_filepath))
        return None, None
    if not os.path.exists(gt_polygons_filepath):
        print("{} does not exist".format(gt_polygons_filepath))
        return None, None

    image = image_utils.load_image(image_filepath)
    gt_polygons = np.load(gt_polygons_filepath)

    # try:
    #     geo_utils.save_shapefile_from_polygons(gt_polygons, image_filepath, gt_polygons_filepath + ".shp")
    # except:
    #     pass

    return image, gt_polygons


def downsample_gt_data(image, gt_polygons, normed_disp_field_maps, downsampling_factor):
    downsampled_image = skimage.transform.rescale(image, 1 / downsampling_factor, order=3, preserve_range=True)
    downsampled_image = downsampled_image.astype(image.dtype)
    downsampled_gt_polygons = polygon_utils.rescale_polygon(gt_polygons, 1 / downsampling_factor)
    downsampled_normed_disp_field_maps = np.empty((normed_disp_field_maps.shape[0],
                                                   round(normed_disp_field_maps.shape[1] / downsampling_factor),
                                                   round(normed_disp_field_maps.shape[2] / downsampling_factor),
                                                   normed_disp_field_maps.shape[3]))
    for i in range(normed_disp_field_maps.shape[0]):
        downsampled_normed_disp_field_maps[i] = skimage.transform.rescale(normed_disp_field_maps[i], 1 / downsampling_factor, order=3, preserve_range=True)
    return downsampled_image, downsampled_gt_polygons, downsampled_normed_disp_field_maps


def generate_disp_data(normed_disp_field_maps, gt_polygons, disp_max_abs_value, spatial_shape):
    scaled_disp_field_maps = normed_disp_field_maps * disp_max_abs_value
    disp_polygons_list = polygon_utils.apply_displacement_fields_to_polygons(gt_polygons,
                                                                             scaled_disp_field_maps)
    disp_polygon_maps = polygon_utils.draw_polygon_maps(disp_polygons_list, spatial_shape, fill=True,
                                                        edges=True, vertices=True)
    return disp_polygons_list, disp_polygon_maps


def process_sample_into_patches(patch_stride, patch_res, image, gt_polygon_map, disp_field_maps, disp_polygon_maps,
                                gt_polygons=None, disp_polygons_list=None):
    """
    Crops all inputs to patches generated with patch_stride and patch_res

    :param patch_stride:
    :param patch_res:
    :param image:
    :param gt_polygon_map:
    :param disp_field_maps:
    :param disp_polygon_maps:
    :param gt_polygons:
    :param disp_polygons_list:
    :return:
    """
    include_polygons = gt_polygons is not None and disp_polygons_list is not None
    patches = []
    patch_boundingboxes = image_utils.compute_patch_boundingboxes(image.shape[0:2],
                                                                  stride=patch_stride,
                                                                  patch_res=patch_res)
    # print(patch_boundingboxes)
    for patch_boundingbox in patch_boundingboxes:
        # Crop image
        patch_image = image[patch_boundingbox[0]:patch_boundingbox[2], patch_boundingbox[1]:patch_boundingbox[3], :]

        if include_polygons:
            patch_gt_polygons, \
            patch_disp_polygons_array = polygon_utils.prepare_polygons_for_tfrecord(gt_polygons, disp_polygons_list,
                                                                                    patch_boundingbox)
        else:
            patch_gt_polygons = patch_disp_polygons_array = None

        patch_gt_polygon_map = gt_polygon_map[patch_boundingbox[0]:patch_boundingbox[2],
                               patch_boundingbox[1]:patch_boundingbox[3], :]
        patch_disp_field_maps = disp_field_maps[:,
                                patch_boundingbox[0]:patch_boundingbox[2],
                                patch_boundingbox[1]:patch_boundingbox[3], :]
        patch_disp_polygon_maps_array = disp_polygon_maps[:,
                                        patch_boundingbox[0]:patch_boundingbox[2],
                                        patch_boundingbox[1]:patch_boundingbox[3], :]

        # Filter out patches based on presence of polygon and area ratio inside inner patch =
        patch_inner_res = 2 * patch_stride
        patch_padding = (patch_res - patch_inner_res) // 2
        inner_patch_gt_polygon_map_corners = patch_gt_polygon_map[patch_padding:-patch_padding, patch_padding:-patch_padding, 2]
        if np.sum(inner_patch_gt_polygon_map_corners) \
                and (not include_polygons or (include_polygons and patch_gt_polygons is not None)):

            assert patch_image.shape[0] == patch_image.shape[1], "image should be square otherwise tile_res cannot be defined"
            tile_res = patch_image.shape[0]
            disp_map_count = patch_disp_polygon_maps_array.shape[0]

            patches.append({
                "tile_res": tile_res,
                "disp_map_count": disp_map_count,
                "image": patch_image,
                "gt_polygons": patch_gt_polygons,
                "disp_polygons": patch_disp_polygons_array,
                "gt_polygon_map": patch_gt_polygon_map,
                "disp_field_maps": patch_disp_field_maps,
                "disp_polygon_maps": patch_disp_polygon_maps_array,
            })

    return patches


def save_patch_to_tfrecord(patch, shard_writer):
    # print(patch["disp_field_maps"].min() / 2147483647, patch["disp_field_maps"].max() / 2147483647)

    # visualization.plot_field_map("disp_field_map", patch["disp_field_maps"][0])

    # Compress image into jpg
    image_raw = image_utils.convert_array_to_jpg_bytes(patch["image"], mode="RGB")
    gt_polygon_map_raw = patch["gt_polygon_map"].tostring()  # TODO: convert to png
    disp_field_maps_raw = patch["disp_field_maps"].tostring()
    disp_polygon_maps_raw = patch["disp_polygon_maps"].tostring()  # TODO: convert to png (have to handle several png images...)

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


def process_image(image_filepath, gt_polygons_filepath, patch_stride, patch_res, downsampling_factors, disp_max_abs_value, include_polygons,
                  downsampling_factor_writers):
    """
    Writes to all the writers (one for each resolution) all sample patches extracted from the image at location image_filepath.

    :param image_filepath:
    :param patch_stride:
    :param patch_res:
    :param downsampling_factors:
    :param disp_max_abs_value:
    :param include_polygons:
    :param downsampling_factor_writers:
    :return:
    """
    ori_image, ori_gt_polygons = load_gt_data(image_filepath, gt_polygons_filepath)

    if ori_gt_polygons is None:
        return False

    # visualization.init_figures(["gt_data"], figsize=(60, 40))
    # visualization.plot_example_polygons("gt_data", ori_image, ori_gt_polygons)

    # Create displacement maps
    ori_normed_disp_field_maps = math_utils.create_displacement_field_maps(ori_image.shape[:2], config.DISP_MAP_COUNT,
                                                                       config.DISP_MODES, config.DISP_GAUSS_MU_RANGE,
                                                                       config.DISP_GAUSS_SIG_SCALING)  # TODO: uncomment
    # ori_normed_disp_field_maps = np.zeros((config.DISP_MAP_COUNT, ori_image.shape[0], ori_image.shape[1], 2))  # TODO: remove

    # # TODO: remove
    # np.random.seed(seed=0)
    # colors = np.random.randint(0, 255, size=(len(downsampling_factors), 3), dtype=np.uint8)

    for index, downsampling_factor in enumerate(downsampling_factors):
        print("downsampling_factor: {}".format(downsampling_factor))
        # Downsample ground-truth
        image, gt_polygons, normed_disp_field_maps = downsample_gt_data(ori_image, ori_gt_polygons, ori_normed_disp_field_maps, downsampling_factor)
        spatial_shape = image.shape[:2]

        # Random color
        # image = np.tile(colors[index], reps=[image.shape[0], image.shape[1], 1])  # TODO: remove

        # Draw gt polygon map
        gt_polygon_map = polygon_utils.draw_polygon_map(gt_polygons, spatial_shape, fill=True, edges=True,
                                                        vertices=True)

        # Generate final displacement
        disp_polygons_list, disp_polygon_maps = generate_disp_data(normed_disp_field_maps, gt_polygons,
                                                                   disp_max_abs_value, spatial_shape)

        # Compress data
        gt_polygons = [polygon.astype(np.float16) for polygon in gt_polygons]
        disp_polygons_list = [[polygon.astype(np.float16) for polygon in polygons] for polygons in disp_polygons_list]
        disp_field_maps = normed_disp_field_maps * 32767  # int16 max value = 32767
        disp_field_maps = np.round(disp_field_maps)
        disp_field_maps = disp_field_maps.astype(np.int16)

        # Cut sample into patches
        if include_polygons:
            patches = process_sample_into_patches(patch_stride, patch_res, image, gt_polygon_map,
                                                  disp_field_maps, disp_polygon_maps,
                                                  gt_polygons, disp_polygons_list)
        else:
            patches = process_sample_into_patches(patch_stride, patch_res, image, gt_polygon_map,
                                                  disp_field_maps, disp_polygon_maps)

        for patch in patches:
            save_patch_to_tfrecord(patch, downsampling_factor_writers[downsampling_factor])

    return True


def process_dataset(images_dir_list, image_extension, gt_polygons_dir_name, gt_polygons_extension,
                    patch_stride, patch_res,
                    train_count, val_count, test_count,
                    data_aug_rot,
                    downsampling_factors, city_min_downsampling_factor,
                    disp_max_abs_value,
                    image_index_start=0, image_index_end=-1):
    """

    :param images_dir:
    :param image_extension:
    :param patch_stride:
    :param patch_res:
    :param train_count:
    :param val_count:
    :param test_count:
    :param data_aug_rot:
    :param downsampling_factors:
    :param city_min_downsampling_factor:
    :param disp_max_abs_value:
    :param image_index_start:
    :param image_index_end: Exclusive
    :return:
    """
    print("Processing images from {}".format(images_dir_list))
    image_filepaths = python_utils.get_dir_list_filepaths(images_dir_list, image_extension)
    random.shuffle(image_filepaths)

    if -1 < image_index_end:
        image_index_end = min(len(image_filepaths), image_index_end)
    else:
        image_index_end = len(image_filepaths)

    # Create writers
    fold_writers = {}
    for dataset_fold in ["train", "val", "test"]:
        # Create shard writers
        shard_writers = {}
        for downsampling_factor in downsampling_factors:
            filename_format = os.path.join(config.TFRECORDS_DIR,
                                           config.TFRECORD_FILENAME_FORMAT.format(dataset_fold, downsampling_factor))
            shard_writer = dataset_utils.TFRecordShardWriter(filename_format, config.RECORDS_PER_SHARD)
            shard_writers[downsampling_factor] = shard_writer
        fold_writers[dataset_fold] = shard_writers

    for image_index in range(image_index_start, image_index_end):
        image_filepath = image_filepaths[image_index]
        base_filepath = os.path.splitext(image_filepath)[0]
        image_basename = os.path.basename(base_filepath)
        gt_polygons_filepath = os.path.join(
            os.path.dirname(
                os.path.dirname(image_filepath)
            ),
            gt_polygons_dir_name,
            "{}.{}".format(image_basename, gt_polygons_extension))
        print("Processing image {}. Progression: {}/{}"
              .format(image_basename, image_index + 1, len(image_filepaths)))

        # Decide in which dataset to put this image
        if image_index < train_count:
            chosen_fold = "train"
        elif image_index < train_count + val_count:
            chosen_fold = "val"
        elif image_index < train_count + val_count + test_count:
            chosen_fold = "test"
        else:
            print("This image is not going to be included!")

        include_polygons = (chosen_fold == "val" or chosen_fold == "test")
        if data_aug_rot and chosen_fold == "train":
            # Account for data augmentation when rotating patches on the training set
            adjusted_patch_res = math.ceil(patch_res * math.sqrt(2))
            adjusted_patch_stride = math.floor(
                patch_stride * math.sqrt(
                    2) / 2)  # Divided by 2 so that no pixels are left out when rotating by 45 degrees
        else:
            adjusted_patch_res = patch_res
            adjusted_patch_stride = patch_stride

        # Filter out downsampling_factors that are lower than city_min_downsampling_factor
        city = re.sub("[0-9]+", "", image_basename)
        image_downsampling_factors = [downsampling_factor for downsampling_factor in downsampling_factors if city_min_downsampling_factor[city] <= downsampling_factor]

        process_image(image_filepath, gt_polygons_filepath,
                      adjusted_patch_stride, adjusted_patch_res,
                      image_downsampling_factors,
                      disp_max_abs_value,
                      include_polygons,
                      fold_writers[chosen_fold])

    # Close writers
    for dataset in ["train", "val", "test"]:
        for downsampling_factor in downsampling_factors:
            fold_writers[dataset][downsampling_factor].close()


def save_metadata(meta_data_filepath, disp_max_abs_value, downsampling_factors):
    data = {
        "disp_max_abs_value": disp_max_abs_value,
        "downsampling_factors": downsampling_factors,
    }
    with open(meta_data_filepath, 'w') as outfile:
        json.dump(data, outfile)


def main():
    random.seed(0)

    # Create dataset tfrecords directory of it does not exist
    if not os.path.exists(config.TFRECORDS_DIR):
        os.makedirs(config.TFRECORDS_DIR)

    # Save meta-data
    meta_data_filepath = os.path.join(config.TFRECORDS_DIR, "metadata.txt")
    save_metadata(meta_data_filepath, config.DISP_MAX_ABS_VALUE, config.DOWNSAMPLING_FACTORS)

    # Save data
    process_dataset(config.IMAGES_DIR_LIST,
                    config.IMAGE_EXTENSION,
                    config.GT_POLYGONS_DIR_NAME,
                    config.GT_POLYGONS_EXTENSION,
                    config.TILE_STRIDE,
                    config.TILE_RES,
                    config.TRAIN_COUNT,
                    config.VAL_COUNT,
                    config.TEST_COUNT,
                    config.DATA_AUG_ROT,
                    config.DOWNSAMPLING_FACTORS,
                    config.CITY_MIN_DOWNSAMPLING_FACTOR,
                    config.DISP_MAX_ABS_VALUE,
                    image_index_start=config.IMAGE_INDEX_START,
                    image_index_end=config.IMAGE_INDEX_END)


if __name__ == "__main__":
    main()
