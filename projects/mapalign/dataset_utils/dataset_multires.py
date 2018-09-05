import sys
import os
import math
import json

import tensorflow as tf

sys.path.append("../utils")  # Mapalign sub-projects utils
import visualization

import skimage.io

sys.path.append("../../utils")  # Projects utils
import tf_utils
import python_utils

# --- Param --- #
STRING_QUEUE_CAPACITY = 4000
MIN_QUEUE_EXAMPLES = 2000

# --- --- #


def all_items_are_integers(l):
    result = True
    for i in l:
        if type(i) is not int:
            result = False
            break
    return result


def get_all_shards(shard_filepath_format):
    shard_filepath_list = []
    shard_index = 0
    stop = False
    while not stop:
        shard_filepath = shard_filepath_format.format(shard_index)
        if os.path.exists(shard_filepath):
            shard_filepath_list.append(shard_filepath)
            shard_index += 1
        else:
            stop = True
    return shard_filepath_list


def create_dataset_filename_list(tfrecords_dir_list, tfrecord_filename_format, downsampling_factors, dataset="train",
                                 resolution_file_repeats=None):
    if resolution_file_repeats is None:
        resolution_file_repeats = [1] * len(downsampling_factors)
    assert len(downsampling_factors) == len(resolution_file_repeats), \
        "Downsampling_factors and sample_resolution_prob_weights must have the same number of elements"
    assert all_items_are_integers(resolution_file_repeats), "All repeat count should be integers"
    dataset_filename_list = []
    for tfrecords_dir in tfrecords_dir_list:
        for downsampling_factor, resolution_file_repeat in zip(downsampling_factors, resolution_file_repeats):
            shard_filepath_format = os.path.join(tfrecords_dir, tfrecord_filename_format.format(dataset, downsampling_factor))
            shard_filepath_list = get_all_shards(shard_filepath_format)
            repeated_filepaths = shard_filepath_list * resolution_file_repeat  # Repeat filepaths
            dataset_filename_list.extend(repeated_filepaths)
    return dataset_filename_list


def rotate_poly_map(poly_map, angle):
    # Apply NEAREST to corner channels and BILINEAR to the others
    gt_polygon_map_area, gt_polygon_map_edges, gt_polygon_corners = tf.unstack(poly_map, axis=-1)
    gt_polygon_map_area = tf.contrib.image.rotate(gt_polygon_map_area, angle, interpolation='BILINEAR')
    gt_polygon_map_edges = tf.contrib.image.rotate(gt_polygon_map_edges, angle, interpolation='BILINEAR')
    gt_polygon_corners = tf.contrib.image.rotate(gt_polygon_corners, angle, interpolation='NEAREST')
    poly_map = tf.stack([gt_polygon_map_area, gt_polygon_map_edges, gt_polygon_corners], axis=-1)
    return poly_map


def rotate_field_vectors(field_map, angle):
    """
    Just rotates every vector of disp_field_map by angle. Does not rotate the spatial support (which is rotated in rotate_poly_map())

    :param field_map:
    :param angle: (in rad.)
    :return:
    """
    field_map_shape = tf.shape(field_map)  # Save shape for later reshape
    tile_resfield_map = tf.reshape(field_map, [-1, 2])  # Convert to a list of vectors
    rot_mat = tf.cast(tf.stack([(tf.cos(-angle), -tf.sin(-angle)), (tf.sin(-angle), tf.cos(-angle))], axis=0),
                      tf.float32)
    tile_resfield_map = tf.cast(tile_resfield_map, tf.float32)
    tile_resfield_map = tf.matmul(tile_resfield_map, rot_mat)
    tile_resfield_map = tf.reshape(tile_resfield_map,
                                   field_map_shape)  # Reshape back to field of vectors
    return tile_resfield_map


def crop_or_pad_many(image_list, res):
    assert type(res) == int, "type(res) should be int"
    image_batch = tf.stack(image_list, axis=0)
    cropped_image_batch = tf.image.resize_image_with_crop_or_pad(image=image_batch, target_height=res, target_width=res)
    cropped_image_list = tf.unstack(cropped_image_batch, axis=0)
    return cropped_image_list


def corners_in_inner_patch(poly_map, patch_inner_res):
    cropped_disp_polygon_map = tf.image.resize_image_with_crop_or_pad(image=poly_map,
                                                                      target_height=patch_inner_res,
                                                                      target_width=patch_inner_res)
    _, _, disp_polygon_map_corners = tf.unstack(cropped_disp_polygon_map, axis=-1)
    result = tf.cast(tf.reduce_sum(disp_polygon_map_corners), dtype=tf.bool)
    return result


def field_map_flip_up_down(field_map):
    field_map = tf.image.flip_up_down(field_map)
    field_map_row, field_map_col = tf.unstack(field_map, axis=-1)
    field_map = tf.stack([-field_map_row, field_map_col], axis=-1)
    return field_map


def drop_components(polygon_map, keep_poly_prob, seed=None):
    """
    Randomly removes some connected components from polygon_map (which amounts to removing some polygons).

    :param polygon_map: The filtered polygon map raster
    :param keep_poly_prob: Probability of a polygon to be kept
    :param seed:
    :return:
    """
    if keep_poly_prob == 1:
        # Keep all
        return polygon_map
    elif keep_poly_prob == 0:
        # Remove all
        zeroed_polygon_map_zeros = tf.zeros_like(polygon_map)
        return zeroed_polygon_map_zeros

    try:
        with tf.name_scope('drop_components'):
            # Compute connected components on the first channel of polygon_map (the polygon fill channel):
            connected_components = tf.contrib.image.connected_components(polygon_map[:, :, 0])
            # Get maximum component label:
            connected_component_max = tf.reduce_max(connected_components)

            # Randomize component labels (but keep the background label "0" the same):
            connected_components_shape = tf.shape(connected_components)
            connected_components = tf.reshape(connected_components, [-1])

            ## --- Keep a polygon with probability keep_poly_prob --- ##
            random_values = tf.random_uniform((connected_component_max,), dtype=tf.float32,
                                              seed=seed)  # Don't draw a random number for the background label 0.
            random_values = tf.pad(random_values, [[1, 0]], "CONSTANT",
                                   constant_values=1)  # Add 1 at the beginning of the array so that the background has a zero probability to be kept
            connected_component_random_values = tf.gather(random_values, connected_components)
            connected_component_random_values = tf.reshape(connected_component_random_values,
                                                           connected_components_shape)

            # Threshold randomized components:
            mask = tf.expand_dims(
                tf.cast(
                    tf.less(connected_component_random_values, keep_poly_prob),
                    dtype=tf.float32
                ),
                axis=-1)

            # Filter polygon_map with mask:
            mask = tf_utils.dilate(mask, filter_size=3)  # Dilate to take polygon outlines inside the mask
            masked_polygon_map = mask * polygon_map
            return masked_polygon_map
    except AttributeError:
        print(
            "WARNING: Tensorflow {} does not have connected_components() implemented. Keeping all components regardless of keep_poly_prob.".format(
                tf.__version__))
        return polygon_map


def read_and_decode(tfrecord_filepaths, patch_inner_res, patch_outer_res, batch_size,
                    dynamic_range, disp_map_dynamic_range_fac=0.5, keep_poly_prob=None, data_aug=False, train=True,
                    seed=None):
    """
    Reads examples from the tfrecord.
    If train = True, polygon data will not be served as it cannot be shuffled easily (varying-sized tensors).
    Set to False for validation and test only (where shuffling does not matter)

    :param tfrecord_filepaths:
    :param patch_inner_res:
    :param patch_outer_res:
    :param batch_size:
    :param dynamic_range:
    :param disp_map_dynamic_range_fac:
    :param keep_poly_prob: If not None, the fraction of disp_polygon that are kept
    :param data_aug:
    :param train:
    :return:
    """
    assert 0 < len(tfrecord_filepaths), "tfrecord_filepaths should contain at least one element"

    with tf.name_scope('read_and_decode'):

        filename_queue = tf.train.string_input_producer(tfrecord_filepaths, shuffle=True, seed=seed, capacity=STRING_QUEUE_CAPACITY + 3 * batch_size)

        # reader_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        # reader = tf.TFRecordReader(options=reader_options)
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        if train:
            features = tf.parse_single_example(
                serialized_example,
                # Defaults are not specified since both keys are required.
                features={
                    'tile_res': tf.FixedLenFeature([], tf.int64),
                    'disp_map_count': tf.FixedLenFeature([], tf.int64),
                    'image': tf.FixedLenFeature([], tf.string),
                    'gt_polygon_map': tf.FixedLenFeature([], tf.string),
                    'disp_field_maps': tf.FixedLenFeature([], tf.string),
                    'disp_polygon_maps': tf.FixedLenFeature([], tf.string)
                })
            disp_map_count = tf.cast(features['disp_map_count'], tf.int64)
            gt_polygons = None
            disp_polygons_array = None
        else:
            features = tf.parse_single_example(
                serialized_example,
                # Defaults are not specified since both keys are required.
                features={
                    'tile_res': tf.FixedLenFeature([], tf.int64),
                    'disp_map_count': tf.FixedLenFeature([], tf.int64),
                    'image': tf.FixedLenFeature([], tf.string),
                    'gt_polygon_count': tf.FixedLenFeature([], tf.int64),
                    'gt_polygon_length': tf.FixedLenFeature([], tf.int64),
                    'gt_polygons': tf.FixedLenFeature([], tf.string),
                    'disp_polygons': tf.FixedLenFeature([], tf.string),
                    'gt_polygon_map': tf.FixedLenFeature([], tf.string),
                    'disp_field_maps': tf.FixedLenFeature([], tf.string),
                    'disp_polygon_maps': tf.FixedLenFeature([], tf.string)
                })
            disp_map_count = tf.cast(features['disp_map_count'], tf.int64)
            gt_polygon_count = tf.cast(features['gt_polygon_count'], tf.int64)
            gt_polygon_length = tf.cast(features['gt_polygon_length'], tf.int64)
            gt_polygons_flat = tf.decode_raw(features['gt_polygons'], tf.float16)
            disp_polygons_flat = tf.decode_raw(features['disp_polygons'], tf.float16)

            gt_polygons_shape = tf.stack([gt_polygon_count, gt_polygon_length, 2])
            gt_polygons = tf.reshape(gt_polygons_flat, gt_polygons_shape)
            disp_polygons_shape = tf.stack([disp_map_count, gt_polygon_count, gt_polygon_length, 2])
            disp_polygons_array = tf.reshape(disp_polygons_flat, disp_polygons_shape)

        tile_res = tf.cast(features['tile_res'], tf.int64)
        image_flat = tf.image.decode_jpeg(features['image'])  # TODO: use dct_method="INTEGER_ACCURATE"?
        gt_polygon_map_flat = tf.decode_raw(features['gt_polygon_map'], tf.uint8)
        disp_field_maps_flat = tf.decode_raw(features['disp_field_maps'], tf.int16)
        disp_polygon_maps_flat = tf.decode_raw(features['disp_polygon_maps'], tf.uint8)

        # return image_flat, None, None, gt_polygon_map_flat, disp_field_maps_flat, disp_polygon_maps_flat

        # Reshape tensors
        image_shape = tf.stack([tile_res, tile_res, 3])
        gt_polygon_map_shape = tf.stack([tile_res, tile_res, 3])
        disp_field_maps_shape = tf.stack([disp_map_count, tile_res, tile_res, 2])
        disp_polygon_maps_shape = tf.stack([disp_map_count, tile_res, tile_res, 3])
        image = tf.reshape(image_flat, image_shape)
        gt_polygon_map = tf.reshape(gt_polygon_map_flat, gt_polygon_map_shape)
        disp_field_maps = tf.reshape(disp_field_maps_flat, disp_field_maps_shape)
        disp_polygon_maps = tf.reshape(disp_polygon_maps_flat, disp_polygon_maps_shape)

        # return image, None, None, gt_polygon_map, disp_field_maps, disp_polygon_maps

        # Choose disp map:
        disp_map_index = tf.random_uniform([], maxval=disp_map_count, dtype=tf.int64, seed=seed)
        disp_polygons = None
        if not train:
            disp_polygons = disp_polygons_array[disp_map_index, :, :, :]
        disp_field_map = disp_field_maps[disp_map_index, :, :, :]
        disp_polygon_map = disp_polygon_maps[disp_map_index, :, :, :]

        # return image, None, None, gt_polygon_map, tf.expand_dims(disp_field_map, axis=0),  tf.expand_dims(disp_polygon_map, axis=0)

        # Normalize data
        image = image / 255
        gt_polygon_map = gt_polygon_map / 255
        disp_polygon_map = disp_polygon_map / 255
        disp_field_map = disp_map_dynamic_range_fac * tf.cast(disp_field_map,
                                                              dtype=tf.float32) / 32767  # Within [-disp_map_dynamic_range_fac, disp_map_dynamic_range_fac]

        if keep_poly_prob is not None:
            # Remove some polygons from disp_polygon_map
            disp_polygon_map = drop_components(disp_polygon_map, keep_poly_prob, seed=seed)

        # return tf.expand_dims(image, axis=0), gt_polygons, disp_polygons, tf.expand_dims(gt_polygon_map, axis=0), tf.expand_dims(disp_field_map, axis=0),  tf.expand_dims(disp_polygon_map, axis=0)

        # Perturb image brightness, contrast, saturation, etc.
        if data_aug:
            image = tf.image.random_brightness(image, 0.25)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)

        # Rotate
        if train and data_aug:  # data_aug rototation only applies to train (val includes polygons that should be also rotated if you want to augment val as well)
            # Pad to avoid losing parts of the image after rotation
            rot_patch_outer_res = int(math.ceil(patch_outer_res * math.sqrt(2)))
            rot_patch_inner_res = int(math.ceil(patch_inner_res * math.sqrt(2)))
            image, gt_polygon_map, disp_polygon_map = crop_or_pad_many([image, gt_polygon_map, disp_polygon_map],
                                                                       rot_patch_outer_res)
            disp_field_map = tf.image.resize_image_with_crop_or_pad(
                image=disp_field_map,
                target_height=rot_patch_inner_res,
                target_width=rot_patch_inner_res)

            #  Apply the rotations on the spatial support
            angle = tf.random_uniform([], maxval=2 * math.pi, dtype=tf.float32, seed=seed)
            image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')
            gt_polygon_map = rotate_poly_map(gt_polygon_map, angle)
            disp_polygon_map = rotate_poly_map(disp_polygon_map, angle)
            disp_field_map = tf.contrib.image.rotate(disp_field_map, angle, interpolation='BILINEAR')

            # Rotate only the vectors for every pixel of disp_field_map
            disp_field_map = rotate_field_vectors(disp_field_map, angle)

        # Crop to final patch_res
        # patch_outer_res = 312
        image, gt_polygon_map, disp_polygon_map = crop_or_pad_many([image, gt_polygon_map, disp_polygon_map],
                                                                   patch_outer_res)
        disp_field_map = tf.image.resize_image_with_crop_or_pad(
            image=disp_field_map,
            target_height=patch_inner_res,
            target_width=patch_inner_res)

        # Shift dynamic range of image to be in [-1, 1]
        image = image * (dynamic_range[1] - dynamic_range[0]) + dynamic_range[0]
        image = tf.clip_by_value(image, dynamic_range[0], dynamic_range[1])

        # return image, gt_polygons, disp_polygons, gt_polygon_map, disp_field_map, disp_polygon_map

        # # Dilate polygon maps
        # gt_polygon_map = tf_utils.dilate(gt_polygon_map, filter_size=2)
        # disp_polygon_map = tf_utils.dilate(disp_polygon_map, filter_size=2)

        if data_aug:
            # Apply random flips
            flip = tf.random_uniform([], dtype=tf.float16, seed=seed)
            flip_outputs = tf.cond(0.5 <= flip,
                                   lambda: (tf.image.flip_up_down(image),
                                            tf.image.flip_up_down(gt_polygon_map),
                                            field_map_flip_up_down(disp_field_map),
                                            tf.image.flip_up_down(disp_polygon_map)),
                                   lambda: (image, gt_polygon_map, disp_field_map, disp_polygon_map))
            image, gt_polygon_map, disp_field_map, disp_polygon_map = flip_outputs

        # Add batch dimension (to be able to use enqueue_many=True)
        image = tf.expand_dims(image, 0)
        if not train:
            gt_polygons = tf.expand_dims(gt_polygons, 0)
            disp_polygons = tf.expand_dims(disp_polygons, 0)
        gt_polygon_map = tf.expand_dims(gt_polygon_map, 0)
        disp_field_map = tf.expand_dims(disp_field_map, 0)
        disp_polygon_map = tf.expand_dims(disp_polygon_map, 0)

        # Remove patches with too little data for training (that have no corners in inner patch)
        include_patch = corners_in_inner_patch(gt_polygon_map, patch_inner_res)
        empty = tf.constant([], tf.int32)
        if train:
            image, \
            gt_polygon_map, \
            disp_field_map, \
            disp_polygon_map = tf.cond(include_patch,
                                       lambda: [image, gt_polygon_map,
                                                disp_field_map, disp_polygon_map],
                                       lambda: [tf.gather(image, empty),
                                                tf.gather(gt_polygon_map, empty),
                                                tf.gather(disp_field_map, empty),
                                                tf.gather(disp_polygon_map, empty)])
        else:
            image, \
            gt_polygons, \
            disp_polygons, \
            gt_polygon_map, \
            disp_field_map, \
            disp_polygon_map = tf.cond(include_patch,
                                       lambda: [image, gt_polygons, disp_polygons, gt_polygon_map, disp_field_map,
                                                disp_polygon_map],
                                       lambda: [
                                           tf.gather(image, empty),
                                           tf.gather(gt_polygons, empty),
                                           tf.gather(disp_polygons, empty),
                                           tf.gather(gt_polygon_map, empty),
                                           tf.gather(disp_field_map, empty),
                                           tf.gather(disp_polygon_map, empty)])

        if train:
            image_batch, gt_polygon_map_batch, disp_field_map_batch, disp_polygon_map_batch = tf.train.shuffle_batch(
                [image, gt_polygon_map, disp_field_map, disp_polygon_map],
                batch_size=batch_size,
                capacity=MIN_QUEUE_EXAMPLES + 3 * batch_size,
                min_after_dequeue=MIN_QUEUE_EXAMPLES,
                num_threads=8,
                seed=seed,
                enqueue_many=True,
                allow_smaller_final_batch=False)
            return image_batch, None, None, gt_polygon_map_batch, disp_field_map_batch, disp_polygon_map_batch
        else:
            image_batch, gt_polygons_batch, disp_polygons_batch, gt_polygon_map_batch, disp_field_map_batch, disp_polygon_map_batch = tf.train.batch(
                [image, gt_polygons, disp_polygons, gt_polygon_map, disp_field_map, disp_polygon_map],
                batch_size=batch_size,
                num_threads=8,
                dynamic_pad=True,
                enqueue_many=True,
                allow_smaller_final_batch=False)
            return image_batch, gt_polygons_batch, disp_polygons_batch, gt_polygon_map_batch, disp_field_map_batch, disp_polygon_map_batch


def main():
    # --- Params --- #
    seed = 0

    data_dir = python_utils.choose_first_existing_path([
        "/local/shared/epitome-polygon-deep-learning/data",  # Try local node first
        "/home/nigirard/epitome-polygon-deep-learning/data",

        "/workspace/data",  # Try inside Docker image
    ])

    tfrecords_dir_list = [
        # os.path.join(data_dir, "AerialImageDataset/tfrecords.mapalign.multires"),
        os.path.join(data_dir, "bradbury_buildings_roads_height_dataset/tfrecords.mapalign.multires"),
        # os.path.join(data_dir, "mapping_challenge_dataset/tfrecords.mapalign.multires"),
    ]
    print("tfrecords_dir_list:")
    print(tfrecords_dir_list)
    # downsampling_factors = [1, 2, 4, 8]
    # resolution_file_repeats = [1, 4, 16, 64]
    tfrecord_filename_format = "{}.ds_fac_{:02d}.{{:06d}}.tfrecord"
    downsampling_factors = [1]
    resolution_file_repeats = [1]
    dataset_filename_list = create_dataset_filename_list(tfrecords_dir_list, tfrecord_filename_format,
                                                         downsampling_factors,
                                                         dataset="train",
                                                         resolution_file_repeats=resolution_file_repeats)
    print("dataset_filename_list:")
    print(dataset_filename_list)
    patch_outer_res = 220
    patch_inner_res = 100
    padding = (patch_outer_res - patch_inner_res) // 2
    disp_max_abs_value = 4

    batch_size = 32
    dynamic_range = [-1, 1]
    keep_poly_prob = 0.1  # Default: 0.1
    data_aug = True
    train = True
    # --- --- #

    # Even when reading in multiple threads, share the filename
    # queue.
    image, gt_polygons, disp_polygons, gt_polygon_map, disp_field_map, disp_polygon_map = read_and_decode(
        dataset_filename_list,
        patch_inner_res,
        patch_outer_res,
        batch_size,
        dynamic_range,
        keep_poly_prob=keep_poly_prob,
        data_aug=data_aug,
        train=train,
        seed=seed)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Let's read off 3 batches just for example
        for i in range(30000):
            print("---- {} ---".format(i))
            if train:
                image_batch, gt_polygon_map_batch, disp_field_map_batch, disp_polygon_map_batch = sess.run(
                    [image, gt_polygon_map, disp_field_map, disp_polygon_map])
            else:
                image_batch, gt_polygons_batch, disp_polygons_batch, gt_polygon_map_batch, disp_field_map_batch, disp_polygon_map_batch = sess.run(
                    [image, gt_polygons, disp_polygons, gt_polygon_map, disp_field_map, disp_polygon_map])
                print(gt_polygons_batch[0, 0, 0, :])
                print(disp_polygons_batch[0, 0, 0, :])
                print(gt_polygons_batch.shape)
                print(disp_polygons_batch.shape)

            print(image_batch.shape)
            print(gt_polygon_map_batch.shape)
            print(disp_field_map_batch.shape)
            print(disp_polygon_map_batch.shape)

            # np.set_printoptions(threshold=np.nan)
            # print(image_batch)
            # print(gt_polygon_map_batch)
            # print(disp_field_map_batch)
            # print(disp_polygon_map_batch)

            print("image_batch:")
            print(image_batch.min())
            print(image_batch.max())

            print("gt_polygon_map_batch:")
            print(gt_polygon_map_batch.min())
            print(gt_polygon_map_batch.max())

            try:
                print(disp_field_map_batch[:, :, :, 0].min())
                print(disp_field_map_batch[:, :, :, 0].max())
            except IndexError:
                print("Skip min and max of disp_field_map_batch because of wrong rank")

            # visualization.plot_field_map("disp_field_map", disp_field_map_batch[0])

            print("disp_polygon_map_batch:")
            print(disp_polygon_map_batch.min())
            print(disp_polygon_map_batch.max())

            dynamic_range = [-1, 1]
            image_batch = (image_batch - dynamic_range[0]) / (
                    dynamic_range[1] - dynamic_range[0])

            disp_field_map_batch = disp_field_map_batch * 2  # Within [-1, 1]
            disp_field_map_batch = disp_field_map_batch * disp_max_abs_value  # Within [-disp_max_abs_value, disp_max_abs_value]

            # gt_polygon_map_batch *= 0  # TODO: Remove

            # for batch_index in range(batch_size):
            #     if train:
            #         visualization.init_figures(["example"])
            #         # visualization.plot_example("example",
            #         #                            image_batch[batch_index],
            #         #                            gt_polygon_map_batch[batch_index],
            #         #                            disp_field_map_batch[batch_index],
            #         #                            disp_polygon_map_batch[batch_index])
            #         visualization.plot_example("example",
            #                                    image_batch[batch_index],
            #                                    disp_polygon_map_batch[batch_index])
            #     else:
            #         visualization.init_figures(["example", "example polygons"])
            #         visualization.plot_example("example",
            #                                    image_batch[batch_index],
            #                                    gt_polygon_map_batch[batch_index],
            #                                    disp_field_map_batch[batch_index],
            #                                    disp_polygon_map_batch[batch_index])
            #         visualization.plot_example_polygons("example polygons",
            #                                             image_batch[batch_index],
            #                                             gt_polygons_batch[batch_index],
            #                                             disp_polygons_batch[batch_index])
            # input("Press <Enter> to continue...")

            skimage.io.imsave("misaligned_polygon_raster.png", disp_polygon_map_batch[0])
            skimage.io.imsave("image.png", image_batch[0])
            disp_field_map_image = visualization.flow_to_image(disp_field_map_batch[0])
            skimage.io.imsave("displacement_field_map.png", disp_field_map_image)
            segmentation = gt_polygon_map_batch[0][padding:-padding, padding:-padding, :]
            skimage.io.imsave("segmentation.png", segmentation)

            input("Press <Enter> to continue...")

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()
