from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tensorflow as tf

sys.path.append("../../utils")
import tf_utils
import print_utils

# --- Params --- #
DEBUG = False
SUMMARY = False

# --- --- #


def print_debug(obj):
    if DEBUG:
        print_utils.print_debug(obj.__str__())


def conv_conv_pool(input_, n_filters, name="", pool=True, activation=tf.nn.elu, weight_decay=None,
                   dropout_keep_prob=None):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activation function
        weight_decay: Weight decay rate

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer_{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf_utils.complete_conv2d(net, F, (3, 3), padding="VALID", activation=activation,
                                           bias_init_value=-0.01,
                                           weight_decay=weight_decay,
                                           summary=SUMMARY)
        if pool is False:
            return net, None
        else:
            pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))
            return net, pool


def upsample_crop_concat(to_upsample, input_to_crop, size=(2, 2), weight_decay=None, name=None):
    """Upsample `to_upsample`, crop to match resolution of `input_to_crop` and concat the two.

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_to_crop (4-D Tensor): (N, 2*H + padding, 2*W + padding, C2)
        size (tuple): (height_multiplier, width_multiplier) (default: (2, 2))
        name (str): name of the concat operation (default: None)

    Returns:
        output (4-D Tensor): (N, size[0]*H, size[1]*W, 2*C2)
    """
    H, W, _ = to_upsample.get_shape().as_list()[1:]
    _, _, target_C = input_to_crop.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    upsample = tf.image.resize_bilinear(to_upsample, (target_H, target_W), name="upsample_{}".format(name))
    upsample = tf_utils.complete_conv2d(upsample, target_C, (3, 3), padding="SAME", bias_init_value=-0.01,
                                        weight_decay=weight_decay,
                                        summary=SUMMARY)

    # TODO: initialize upsample with bilinear weights
    # upsample = tf.layers.conv2d_transpose(to_upsample, target_C, kernel_size=2, strides=1, padding="valid", name="deconv{}".format(name))

    crop = tf.image.resize_image_with_crop_or_pad(input_to_crop, target_H, target_W)

    return tf.concat([upsample, crop], axis=-1, name="concat_{}".format(name))


def upsample_crop(input, resolution, factor=(2, 2), name=None):
    """
    Scales the input displacement field map by factor.
    First upsamples by factor,
    then crops to resolution.

    :param input: Tensor to upsample and then crop
    :param resolution: Output resolution (row_count, col_count)
    :param factor: Factor of scaling (row_factor, col_factor)
    :param name: Name of op
    :return:  Upsampled + cropped tensor
    """
    # Upsample
    up_size = (input.shape[1] * factor[0], input.shape[2] * factor[1])
    input_upsampled = tf.image.resize_bilinear(input, up_size, name="upsample_{}".format(name))

    # Crop
    input_cropped = tf.image.resize_image_with_crop_or_pad(input_upsampled, resolution[0], resolution[1])

    return input_cropped


def build_input_branch(input, feature_base_count, pool_count, name="", weight_decay=None):
    res_levels = pool_count + 1
    with tf.variable_scope(name):
        print_debug(name)
        levels = []
        for res_level_index in range(res_levels):
            print_debug("\tlevel {}:".format(res_level_index))
            feature_count = feature_base_count * math.pow(2, res_level_index)
            if res_level_index == 0:
                # Add first level
                conv, pool = conv_conv_pool(input, [feature_count, feature_count],
                                            name="conv_pool_{}".format(res_level_index), weight_decay=weight_decay)
            elif res_level_index < res_levels - 1:
                # Add all other levels (except the last one)
                level_input = levels[-1][1]  # Select the previous pool
                conv, pool = conv_conv_pool(level_input, [feature_count, feature_count],
                                            name="conv_pool_{}".format(res_level_index), weight_decay=weight_decay)
            elif res_level_index == res_levels - 1:
                # Add last level
                level_input = levels[-1][1]  # Select the previous pool
                conv, pool = conv_conv_pool(level_input, [feature_count, feature_count],
                                            name="conv_pool_{}".format(res_level_index), pool=False,
                                            weight_decay=weight_decay)
            else:
                print("WARNING: Should be impossible to get here!")
                conv = pool = None
            print_debug("\t\tconv: {}".format(conv))
            print_debug("\t\tpool: {}".format(pool))
            levels.append((conv, pool))

    return levels


def build_common_part(branch_levels_list, feature_base_count,
                      name="", weight_decay=None):
    """
    Merges the two branches level by level in a U-Net fashion

    :param branch_levels_list:
    :param feature_base_count:
    :param name:
    :param weight_decay:
    :return:
    """
    res_levels = len(branch_levels_list[0])
    with tf.variable_scope(name):
        print_debug(name)
        # Concat branches at each level + add conv layers
        levels = []
        for level_index in range(res_levels):
            print_debug("\tlevel {}:".format(level_index))
            concat_a_b = tf.concat([branch_levels[level_index][0] for branch_levels in branch_levels_list], axis=-1,
                                   name="concat_a_b_{}".format(level_index))
            print_debug("\t\tconcat_a_b: {}".format(concat_a_b))
            feature_count = feature_base_count * math.pow(2, level_index)
            concat_a_b_conv, _ = conv_conv_pool(concat_a_b, [feature_count, feature_count],
                                                name="concat_a_b_conv{}".format(level_index), pool=False,
                                                weight_decay=weight_decay)
            print_debug("\t\tconcat_a_b_conv: {}".format(concat_a_b_conv))
            levels.append(concat_a_b_conv)

    return levels


def build_output_branch(input_levels, feature_base_count, name="", weight_decay=None):
    with tf.variable_scope(name):
        print_debug(name)
        res_levels = len(input_levels)
        prev_level_output = None
        output_levels = []
        for level_index in range(res_levels - 1, -1, -1):
            print_debug("\tlevel {}:".format(level_index))
            if prev_level_output is None:
                # This means we are at the bottom of the "U" of the U-Net
                prev_level_output = input_levels[level_index]
            else:
                # Now concat prev_level_output with current input level
                up = upsample_crop_concat(prev_level_output, input_levels[level_index], weight_decay=weight_decay,
                                          name="up_{}".format(level_index))
                print_debug("\t\tup: {}".format(up))
                feature_count = feature_base_count * math.pow(2, level_index)
                final_conv, _ = conv_conv_pool(up, [feature_count, feature_count],
                                               name="final_conv_{}".format(level_index), pool=False,
                                               weight_decay=weight_decay)
                print_debug("\t\tfinal_conv: {}".format(final_conv))
                output_levels.insert(0, final_conv)  # Insert at the beginning because we are iterating in reverse order
                prev_level_output = final_conv

    return output_levels


def build_pred_branch(input_levels, output_channels, name=""):
    with tf.variable_scope(name):
        print_debug(name)
        output_levels = []
        output_level_0 = None
        level_0_resolution = None
        for level_index, input in enumerate(input_levels):
            print_debug("\tlevel {}:".format(level_index))
            # Add prediction layer then upsample prediction to match level 0's prediction resolution
            pred = tf.layers.conv2d(input, output_channels, (1, 1), name="pred_conv1x1_level_{}".format(level_index),
                                    padding='VALID')
            tf.summary.histogram("pred_{}".format(level_index), pred)
            print_debug("\t\tpred: {}".format(pred))

            if level_index == 0:
                output_level_0 = pred
                level_0_resolution = pred.get_shape().as_list()[1:3]
            else:
                # Upsample pred and crop to the resolution of the first level
                single_factor = math.pow(2, level_index)
                pred = upsample_crop(pred, level_0_resolution, (single_factor, single_factor),
                                     name="convert_disp_pred_{}".format(level_index))
            output_levels.append(pred)

        stacked_output_levels = tf.stack(output_levels, axis=1, name="stacked_preds")
        print_debug("\tstacked_output_levels: {}".format(stacked_output_levels))

    return output_level_0, stacked_output_levels


def build_multibranch_unet(input_branch_params_list, pool_count, common_feature_base_count, output_branch_params_list,
                           weight_decay=None):
    """
    Builds a multi-branch U-Net network. Has len(input_tensors) input branches and len(output_channel_counts) output branches.

    :param input_branch_params_list: [
        {
            "tensor": input Tensorflow tensor,
            "name": name used in internal scope of the graph,
            "feature_base_count": number of features of the first conv for the each input branch. Multiplied by 2 after each conv_conv block,
        },
        ...
        ]
    :param pool_count: number of 2x2 pooling operations. Results in (pool_count+1) resolution levels
    :param common_feature_base_count: number of features of the first conv for the common part of the network. Multiplied by 2 after each conv_conv block
    :param output_branch_params_list: [
        {
            "feature_base_count": like input feature_base_counts but for outputs,
            "channel_count": integer for the final channel count,
            "activation": final activation function,
            "name": name used in internal scope of the graph,
        },
        ...
    ]
    :param weight_decay: (Default: None). Weight decay rate
    :return: output heads, keep_prob (not used)
    """
    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)

    # Build the separate simple convolution networks for each input:
    input_branch_levels_list = []
    for params in input_branch_params_list:
        tf.summary.histogram("input_{}".format(params["name"]), params["tensor"])
        branch_levels = build_input_branch(params["tensor"], params["feature_base_count"], pool_count,
                                           name="branch_{}".format(params["name"]),
                                           weight_decay=weight_decay)
        input_branch_levels_list.append(branch_levels)

    # Build the common part of the network, concatenating inout branches at all levels
    common_part_levels = build_common_part(input_branch_levels_list,
                                           common_feature_base_count,
                                           name="common_part",
                                           weight_decay=weight_decay)

    # Build the splitting part of the network, each level (except the last one) finishing with output branches.
    # Each branch is like the upsampling part of a U-Net
    outputs = []
    for params in output_branch_params_list:
        branch_levels = build_output_branch(common_part_levels,
                                            params["feature_base_count"],
                                            name="branch_{}".format(params["name"]),
                                            weight_decay=weight_decay)
        # Add the last layers for prediction, then upsample each levels' prediction to level 0's resolution
        # TODO: keep this for legacy reasons:
        if params["activation"] == tf.identity:
            name = "branch_{}_pred_logit".format(params["name"])
        else:
            name = "branch_{}_pred".format(params["name"])
        level_0_pred, stacked_pred_logits = build_pred_branch(branch_levels,
                                                              output_channels=params["channel_count"],
                                                              name=name)
        # Apply activation function to logits
        stacked_preds = params["activation"](stacked_pred_logits)
        output = (stacked_pred_logits, stacked_preds, stacked_preds[:, 0, ...])
        outputs.append(output)

    return outputs, keep_prob


def build_double_unet(input_image, input_poly_map,
                      image_feature_base_count, poly_map_feature_base_count, common_feature_base_count, pool_count,
                      disp_output_channels, add_seg_output=True, seg_output_channels=1,
                      weight_decay=None):
    """
    Build the double U-Net network. Has two input branches and two output branches (actually, each resolution level
    except the last one have two output branches).

    :param input_image: image
    :param input_poly_map: polygon_map
    :param image_feature_base_count: number of features of the first conv for the image branch. Multiplied by 2 after each conv_conv block
    :param poly_map_feature_base_count: number of features of the first conv for the polygon map branch. Multiplied by 2 after each conv_conv block
    :param common_feature_base_count: number of features of the first conv for the common part of the network. Multiplied by 2 after each conv_conv block
    :param pool_count: number of 2x2 pooling operations. Results in (pool_count+1) resolution levels
    :param disp_output_channels: Output dimension for the displacement prediction
    :param add_seg_output: (Default: True). If True, a segmentation output branch is built. If False, no additional branch is built and the seg_output_channels argument is ignored.
    :param seg_output_channels: Output dimension for the segmentation prediction
    :param weight_decay: (Default: None). Weight decay rate
    :return: Network
    """
    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)

    tf.summary.histogram("input_image", input_image)
    tf.summary.histogram("input_poly_map", input_poly_map)

    # Build the two separate simple convolution networks for each input
    branch_image_levels = build_input_branch(input_image, image_feature_base_count, pool_count,
                                             name="branch_image",
                                             weight_decay=weight_decay)

    branch_poly_map_levels = build_input_branch(input_poly_map, poly_map_feature_base_count, pool_count,
                                                name="branch_poly_map",
                                                weight_decay=weight_decay)

    # Build the common part of the network, concatenating the image and polygon map branches at all levels
    common_part_levels = build_common_part([branch_image_levels, branch_poly_map_levels],
                                           common_feature_base_count,
                                           name="common_part",
                                           weight_decay=weight_decay)

    # Build the splitting part of the network, each level (except the last one) finishing with two branches: one for
    # displacement map prediction and the other for segmentation prediction. Each branch is like the upsampling part of
    # A U-Net
    disp_levels = build_output_branch(common_part_levels,
                                      common_feature_base_count,
                                      name="branch_disp",
                                      weight_decay=weight_decay)
    if add_seg_output:
        seg_levels = build_output_branch(common_part_levels,
                                         common_feature_base_count,
                                         name="branch_seg",
                                         weight_decay=weight_decay)
    else:
        seg_levels = None

    # Add the last layers for prediction, then upsample each levels' prediction to level 0's resolution
    level_0_disp_pred_logit, stacked_disp_pred_logits = build_pred_branch(disp_levels,
                                                                          output_channels=disp_output_channels,
                                                                          name="branch_disp_pred")
    level_0_disp_pred = tf.nn.tanh(level_0_disp_pred_logit)
    stacked_disp_preds = tf.nn.tanh(stacked_disp_pred_logits)

    if add_seg_output:
        level_0_seg_pred_logit, stacked_seg_pred_logits = build_pred_branch(seg_levels,
                                                                            output_channels=seg_output_channels,
                                                                            name="branch_seg_pred_logit")
        # Apply sigmoid to level_0_seg_pred_logit
        level_0_seg_pred = tf.nn.sigmoid(level_0_seg_pred_logit)
    else:
        stacked_seg_pred_logits = None
        level_0_seg_pred = None

    return level_0_disp_pred, stacked_disp_preds, level_0_seg_pred, stacked_seg_pred_logits, keep_prob


def get_output_res(input_res, pool_count):
    """
    This function has to be re-written if the model architecture changes

    :param input_res:
    :param pool_count:
    :return:
    """
    current_res = input_res
    warning_non_zero_remainder = False
    # branch_image
    for i in range(pool_count):
        current_res -= 4  # 2 conv3x3
        current_res, r = divmod(current_res, 2)  # pool
        warning_non_zero_remainder = warning_non_zero_remainder or bool(r)
    current_res -= 4  # 2 conv3x3 of the last layer
    # common_part
    current_res -= 4  # 2 conv3x3
    # branch_disp
    for i in range(pool_count):
        current_res *= 2  # upsample
        current_res -= 4  # 2 conv3x3
    if warning_non_zero_remainder:
        print(
            "WARNING: a pooling operation will result in a non integer res, the network will automatically add padding there. The output of this function is not garanteed to be exact.")
    return int(current_res)


def get_input_res(output_res, pool_count):
    """
    This function has to be re-written if the model architecture changes

    :param output_res:
    :param pool_count:
    :return:
    """
    current_res = output_res
    warning_non_zero_remainder = False
    # branch_disp
    for i in range(pool_count):
        current_res += 4  # 2 conv3x3
        current_res, r = divmod(current_res, 2)  # upsample
        warning_non_zero_remainder = warning_non_zero_remainder or bool(r)
    # common_part
    current_res += 4  # 2 conv3x3
    # branch_image
    current_res += 4  # 2 conv3x3 of the last layer
    for i in range(pool_count):
        current_res *= 2  # pool
        current_res += 4  # 2 conv3x3
    if warning_non_zero_remainder:
        print(
            "WARNING: a pooling operation will result in a non integer res, the network will automatically add padding there. The output of this function is not garanteed to be exact.")
    return int(current_res)
