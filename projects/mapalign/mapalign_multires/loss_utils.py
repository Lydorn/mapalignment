from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

sys.path.append("../../utils")
import tf_utils


def displacement_error(gt, preds, level_loss_coefs, polygon_map, disp_loss_params):
    """

    :param gt: Groundtruth displacement map bounded between -1 and 1. Shape [batch, height, width, channels (3)]
    :param preds: Predicted displacement maps bounded between -1 and 1. Shape [batch, levels, height, width, channels (2)]
    :param level_loss_coefs: Loss coefficients to apply to each level
    :param polygon_map: Used as mask for fill, outline and vertex. Shape [batch, height, width, channels (3)]
    :return: error
    """
    height, width, _ = gt.get_shape().as_list()[1:]

    with tf.name_scope("euclidean_error"):
        # Compute weight mask
        cropped_polygon_map = tf.image.resize_image_with_crop_or_pad(polygon_map, height, width)
        # TODO: normalize correction_weights
        correction_weights = 1 / (
                    tf.reduce_sum(tf.reduce_sum(cropped_polygon_map, axis=1), axis=1) + tf.keras.backend.epsilon())
        weigths = tf.constant(
            [disp_loss_params["fill_coef"], disp_loss_params["edge_coef"], disp_loss_params["vertex_coef"]],
            dtype=tf.float32)
        corrected_weights = weigths * correction_weights
        corrected_weights = tf.expand_dims(tf.expand_dims(corrected_weights, axis=1), axis=1)
        weighted_mask = tf.reduce_sum(cropped_polygon_map * corrected_weights, axis=-1)
        weighted_mask = tf.expand_dims(weighted_mask, axis=1)  # Add levels dimension

        # Compute errors
        gt = tf.expand_dims(gt, axis=1)  # Add levels dimension
        pixelwise_euclidean_error = tf.reduce_sum(tf.square(gt - preds), axis=-1)
        masked_pixelwise_euclidean_error = pixelwise_euclidean_error * weighted_mask
        # Sum errors
        summed_error = tf.reduce_sum(masked_pixelwise_euclidean_error, axis=0)  # Batch sum
        summed_error = tf.reduce_sum(summed_error, axis=-1)  # Col/Width sum
        summed_error = tf.reduce_sum(summed_error, axis=-1)  # Row/Height sum
        summed_error = summed_error * level_loss_coefs  # Apply Level loss coefficients
        summed_error = tf.reduce_sum(summed_error)
        # Sum weights
        summed_weighted_mask = tf.reduce_sum(weighted_mask)
        loss = summed_error / (summed_weighted_mask + tf.keras.backend.epsilon())

    return loss


def segmentation_error(seg_gt, seg_pred_logits, level_loss_coefs, seg_loss_params):
    """

    :param seg_gt:
    :param seg_pred_logits:
    :param level_loss_coefs:
    :return:
    """
    # print("--- segmentation_error ---")
    _, levels, height, width, _ = seg_pred_logits.get_shape().as_list()
    # Crop seg_gt to match resolution of seg_pred_logits
    seg_gt = tf.image.resize_image_with_crop_or_pad(seg_gt, height, width)
    # Add background class to gt segmentation
    if tf_utils.get_tf_version() == "1.4.0":
        seg_gt_bg = tf.reduce_prod(1 - seg_gt, axis=-1,
                                   keep_dims=True)  # Equals 0 if pixel is either fill, outline or vertex. Equals 1 otherwise
    else:
        seg_gt_bg = tf.reduce_prod(1 - seg_gt, axis=-1,
                                   keepdims=True)  # Equals 0 if pixel is either fill, outline or vertex. Equals 1 otherwise
    seg_gt = tf.concat([seg_gt_bg, seg_gt], axis=-1)

    # Compute weight mask
    # class_sums = tf.reduce_sum(tf.reduce_sum(seg_gt, axis=1), axis=1)
    # seg_class_balance_weights = 1 / (
    #         class_sums + tf.keras.backend.epsilon())
    seg_class_weights = tf.constant([[seg_loss_params["background_coef"], seg_loss_params["fill_coef"],
                                      seg_loss_params["edge_coef"], seg_loss_params["vertex_coef"]]],
                                    dtype=tf.float32)
    # balanced_class_weights = seg_class_balance_weights * seg_class_weights
    balanced_class_weights = seg_class_weights
    balanced_class_weights = tf.expand_dims(balanced_class_weights, axis=1)  # Add levels dimension
    balanced_class_weights = tf.tile(balanced_class_weights, multiples=[1, levels, 1])  # Repeat on levels dimension
    level_loss_coefs = tf.expand_dims(level_loss_coefs, axis=-1)  # Add channels dimension
    final_weights = balanced_class_weights * level_loss_coefs
    final_weights = tf.expand_dims(tf.expand_dims(final_weights, axis=2), axis=2)  # Add spatial dimensions

    # Adapt seg_gt shape to seg_pred_logits
    seg_gt = tf.expand_dims(seg_gt, axis=1)  # Add levels dimension
    seg_gt = tf.tile(seg_gt, multiples=[1, levels, 1, 1, 1])  # Add levels dimension

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=seg_gt, logits=seg_pred_logits)

    # Now apply the various weights
    weighted_loss = loss * final_weights

    final_loss = tf.reduce_mean(weighted_loss)

    return final_loss


def laplacian_penalty(preds, level_loss_coefs):
    in_channels = preds.shape[-1]

    with tf.name_scope("laplacian_penalty"):
        laplace_k = tf_utils.make_depthwise_kernel([[0.5, 1.0, 0.5],
                                                    [1.0, -6., 1.0],
                                                    [0.5, 1.0, 0.5]], in_channels)
        # Reshape preds to respect the input format of the depthwise_conv2d op
        shape = [preds.shape[0] * preds.shape[1]] + preds.get_shape().as_list()[2:]
        reshaped_preds = tf.reshape(preds, shape)
        laplacians = tf.nn.depthwise_conv2d(reshaped_preds, laplace_k, [1, 1, 1, 1], padding='SAME')
        penalty_map = tf.reduce_sum(tf.square(laplacians), axis=-1)
        # Reshape penalty_map to shape compatible with preds
        shape = preds.get_shape().as_list()[:-1]
        reshaped_penalty_map = tf.reshape(penalty_map, shape)

        # Compute mean penalty per level over spatial dimension as well as over batches
        level_penalties = tf.reduce_mean(reshaped_penalty_map, axis=0)  # Batch mean
        level_penalties = tf.reduce_mean(level_penalties, axis=-1)  # Col/Width mean
        level_penalties = tf.reduce_mean(level_penalties, axis=-1)  # Row/Height mean

        # Apply level_loss_coefs
        weighted_penalties = level_penalties * level_loss_coefs

        penalty = tf.reduce_mean(weighted_penalties)  # Levels mean

    return penalty


def main(_):
    batch_size = 1
    levels = 2
    patch_inner_res = 3
    patch_outer_res = 5

    disp_ = tf.placeholder(tf.float32, [batch_size, patch_inner_res, patch_inner_res, 2])
    disps = tf.placeholder(tf.float32, [batch_size, levels, patch_inner_res, patch_inner_res, 2])
    seg_ = tf.placeholder(tf.float32, [batch_size, patch_inner_res, patch_inner_res, 3])
    seg_logits = tf.placeholder(tf.float32, [batch_size, levels, patch_inner_res, patch_inner_res, 3])
    level_loss_coefs = tf.placeholder(tf.float32, [levels])
    mask = tf.placeholder(tf.float32, [batch_size, patch_outer_res, patch_outer_res, 3])

    disp_loss = displacement_error(disp_, disps, level_loss_coefs, mask)
    seg_loss = segmentation_error(seg_, seg_logits, level_loss_coefs)
    penalty = laplacian_penalty(disps, level_loss_coefs)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        disp_gt = np.zeros([batch_size, patch_inner_res, patch_inner_res, 2])
        disp_gt[0, 0, 0, 0] = 1
        disp_preds = np.zeros([batch_size, levels, patch_inner_res, patch_inner_res, 2])
        disp_preds[0, 0, 0, 0, 0] = 1
        disp_preds[0, 1, 0, 0, 0] = 1

        seg_gt = np.zeros([batch_size, patch_inner_res, patch_inner_res, 3])
        # seg_gt += 0.5
        seg_gt[0, 0, 0, 0] = 1.0
        seg_gt[0, 0, 1, 1] = 1.0
        seg_gt[0, 0, 2, 2] = 1.0

        seg_gt[0, 1, 0, 0] = 1.0
        seg_gt[0, 1, 1, 1] = 1.0
        seg_gt[0, 1, 2, 2] = 1.0
        seg_pred_logits = np.zeros([batch_size, levels, patch_inner_res, patch_inner_res, 3])
        seg_pred_logits += -100
        seg_pred_logits[0, 0, 0, 0, 0] = 100
        seg_pred_logits[0, 0, 0, 1, 1] = 100
        seg_pred_logits[0, 0, 0, 2, 2] = -100
        seg_pred_logits[0, 1, 0, 0, 0] = 100
        seg_pred_logits[0, 1, 0, 1, 1] = 100
        seg_pred_logits[0, 1, 0, 2, 2] = -100

        seg_pred_logits[0, 0, 1, 0, 0] = 100
        seg_pred_logits[0, 0, 1, 1, 1] = 100
        seg_pred_logits[0, 0, 1, 2, 2] = -100
        seg_pred_logits[0, 1, 1, 0, 0] = 100
        seg_pred_logits[0, 1, 1, 1, 1] = 100
        seg_pred_logits[0, 1, 1, 2, 2] = -100

        coefs = np.array([1, 0.5])
        poly_mask = np.zeros([batch_size, patch_outer_res, patch_outer_res, 3])
        poly_mask[0, 1, 1, 0] = 1

        computed_disp_loss, computed_seg_loss, computed_penalty = sess.run(
            [disp_loss, seg_loss, penalty], feed_dict={disp_: disp_gt, disps: disp_preds,
                                                       seg_: seg_gt, seg_logits: seg_pred_logits,
                                                       level_loss_coefs: coefs, mask: poly_mask})
        print("computed_disp_loss:")
        print(computed_disp_loss)
        print("computed_seg_loss:")
        print(computed_seg_loss)
        print("computed_penalty:")
        print(computed_penalty)


if __name__ == '__main__':
    tf.app.run(main=main)
