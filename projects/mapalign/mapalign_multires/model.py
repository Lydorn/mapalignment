import sys
import os
import time
import tensorflow as tf
import numpy as np

import model_utils
# import model_utils_concat_interm_outputs
import loss_utils

sys.path.append("../evaluate_funcs")  # Evaluation functions
import evaluate_utils

sys.path.append("../utils")  # Mapalign utils
import visualization

sys.path.append("../../utils")  # All project utils
import python_utils
import polygonization_utils
import tf_utils
import image_utils
import print_utils


class MapAlignModel:
    def __init__(self, model_name, input_res,

                 add_image_input, image_channel_count,
                 image_feature_base_count,

                 add_poly_map_input, poly_map_channel_count,
                 poly_map_feature_base_count,

                 common_feature_base_count, pool_count,

                 add_disp_output, disp_channel_count,

                 add_seg_output, seg_channel_count,

                 output_res,
                 batch_size,

                 loss_params,
                 level_loss_coefs_params,

                 learning_rate_params,
                 weight_decay,

                 image_dynamic_range, disp_map_dynamic_range_fac,
                 disp_max_abs_value):
        """
         Methods that may need a re-write if changing this class's code:
        - get_input_res
        - get_output_res

        :param model_name:
        :param input_res:
        :param add_image_input:
        :param image_channel_count:
        :param image_feature_base_count:
        :param add_poly_map_input:
        :param poly_map_channel_count:
        :param poly_map_feature_base_count:
        :param common_feature_base_count:
        :param pool_count:
        :param add_disp_output:
        :param disp_channel_count:
        :param add_seg_output:
        :param seg_channel_count:
        :param output_res:
        :param batch_size:
        :param loss_params:
        :param level_loss_coefs_params:
        :param learning_rate_params:
        :param weight_decay:
        :param image_dynamic_range:
        :param disp_map_dynamic_range_fac:
        :param disp_max_abs_value:
        """
        assert type(model_name) == str, "model_name should be a string, not a {}".format(type(model_name))
        assert type(input_res) == int, "input_res should be an int, not a {}".format(type(input_res))
        assert type(add_image_input) == bool, "add_image_input should be a bool, not a {}".format(type(add_image_input))
        assert type(image_channel_count) == int, "image_channel_count should be an int, not a {}".format(type(image_channel_count))
        assert type(image_feature_base_count) == int, "image_feature_base_count should be an int, not a {}".format(type(image_feature_base_count))
        assert type(add_poly_map_input) == bool, "add_poly_map_input should be a bool, not a {}".format(type(add_poly_map_input))
        assert type(poly_map_channel_count) == int, "poly_map_channel_count should be an int, not a {}".format(type(poly_map_channel_count))
        assert type(poly_map_feature_base_count) == int, "poly_map_feature_base_count should be an int, not a {}".format(type(poly_map_feature_base_count))
        assert type(common_feature_base_count) == int, "common_feature_base_count should be an int, not a {}".format(type(common_feature_base_count))
        assert type(pool_count) == int, "pool_count should be an int, not a {}".format(type(pool_count))
        assert type(add_disp_output) == bool, "add_disp_output should be a bool, not a {}".format(type(add_disp_output))
        assert type(disp_channel_count) == int, "disp_channel_count should be an int, not a {}".format(type(disp_channel_count))
        assert type(add_seg_output) == bool, "add_seg_output should be a bool, not a {}".format(type(add_seg_output))
        assert type(seg_channel_count) == int, "seg_channel_count should be an int, not a {}".format(type(seg_channel_count))
        assert type(output_res) == int, "output_res should be an int, not a {}".format(type(output_res))
        assert type(batch_size) == int, "batch_size should be an int, not a {}".format(type(batch_size))
        assert type(loss_params) == dict, "loss_params should be a dict, not a {}".format(type(loss_params))
        assert type(level_loss_coefs_params) == list, "level_loss_coefs_params should be a list, not a {}".format(type(level_loss_coefs_params))
        assert type(learning_rate_params) == dict, "learning_rate_params should be a dict, not a {}".format(type(learning_rate_params))
        assert type(weight_decay) == float, "weight_decay should be a float, not a {}".format(type(weight_decay))
        assert type(image_dynamic_range) == list, "image_dynamic_range should be a string, not a {}".format(type(image_dynamic_range))
        assert type(disp_map_dynamic_range_fac) == float, "disp_map_dynamic_range_fac should be a float, not a {}".format(type(disp_map_dynamic_range_fac))
        assert type(disp_max_abs_value) == float or type(disp_max_abs_value) == int, "disp_max_abs_value should be a number, not a {}".format(type(disp_max_abs_value))

        # Re-init Tensorflow
        self.init_tf()
        # Init attributes from arguments
        self.model_name = model_name
        self.input_res = input_res

        self.add_image_input = add_image_input
        self.image_channel_count = image_channel_count
        self.image_feature_base_count = image_feature_base_count

        self.add_poly_map_input = add_poly_map_input
        self.poly_map_channel_count = poly_map_channel_count
        self.poly_map_feature_base_count = poly_map_feature_base_count

        self.common_feature_base_count = common_feature_base_count
        self.pool_count = pool_count

        self.add_disp_output = add_disp_output
        self.disp_channel_count = disp_channel_count

        self.add_seg_output = add_seg_output
        self.seg_channel_count = seg_channel_count

        self.output_res = output_res
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.image_dynamic_range = image_dynamic_range
        self.disp_map_dynamic_range_fac = disp_map_dynamic_range_fac
        self.disp_max_abs_value = disp_max_abs_value

        # Create placeholders
        self.input_image, \
        self.input_disp_polygon_map, \
        self.gt_disp_field_map, \
        self.gt_seg, \
        self.gt_polygons, \
        self.disp_polygons = self.create_placeholders()

        # --- Create model --- #
        # # concat_interm_outputs:
        # self.level_0_disp_pred, \
        # self.stacked_disp_preds, \
        # self.level_0_seg_pred, \
        # self.stacked_seg_pred_logits, \
        # self.keep_prob = model_utils_concat_interm_outputs.build_double_unet(self.input_image,
        #                                                                      self.input_disp_polygon_map,
        #                                                                      self.image_feature_base_count,
        #                                                                      self.poly_map_feature_base_count,
        #                                                                      self.common_feature_base_count,
        #                                                                      self.pool_count,
        #                                                                      self.disp_channel_count,
        #                                                                      add_seg_output=self.add_seg_output,
        #                                                                      seg_channel_count=self.seg_channel_count,
        #                                                                      weight_decay=self.weight_decay)

        # # Old way:
        # self.level_0_disp_pred, \
        # self.stacked_disp_preds, \
        # self.level_0_seg_pred, \
        # self.stacked_seg_pred_logits, \
        # self.keep_prob = model_utils.build_double_unet(self.input_image, self.input_disp_polygon_map,
        #                                                self.image_feature_base_count,
        #                                                self.poly_map_feature_base_count,
        #                                                self.common_feature_base_count, self.pool_count,
        #                                                self.disp_channel_count,
        #                                                add_seg_output=self.add_seg_output,
        #                                                seg_channel_count=self.seg_channel_count,
        #                                                weight_decay=self.weight_decay)

        # New way:
        input_branch_params_list = []
        if self.add_image_input:
            input_branch_params_list.append({
                "tensor": self.input_image,
                "name": "image",
                "feature_base_count": self.image_feature_base_count,
            })
        if self.add_poly_map_input:
            input_branch_params_list.append({
                "tensor": self.input_disp_polygon_map,
                "name": "poly_map",
                "feature_base_count": self.poly_map_feature_base_count,
            })
        output_branch_params_list = []
        if self.add_disp_output:
            output_branch_params_list.append({
                "feature_base_count": self.common_feature_base_count,
                "channel_count": self.disp_channel_count,
                "activation": tf.nn.tanh,
                "name": "disp",
            })
        if self.add_seg_output:
            output_branch_params_list.append({
                "feature_base_count": self.common_feature_base_count,
                "channel_count": self.seg_channel_count,
                "activation": tf.identity,
                "name": "seg",
            })

        outputs, self.keep_prob = model_utils.build_multibranch_unet(input_branch_params_list, self.pool_count,
                                                                     self.common_feature_base_count,
                                                                     output_branch_params_list,
                                                                     weight_decay=self.weight_decay)
        if self.add_disp_output:
            index = 0
            _, self.stacked_disp_preds, self.level_0_disp_pred = outputs[index]
        else:
            self.stacked_disp_preds = self.level_0_disp_pred = None
        if self.add_seg_output:
            index = self.add_disp_output  # 0 if there is no disp_output, 1 if there is
            self.stacked_seg_pred_logits, _, self.level_0_seg_pred = outputs[index]
            # # --- Add polygonization module --- #
            # print_utils.print_info(" --- Add polygonization module: --- #")
            # polygonization_utils.build_polygonization_module(self.level_0_seg_pred)
            # print_utils.print_info(" --- --- #")
        else:
            self.stacked_seg_pred_logits = self.level_0_seg_pred = None

        # --- --- #

        # Create training attributes
        self.global_step = self.create_global_step()
        self.learning_rate = self.build_learning_rate(learning_rate_params)
        # Create level_coefs tensor
        self.level_loss_coefs = self.build_level_coefs(level_loss_coefs_params)

        # Build losses
        self.total_loss = self.build_losses(loss_params)

        # # Build evaluator
        # self.aligned_disp_polygons_batch, self.threshold_accuracies = self.build_evaluator()

        # Create optimizer
        self.train_step = self.build_optimizer()

    @staticmethod
    def init_tf():
        tf.reset_default_graph()

    def create_placeholders(self):
        input_image = tf.placeholder(tf.float32, [self.batch_size, self.input_res, self.input_res,
                                                  self.image_channel_count])
        input_disp_polygon_map = tf.placeholder(tf.float32, [self.batch_size, self.input_res,
                                                             self.input_res,
                                                             self.poly_map_channel_count])

        gt_disp_field_map = tf.placeholder(tf.float32, [self.batch_size, self.output_res, self.output_res,
                                                        self.disp_channel_count])
        gt_seg = tf.placeholder(tf.float32, [self.batch_size, self.input_res, self.input_res,
                                             self.poly_map_channel_count])

        gt_polygons = tf.placeholder(tf.float32, [self.batch_size, None, None, 2])
        disp_polygons = tf.placeholder(tf.float32, [self.batch_size, None, None, 2])

        return input_image, input_disp_polygon_map, gt_disp_field_map, gt_seg, gt_polygons, disp_polygons

    @staticmethod
    def create_global_step():
        return tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def build_learning_rate(self, learning_rate_params):
        return tf.train.piecewise_constant(self.global_step, learning_rate_params["boundaries"],
                                           learning_rate_params["values"])

    def build_level_coefs(self, level_loss_coefs_params):
        with tf.name_scope('level_coefs'):
            level_loss_coefs_list = []
            for level_index, level_coef_params in enumerate(level_loss_coefs_params):
                level_loss_coef = tf.train.piecewise_constant(self.global_step,
                                                              level_coef_params["boundaries"],
                                                              level_coef_params["values"],
                                                              name="{}".format(level_index))
                tf.summary.scalar("{}".format(level_index), level_loss_coef)
                level_loss_coefs_list.append(level_loss_coef)
            level_loss_coefs = tf.stack(level_loss_coefs_list)
        return level_loss_coefs

    def build_losses(self, loss_params):
        with tf.name_scope('losses'):
            if self.add_disp_output:
                # Displacement loss
                displacement_error = loss_utils.displacement_error(self.gt_disp_field_map,
                                                                   self.stacked_disp_preds,
                                                                   self.level_loss_coefs,
                                                                   self.input_disp_polygon_map,
                                                                   loss_params["disp"])
                tf.summary.scalar('displacement_error', displacement_error)
                weighted_displacement_error = loss_params["disp"]["coef"] * displacement_error
                tf.summary.scalar('weighted_displacement_error', weighted_displacement_error)
                tf.add_to_collection('losses', weighted_displacement_error)

                # Laplacian penalty
                laplacian_penalty = loss_utils.laplacian_penalty(self.stacked_disp_preds,
                                                                 self.level_loss_coefs)
                tf.summary.scalar('laplacian_penalty', laplacian_penalty)
                weighted_laplacian_penalty = loss_params["laplacian_penalty_coef"] * laplacian_penalty
                tf.summary.scalar('weighted_laplacian_penalty', weighted_laplacian_penalty)
                tf.add_to_collection('losses', weighted_laplacian_penalty)

            if self.add_seg_output:
                # Segmentation loss
                segmentation_error = loss_utils.segmentation_error(self.gt_seg,
                                                                   self.stacked_seg_pred_logits,
                                                                   self.level_loss_coefs,
                                                                   loss_params["seg"])
                tf.summary.scalar('segmentation_error', segmentation_error)
                weighted_segmentation_error = loss_params["seg"]["coef"] * segmentation_error
                tf.summary.scalar('weighted_segmentation_error', weighted_segmentation_error)
                tf.add_to_collection('losses', weighted_segmentation_error)

            # Add up all losses (objective loss + weigh loss for now)
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            tf.summary.scalar('total_loss', total_loss)

        with tf.name_scope('losses_baseline'):
            if self.add_disp_output:
                # Baseline displacement loss
                baseline_stacked_disp_preds = tf.zeros_like(self.stacked_disp_preds)
                baseline_displacement_error = loss_utils.displacement_error(self.gt_disp_field_map,
                                                                            baseline_stacked_disp_preds,
                                                                            self.level_loss_coefs,
                                                                            self.input_disp_polygon_map,
                                                                            loss_params["disp"])
                tf.summary.scalar('baseline_displacement_error', baseline_displacement_error)
        return total_loss

    # def build_evaluator(self):
    #     thresholds = np.arange(0, 8.0, 0.5)
    #     disp_max_abs_value = self.disp_max_abs_value
    #
    #     def evaluate(pred_disp_field_map_batch, disp_polygons_batch, gt_polygons_batch):
    #         # val_gt_disp_field_map_batch *= 2*DISP_MAX_ABS_VALUE  # Denormalize
    #         # val_aligned_disp_polygons_batch = polygon_utils.apply_batch_disp_map_to_polygons(
    #         #     val_gt_disp_field_map_batch, val_disp_polygons_batch)
    #         pred_disp_field_map_batch *= 2 * disp_max_abs_value  # Denormalize
    #         aligned_disp_polygons_batch = polygon_utils.apply_batch_disp_map_to_polygons(
    #             pred_disp_field_map_batch, disp_polygons_batch)
    #         threshold_accuracies = evaluate_utils.compute_threshold_accuracies(gt_polygons_batch,
    #                                                                            aligned_disp_polygons_batch,
    #                                                                            thresholds)  # TODO: add padding information to filter out vertices outside output image
    #         aligned_disp_polygons_batch = aligned_disp_polygons_batch.astype(np.float32)
    #         threshold_accuracies = np.array(threshold_accuracies).astype(np.float32)
    #         return aligned_disp_polygons_batch, threshold_accuracies
    #
    #     with tf.name_scope('evaluator'):
    #         aligned_disp_polygons_batch, threshold_accuracies = tf.py_func(
    #             evaluate,
    #             [self.level_0_disp_pred, self.disp_polygons, self.gt_polygons],
    #             Tout=(tf.float32, tf.float32),
    #             name="evaluator"
    #         )
    #
    #         threshold_accuracies.set_shape((1, len(thresholds)))
    #
    #         # tf.summary.scalar('accuracy with threshold 1', threshold_accuracies[0])
    #         # # tf.summary.scalar('accuracy with threshold 2', threshold_accuracy_2)
    #         # # tf.summary.scalar('accuracy with threshold 3', threshold_accuracy_3)
    #         # # tf.summary.scalar('accuracy with threshold 4', threshold_accuracy_4)
    #         # # tf.summary.scalar('accuracy with threshold 5', threshold_accuracy_5)
    #         # # tf.summary.scalar('accuracy with threshold 6', threshold_accuracy_6)
    #         # # tf.summary.scalar('accuracy with threshold 7', threshold_accuracy_7)
    #         # # tf.summary.scalar('accuracy with threshold 8', threshold_accuracy_8)
    #
    #     return aligned_disp_polygons_batch, threshold_accuracies

    def build_optimizer(self):
        with tf.name_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_step = optimizer.minimize(self.total_loss, global_step=self.global_step)
            current_adam_lr = tf_utils.compute_current_adam_lr(optimizer)
            tf.summary.scalar('lr', current_adam_lr)
        return train_step

    def train(self, sess, dataset_tensors, dropout_keep_prob, with_summaries=False, merged_summaries=None,
              summaries_writer=None, summary_index=None, plot=False):
        """

        :param sess:
        :param with_summaries: (Default: False)
        :param merged_summaries: Must be not None if with_summaries is True
        :param summaries_writer: Must be not None if with_summaries is True
        :return:
        """
        if with_summaries:
            assert merged_summaries is not None and summaries_writer is not None, \
                "merged_summaries and writer should be specified if with_summaries is True"
        train_image, \
        _, \
        _, \
        train_gt_polygon_map, \
        train_gt_disp_field_map, \
        train_disp_polygon_map = dataset_tensors
        train_image_batch, train_gt_polygon_map_batch, train_gt_disp_field_map_batch, train_disp_polygon_map_batch = sess.run(
            [train_image, train_gt_polygon_map, train_gt_disp_field_map, train_disp_polygon_map])

        feed_dict = {
            self.input_image: train_image_batch,
            self.input_disp_polygon_map: train_disp_polygon_map_batch,
            self.gt_disp_field_map: train_gt_disp_field_map_batch,
            self.gt_seg: train_gt_polygon_map_batch,
            self.gt_polygons: tf_utils.create_array_to_feed_placeholder(self.gt_polygons),
            self.disp_polygons: tf_utils.create_array_to_feed_placeholder(self.disp_polygons),
            self.keep_prob: dropout_keep_prob,
        }

        if with_summaries:
            if summary_index == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = run_metadata = None

            input_list = [merged_summaries, self.train_step, self.total_loss]
            if self.add_disp_output:
                input_list.append(self.level_0_disp_pred)
            if self.add_seg_output:
                input_list.append(self.level_0_seg_pred)

            output_list = sess.run(input_list, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

            extra_output_count = self.add_disp_output + self.add_seg_output
            train_summary, _, train_loss = output_list[:-extra_output_count]

            train_pred_disp_field_map_batch = train_pred_seg_batch = None
            if self.add_disp_output:
                index = -extra_output_count
                train_pred_disp_field_map_batch = output_list[index]
            if self.add_seg_output:
                index = -extra_output_count + self.add_disp_output
                train_pred_seg_batch = output_list[index]

            # TODO: If uncommenting below code, also add relevant code to the "else" block below
            # train_summary, _, train_loss, train_pred_disp_field_map_batch = sess.run(
            #     [merged_summaries, train_step, total_loss, pred_disp_field_map],
            #     feed_dict={input_image: train_gt_polygon_map_batch, input_disp_polygon_map: train_disp_polygon_map_batch,
            #                gt_disp_field_map: train_gt_disp_field_map_batch,
            #                keep_prob: DROPOUT_KEEP_PROB,
            #                mode_training: True}, options=run_options, run_metadata=run_metadata)
            summaries_writer.add_summary(train_summary, summary_index)
            if summary_index == 0:
                summaries_writer.add_run_metadata(run_metadata, 'step%03d' % summary_index)
            print_utils.print_info("step {}, training loss = {}".format(summary_index, train_loss))

            if plot:
                train_image_batch = (train_image_batch - self.image_dynamic_range[0]) / (
                        self.image_dynamic_range[1] - self.image_dynamic_range[0])
                # train_gt_disp_field_map_batch = train_gt_disp_field_map_batch * 2  # Within [-1, 1]
                # train_gt_disp_field_map_batch = train_gt_disp_field_map_batch * self.disp_max_abs_value  # Within [-disp_max_abs_value, disp_max_abs_value]
                # train_pred_disp_field_map_batch = train_pred_disp_field_map_batch * 2  # Within [-1, 1]
                # train_pred_disp_field_map_batch = train_pred_disp_field_map_batch * self.disp_max_abs_value  # Within [-disp_max_abs_value, disp_max_abs_value]
                # visualization.plot_batch(["Training gt disp", "Training pred disp"], train_image_batch,
                #                          train_gt_polygon_map_batch,
                #                          [train_gt_disp_field_map_batch, train_pred_disp_field_map_batch],
                #                          train_disp_polygon_map_batch)
                if self.add_seg_output:
                    visualization.plot_batch_seg("Training pred seg", train_image_batch, train_pred_seg_batch)

            return train_image_batch, train_gt_polygon_map_batch, train_gt_disp_field_map_batch, train_disp_polygon_map_batch, train_pred_disp_field_map_batch, train_pred_seg_batch
        else:
            _ = sess.run([self.train_step], feed_dict=feed_dict)
            return train_image_batch, train_gt_polygon_map_batch, train_gt_disp_field_map_batch, train_disp_polygon_map_batch, None, None

    def validate(self, sess, dataset_tensors, merged_summaries, summaries_writer, summary_index, plot=False):
        val_image, \
        val_gt_polygons, \
        val_disp_polygons, \
        val_gt_polygon_map, \
        val_gt_disp_field_map, \
        val_disp_polygon_map = dataset_tensors
        val_image_batch, val_gt_polygons_batch, val_disp_polygons_batch, val_gt_polygon_map_batch, val_gt_disp_field_map_batch, val_disp_polygon_map_batch = sess.run(
            [val_image, val_gt_polygons, val_disp_polygons, val_gt_polygon_map, val_gt_disp_field_map,
             val_disp_polygon_map])

        feed_dict = {
            self.input_image: val_image_batch,
            self.input_disp_polygon_map: val_disp_polygon_map_batch,
            self.gt_disp_field_map: val_gt_disp_field_map_batch,
            self.gt_seg: val_gt_polygon_map_batch,
            self.gt_polygons: val_gt_polygons_batch,
            self.disp_polygons: val_disp_polygons_batch,
            self.keep_prob: 1.0
        }

        input_list = [merged_summaries, self.total_loss]
        if self.add_disp_output:
            input_list.append(self.level_0_disp_pred)
        if self.add_seg_output:
            input_list.append(self.level_0_seg_pred)

        output_list = sess.run(input_list, feed_dict=feed_dict)

        extra_output_count = self.add_disp_output + self.add_seg_output
        val_summary, val_loss, = output_list[:-extra_output_count]

        val_pred_disp_field_map_batch = val_pred_seg_batch = None
        if self.add_disp_output:
            index = -extra_output_count
            val_pred_disp_field_map_batch = output_list[index]
        if self.add_seg_output:
            index = -extra_output_count + self.add_disp_output
            val_pred_seg_batch = output_list[index]

        if plot:
            val_image_batch = (val_image_batch - self.image_dynamic_range[0]) / (
                    self.image_dynamic_range[1] - self.image_dynamic_range[0])
            # visualization.plot_batch_polygons("Validation plot", val_image_batch, val_gt_polygons_batch,
            #                                   val_disp_polygons_batch, val_aligned_disp_polygons_batch)
            if self.add_seg_output:
                visualization.plot_batch_seg("Validation pred seg", val_image_batch, val_pred_seg_batch)

        summaries_writer.add_summary(val_summary, summary_index)
        print_utils.print_info("step {}, validation loss = {}".format(summary_index, val_loss))
        # print("\t validation threshold accuracies = {}".format(val_threshold_accuracies))
        return val_image_batch, val_gt_polygons_batch, val_disp_polygons_batch, val_gt_polygon_map_batch, val_gt_disp_field_map_batch, val_disp_polygon_map_batch, val_pred_disp_field_map_batch, val_pred_seg_batch

    def restore_checkpoint(self, sess, saver, checkpoints_dir):
        """

        :param sess:
        :param saver:
        :param checkpoints_dir:
        :return: True if a checkpoint was found and restored, False if no checkpoint was found
        """
        checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
        if checkpoint and checkpoint.model_checkpoint_path:  # Check if the model has a checkpoint
            print_utils.print_info("Restoring {} checkpoint {}".format(self.model_name, checkpoint.model_checkpoint_path))
            try:
                saver.restore(sess, checkpoint.model_checkpoint_path)
            except tf.errors.InvalidArgumentError:
                print_utils.print_error("ERROR: could not load checkpoint.\n"
                      "\tThis is likely due to: .\n"
                      "\t\t -  the model graph definition has changed from the checkpoint thus weights do not match\n"
                      .format(checkpoints_dir)
                      )
                exit()
            return True
        else:
            return False

    # def get_weight_variables(self, starts_with):
    #     """
    #
    #     :return: A filtered list of all trainable variables whose names start with starts_with.
    #     """
    #     trainable_variables = tf.trainable_variables()
    #     weight_variables = []
    #     for var in trainable_variables:
    #         if var.name.startswith(starts_with):
    #             weight_variables.append(var)
    #     return weight_variables

    def optimize(self, train_dataset_tensors, val_dataset_tensors,
                 max_iter, dropout_keep_prob,
                 logs_dir, train_summary_step, val_summary_step,
                 checkpoints_dir, checkpoint_step,
                 init_checkpoints_dirpath=None,
                 plot_results=False):
        """

        :param train_dataset_tensors:
        :param val_dataset_tensors: (If None: do not perform validation step)
        :param max_iter:
        :param dropout_keep_prob:
        :param logs_dir:
        :param train_summary_step:
        :param val_summary_step:
        :param checkpoints_dir: Directory to save checkpoints. If this is not the first time launching the optimization,
                                the weights will be restored form the last checkpoint in that directory
        :param checkpoint_step:
        :param init_checkpoints_dirpath: If this is the first time launching the optimization, the weights will be
                                     initialized with the last checkpoint in init_checkpoints_dirpath (optional)
        :param plot_results: (optional)
        :return:
        """
        # Summaries
        merged_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(logs_dir, "train"), tf.get_default_graph())
        val_writer = tf.summary.FileWriter(os.path.join(logs_dir, "val"), tf.get_default_graph())

        # Savers
        saver = tf.train.Saver(save_relative_paths=True)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)

            # Restore checkpoint if one exists
            restore_checkpoint_success = self.restore_checkpoint(sess, saver, checkpoints_dir)
            if not restore_checkpoint_success and init_checkpoints_dirpath is not None:
                # This is the first time launching this optimization.
                # Create saver with only trainable variables:
                init_variables_saver = tf.train.Saver(tf.trainable_variables())
                # Restore from init_checkpoints_dirpath if it exists:
                restore_checkpoint_success = self.restore_checkpoint(sess, init_variables_saver,
                                                                     init_checkpoints_dirpath)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            if plot_results:
                visualization.init_figures(["Training gt disp", "Training pred disp", "Training pred seg",
                                            "Training polygonization",
                                            "Validation plot", "Validation pred seg"])

            print("Model has {} trainable variables".format(
                tf_utils.count_number_trainable_params())
            )

            i = tf.train.global_step(sess, self.global_step)
            while i <= max_iter:
                if i % train_summary_step == 0:
                    time_start = time.time()

                    train_image_batch, \
                    train_gt_polygon_map_batch, \
                    train_gt_disp_field_map_batch, \
                    train_disp_polygon_map_batch, \
                    train_pred_disp_field_map_batch, \
                    train_pred_seg_batch = self.train(sess, train_dataset_tensors, dropout_keep_prob,
                                                      with_summaries=True, merged_summaries=merged_summaries,
                                                      summaries_writer=train_writer, summary_index=i, plot=plot_results)

                    time_end = time.time()
                    print("\tIteration done in {}s".format(time_end - time_start))

                else:
                    self.train(sess, train_dataset_tensors, dropout_keep_prob)

                if val_dataset_tensors is not None:
                    # i += 1
                    # Measure validation loss and accuracy
                    if i % val_summary_step == 1:
                        val_image_batch, \
                        val_gt_polygons_batch, \
                        val_disp_polygons_batch, \
                        val_gt_polygon_map_batch, \
                        val_gt_disp_field_map_batch, \
                        val_disp_polygon_map_batch, \
                        val_pred_disp_field_map_batch, val_pred_seg_batch = self.validate(sess, val_dataset_tensors,
                                                                                          merged_summaries, val_writer, i,
                                                                                          plot=plot_results)
                # Save checkpoint
                if i % checkpoint_step == (checkpoint_step - 1):
                    saver.save(sess, os.path.join(checkpoints_dir, self.model_name),
                               global_step=self.global_step)

                i = tf.train.global_step(sess, self.global_step)

            coord.request_stop()
            coord.join(threads)

            train_writer.close()
            val_writer.close()

    def make_batches_patch_boundingboxes(self, patch_boundingboxes, batch_size):
        batches_patch_boundingboxes = []
        batch_patch_boundingboxes = []
        for patch_boundingbox in patch_boundingboxes:
            if len(batch_patch_boundingboxes) < batch_size:
                batch_patch_boundingboxes.append(patch_boundingbox)
            else:
                batches_patch_boundingboxes.append(batch_patch_boundingboxes)
                batch_patch_boundingboxes = []
        return batches_patch_boundingboxes

    def inference(self, image_array, ori_gt_array, checkpoints_dir):
        """
        Runs inference on image_array and ori_gt_array with model checkpoint in checkpoints_dir

        :param image_array:
        :param ori_gt_array:
        :param checkpoints_dir:
        :return:
        """
        spatial_shape = image_array.shape[:2]
        # Format inputs
        image_array = image_array[:, :, :3]  # Remove alpha channel if any
        image_array = (image_array / 255) * (self.image_dynamic_range[1] - self.image_dynamic_range[0]) + \
                      self.image_dynamic_range[0]

        ori_gt_array = ori_gt_array / 255

        padding = (self.input_res - self.output_res) // 2

        # Init displacement field and segmentation image
        complete_pred_field_map = np.zeros(
            (spatial_shape[0] - 2 * padding, spatial_shape[1] - 2 * padding, self.disp_channel_count))
        complete_segmentation_image = np.zeros(
            (spatial_shape[0] - 2 * padding, spatial_shape[1] - 2 * padding, self.seg_channel_count))

        # visualization.init_figures(["example"])

        # Iterate over every patch and predict displacement field for this patch
        patch_boundingboxes = image_utils.compute_patch_boundingboxes(spatial_shape,
                                                                      stride=self.output_res,
                                                                      patch_res=self.input_res)
        batch_boundingboxes_list = list(
            python_utils.split_list_into_chunks(patch_boundingboxes, self.batch_size, pad=True))

        # Saver
        saver = tf.train.Saver(save_relative_paths=True)
        with tf.Session() as sess:
            # Restore checkpoint
            restore_checkpoint_success = self.restore_checkpoint(sess, saver, checkpoints_dir)
            if not restore_checkpoint_success:
                sys.exit('No checkpoint found in {}'.format(checkpoints_dir))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Loop over every batch
            for batch_index, batch_boundingboxes in enumerate(batch_boundingboxes_list):
                if batch_index % 10 == 0:
                    print("Processing batch {}/{}"
                          .format(batch_index + 1, len(batch_boundingboxes_list)))
                # Form batch
                batch_image_list = []
                batch_ori_gt_list = []
                for boundingbox in batch_boundingboxes:
                    patch_image = image_array[boundingbox[0]:boundingbox[2],
                                  boundingbox[1]:boundingbox[3], :]
                    patch_ori_gt = ori_gt_array[boundingbox[0]:boundingbox[2],
                                   boundingbox[1]:boundingbox[3], :]
                    batch_image_list.append(patch_image)
                    batch_ori_gt_list.append(patch_ori_gt)
                batch_image = np.stack(batch_image_list, axis=0)
                batch_ori_gt = np.stack(batch_ori_gt_list, axis=0)

                if self.add_seg_output:
                    batch_pred_disp_field_map, batch_pred_seg = sess.run(
                        [self.level_0_disp_pred, self.level_0_seg_pred], feed_dict={
                            self.input_image: batch_image,
                            self.input_disp_polygon_map: batch_ori_gt,
                            self.keep_prob: 1.0
                        })
                else:
                    batch_pred_disp_field_map = sess.run(
                        self.level_0_disp_pred, feed_dict={
                            self.input_image: batch_image,
                            self.input_disp_polygon_map: batch_ori_gt,
                            self.keep_prob: 1.0
                        })
                    batch_pred_seg = np.zeros((batch_pred_disp_field_map.shape[0], batch_pred_disp_field_map.shape[1],
                                               batch_pred_disp_field_map.shape[2], self.seg_channel_count))

                # Fill complete outputs
                for batch_index, boundingbox in enumerate(batch_boundingboxes):
                    patch_pred_disp_field_map = batch_pred_disp_field_map[batch_index]
                    patch_pred_seg = batch_pred_seg[batch_index]
                    # print("--- patch_pred_seg: ---")
                    # print(patch_pred_seg[:, :, 0])
                    # print("---")
                    # print(patch_pred_seg[:, :, 1])
                    # print("---")
                    # print(patch_pred_seg[:, :, 2])
                    # print("---")
                    # print(patch_pred_seg[:, :, 3])
                    # print("---")

                    # # visualization.init_figures(["example", "example 2"])
                    # visualization.init_figures(["example"])
                    # patch_image = image_array[boundingbox[0]:boundingbox[2],
                    #               boundingbox[1]:boundingbox[3], :]
                    # patch_image = (patch_image - self.image_dynamic_range[0]) / (
                    #         self.image_dynamic_range[1] - self.image_dynamic_range[0])
                    # visualization.plot_seg("example", patch_image, patch_pred_seg)

                    padded_boundingbox = image_utils.padded_boundingbox(boundingbox, padding)
                    translated_padded_boundingbox = [x - padding for x in padded_boundingbox]
                    complete_pred_field_map[
                    translated_padded_boundingbox[0]:translated_padded_boundingbox[2],
                    translated_padded_boundingbox[1]:translated_padded_boundingbox[3], :] = patch_pred_disp_field_map
                    complete_segmentation_image[
                    translated_padded_boundingbox[0]:translated_padded_boundingbox[2],
                    translated_padded_boundingbox[1]:translated_padded_boundingbox[3],
                    :] = patch_pred_seg

                    # visualization.plot_seg("example 2", patch_image, complete_segmentation_image[
                    # translated_padded_boundingbox[0]:translated_padded_boundingbox[2],
                    # translated_padded_boundingbox[1]:translated_padded_boundingbox[3],
                    # :])

            # visualization.plot_example("example",
            #                            patch_image[0],
            #                            patch_ori_gt[0],
            #                            patch_pred_disp_field_map[0],
            #                            patch_ori_gt[0])

            coord.request_stop()
            coord.join(threads)

        # De-normalize field map
        complete_pred_field_map = complete_pred_field_map / self.disp_map_dynamic_range_fac  # Within [-1, 1]
        complete_pred_field_map = complete_pred_field_map * self.disp_max_abs_value  # Within [-DISP_MAX_ABS_VALUE, DISP_MAX_ABS_VALUE]

        # # De-normalize segmentation image
        # complete_segmentation_image = complete_segmentation_image * 255
        # complete_segmentation_image = complete_segmentation_image.astype(dtype=np.uint8)

        return complete_pred_field_map, complete_segmentation_image

    @staticmethod
    def get_output_res(input_res, pool_count):
        """
        This function has to be re-written if the model architecture changes

        :param input_res:
        :param pool_count:
        :return:
        """
        return model_utils.get_output_res(input_res, pool_count)

    @staticmethod
    def get_input_res(output_res, pool_count):
        """
        This function has to be re-written if the model architecture changes

        :param output_res:
        :param pool_count:
        :return:
        """
        return model_utils.get_input_res(output_res, pool_count)


def main(_):
    input_res = 348
    pool_count = 3
    print("Get output_res from input_res = {}".format(input_res))
    output_res = MapAlignModel.get_output_res(input_res, pool_count)
    print("output_res = {}".format(output_res))

    output_res = 228
    pool_count = 3
    print("Get input_res from output_res = {}".format(output_res))
    input_res = MapAlignModel.get_input_res(output_res, pool_count)
    print("input_res = {}".format(input_res))


if __name__ == '__main__':
    tf.app.run(main=main)
