from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import os

import config
import model

sys.path.append(os.path.join("../dataset_utils"))
import dataset_multires

sys.path.append("../../utils")
import python_utils
import run_utils

# --- Command-line flags --- #

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('new_run', False,
                     "Train from scratch (when True) or train from the last checkpoint (when False)")

flags.DEFINE_string('init_run_name', None,
                    "This is the run_name to initialize the weights from. If None, weights will be initialized randomly. This is a single word, without the timestamp.")

flags.DEFINE_string('run_name', None,
                    "Continue training from run_name. This is a single word, without the timestamp.")
# If not specified, the last run is used (unless new_run is True or no runs are in the runs directory).
# If new_run is True, creates the new run with name equal run_name.

flags.DEFINE_integer('batch_size', 8, "Batch size. Generally set as large as the VRAM can handle.")

flags.DEFINE_integer('ds_fac', 8, "Downsampling factor. Choose from which resolution sub-dataset to train on.")

# Some examples:
# On Quadro M2200, 4GB VRAM: python 1_train.py --new_run --run_name=ds_fac_8 --batch_size 8  --ds_fac 8
# On Quadro M2200, 4GB VRAM: python 1_train.py --new_run  --init_run_name=ds_fac_8 --run_name=ds_fac_4_with_init --batch_size 8  --ds_fac_4
# On Quadro M2200, 4GB VRAM: python 1_train.py --new_run --batch_size 8  --ds_fac 2

# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --new_run --run_name=ds_fac_8_no_seg --batch_size 32 --ds_fac 8
# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --new_run --run_name=ds_fac_4_no_seg --batch_size 32 --ds_fac 4

# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --new_run --init_run_name=ds_fac_4_double --run_name=ds_fac_8_double --batch_size 32 --ds_fac 8
# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --new_run --init_run_name=ds_fac_4_double --run_name=ds_fac_2_double --batch_size 32 --ds_fac 2
# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --new_run --init_run_name=ds_fac_1_double --run_name=ds_fac_1_double_seg --batch_size 32 --ds_fac 1

# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --run_name=ds_fac_8_double --batch_size 32 --ds_fac 8
# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --run_name=ds_fac_2_double --batch_size 32 --ds_fac 2

# --- --- #


def train(init_run_dirpath, run_dirpath, batch_size, ds_fac_list, ds_repeat_list):
    # Setup init checkpoints directory path if one is specified:
    if init_run_dirpath is not None:
        _, init_checkpoints_dirpath = run_utils.setup_run_subdirs(init_run_dirpath, config.LOGS_DIRNAME,
                                                                     config.CHECKPOINTS_DIRNAME)
    else:
        init_checkpoints_dirpath = None

    # Setup stage run dirs
    # Create run subdirectories if they do not exist
    logs_dirpath, checkpoints_dirpath = run_utils.setup_run_subdirs(run_dirpath, config.LOGS_DIRNAME,
                                                                       config.CHECKPOINTS_DIRNAME)

    # Compute output_res
    output_res = model.MapAlignModel.get_output_res(config.INPUT_RES, config.POOL_COUNT)
    print("output_res: {}".format(output_res))

    # Instantiate model object (resets the default graph)
    map_align_model = model.MapAlignModel(config.MODEL_NAME, config.INPUT_RES, config.IMAGE_INPUT_CHANNELS,
                                          config.IMAGE_DYNAMIC_RANGE, config.DISP_MAP_DYNAMIC_RANGE_FAC, config.POLY_MAP_INPUT_CHANNELS,
                                          config.IMAGE_FEATURE_BASE_COUNT, config.POLY_MAP_FEATURE_BASE_COUNT,
                                          config.COMMON_FEATURE_BASE_COUNT, config.POOL_COUNT,
                                          output_res, config.DISP_OUTPUT_CHANNELS, config.DISP_MAX_ABS_VALUE,
                                          config.ADD_SEG_OUTPUT,
                                          config.SEG_OUTPUT_CHANNELS,
                                          batch_size, config.LEARNING_RATE_PARAMS,
                                          config.LEVEL_LOSS_COEFS_PARAMS,
                                          config.DISP_LOSS_COEF, config.SEG_LOSS_COEF, config.LAPLACIAN_PENALTY_COEF,
                                          config.WEIGHT_DECAY)

    # Train dataset
    train_dataset_filename_list = dataset_multires.create_dataset_filename_list(config.TFRECORDS_DIR_LIST, config.TFRECORD_FILENAME_FORMAT,
                                                                                ds_fac_list,
                                                                                dataset="train",
                                                                                resolution_file_repeats=ds_repeat_list)
    train_dataset_tensors = dataset_multires.read_and_decode(
        train_dataset_filename_list,
        output_res,
        config.INPUT_RES,
        batch_size,
        config.IMAGE_DYNAMIC_RANGE,
        disp_map_dynamic_range_fac=config.DISP_MAP_DYNAMIC_RANGE_FAC,
        keep_poly_prob=config.KEEP_POLY_PROB,
        data_aug=config.DATA_AUG,
        train=True)

    # Val dataset
    val_dataset_filename_list = dataset_multires.create_dataset_filename_list(config.TFRECORDS_DIR_LIST, config.TFRECORD_FILENAME_FORMAT,
                                                                              ds_fac_list,
                                                                              dataset="val",
                                                                              resolution_file_repeats=ds_repeat_list)
    val_dataset_tensors = dataset_multires.read_and_decode(
        val_dataset_filename_list,
        output_res,
        config.INPUT_RES,
        batch_size,
        config.IMAGE_DYNAMIC_RANGE,
        disp_map_dynamic_range_fac=config.DISP_MAP_DYNAMIC_RANGE_FAC,
        keep_poly_prob=config.KEEP_POLY_PROB,
        data_aug=False,
        train=False)

    # Launch training
    map_align_model.optimize(train_dataset_tensors, val_dataset_tensors,
                             config.MAX_ITER, config.DROPOUT_KEEP_PROB,
                             logs_dirpath, config.TRAIN_SUMMARY_STEP, config.VAL_SUMMARY_STEP,
                             checkpoints_dirpath, config.CHECKPOINT_STEP,
                             init_checkpoints_dirpath=init_checkpoints_dirpath,
                             plot_results=config.PLOT_RESULTS)


def main(_):
    # Print flags
    print("#--- Flags: ---#")
    print("new_run: {}".format(FLAGS.new_run))
    print("init_run_name: {}".format(FLAGS.init_run_name))
    print("run_name: {}".format(FLAGS.run_name))
    print("batch_size: {}".format(FLAGS.batch_size))
    print("ds_fac: {}".format(FLAGS.ds_fac))

    if FLAGS.ds_fac is not None:
        ds_fac_list = [FLAGS.ds_fac]
        ds_repeat_list = [1]
    else:
        ds_fac_list = config.DS_FAC_LIST
        ds_repeat_list = config.DS_REPEAT_LIST

    # Setup init run directory of one is specified:
    if FLAGS.init_run_name is not None:
        init_run_dirpath = run_utils.setup_run_dir(config.RUNS_DIR, FLAGS.init_run_name)
    else:
        init_run_dirpath = None

    # Setup run directory:
    current_run_dirpath = run_utils.setup_run_dir(config.RUNS_DIR, FLAGS.run_name, FLAGS.new_run)

    # Save config.py in logs directory
    run_utils.save_config(config.PROJECT_DIR, current_run_dirpath)

    # Save flags
    flags_filepath = os.path.join(current_run_dirpath, "flags.json")
    python_utils.save_json(flags_filepath, {
        "run_name": FLAGS.run_name,
        "new_run": FLAGS.new_run,
        "batch_size": FLAGS.batch_size,
        "ds_fac": FLAGS.ds_fac,
    })

    train(init_run_dirpath, current_run_dirpath, FLAGS.batch_size, ds_fac_list, ds_repeat_list)


if __name__ == '__main__':
    tf.app.run(main=main)
