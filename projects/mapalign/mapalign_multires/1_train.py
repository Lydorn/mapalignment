from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import os

import model

sys.path.append(os.path.join("../dataset_utils"))
import dataset_multires

sys.path.append("../../utils")
import python_utils
import run_utils

# --- Command-line FLAGS --- #

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('config', "config",
                    "Name of the config file, excluding the .json file extension")

flags.DEFINE_boolean('new_run', False,
                     "Train from scratch (when True) or train from the last checkpoint (when False)")

flags.DEFINE_string('init_run_name', None,
                    "This is the run_name to initialize the weights from. "
                    "If None, weights will be initialized randomly."
                    "This is a single word, without the timestamp.")

flags.DEFINE_string('run_name', None,
                    "Continue training from run_name. This is a single word, without the timestamp.")
# If not specified, the last run is used (unless new_run is True or no runs are in the runs directory).
# If new_run is True, creates the new run with name equal run_name.

flags.DEFINE_integer('batch_size', 8, "Batch size. Generally set as large as the VRAM can handle.")

flags.DEFINE_integer('ds_fac', 8, "Downsampling factor. Choose from which resolution sub-dataset to train on.")


# Some examples:
# On Quadro M2200, 4GB VRAM: python 1_train.py --new_run --run_name=ds_fac_8 --batch_size 4 --ds_fac 8
# On Quadro M2200, 4GB VRAM: python 1_train.py --new_run  --init_run_name=ds_fac_8 --run_name=ds_fac_4_with_init --batch_size 4 --ds_fac_4
# On Quadro M2200, 4GB VRAM: python 1_train.py --new_run --batch_size 4 --ds_fac 2

# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --new_run --run_name=ds_fac_8_no_seg --batch_size 32 --ds_fac 8
# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --new_run --run_name=ds_fac_4_no_seg --batch_size 32 --ds_fac 4

# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --new_run --init_run_name=ds_fac_4_double --run_name=ds_fac_8_double --batch_size 32 --ds_fac 8
# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --new_run --init_run_name=ds_fac_4_double --run_name=ds_fac_2_double --batch_size 32 --ds_fac 2
# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --new_run --init_run_name=ds_fac_1_double --run_name=ds_fac_1_double_seg --batch_size 32 --ds_fac 1

# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --run_name=ds_fac_8_double --batch_size 32 --ds_fac 8
# On GTX 1080 Ti, 11GB VRAM: python 1_train.py --run_name=ds_fac_2_double --batch_size 32 --ds_fac 2

# --- --- #


def train(config, tfrecords_dirpath_list, init_run_dirpath, run_dirpath, batch_size, ds_fac_list, ds_repeat_list):
    # setup init checkpoints directory path if one is specified:
    if init_run_dirpath is not None:
        _, init_checkpoints_dirpath = run_utils.setup_run_subdirs(init_run_dirpath, config["logs_dirname"],
                                                                  config["checkpoints_dirname"])
    else:
        init_checkpoints_dirpath = None

    # setup stage run dirs
    # create run subdirectories if they do not exist
    logs_dirpath, checkpoints_dirpath = run_utils.setup_run_subdirs(run_dirpath, config["logs_dirname"],
                                                                    config["checkpoints_dirname"])

    # compute output_res
    output_res = model.MapAlignModel.get_output_res(config["input_res"], config["pool_count"])
    print("output_res: {}".format(output_res))

    # instantiate model object (resets the default graph)
    map_align_model = model.MapAlignModel(config["model_name"], config["input_res"],

                                          config["add_image_input"], config["image_channel_count"],
                                          config["image_feature_base_count"],

                                          config["add_poly_map_input"], config["poly_map_channel_count"],
                                          config["poly_map_feature_base_count"],

                                          config["common_feature_base_count"], config["pool_count"],

                                          config["add_disp_output"], config["disp_channel_count"],

                                          config["add_seg_output"], config["seg_channel_count"],

                                          output_res,
                                          batch_size,

                                          config["loss_params"],
                                          config["level_loss_coefs_params"],

                                          config["learning_rate_params"],
                                          config["weight_decay"],

                                          config["image_dynamic_range"], config["disp_map_dynamic_range_fac"],
                                          config["disp_max_abs_value"])

    # train dataset
    train_dataset_filename_list = dataset_multires.create_dataset_filename_list(tfrecords_dirpath_list,
                                                                                config["tfrecord_filename_format"],
                                                                                ds_fac_list,
                                                                                dataset="train",
                                                                                resolution_file_repeats=ds_repeat_list)
    train_dataset_tensors = dataset_multires.read_and_decode(
        train_dataset_filename_list,
        output_res,
        config["input_res"],
        batch_size,
        config["image_dynamic_range"],
        disp_map_dynamic_range_fac=config["disp_map_dynamic_range_fac"],
        keep_poly_prob=config["keep_poly_prob"],
        data_aug=config["data_aug"],
        train=True)

    if config["perform_validation_step"]:
        # val dataset
        val_dataset_filename_list = dataset_multires.create_dataset_filename_list(tfrecords_dirpath_list,
                                                                                  config["tfrecord_filename_format"],
                                                                                  ds_fac_list,
                                                                                  dataset="val",
                                                                                  resolution_file_repeats=ds_repeat_list)
        val_dataset_tensors = dataset_multires.read_and_decode(
            val_dataset_filename_list,
            output_res,
            config["input_res"],
            batch_size,
            config["image_dynamic_range"],
            disp_map_dynamic_range_fac=config["disp_map_dynamic_range_fac"],
            keep_poly_prob=config["keep_poly_prob"],
            data_aug=False,
            train=False)
    else:
        val_dataset_tensors = None

    # launch training
    map_align_model.optimize(train_dataset_tensors, val_dataset_tensors,
                             config["max_iter"], config["dropout_keep_prob"],
                             logs_dirpath, config["train_summary_step"], config["val_summary_step"],
                             checkpoints_dirpath, config["checkpoint_step"],
                             init_checkpoints_dirpath=init_checkpoints_dirpath,
                             plot_results=config["plot_results"])


def main(_):
    working_dir = os.path.dirname(os.path.abspath(__file__))

    # print FLAGS
    print("#--- FLAGS: ---#")
    print("config: {}".format(FLAGS.config))
    print("new_run: {}".format(FLAGS.new_run))
    print("init_run_name: {}".format(FLAGS.init_run_name))
    print("run_name: {}".format(FLAGS.run_name))
    print("batch_size: {}".format(FLAGS.batch_size))
    print("ds_fac: {}".format(FLAGS.ds_fac))

    # load config file
    config = run_utils.load_config(FLAGS.config)

    # Check config setting coherences
    assert len(config["level_loss_coefs_params"]) == config["pool_count"], \
        "level_loss_coefs_params ({} elements) must have model_res_levels ({}) elements".format(
            len(config["level_loss_coefs_params"]), config["pool_count"])

    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dir is None:
        print("ERROR: Data directory not found!")
        exit()
    else:
        print("Using data from {}".format(data_dir))

    # Setup dataset dirpaths
    tfrecords_dirpath_list = [os.path.join(data_dir, tfrecords_dirpath) for tfrecords_dirpath in
                              config["tfrecords_partial_dirpath_list"]]

    # Overwrite config ds_fac if FLAGS specify them
    if FLAGS.ds_fac is not None:
        ds_fac_list = [FLAGS.ds_fac]
        ds_repeat_list = [1]
    else:
        ds_fac_list = config["ds_fac_list"]
        ds_repeat_list = config["ds_repeat_list"]

    # setup init run directory of one is specified:
    if FLAGS.init_run_name is not None:
        init_run_dirpath = run_utils.setup_run_dir(config["runs_dirname"], FLAGS.init_run_name)
    else:
        init_run_dirpath = None

    # setup run directory:
    runs_dir = os.path.join(working_dir, config["runs_dirname"])
    current_run_dirpath = run_utils.setup_run_dir(runs_dir, FLAGS.run_name, FLAGS.new_run)

    # save config in logs directory
    run_utils.save_config(config, current_run_dirpath)

    # save FLAGS
    FLAGS_filepath = os.path.join(current_run_dirpath, "FLAGS.json")
    python_utils.save_json(FLAGS_filepath, {
        "run_name": FLAGS.run_name,
        "new_run": FLAGS.new_run,
        "batch_size": FLAGS.batch_size,
        "ds_fac": FLAGS.ds_fac,
    })

    train(config, tfrecords_dirpath_list, init_run_dirpath, current_run_dirpath, FLAGS.batch_size, ds_fac_list,
          ds_repeat_list)


if __name__ == '__main__':
    tf.app.run(main=main)
