import sys

import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../utils")
import python_utils

# --- Params --- #
ACCURACIES_FILENAME_EXTENSION = ".accuracy.npy"

SOURCE_PARAMS_LIST = [
    # # --- Stereo real disps --- #
    # {
    #     "name": "Aligned image 1",
    #     "path": "test/stereo_dataset_real_displacements.align.ds_fac_8.ds_fac_4.ds_fac_2.image_ref",
    #     "plot_color": "royalblue"
    # },
    # {
    #     "name": "Aligned image 2",
    #     "path": "test/stereo_dataset_real_displacements.align.ds_fac_8.ds_fac_4.ds_fac_2.image_rec",
    #     "plot_color": "seagreen"
    # },
    #
    # # --- Stereo real disps no align --- #
    #
    # {
    #     "name": "No alignment",
    #     "path": "test/stereo_dataset_real_displacements.noalign",
    #     "plot_color": "gray"
    # },

    # --- New/Old training (without/with SanFrancisco in train set) --- #

    # {
    #     "name": "Aligned SanFrancisco After",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1.SanFrancisco.new",
    #     "plot_color": "royalblue"
    # },
    # {
    #     "name": "Aligned SanFrancisco Before",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1.SanFrancisco.old",
    #     "plot_color": "green"
    # },
    #
    # {
    #     "name": "Aligned Norfolk After",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1.Norfolk.new",
    #     "plot_color": "orange"
    # },
    # {
    #     "name": "Aligned Norfolk Before",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1.Norfolk.old",
    #     "plot_color": "tomato"
    # },

    # --- Individual images --- #

    # {
    #     "name": "Aligned SanFrancisco_01",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1.SanFrancisco_01",
    #     "plot_color": "royalblue"
    # },
    # {
    #     "name": "Aligned SanFrancisco_02",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1.SanFrancisco_02",
    #     "plot_color": "seagreen"
    # },
    # {
    #     "name": "Aligned SanFrancisco_03",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1.SanFrancisco_03",
    #     "plot_color": "tomato"
    # },
    # {
    #     "name": "Aligned Norfolk_01",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1.Norfolk_01",
    #     "plot_color": "orange"
    # },
    # {
    #     "name": "Aligned Norfolk_02",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1.Norfolk_02",
    #     "plot_color": "green"
    # },
    #
    # {
    #     "name": "Not aligned SanFrancisco_01",
    #     "path": "test/bradbury_buildings.disp_maps.SanFrancisco_01",
    #     "plot_color": "royalblue",
    #     "plot_dashes": (6, 1),
    # },
    # {
    #     "name": "Not aligned SanFrancisco_02",
    #     "path": "test/bradbury_buildings.disp_maps.SanFrancisco_02",
    #     "plot_color": "seagreen",
    #     "plot_dashes": (6, 1),
    # },
    # {
    #     "name": "Not aligned SanFrancisco_03",
    #     "path": "test/bradbury_buildings.disp_maps.SanFrancisco_03",
    #     "plot_color": "tomato",
    #     "plot_dashes": (6, 1),
    # },
    # {
    #     "name": "Not aligned Norfolk_01",
    #     "path": "test/bradbury_buildings.disp_maps.Norfolk_01",
    #     "plot_color": "orange",
    #     "plot_dashes": (6, 1),
    # },
    # {
    #     "name": "Not aligned Norfolk_02",
    #     "path": "test/bradbury_buildings.disp_maps.Norfolk_02",
    #     "plot_color": "green",
    #     "plot_dashes": (6, 1),
    # },

    # --- Ablation studies and comparison --- #

    # {
    #     "name": "No dropping of input polygons",
    #     "path": "test/bradbury_buildings.align.ds_fac_8_keep_poly_1.ds_fac_4_keep_poly_1.ds_fac_2_keep_poly_1.ds_fac_1_keep_poly_1",
    #     "plot_color": "tomato"
    # },
    {
        "name": "Zampieri et al.",
        "path": "test/bradbury_buildings.align.ds_fac_8_zampieri.ds_fac_4_zampieri.ds_fac_2_zampieri.ds_fac_1_zampieri",
        "plot_color": "black"
    },
    {
        "name": "Full method",
        "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1",
        "plot_color": "royalblue"
    },
    # {
    #     "name": "Full method ds_fac >= 2",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2",
    #     "plot_color": "orange"
    # },
    # {
    #     "name": "Full method ds_fac >= 4",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4",
    #     "plot_color": "darkorchid"
    # },
    # {
    #     "name": "Full method ds_fac = 8",
    #     "path": "test/bradbury_buildings.align.ds_fac_8",
    #     "plot_color": "green"
    # },
    # {
    #     "name": "No segmentation branch",
    #     "path": "test/bradbury_buildings.align.ds_fac_8_no_seg.ds_fac_4_no_seg.ds_fac_2_no_seg.ds_fac_1_no_seg",
    #     "plot_color": "orange"
    # },
    # {
    #     "name": "No intermediary losses",
    #     "path": "test/bradbury_buildings.align.ds_fac_8_no_interm_loss.ds_fac_4_no_interm_loss.ds_fac_2_no_interm_loss.ds_fac_1_no_interm_loss",
    #     "plot_color": "darkorchid"
    # },

    # # --- Comparison to Quicksilver --- #
    #
    # {
    #     "name": "Our model (scaling = 4)",
    #     "path": "test/bradbury_buildings.align.ds_fac_4_disp_max_16",
    #     "plot_color": "blue"
    # },
    # {
    #     "name": "Quicksilver (scaling = 4)",
    #     "path": "test/bradbury_buildings.align.ds_fac_4_disp_max_16_quicksilver",
    #     "plot_color": "seagreen"
    # },
    #
    # # --- Bradbury buildings no align --- #

    # {
    #     "name": "No alignment",
    #     "path": "test/bradbury_buildings.disp_maps",
    #     "plot_color": "gray"
    # },

    # # --- Adding the Mapping Challenge (from Crowd AI) dataset --- #

    # {
    #     "name": "ds_fac_8",
    #     "path": "test/bradbury_buildings.align.ds_fac_8",
    #     "plot_color": "blue"
    # },
    # {
    #     "name": "ds_fac_8_zampieri",
    #     "path": "test/bradbury_buildings.align.ds_fac_8_zampieri",
    #     "plot_color": "black"
    # },
    # {
    #     "name": "ds_fac_8_bradbury",
    #     "path": "test/bradbury_buildings.align.ds_fac_8_bradbury",
    #     "plot_color": "seagreen"
    # },
    # {
    #     "name": "ds_fac_8_inria",
    #     "path": "test/bradbury_buildings.align.ds_fac_8_inria",
    #     "plot_color": "tomato"
    # },
    # {
    #     "name": "ds_fac_8_mapping",
    #     "path": "test/bradbury_buildings.align.ds_fac_8_mapping",
    #     "plot_color": "orange"
    # },
    # {
    #     "name": "ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1",
    #     "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1",
    #     "plot_color": "royalblue"
    # },
]

PLOT_ALL = False
PLOT_MIN_MAX = True
PLOT_AVERAGE = True

ALPHA_MAIN = 1.0
ALPHA_MIN_MAX = 0.5
ALPHA_INDIVIDUAL = 0.2  # Default: 0.2
COLOR = 'cornflowerblue'

X_LIM = 16  # Default: 12, can go up to 16

FILEPATH = "test/accuracies.png"

# ---  --- #


def main():
    plt.figure(1, figsize=(7, 4))
    handles = []
    for source_params in SOURCE_PARAMS_LIST:
        if "plot_dashes" in source_params:
            plot_dashes = source_params["plot_dashes"]
        else:
            plot_dashes = (None, None)

        threshold_accuracies_filepath_list = python_utils.get_filepaths(source_params["path"], ACCURACIES_FILENAME_EXTENSION)

        threshold_accuracies_list = []
        for threshold_accuracies_filepath in threshold_accuracies_filepath_list:
            threshold_accuracies = np.load(threshold_accuracies_filepath).item()
            threshold_accuracies_list.append(threshold_accuracies)

        # Plot main, min and max curves
        accuracies_list = []
        for threshold_accuracies in threshold_accuracies_list:
            accuracies_list.append(threshold_accuracies["accuracies"])
        accuracies_table = np.stack(accuracies_list, axis=0)
        accuracies_min = np.min(accuracies_table, axis=0)
        accuracies_average = np.mean(accuracies_table, axis=0)
        accuracies_max = np.max(accuracies_table, axis=0)
        if PLOT_AVERAGE:
            plt.plot(threshold_accuracies_list[0]["thresholds"], accuracies_average, color=source_params["plot_color"], dashes=plot_dashes, alpha=ALPHA_MAIN, label=source_params["name"])
        if PLOT_MIN_MAX:
            plt.plot(threshold_accuracies_list[0]["thresholds"], accuracies_min, color=source_params["plot_color"], dashes=(6, 1), alpha=ALPHA_MIN_MAX, label=source_params["name"])
            plt.plot(threshold_accuracies_list[0]["thresholds"], accuracies_max, color=source_params["plot_color"], dashes=(6, 1), alpha=ALPHA_MIN_MAX, label=source_params["name"])

        if PLOT_ALL:
            # Plot all curves:
            for threshold_accuracies in threshold_accuracies_list:
                plt.plot(threshold_accuracies["thresholds"], threshold_accuracies["accuracies"],
                         color=source_params["plot_color"], dashes=plot_dashes, alpha=ALPHA_INDIVIDUAL, label=source_params["name"])

        # Legend
        handles.append(plt.Line2D([0], [0], color=source_params["plot_color"], dashes=plot_dashes))

    plt.grid(True)
    axes = plt.gca()
    axes.set_xlim([0, X_LIM])
    axes.set_ylim([0.0, 1.0])
    # plt.title("Fraction of vertices whose ground truth point distance is less than the threshold (higher is better)")
    plt.xlabel('Threshold (in pixels)')
    plt.ylabel('Fraction of vertices')
    # Add legends in top-left
    labels = [source_params["name"] for source_params in SOURCE_PARAMS_LIST]
    plt.legend(handles, labels)

    # Plot
    plt.tight_layout()
    plt.savefig(FILEPATH)
    plt.show()


if __name__ == '__main__':
    main()
