import sys

import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../utils")
import python_utils

# --- Params --- #
ACCURACIES_FILENAME_EXTENSION = ".accuracy.npy"

SOURCE_PARAMS_LIST = [
    {
        "name": "Full method",
        "path": "test/bradbury_buildings.align.ds_fac_8.ds_fac_4.ds_fac_2.ds_fac_1",
        "plot_color": "royalblue"
    },

    # --- Bradbury buildings no align --- #

    {
        "name": "No alignment",
        "path": "test/bradbury_buildings.disp_maps",
        "plot_color": "gray"
    },
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
