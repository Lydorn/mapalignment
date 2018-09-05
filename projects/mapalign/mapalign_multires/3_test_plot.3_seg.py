import sys

import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../utils")
import python_utils

# --- Params --- #
IOUS_FILENAME_EXTENSION = ".iou.npy"

SOURCE_PARAMS_LIST = [
    {
        "name": "Full method",
        "path": "test/bradbury_buildings.seg.ds_fac_1_input_poly_coef_1",
        "plot_color": "blue"
    },
    {
        "name": "Full method ds_fac=1 input_poly_coef 0.1",
        "path": "test/bradbury_buildings.seg.ds_fac_1",
        "plot_color": "darkorchid"
    },
    # {
    #     "name": "No intermediary losses",
    #     "path": "test/bradbury_buildings.seg.ds_fac_1_no_interm_loss",
    #     "plot_color": "red"
    # },
    # {
    #     "name": "No dropping of input polygons",
    #     "path": "test/bradbury_buildings.seg.ds_fac_1_keep_poly_1",
    #     "plot_color": "darksalmon"
    # },

    # {
    #     "name": "Full method ds_fac=4",
    #     "path": "test/bradbury_buildings.seg.ds_fac_4",
    #     "plot_color": "blue"
    # },
    # # {
    # #     "name": "Full method ds_fac=4 input_poly_coef_1",
    # #     "path": "test/bradbury_buildings.seg.ds_fac_4_input_poly_coef_1",
    # #     "plot_color": "darkorchid"
    # # },
    # {
    #     "name": "Full method ds_fac=4 keep_poly_1",
    #     "path": "test/bradbury_buildings.seg.ds_fac_4_keep_poly_1",
    #     "plot_color": "darksalmon"
    # },
    # {
    #     "name": "Full method ds_fac=4 no_interm_loss",
    #     "path": "test/bradbury_buildings.seg.ds_fac_4_no_interm_loss",
    #     "plot_color": "red"
    # },
]

ALPHA_MAIN = 1.0
ALPHA_INDIVIDUAL = 0.2
COLOR = 'cornflowerblue'
FILEPATH = "test/ious_compare.png"

# ---  --- #


def main():
    plt.figure(1, figsize=(4, 4))

    for source_params in SOURCE_PARAMS_LIST:
        thresholds_ious_filepath_list = python_utils.get_filepaths(source_params["path"], IOUS_FILENAME_EXTENSION)
        print(thresholds_ious_filepath_list)

        thresholds_ious_list = []
        for thresholds_ious_filepath in thresholds_ious_filepath_list:
            thresholds_ious = np.load(thresholds_ious_filepath).item()
            thresholds_ious_list.append(thresholds_ious)

        print(thresholds_ious_list)

        # Plot main, min and max curves
        ious_list = []
        for thresholds_ious in thresholds_ious_list:
            ious_list.append(thresholds_ious["ious"])
        ious_table = np.stack(ious_list, axis=0)
        ious_average = np.mean(ious_table, axis=0)
        plt.plot(thresholds_ious_list[0]["thresholds"], ious_average, color=source_params["plot_color"], alpha=ALPHA_MAIN, label=source_params["name"])

        # Plot all curves:
        for thresholds_ious in thresholds_ious_list:
            plt.plot(thresholds_ious["thresholds"], thresholds_ious["ious"],
                     color=source_params["plot_color"], alpha=ALPHA_INDIVIDUAL, label=source_params["name"])

    plt.grid(True)
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([-0.01, 1.0])
    # plt.title("IoU relative to the mask threshold")
    plt.xlabel('Mask threshold')
    plt.ylabel('IoU')
    # Add legends in top-left
    handles = [plt.Line2D([0], [0], color=source_params["plot_color"]) for source_params in SOURCE_PARAMS_LIST]
    labels = [source_params["name"] for source_params in SOURCE_PARAMS_LIST]
    plt.legend(handles, labels)

    # Plot
    plt.tight_layout()
    plt.savefig(FILEPATH)
    plt.show()


if __name__ == '__main__':
    main()
