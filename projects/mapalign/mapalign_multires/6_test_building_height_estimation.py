import os.path
import numpy as np
import matplotlib.pyplot as plt

# --- Params --- #

BINS = 50

INPUT_BASE_DIRPATH = "3d_buildings/leibnitz"


# ---  --- #


def compute_polygon_area(polygon):
    return 0.5 * np.abs(
        np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))


def main(input_base_dirpath, bins):
    # --- Loading data --- #
    polygon_array = np.load(os.path.join(input_base_dirpath, "polygons.npy"))
    gt_heights_array = np.load(os.path.join(input_base_dirpath, "gt_heights.npy"))
    pred_heights_array = np.load(os.path.join(input_base_dirpath, "pred_heights.npy"))

    # # Exclude buildings with pred_height < 3:
    # keep_indices = np.where(3 <= pred_heights_array)
    # polygon_array = polygon_array[keep_indices]
    # gt_heights_array = gt_heights_array[keep_indices]
    # pred_heights_array = pred_heights_array[keep_indices]

    mean_gt_height = np.mean(gt_heights_array)
    print("mean_gt_height:")
    print(mean_gt_height)
    mean_pred_height = np.mean(pred_heights_array)
    print("mean_pred_height:")
    print(mean_pred_height)

    diff_array = np.abs(gt_heights_array - pred_heights_array)
    mean_diff = np.mean(diff_array)
    print("mean_diff:")
    print(mean_diff)

    # --- Plot area/height pairs --- #
    polygon_area_list = [compute_polygon_area(polygon) for polygon in polygon_array]

    plt.scatter(polygon_area_list, diff_array, s=1)
    # plt.scatter(polygon_area_list, pred_heights_array, s=1)

    plt.xlabel('Area')
    plt.xlim([0, 1000])
    plt.ylabel('Height difference')
    plt.title('Height difference relative to area')
    plt.grid(True)
    plt.show()


    # --- Plot histograms --- #
    # pred_heights_array_int = np.round(pred_heights_array).astype(int)

    plt.hist(gt_heights_array, bins, alpha=0.5)
    plt.hist(pred_heights_array, bins, alpha=0.5)

    plt.xlabel('Height')
    plt.ylabel('Count')
    plt.title('Histogram of building heights')
    plt.grid(True)
    plt.show()

    # --- Measure results --- #


if __name__ == "__main__":
    main(INPUT_BASE_DIRPATH, BINS)
