import sys

import numpy as np

sys.path.append("../../utils")
import polygon_utils


def compute_batch_polygon_distances(gt_polygons_batch, aligned_disp_polygons_batch):
    # Compute distances
    distances = np.sqrt(np.sum(np.square(aligned_disp_polygons_batch - gt_polygons_batch), axis=-1))

    min = np.nanmin(distances)
    mean = np.nanmean(distances)
    max = np.nanmax(distances)

    return min, mean, max


def compute_threshold_accuracies(gt_vertices_batch, pred_vertices_batch, thresholds):
    stripped_gt_polygons_list = []
    stripped_pred_polygons_list = []

    for gt_vertices, pred_vertices in zip(gt_vertices_batch, pred_vertices_batch):
        for gt_polygon, pred_polygon in zip(gt_vertices, pred_vertices):
            # Find first nan occurance
            nan_indices = np.where(np.isnan(gt_polygon[:, 0]))[0]
            if len(nan_indices):
                nan_index = nan_indices[0]
                if nan_index:
                    gt_polygon = gt_polygon[:nan_index, :]
                    pred_polygon = pred_polygon[:nan_index, :]
                else:
                    # Empty polygon, break the for loop
                    break
            gt_polygon = polygon_utils.strip_redundant_vertex(gt_polygon, epsilon=1e-3)
            pred_polygon = polygon_utils.strip_redundant_vertex(pred_polygon, epsilon=1e-3)
            stripped_gt_polygons_list.append(gt_polygon)
            stripped_pred_polygons_list.append(pred_polygon)

    if len(stripped_gt_polygons_list) == 0 or len(stripped_pred_polygons_list) == 0:
        return []

    stripped_gt_polygons = np.concatenate(stripped_gt_polygons_list)
    stripped_pred_polygons = np.concatenate(stripped_pred_polygons_list)

    distances = np.sqrt(np.sum(np.square(stripped_gt_polygons - stripped_pred_polygons), axis=-1))

    # Compute thresholds count
    threshold_accuracies = []
    for threshold in thresholds:
        accuracy = np.sum(distances <= threshold) / distances.size
        threshold_accuracies.append(accuracy)
    return threshold_accuracies


if __name__ == '__main__':
    batch_size = 1
    poly_count = 3
    vertex_count = 4
    gt_vertices = np.zeros((batch_size, poly_count, vertex_count, 2))
    gt_vertices[0, 0, 0, :] = [1, 2]
    gt_vertices[0, 0, 1, :] = [3, 4]
    gt_vertices[0, 0, 2, :] = np.nan
    gt_vertices[0, 1, 0, :] = np.nan
    pred_vertices = np.zeros((batch_size, poly_count, vertex_count, 2))
    pred_vertices[0, 0, 0, :] = [1, 2]
    pred_vertices[0, 0, 1, :] = [3, 4]
    pred_vertices[0, 0, 2, :] = np.nan
    pred_vertices[0, 1, 0, :] = np.nan
    thresholds = [1, 2, 3, 4, 5, 6, 7, 8]

    threshold_accuracies = compute_threshold_accuracies(gt_vertices, pred_vertices, thresholds)
    print("threshold_accuracies = {}".format(threshold_accuracies))
