import sys
import numpy as np

sys.path.append("../../utils")
import polygon_utils

import skimage.io


def save_plot_image_polygon(filepath, image, polygons):
    spatial_shape = image.shape[:2]
    polygons_map = polygon_utils.draw_polygon_map(polygons, spatial_shape, fill=False, edges=True,
                                                         vertices=False, line_width=1)

    output_image = image[:, :, :3]  # Keep first 3 channels
    output_image = output_image.astype(np.float64)
    output_image[np.where(0 < polygons_map[:, :, 0])] = np.array([0, 0, 255])
    # output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(np.uint8)

    skimage.io.imsave(filepath, output_image)


def save_plot_segmentation_image(filepath, segmentation_image):
    output_image = np.zeros((segmentation_image.shape[0], segmentation_image.shape[1], 4))
    output_image[:, :, :3] = segmentation_image[:, :, 1:4]  # Remove background channel
    output_image[:, :, 3] = np.sum(segmentation_image[:, :, 1:4], axis=-1)  # Add alpha

    output_image = output_image * 255
    output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(np.uint8)

    skimage.io.imsave(filepath, output_image)
