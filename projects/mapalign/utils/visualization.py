import os
import sys
import numpy as np
import cv2

current_filepath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_filepath, "../../utils"))
import python_utils
import polygon_utils

# --- Useful when code is executed inside Docker without a display: --- #
# Try importing pyplot:
display_is_available = python_utils.get_display_availability()
use_pyplot = None
if display_is_available:
    if python_utils.module_exists("matplotlib.pyplot"):
        # This means everything works with the default matplotlib backend
        import matplotlib.pyplot as plt
        use_pyplot = True
    else:
        # matplotlib.pyplot is just not available we cannot plot anything
        use_pyplot = False
else:
    # Try switching backend
    import matplotlib
    matplotlib.use('Agg')
    if python_utils.module_exists("matplotlib.pyplot"):
        # The Agg backend works, pyplot is available, we just can't display plots to the screen (they'll be saved to file anyway)
        import matplotlib.pyplot as plt
        use_pyplot = True
# --- --- #

import skimage.io

print("#--- Visualization ---#")
print("display_is_available: {}".format(display_is_available))
print("use_pyplot: {}".format(use_pyplot))


def flow_to_image(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


if use_pyplot:

    FIGURE_DICT = {}

    def fig_out(figure_name, shape, nonblocking):
        plt.margins(0)
        plt.axis('off')

        axes = plt.gca()
        axes.set_xlim([0, shape[1]])
        axes.set_ylim([0, shape[0]])

        if display_is_available:
            if nonblocking:
                plt.draw()
                plt.pause(0.001)
            else:
                plt.show()
        # plt.savefig("{}.png".format(figure_name), bbox_inches='tight', pad_inches=0)
        plt.savefig("{}.png".format(figure_name), pad_inches=0)


    def init_figures(figure_names, nonblocking=True, figsize=(4, 4)):
        for i, figure_name in enumerate(figure_names):
            fig = plt.figure(i, figsize=figsize)
            fig.canvas.set_window_title(figure_name)
            FIGURE_DICT[figure_name] = i
        if nonblocking:
            plt.ion()
    
    
    def plot_image(image):
        plt.imshow(image[:, :, :3])  # Remove extra channels if any


    def plot_example(figure_name, image, gt_polygon_map, disp_field_map=None, disp_polygon_map=None, nonblocking=True):
        patch_outer_res = image.shape[0]

        gt_polygon_map = gt_polygon_map.astype(np.float32)

        if nonblocking:
            fig = plt.figure(FIGURE_DICT[figure_name])
            plt.cla()
        plot_image(image)

        # Overlay GT polygons with 0.5 alpha
        shown_gt_polygon_map = np.zeros((patch_outer_res, patch_outer_res, 4))
        shown_gt_polygon_map[:, :, :3] = gt_polygon_map
        shown_gt_polygon_map[:, :, 3] = np.any(gt_polygon_map, axis=-1) / 2
        plt.imshow(shown_gt_polygon_map)

        if disp_polygon_map is not None:
            disp_polygon_map = disp_polygon_map.astype(np.float32)
            disp_polygon_map /= 2
            # Overlay displaced polygons with 0.5 alpha
            shown_disp_polygon_map = np.zeros((patch_outer_res, patch_outer_res, 4))
            shown_disp_polygon_map[:, :, :3] = disp_polygon_map
            shown_disp_polygon_map[:, :, 3] = np.any(disp_polygon_map, axis=-1) / 2
            plt.imshow(shown_disp_polygon_map)

        # Overlay displacement map with 0.5 alpha
        if disp_field_map is not None:
            patch_inner_res = disp_field_map.shape[0]
            patch_padding = (patch_outer_res - patch_inner_res) // 2
            shown_disp_field_map_padded = np.zeros((patch_outer_res, patch_outer_res, 4))
            shown_disp_field_map = np.empty_like(disp_field_map)
            maxi = np.max(np.abs(disp_field_map))
            shown_disp_field_map[:, :, 0] = disp_field_map[:, :, 0] / (maxi + 1e-6)
            shown_disp_field_map[:, :, 1] = disp_field_map[:, :, 1] / (maxi + 1e-6)
            shown_disp_field_map = (shown_disp_field_map + 1) / 2
            shown_disp_field_map_padded[patch_padding:-patch_padding, patch_padding:-patch_padding, 1:3] = shown_disp_field_map
            shown_disp_field_map_padded[patch_padding:-patch_padding, patch_padding:-patch_padding, 3] = 0.5
            plt.imshow(shown_disp_field_map_padded)

        # Draw quivers on displaced corners
        if disp_polygon_map is not None:
            disp_polygon_map_cropped_corners = disp_polygon_map[patch_padding:-patch_padding, patch_padding:-patch_padding, 2]
            quiver_indexes = np.where(0 < disp_polygon_map_cropped_corners.max() - 1e-1 < disp_polygon_map_cropped_corners)
            if len(quiver_indexes[0]) and len(quiver_indexes[1]):
                disp_field_map_corners = disp_field_map[quiver_indexes[0], quiver_indexes[1], :]
                plt.quiver(quiver_indexes[1] + patch_padding, quiver_indexes[0] + patch_padding, disp_field_map_corners[:, 1],
                           disp_field_map_corners[:, 0], scale=1, scale_units="xy", angles="xy", width=0.005, color="purple")

        fig_out(figure_name, image.shape, nonblocking)


    def plot_example_homography(figure_name, image, aligned_polygon_raster, misaligned_polygon_raster, nonblocking=True):
        patch_res = image.shape[0]

        aligned_polygon_raster = aligned_polygon_raster.astype(np.float32)
        misaligned_polygon_raster = misaligned_polygon_raster.astype(np.float32)
        # Darken image and gt_polygon_map

        if nonblocking:
            fig = plt.figure(FIGURE_DICT[figure_name])
            plt.cla()
        plot_image(image)

        # Overlay aligned_polygon_raster with 0.5 alpha
        shown_aligned_polygon_raster = np.zeros((patch_res, patch_res, 4))
        shown_aligned_polygon_raster[:, :, 1] = aligned_polygon_raster[:, :, 0]
        shown_aligned_polygon_raster[:, :, 3] = aligned_polygon_raster[:, :, 0] / 8
        plt.imshow(shown_aligned_polygon_raster)

        # Overlay misaligned_polygon_raster with 0.5 alpha
        shown_misaligned_polygon_raster = np.zeros((patch_res, patch_res, 4))
        shown_misaligned_polygon_raster[:, :, 0] = misaligned_polygon_raster[:, :, 0]
        shown_misaligned_polygon_raster[:, :, 3] = misaligned_polygon_raster[:, :, 0] / 8
        plt.imshow(shown_misaligned_polygon_raster)

        fig_out(figure_name, image.shape, nonblocking)


    def plot_polygons(polygons, color):
        # print("plot_polygons(polygons, color)")  # TODO: remove
        for i, polygon in enumerate(polygons):
            # Remove coordinates after nans
            indexes_of_nans = np.where(np.isnan(polygon[:, 0]))[0]
            if len(indexes_of_nans):
                polygon_nans_crop = polygon[:indexes_of_nans[-1], :]
                polygon_utils.plot_polygon(polygon_nans_crop, color=color, draw_labels=False, indexing="ij")
            else:
                polygon_utils.plot_polygon(polygon, color=color, draw_labels=False, indexing="ij")

            # if 10 < i:  # TODO: remove
            #     break  # TODO: remove


    def plot_example_polygons(figure_name, image, gt_polygons, disp_polygons=None, aligned_disp_polygons=None, nonblocking=True):

        if nonblocking:
            fig = plt.figure(FIGURE_DICT[figure_name])
            plt.cla()
        plot_image(image)

        # Draw gt polygons
        plot_polygons(gt_polygons, "green")
        if disp_polygons is not None:
            plot_polygons(disp_polygons, "red")
        if aligned_disp_polygons is not None:
            plot_polygons(aligned_disp_polygons, "blue")

        fig_out(figure_name, image.shape, nonblocking)


    def plot_seg(figure_name, image, seg, nonblocking=True):
        patch_outer_res = image.shape[0]
        patch_inner_res = seg.shape[0]
        patch_padding = (patch_outer_res - patch_inner_res) // 2

        if 3 < seg.shape[2]:
            seg = seg[:, :, 1:4]

        # seg = seg.astype(np.float32)

        # print(seg.dtype)
        # print(seg.shape)
        # print(seg.min())
        # print(seg.max())

        if nonblocking:
            fig = plt.figure(FIGURE_DICT[figure_name])
            plt.cla()
        plot_image(image)

        # Overlay GT polygons
        shown_seg = np.zeros((patch_outer_res, patch_outer_res, 4))
        if 0 < patch_padding:
            shown_seg[patch_padding:-patch_padding, patch_padding:-patch_padding, :3] = seg[:, :, :]
            shown_seg[patch_padding:-patch_padding, patch_padding:-patch_padding, 3] = np.clip(np.sum(seg[:, :, :], axis=-1), 0, 1)
        else:
            shown_seg[:, :, :3] = seg[:, :, :]
            shown_seg[:, :, 3] = np.clip(
                np.sum(seg[:, :, :], axis=-1), 0, 1)
        plt.imshow(shown_seg)

        fig_out(figure_name, image.shape, nonblocking)


    def plot_field_map(figure_name, field_map, nonblocking=True):
        assert len(field_map.shape) == 3 and field_map.shape[2] == 2, "field_map should have 3 dimensions like so: [height, width, 2]"
        from mpl_toolkits.mplot3d import Axes3D

        row = np.linspace(0, 1, field_map.shape[0])
        col = np.linspace(0, 1, field_map.shape[1])
        rr, cc = np.meshgrid(row, col, indexing='ij')

        fig = plt.figure(figsize=(18, 9))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(rr, cc, field_map[:, :, 0], rstride=3, cstride=3, linewidth=1, antialiased=True)

        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(rr, cc, field_map[:, :, 1], rstride=3, cstride=3, linewidth=1, antialiased=True)

        plt.savefig("{}.png".format(figure_name), pad_inches=0)

else:

    def init_figures(figure_names, nonblocking=True):
        print("Graphical interface (matplotlib.pyplot) is not available. Will print out relevant values instead of "
              "plotting.")


    def plot_example(figure_name, image, gt_polygon_map, disp_field_map, disp_polygon_map, nonblocking=True):
        print(figure_name)


    def plot_example_homography(figure_name, image, aligned_polygon_raster, misaligned_polygon_raster,
                                nonblocking=True):
        print(figure_name)


    def plot_example_polygons(figure_name, image, gt_polygons, disp_polygons, aligned_disp_polygons=None, nonblocking=True):
        print(figure_name)
        # print("gt_polygons:")
        # print(gt_polygons)
        # print("aligned_disp_polygons:")
        # print(aligned_disp_polygons)


    def plot_seg(figure_name, image, seg, nonblocking=True):
        print(figure_name)


def plot_batch(figure_names, image_batch, gt_polygon_map_batch, disp_field_map_batches, disp_polygon_map_batch, nonblocking=True):
    assert len(figure_names) == len(disp_field_map_batches)

    # batch_size = gt_polygon_map_batch.shape[0]
    # index = random.randrange(batch_size)
    index = 0

    for figure_name, disp_field_map_batch in zip(figure_names, disp_field_map_batches):
        plot_example(figure_name, image_batch[index], gt_polygon_map_batch[index], disp_field_map_batch[index], disp_polygon_map_batch[index], nonblocking=nonblocking)


def plot_batch_polygons(figure_name, image_batch, gt_polygons_batch, disp_polygons_batch, aligned_disp_polygons_batch, nonblocking=True):

    # batch_size = image_batch.shape[0]
    # index = random.randrange(batch_size)
    index = 0

    plot_example_polygons(figure_name, image_batch[index], gt_polygons_batch[index], disp_polygons_batch[index], aligned_disp_polygons_batch[index], nonblocking=nonblocking)


def plot_batch_seg(figure_name, image_batch, seg_batch):
    # batch_size = image_batch.shape[0]
    # index = random.randrange(batch_size)
    index = 0

    plot_seg(figure_name, image_batch[index], seg_batch[index])


def save_plot_image_polygons(filepath, ori_image, ori_gt_polygons, disp_polygons, aligned_disp_polygons, line_width=1):
    spatial_shape = ori_image.shape[:2]
    ori_gt_polygons_map = polygon_utils.draw_polygon_map(ori_gt_polygons, spatial_shape, fill=False, edges=True,
                                                         vertices=False, line_width=line_width)
    disp_polygons_map = polygon_utils.draw_polygon_map(disp_polygons, spatial_shape, fill=False, edges=True,
                                                       vertices=False, line_width=line_width)
    aligned_disp_polygons_map = polygon_utils.draw_polygon_map(aligned_disp_polygons, spatial_shape, fill=False,
                                                               edges=True, vertices=False, line_width=line_width)

    output_image = ori_image[:, :, :3]  # Keep first 3 channels
    output_image = output_image.astype(np.float64)
    output_image[np.where(0 < ori_gt_polygons_map[:, :, 0])] = np.array([0, 255, 0])
    output_image[np.where(0 < disp_polygons_map[:, :, 0])] = np.array([255, 0, 0])
    output_image[np.where(0 < aligned_disp_polygons_map[:, :, 0])] = np.array([0, 0, 255])
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
