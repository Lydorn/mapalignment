import sys
import os
import numpy as np

# Import visualization first as it might change the matplotlib backend
sys.path.append("../utils")
import visualization

import multires_pipeline

sys.path.append("../evaluate_funcs")
import evaluate_utils

sys.path.append("../../utils")
import polygon_utils
import geo_utils


def generate_disp_data(normed_disp_field_maps, gt_polygons, disp_max_abs_value):
    scaled_disp_field_maps = normed_disp_field_maps * disp_max_abs_value
    disp_polygons_list = polygon_utils.apply_displacement_fields_to_polygons(gt_polygons,
                                                                             scaled_disp_field_maps)
    return disp_polygons_list


def measure_accuracies(polygons_1, polygons_2, thresholds, filepath):
    accuracies = evaluate_utils.compute_threshold_accuracies([polygons_1], [polygons_2], thresholds)
    threshold_accuracies = {
        "thresholds": thresholds,
        "accuracies": accuracies,
    }
    np.save(filepath, threshold_accuracies)
    return accuracies


def measure_ious(gt_polygons, pred_seg, thresholds, filepath):
    padding = (220 - 100) // 2  # TODO: retrieve this programmatically
    gt_seg = polygon_utils.draw_polygon_map(gt_polygons, pred_seg.shape[:2], fill=True, edges=True, vertices=True)
    # Crop both images to remove margin
    pred_seg = pred_seg[padding:-padding, padding:-padding, :]
    gt_seg = gt_seg[padding:-padding, padding:-padding, :]
    # Reduce channels to single max value
    pred_seg = np.max(pred_seg, axis=-1)
    gt_seg = np.max(gt_seg, axis=-1)
    gt_mask = gt_seg.astype(np.bool)
    # Create thresholded masks
    ious = []
    for threshold in thresholds:
        pred_mask = threshold < pred_seg

        # import skimage.io
        # skimage.io.imsave("pred_mask_{:0.02}.png".format(threshold), pred_mask * 255)

        intersection = pred_mask & gt_mask
        union = pred_mask | gt_mask
        intersection_count = np.sum(intersection)
        union_count = np.sum(union)
        if 0 < union_count:
            iou = intersection_count / float(union_count)
        else:
            iou = np.nan
        ious.append(iou)

    thresholds_ious = {
        "thresholds": thresholds,
        "ious": ious,
    }
    np.save(filepath, thresholds_ious)
    return ious


def test(ori_image, ori_metadata, ori_gt_polygons, ori_disp_polygons, batch_size, ds_fac_list, run_name_list,
         model_disp_max_abs_value, thresholds, test_output_dir, output_name, output_shapefiles=True,
         properties_list=None):
    if output_shapefiles:
        assert properties_list is not None and len(ori_disp_polygons) == len(
            properties_list), "ori_disp_polygons and properties_list should have the same length"
    polygons_image_plot_filename_format = "{}.polygons.png"
    shapefile_filename_format = "{}.{}_polygons.shp"
    segmentation_image_plot_filename_format = "{}.segmentation.png"
    accuracies_filename_format = "{}.accuracy.npy"

    # --- Run the model --- #
    print("# --- Run the model --- #")
    aligned_disp_polygons, segmentation_image = multires_pipeline.multires_inference(ori_image, ori_metadata,
                                                                                     ori_disp_polygons,
                                                                                     model_disp_max_abs_value,
                                                                                     batch_size, ds_fac_list,
                                                                                     run_name_list)
    # aligned_disp_polygons = ori_disp_polygons
    # segmentation_image = np.zeros((ori_image.shape[0], ori_image.shape[1], 4))

    # --- Save segmentation_output --- #
    print("# --- Save segmentation_output --- #")
    plot_segmentation_image_filename = segmentation_image_plot_filename_format.format(output_name)
    plot_segmentation_image_filepath = os.path.join(test_output_dir, plot_segmentation_image_filename)
    visualization.save_plot_segmentation_image(plot_segmentation_image_filepath, segmentation_image)

    # --- Save polygons plot --- #
    plot_image_filename = polygons_image_plot_filename_format.format(output_name)
    plot_image_filepath = os.path.join(test_output_dir, plot_image_filename)
    visualization.save_plot_image_polygons(plot_image_filepath, ori_image, ori_gt_polygons, ori_disp_polygons,
                                           aligned_disp_polygons)
    # visualization.save_plot_image_polygons(plot_image_filepath, ori_image, [], ori_disp_polygons,
    #                                        aligned_disp_polygons)

    # --- Save polygons as shapefiles --- #
    if output_shapefiles:
        print("# --- Save polygons as shapefiles --- #")
        output_shapefile_filename = shapefile_filename_format.format(output_name, "ori")
        output_shapefile_filepath = os.path.join(test_output_dir, output_shapefile_filename)
        geo_utils.save_shapefile_from_polygons(ori_gt_polygons, ori_metadata["filepath"],
                                               output_shapefile_filepath, properties_list=properties_list)
        output_shapefile_filename = shapefile_filename_format.format(output_name, "misaligned")
        output_shapefile_filepath = os.path.join(test_output_dir, output_shapefile_filename)
        geo_utils.save_shapefile_from_polygons(ori_disp_polygons, ori_metadata["filepath"],
                                               output_shapefile_filepath,  properties_list=properties_list)
        output_shapefile_filename = shapefile_filename_format.format(output_name, "aligned")
        output_shapefile_filepath = os.path.join(test_output_dir, output_shapefile_filename)
        geo_utils.save_shapefile_from_polygons(aligned_disp_polygons, ori_metadata["filepath"],
                                               output_shapefile_filepath, properties_list=properties_list)

    # --- Measure accuracies --- #
    if len(ori_gt_polygons) == len(aligned_disp_polygons):
        print("# --- Measure accuracies --- #")
        accuracies_filename = accuracies_filename_format.format(output_name)
        accuracies_filepath = os.path.join(test_output_dir, accuracies_filename)
        accuracies = measure_accuracies(ori_gt_polygons, aligned_disp_polygons, thresholds, accuracies_filepath)
        print(accuracies)


def test_image_with_gt_and_disp_polygons(image_name, ori_image, ori_metadata, ori_gt_polygons, ori_disp_polygons,
                                         ori_disp_properties_list,
                                         batch_size, ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds,
                                         test_output_dir, output_shapefiles=True):
    ori_gt_polygons = polygon_utils.polygons_remove_holes(ori_gt_polygons)  # TODO: Remove
    ori_gt_polygons = polygon_utils.simplify_polygons(ori_gt_polygons, tolerance=1)  # Remove redundant vertices
    ori_disp_polygons = polygon_utils.polygons_remove_holes(ori_disp_polygons)  # TODO: Remove
    ori_disp_polygons = polygon_utils.simplify_polygons(ori_disp_polygons, tolerance=1)  # Remove redundant vertices

    output_name = image_name
    test(ori_image, ori_metadata, ori_gt_polygons, ori_disp_polygons, batch_size, ds_fac_list, run_name_list,
         model_disp_max_abs_value, thresholds, test_output_dir, output_name, output_shapefiles=output_shapefiles,
         properties_list=ori_disp_properties_list)


def test_image_with_gt_polygons_and_disp_maps(image_name, ori_image, ori_metadata, ori_gt_polygons,
                                              ori_normed_disp_field_maps, disp_max_abs_value, batch_size,
                                              ds_fac_list, run_name_list, model_disp_max_abs_value, thresholds,
                                              test_output_dir,
                                              output_shapefiles=True):
    output_name_format = "{}.disp_{:02d}"

    ori_gt_polygons = polygon_utils.polygons_remove_holes(ori_gt_polygons)  # TODO: Remove
    # Remove redundant vertices
    ori_gt_polygons = polygon_utils.simplify_polygons(ori_gt_polygons, tolerance=1)

    disp_polygons_list = generate_disp_data(ori_normed_disp_field_maps, ori_gt_polygons,
                                            disp_max_abs_value)

    for i in range(len(disp_polygons_list)):
        print("# --- Testing with disp {:02d} --- #".format(i))
        disp_polygons = disp_polygons_list[i]
        output_name = output_name_format.format(image_name, i)
        test(ori_image, ori_metadata, ori_gt_polygons, disp_polygons, batch_size, ds_fac_list,
             run_name_list,
             model_disp_max_abs_value, thresholds, test_output_dir, output_name, output_shapefiles=output_shapefiles)


def test_detect_new_buildings(image_name, ori_image, ori_metadata, ori_gt_polygons, batch_size, ds_fac_list,
                              run_name_list, model_disp_max_abs_value, polygonization_params, thresholds,
                              test_output_dir, output_shapefiles=True):
    ori_gt_polygons = polygon_utils.polygons_remove_holes(ori_gt_polygons)  # TODO: Remove
    ori_gt_polygons = polygon_utils.simplify_polygons(ori_gt_polygons, tolerance=1)  # Remove redundant vertices
    ori_disp_polygons = []  # Don't input any polygons

    output_name = image_name
    seg_image_plot_filename_format = "{}.segmentation.png"
    ious_filename_format = "{}.iou.npy"
    new_polygons_filename_format = "{}.new_polygons.npy"
    aligned_new_polygons_filename_format = "{}.aligned_new_polygons.npy"
    polygons_image_plot_filename_format = "{}.polygons.png"

    shapefile_filename_format = "{}.{}_polygons.shp"
    accuracies_filename_format = "{}.accuracy.npy"

    # --- Get the segmentation output --- #
    seg_ds_fac_list = ds_fac_list[-1:]
    seg_run_name_list = run_name_list[-1:]
    print("# --- Run the model --- #")
    _, segmentation_image = multires_pipeline.multires_inference(ori_image, ori_metadata,
                                                                 ori_disp_polygons,
                                                                 model_disp_max_abs_value,
                                                                 batch_size, seg_ds_fac_list,
                                                                 seg_run_name_list)
    # segmentation_image = np.zeros((ori_image.shape[0], ori_image.shape[1], 4))
    print("# --- Save segmentation_output --- #")
    plot_segmentation_image_filename = seg_image_plot_filename_format.format(output_name)
    plot_segmentation_image_filepath = os.path.join(test_output_dir, plot_segmentation_image_filename)
    visualization.save_plot_segmentation_image(plot_segmentation_image_filepath, segmentation_image)

    seg_image = segmentation_image[:, :, 1:]  # Remove background channel

    # --- Measure IoUs --- #
    print("# --- Measure accuracies --- #")
    print(seg_image.min())
    print(seg_image.max())
    iou_thresholds = np.arange(0, 1.01, 0.01)
    ious_filename = ious_filename_format.format(output_name)
    ious_filepath = os.path.join(test_output_dir, ious_filename)
    ious = measure_ious(ori_gt_polygons, seg_image, iou_thresholds, ious_filepath)
    print("IoUs:")
    print(ious)

    # --- Polygonize segmentation --- #
    print("# --- Polygonize segmentation --- #")

    # TODO: remove:
    # seg_image_filepath = "test/bradbury_buildings.1_double.only_seg/SanFrancisco_01.disp_00.segmentation.png"
    # seg_image = image_utils.load_image(seg_image_filepath)
    # seg_image = seg_image / 255

    fill_threshold = polygonization_params["fill_threshold"]
    outline_threshold = polygonization_params["outline_threshold"]
    selem_width = polygonization_params["selem_width"]
    iterations = polygonization_params["iterations"]
    # new_polygons = polygonize_buildings.find_building_contours_from_seg(seg_image, fill_threshold,
    #                                                                     outline_threshold, selem_width, iterations)
    # print("# --- Save new polygons--- #")
    # new_polygons_filename = new_polygons_filename_format.format(output_name)
    # new_polygons_filepath = os.path.join(test_output_dir, new_polygons_filename)
    # np.save(new_polygons_filepath, new_polygons)
    #
    # # --- Align new polygons --- #
    # print("# --- Align new polygons --- #")
    # print("# --- Run the model --- #")
    # aligned_new_polygons = new_polygons
    # aligned_new_polygons, segmentation_image = multires_pipeline.multires_inference(ori_image, ori_metadata,
    #                                                                                 aligned_new_polygons,
    #                                                                                 model_disp_max_abs_value,
    #                                                                                 batch_size, ds_fac_list,
    #                                                                                run_name_list)
    # # for i in range(10):
    # #     aligned_new_polygons, segmentation_image = multires_pipeline.multires_inference(ori_image, ori_metadata,
    # #                                                                                     aligned_new_polygons,
    # #                                                                                     model_disp_max_abs_value,
    # #                                                                                     batch_size, ds_fac_list[-1:],
    # #                                                                                     run_name_list[-1:])
    # print("# --- Save aligned new polygons--- #")
    # aligned_new_polygons_filename = aligned_new_polygons_filename_format.format(output_name)
    # aligned_new_polygons_filepath = os.path.join(test_output_dir, aligned_new_polygons_filename)
    # np.save(aligned_new_polygons_filepath, aligned_new_polygons)
    # print("# --- Save polygons plot--- #")
    # plot_image_filename = polygons_image_plot_filename_format.format(output_name)
    # plot_image_filepath = os.path.join(test_output_dir, plot_image_filename)
    # visualization.save_plot_image_polygons(plot_image_filepath, ori_image, ori_gt_polygons, new_polygons,
    #                                        aligned_new_polygons)
    #
    # # --- Save polygons as shapefiles --- #
    # if output_shapefiles:
    #     print("# --- Save polygons as shapefiles --- #")
    #     output_shapefile_filename = shapefile_filename_format.format(output_name, "new_polygons")
    #     output_shapefile_filepath = os.path.join(test_output_dir, output_shapefile_filename)
    #     geo_utils.save_shapefile_from_polygons(new_polygons, ori_metadata["filepath"], output_shapefile_filepath)
    #     output_shapefile_filename = shapefile_filename_format.format(output_name, "aligned_new_polygons")
    #     output_shapefile_filepath = os.path.join(test_output_dir, output_shapefile_filename)
    #     geo_utils.save_shapefile_from_polygons(aligned_new_polygons, ori_metadata["filepath"],
    #                                            output_shapefile_filepath)

    # # --- Measure accuracies --- #
    # print("# --- Measure accuracies --- #")
    # accuracies_filename = accuracies_filename_format.format(output_name)
    # accuracies_filepath = os.path.join(test_output_dir, accuracies_filename)
    # accuracies = measure_accuracies(ori_gt_polygons, new_polygons, thresholds, accuracies_filepath)
    # print("New polygons:")
    # print(accuracies)
    #
    # accuracies_filename = accuracies_filename_format.format(output_name)
    # accuracies_filepath = os.path.join(test_output_dir + ".no_align", accuracies_filename)
    # integer_thresholds = [threshold for threshold in thresholds if (int(threshold) == threshold)]
    # accuracies = measure_accuracies(ori_gt_polygons, aligned_new_polygons, integer_thresholds, accuracies_filepath)
    # print("Aligned new polygons:")
    # print(accuracies)


def main():
    pass


if __name__ == '__main__':
    main()
