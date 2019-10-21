import sys
import skimage.transform
import skimage.io
import numpy as np

import model

sys.path.append("../../utils")
import run_utils
import polygon_utils
import print_utils


def rescale_data(image, polygons, scale):
    downsampled_image = skimage.transform.rescale(image, scale, order=3, preserve_range=True, multichannel=True, anti_aliasing=True)
    downsampled_image = downsampled_image.astype(image.dtype)
    downsampled_polygons = polygon_utils.rescale_polygon(polygons, scale)
    return downsampled_image, downsampled_polygons


def downsample_data(image, metadata, polygons, factor, reference_pixel_size):
    corrected_factor = factor * reference_pixel_size / metadata["pixelsize"]
    scale = 1 / corrected_factor
    downsampled_image, downsampled_polygons = rescale_data(image, polygons, scale)
    return downsampled_image, downsampled_polygons


def upsample_data(image, metadata, polygons, factor, reference_pixel_size):
    # TODO: test with metadata["pixelsize"] != config.REFERENCE_PIXEL_SIZE
    corrected_factor = factor * reference_pixel_size / metadata["pixelsize"]
    upsampled_image, upsampled_polygons = rescale_data(image, polygons, corrected_factor)
    return upsampled_image, upsampled_polygons


def inference(runs_dirpath, ori_image, ori_metadata, ori_disp_polygons, model_disp_max_abs_value, batch_size, scale_factor, run_name):
    # Setup run dir and load config file
    run_dir = run_utils.setup_run_dir(runs_dirpath, run_name)
    _, checkpoints_dir = run_utils.setup_run_subdirs(run_dir)

    config = run_utils.load_config(config_dirpath=run_dir)

    # Downsample
    image, disp_polygons = downsample_data(ori_image, ori_metadata, ori_disp_polygons, scale_factor, config["reference_pixel_size"])
    spatial_shape = image.shape[:2]

    # Draw displaced polygon map
    # disp_polygons_to_rasterize = []
    disp_polygons_to_rasterize = disp_polygons
    disp_polygon_map = polygon_utils.draw_polygon_map(disp_polygons_to_rasterize, spatial_shape, fill=True, edges=True,
                                                      vertices=True)

    # Compute output_res
    output_res = model.MapAlignModel.get_output_res(config["input_res"], config["pool_count"])
    # print("output_res: {}".format(output_res))

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
                                          model_disp_max_abs_value)

    pred_field_map, segmentation_image = map_align_model.inference(image, disp_polygon_map, checkpoints_dir)

    # --- align disp_polygon according to pred_field_map --- #
    # print("# --- Align disp_polygon according to pred_field_map --- #")
    aligned_disp_polygons = disp_polygons
    # First remove polygons that are not fully inside the inner_image
    padding = (spatial_shape[0] - pred_field_map.shape[0]) // 2
    bounding_box = [padding, padding, spatial_shape[0] - padding, spatial_shape[1] - padding]
    # aligned_disp_polygons = polygon_utils.filter_polygons_in_bounding_box(aligned_disp_polygons, bounding_box)  # TODO: reimplement? But also filter out ori_gt_polygons for comparaison
    aligned_disp_polygons = polygon_utils.transform_polygons_to_bounding_box_space(aligned_disp_polygons, bounding_box)
    # Then apply displacement field map to aligned_disp_polygons
    aligned_disp_polygons = polygon_utils.apply_disp_map_to_polygons(pred_field_map, aligned_disp_polygons)
    # Restore polygons to original image space
    bounding_box = [-padding, -padding, spatial_shape[0] + padding, spatial_shape[1] + padding]
    aligned_disp_polygons = polygon_utils.transform_polygons_to_bounding_box_space(aligned_disp_polygons, bounding_box)

    # Add padding to segmentation_image
    final_segmentation_image = np.zeros((spatial_shape[0], spatial_shape[1], segmentation_image.shape[2]))
    final_segmentation_image[padding:-padding, padding:-padding, :] = segmentation_image

    # --- Upsample outputs --- #
    # print("# --- Upsample outputs --- #")
    final_segmentation_image, aligned_disp_polygons = upsample_data(final_segmentation_image, ori_metadata, aligned_disp_polygons, scale_factor, config["reference_pixel_size"])

    return aligned_disp_polygons, final_segmentation_image


def multires_inference(runs_dirpath, ori_image, ori_metadata, ori_disp_polygons, model_disp_max_abs_value, batch_size, ds_fac_list, run_name_list):
    """
    Returns the last segmentation image that was computed (from the finest resolution)

    :param ori_image:
    :param ori_metadata:
    :param ori_disp_polygons:
    :param model_disp_max_abs_value:
    :param ds_fac_list:
    :param run_name_list:
    :return:
    """
    aligned_disp_polygons = ori_disp_polygons  # init
    segmentation_image = None
    # Launch the resolution chain pipeline:
    for index, (ds_fac, run_name) in enumerate(zip(ds_fac_list, run_name_list)):
        print("# --- downsampling_factor: {} --- #".format(ds_fac))
        try:
            aligned_disp_polygons, segmentation_image = inference(runs_dirpath, ori_image, ori_metadata, aligned_disp_polygons, model_disp_max_abs_value, batch_size, ds_fac, run_name)
        except ValueError as e:
            print_utils.print_warning(str(e))

    return aligned_disp_polygons, segmentation_image

