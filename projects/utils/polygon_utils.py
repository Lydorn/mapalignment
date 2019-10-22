import math
import random
import numpy as np
import scipy.spatial
from PIL import Image, ImageDraw, ImageFilter

import skimage

import python_utils

if python_utils.module_exists("skimage.measure"):
    from skimage.measure import approximate_polygon

if python_utils.module_exists("shapely"):
    from shapely import geometry


def is_polygon_clockwise(polygon):
    rolled_polygon = np.roll(polygon, shift=1, axis=0)
    double_signed_area = np.sum((rolled_polygon[:, 0] - polygon[:, 0]) * (rolled_polygon[:, 1] + polygon[:, 1]))
    if 0 < double_signed_area:
        return True
    else:
        return False


def orient_polygon(polygon, orientation="CW"):
    poly_is_orientated_cw = is_polygon_clockwise(polygon)
    if (poly_is_orientated_cw and orientation == "CCW") or (not poly_is_orientated_cw and orientation == "CW"):
        return np.flip(polygon, axis=0)
    else:
        return polygon


def orient_polygons(polygons, orientation="CW"):
    return [orient_polygon(polygon, orientation=orientation) for polygon in polygons]


def raster_to_polygon(image, vertex_count):
    contours = skimage.measure.find_contours(image, 0.5)
    contour = np.empty_like(contours[0])
    contour[:, 0] = contours[0][:, 1]
    contour[:, 1] = contours[0][:, 0]

    # Simplify until vertex_count
    tolerance = 0.1
    tolerance_step = 0.1
    simplified_contour = contour
    while 1 + vertex_count < len(simplified_contour):
        simplified_contour = approximate_polygon(contour, tolerance=tolerance)
        tolerance += tolerance_step

    simplified_contour = simplified_contour[:-1]

    # plt.imshow(image, cmap="gray")
    # plot_polygon(simplified_contour, draw_labels=False)
    # plt.show()

    return simplified_contour


def l2diffs(polygon1, polygon2):
    """
    Computes vertex-wise L2 difference between the two polygons.
    As the two polygons may not have the same starting vertex,
    all shifts are considred and the shift resulting in the minimum mean L2 difference is chosen

    :param polygon1:
    :param polygon2:
    :return:
    """
    # Make polygons of equal length
    if len(polygon1) != len(polygon2):
        while len(polygon1) < len(polygon2):
            polygon1 = np.append(polygon1, [polygon1[-1, :]], axis=0)
        while len(polygon2) < len(polygon1):
            polygon2 = np.append(polygon2, [polygon2[-1, :]], axis=0)
    vertex_count = len(polygon1)

    def naive_l2diffs(polygon1, polygon2):
        naive_l2diffs_result = np.sqrt(np.power(np.sum(polygon1 - polygon2, axis=1), 2))
        return naive_l2diffs_result

    min_l2_diffs = naive_l2diffs(polygon1, polygon2)
    min_mean_l2_diffs = np.mean(min_l2_diffs, axis=0)
    for i in range(1, vertex_count):
        current_naive_l2diffs = naive_l2diffs(np.roll(polygon1, shift=i, axis=0), polygon2)
        current_naive_mean_l2diffs = np.mean(current_naive_l2diffs, axis=0)
        if current_naive_mean_l2diffs < min_mean_l2_diffs:
            min_l2_diffs = current_naive_l2diffs
            min_mean_l2_diffs = current_naive_mean_l2diffs
    return min_l2_diffs


def check_intersection_with_polygon(input_polygon, target_polygon):
    poly1 = geometry.Polygon(input_polygon).buffer(0)
    poly2 = geometry.Polygon(target_polygon).buffer(0)
    intersection_poly = poly1.intersection(poly2)
    intersection_area = intersection_poly.area
    is_intersection = 0 < intersection_area
    return is_intersection


def check_intersection_with_polygons(input_polygon, target_polygons):
    """
    Returns True if there is an intersection with at least one polygon in target_polygons
    :param input_polygon:
    :param target_polygons:
    :return:
    """
    for target_polygon in target_polygons:
        if check_intersection_with_polygon(input_polygon, target_polygon):
            return True
    return False


def polygon_area(polygon):
    poly = geometry.Polygon(polygon).buffer(0)
    return poly.area


def polygon_union(polygon1, polygon2):
    poly1 = geometry.Polygon(polygon1).buffer(0)
    poly2 = geometry.Polygon(polygon2).buffer(0)
    union_poly = poly1.union(poly2)
    return np.array(union_poly.exterior.coords)


def polygon_iou(polygon1, polygon2):
    poly1 = geometry.Polygon(polygon1).buffer(0)
    poly2 = geometry.Polygon(polygon2).buffer(0)
    intersection_poly = poly1.intersection(poly2)
    union_poly = poly1.union(poly2)
    intersection_area = intersection_poly.area
    union_area = union_poly.area
    if union_area:
        iou = intersection_area / union_area
    else:
        iou = 0
    return iou


def generate_polygon(cx, cy, ave_radius, irregularity, spikeyness, vertex_count):
    """
    Start with the centre of the polygon at cx, cy,
    then creates the polygon by sampling points on a circle around the centre.
    Random noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    cx, cy - coordinates of the "centre" of the polygon
    ave_radius - in px, the average radius of this polygon, this roughly controls how large the polygon is,
        really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to
        [0, 2 * pi / vertex_count]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius ave_radius.
        [0,1] will map to [0, ave_radius]
    vertex_count - self-explanatory

    Returns a list of vertices, in CCW order.
    """

    irregularity = clip(irregularity, 0, 1) * 2 * math.pi / vertex_count
    spikeyness = clip(spikeyness, 0, 1) * ave_radius

    # generate n angle steps
    angle_steps = []
    lower = (2 * math.pi / vertex_count) - irregularity
    upper = (2 * math.pi / vertex_count) + irregularity
    angle_sum = 0
    for i in range(vertex_count):
        tmp = random.uniform(lower, upper)
        angle_steps.append(tmp)
        angle_sum = angle_sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = angle_sum / (2 * math.pi)
    for i in range(vertex_count):
        angle_steps[i] = angle_steps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(vertex_count):
        r_i = clip(random.gauss(ave_radius, spikeyness), 0, 2 * ave_radius)
        x = cx + r_i * math.cos(angle)
        y = cy + r_i * math.sin(angle)
        points.append((x, y))

        angle = angle + angle_steps[i]

    return points


def clip(x, mini, maxi):
    if mini > maxi:
        return x
    elif x < mini:
        return mini
    elif x > maxi:
        return maxi
    else:
        return x


def scale_bounding_box(bounding_box, scale):
    half_width = math.ceil((bounding_box[2] - bounding_box[0]) * scale / 2)
    half_height = math.ceil((bounding_box[3] - bounding_box[1]) * scale / 2)
    center = [round((bounding_box[0] + bounding_box[2]) / 2), round((bounding_box[1] + bounding_box[3]) / 2)]
    scaled_bounding_box = [int(center[0] - half_width), int(center[1] - half_height), int(center[0] + half_width),
                           int(center[1] + half_height)]
    return scaled_bounding_box


def pad_bounding_box(bbox, pad):
    return [bbox[0] + pad, bbox[1] + pad, bbox[2] - pad, bbox[3] - pad]


def compute_bounding_box(polygon, scale=1, boundingbox_margin=0, fit=None):
    # Compute base bounding box
    bounding_box = [np.min(polygon[:, 0]), np.min(polygon[:, 1]), np.max(polygon[:, 0]), np.max(polygon[:, 1])]
    # Scale
    half_width = math.ceil((bounding_box[2] - bounding_box[0]) * scale / 2)
    half_height = math.ceil((bounding_box[3] - bounding_box[1]) * scale / 2)
    # Add margin
    half_width += boundingbox_margin
    half_height += boundingbox_margin
    # Compute square bounding box
    if fit == "square":
        half_width = half_height = max(half_width, half_height)
    center = [round((bounding_box[0] + bounding_box[2]) / 2), round((bounding_box[1] + bounding_box[3]) / 2)]
    bounding_box = [int(center[0] - half_width), int(center[1] - half_height), int(center[0] + half_width),
                    int(center[1] + half_height)]
    return bounding_box


def compute_patch(polygon, patch_size):
    centroid = np.mean(polygon, axis=0)
    half_height = half_width = patch_size / 2
    bounding_box = [math.ceil(centroid[0] - half_width), math.ceil(centroid[1] - half_height),
                    math.ceil(centroid[0] + half_width), math.ceil(centroid[1] + half_height)]
    return bounding_box


def bounding_box_within_bounds(bounding_box, bounds):
    return bounds[0] <= bounding_box[0] and bounds[1] <= bounding_box[1] and bounding_box[2] <= bounds[2] and \
           bounding_box[3] <= bounds[3]


def vertex_within_bounds(vertex, bounds):
    return bounds[0] <= vertex[0] <= bounds[2] and \
           bounds[1] <= vertex[1] <= bounds[3]


def edge_within_bounds(edge, bounds):
    return vertex_within_bounds(edge[0], bounds) and vertex_within_bounds(edge[1], bounds)


def bounding_box_area(bounding_box):
    return (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])


def convert_to_image_patch_space(polygon_image_space, bounding_box):
    polygon_image_patch_space = np.empty_like(polygon_image_space)
    polygon_image_patch_space[:, 0] = polygon_image_space[:, 0] - bounding_box[0]
    polygon_image_patch_space[:, 1] = polygon_image_space[:, 1] - bounding_box[1]
    return polygon_image_patch_space


def strip_redundant_vertex(vertices, epsilon=1):
    assert len(vertices.shape) == 2  # Is a polygon
    new_vertices = vertices
    if 1 < vertices.shape[0]:
        if np.sum(np.absolute(vertices[0, :] - vertices[-1, :])) < epsilon:
            new_vertices = vertices[:-1, :]
    return new_vertices


def remove_doubles(vertices, epsilon=0.1):
    dists = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=-1)
    new_vertices = vertices[epsilon < dists]
    return new_vertices


def simplify_polygon(polygon, tolerance=1):
    approx_polygon = approximate_polygon(polygon, tolerance=tolerance)
    return approx_polygon


def simplify_polygons(polygons, tolerance=1):
    approx_polygons = []
    for polygon in polygons:
        approx_polygon = approximate_polygon(polygon, tolerance=tolerance)
        approx_polygons.append(approx_polygon)
    return approx_polygons


def pad_polygon(vertices, target_length):
    assert len(vertices.shape) == 2  # Is a polygon
    assert vertices.shape[0] <= target_length
    padding_length = target_length - vertices.shape[0]
    padding = np.tile(vertices[-1], [padding_length, 1])
    padded_vertices = np.append(vertices, padding, axis=0)
    return padded_vertices


def compute_diameter(polygon):
    dist = scipy.spatial.distance.cdist(polygon, polygon)
    return dist.max()


def plot_polygon(polygon, color=None, draw_labels=True, label_direction=1, indexing="xy", axis=None):
    if python_utils.module_exists("matplotlib.pyplot"):
        import matplotlib.pyplot as plt

        if axis is None:
            axis = plt.gca()

        polygon_closed = np.append(polygon, [polygon[0, :]], axis=0)
        if indexing == "xy=":
            axis.plot(polygon_closed[:, 0], polygon_closed[:, 1], color=color, linewidth=3.0)
        elif indexing == "ij":
            axis.plot(polygon_closed[:, 1], polygon_closed[:, 0], color=color, linewidth=3.0)
        else:
            print("WARNING: Invalid indexing argument")

        if draw_labels:
            labels = range(1, polygon.shape[0] + 1)
            for label, x, y in zip(labels, polygon[:, 0], polygon[:, 1]):
                axis.annotate(
                    label,
                    xy=(x, y), xytext=(-20 * label_direction, 20 * label_direction),
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.25', fc=color, alpha=0.75),
                    arrowprops=dict(arrowstyle='->', color=color, connectionstyle='arc3,rad=0'))


def plot_polygons(polygons, color=None, draw_labels=True, label_direction=1, indexing="xy", axis=None):
    for polygon in polygons:
        plot_polygon(polygon, color=color, draw_labels=draw_labels, label_direction=label_direction, indexing=indexing,
                     axis=axis)


def compute_edge_normal(edge):
    normal = np.array([- (edge[1][1] - edge[0][1]),
                       edge[1][0] - edge[0][0]])
    normal_norm = np.sqrt(np.sum(np.square(normal)))
    normal /= normal_norm
    return normal


def compute_vector_angle(x, y):
    if x < 0.0:
        slope = y / x
        angle = np.pi + np.arctan(slope)
    elif 0.0 < x:
        slope = y / x
        angle = np.arctan(slope)
    else:
        if 0 < y:
            angle = np.pi / 2
        else:
            angle = 3 * np.pi / 2
    if angle < 0.0:
        angle += 2 * np.pi
    return angle


def compute_edge_normal_angle_edge(edge):
    normal = compute_edge_normal(edge)
    normal_x = normal[1]
    normal_y = normal[0]
    angle = compute_vector_angle(normal_x, normal_y)
    return angle


def polygon_in_bounding_box(polygon, bounding_box):
    """
    Returns True if all vertices of polygons are inside bounding_box
    :param polygon: [N, 2]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    result = np.all(
        np.logical_and(
            np.logical_and(bounding_box[0] <= polygon[:, 0], polygon[:, 0] <= bounding_box[2]),
            np.logical_and(bounding_box[1] <= polygon[:, 1], polygon[:, 1] <= bounding_box[3])
        )
    )
    return result


def filter_polygons_in_bounding_box(polygons, bounding_box):
    """
    Only keep polygons that are fully inside bounding_box

    :param polygons: [shape(N, 2), ...]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    filtered_polygons = []
    for polygon in polygons:
        if polygon_in_bounding_box(polygon, bounding_box):
            filtered_polygons.append(polygon)
    return filtered_polygons


def transform_polygon_to_bounding_box_space(polygon, bounding_box):
    """

    :param polygon: shape(N, 2)
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    assert len(polygon.shape) and polygon.shape[1] == 2, "polygon should have shape (N, 2), not shape {}".format(
        polygon.shape)
    assert len(bounding_box) == 4, "bounding_box should have 4 elements: [row_min, col_min, row_max, col_max]"
    transformed_polygon = polygon.copy()
    transformed_polygon[:, 0] -= bounding_box[0]
    transformed_polygon[:, 1] -= bounding_box[1]
    return transformed_polygon


def transform_polygons_to_bounding_box_space(polygons, bounding_box):
    transformed_polygons = []
    for polygon in polygons:
        transformed_polygons.append(transform_polygon_to_bounding_box_space(polygon, bounding_box))
    return transformed_polygons


def crop_polygon_to_patch(polygon, bounding_box):
    return transform_polygon_to_bounding_box_space(polygon, bounding_box)


def crop_polygon_to_patch_if_touch(polygon, bounding_box):
    # Verify that at least one vertex is inside bounding_box
    polygon_touches_patch = np.any(
        np.logical_and(
            np.logical_and(bounding_box[0] <= polygon[:, 0], polygon[:, 0] <= bounding_box[2]),
            np.logical_and(bounding_box[1] <= polygon[:, 1], polygon[:, 1] <= bounding_box[3])
        )
    )
    if polygon_touches_patch:
        return crop_polygon_to_patch(polygon, bounding_box)
    else:
        return None


def crop_polygons_to_patch_if_touch(polygons, bounding_box, return_indices=False):
    if return_indices:
        indices = []
    cropped_polygons = []
    for i, polygon in enumerate(polygons):
        cropped_polygon = crop_polygon_to_patch_if_touch(polygon, bounding_box)
        if cropped_polygon is not None:
            cropped_polygons.append(cropped_polygon)
            if return_indices:
                indices.append(i)
    if return_indices:
        return cropped_polygons, indices
    else:
        return cropped_polygons


def crop_polygons_to_patch(polygons, bounding_box):
    cropped_polygons = []
    for polygon in polygons:
        cropped_polygon = crop_polygon_to_patch(polygon, bounding_box)
        if cropped_polygon is not None:
            cropped_polygons.append(cropped_polygon)
    return cropped_polygons


def polygon_remove_holes(polygon):
    polygon_no_holes = []
    for coords in polygon:
        if not np.isnan(coords[0]) and not np.isnan(coords[1]):
            polygon_no_holes.append(coords)
        else:
            break
    return np.array(polygon_no_holes)


def polygons_remove_holes(polygons):
    gt_polygons_no_holes = []
    for polygon in polygons:
        gt_polygons_no_holes.append(polygon_remove_holes(polygon))
    return gt_polygons_no_holes


def apply_batch_disp_map_to_polygons(pred_disp_field_map_batch, disp_polygons_batch):
    """

    :param pred_disp_field_map_batch: shape(batch_size, height, width, 2)
    :param disp_polygons_batch: shape(batch_size, polygon_count, vertex_count, 2)
    :return:
    """

    # Apply all displacements at once
    batch_count = pred_disp_field_map_batch.shape[0]
    row_count = pred_disp_field_map_batch.shape[1]
    col_count = pred_disp_field_map_batch.shape[2]

    disp_polygons_batch_int = np.round(disp_polygons_batch).astype(np.int)
    # Clip coordinates to the field map:
    disp_polygons_batch_int_nearest_valid_field = np.maximum(0, disp_polygons_batch_int)
    disp_polygons_batch_int_nearest_valid_field[:, :, :, 0] = np.minimum(
        disp_polygons_batch_int_nearest_valid_field[:, :, :, 0], row_count - 1)
    disp_polygons_batch_int_nearest_valid_field[:, :, :, 1] = np.minimum(
        disp_polygons_batch_int_nearest_valid_field[:, :, :, 1], col_count - 1)

    aligned_disp_polygons_batch = disp_polygons_batch.copy()
    for batch_index in range(batch_count):
        mask = ~np.isnan(disp_polygons_batch[batch_index, :, :, 0])  # Checking one coordinate is enough
        aligned_disp_polygons_batch[batch_index, mask, 0] += pred_disp_field_map_batch[batch_index,
                                                                                       disp_polygons_batch_int_nearest_valid_field[
                                                                                           batch_index, mask, 0],
                                                                                       disp_polygons_batch_int_nearest_valid_field[
                                                                                           batch_index, mask, 1], 0].flatten()
        aligned_disp_polygons_batch[batch_index, mask, 1] += pred_disp_field_map_batch[batch_index,
                                                                                       disp_polygons_batch_int_nearest_valid_field[
                                                                                           batch_index, mask, 0],
                                                                                       disp_polygons_batch_int_nearest_valid_field[
                                                                                           batch_index, mask, 1], 1].flatten()
    return aligned_disp_polygons_batch


def apply_disp_map_to_polygons(disp_field_map, polygons):
    """

    :param disp_field_map: shape(height, width, 2)
    :param polygon_list: [shape(N, 2), shape(M, 2), ...]
    :return:
    """
    disp_field_map_batch = np.expand_dims(disp_field_map, axis=0)
    disp_polygons = []
    for polygon in polygons:
        polygon_batch = np.expand_dims(np.expand_dims(polygon, axis=0), axis=0)  # Add batch and polygon_count dims
        disp_polygon_batch = apply_batch_disp_map_to_polygons(disp_field_map_batch, polygon_batch)
        disp_polygon_batch = disp_polygon_batch[0, 0]  # Remove batch and polygon_count dims
        disp_polygons.append(disp_polygon_batch)
    return disp_polygons


# This next function is somewhat redundant with apply_disp_map_to_polygons... (but displaces in the opposite direction)
def apply_displacement_field_to_polygons(polygons, disp_field_map):
    disp_polygons = []
    for polygon in polygons:
        mask_nans = np.isnan(polygon)  # Will be necessary when polygons with holes are handled
        polygon_int = np.round(polygon).astype(np.int)
        polygon_int_clipped = np.maximum(0, polygon_int)
        polygon_int_clipped[:, 0] = np.minimum(disp_field_map.shape[0] - 1, polygon_int_clipped[:, 0])
        polygon_int_clipped[:, 1] = np.minimum(disp_field_map.shape[1] - 1, polygon_int_clipped[:, 1])
        disp_polygon = polygon.copy()
        disp_polygon[~mask_nans[:, 0], 0] -= disp_field_map[polygon_int_clipped[~mask_nans[:, 0], 0],
                                                            polygon_int_clipped[~mask_nans[:, 0], 1], 0]
        disp_polygon[~mask_nans[:, 1], 1] -= disp_field_map[polygon_int_clipped[~mask_nans[:, 1], 0],
                                                            polygon_int_clipped[~mask_nans[:, 1], 1], 1]
        disp_polygons.append(disp_polygon)
    return disp_polygons


def apply_displacement_fields_to_polygons(polygons, disp_field_maps):
    disp_field_map_count = disp_field_maps.shape[0]
    disp_polygons_list = []
    for i in range(disp_field_map_count):
        disp_polygons = apply_displacement_field_to_polygons(polygons, disp_field_maps[i, :, :, :])
        disp_polygons_list.append(disp_polygons)
    return disp_polygons_list


def draw_line(shape, line, width, blur_radius=0):
    im = Image.new("L", (shape[1], shape[0]))
    # im_px_access = im.load()
    draw = ImageDraw.Draw(im)
    vertex_list = []
    for coords in line:
        vertex = (coords[1], coords[0])
        vertex_list.append(vertex)
    draw.line(vertex_list, fill=255, width=width)
    if 0 < blur_radius:
        im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    array = np.array(im) / 255
    return array


def draw_triangle(shape, triangle, blur_radius=0):
    im = Image.new("L", (shape[1], shape[0]))
    # im_px_access = im.load()
    draw = ImageDraw.Draw(im)
    vertex_list = []
    for coords in triangle:
        vertex = (coords[1], coords[0])
        vertex_list.append(vertex)
    draw.polygon(vertex_list, fill=255)
    if 0 < blur_radius:
        im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    array = np.array(im) / 255
    return array


def draw_polygon(polygon, shape, fill=True, edges=True, vertices=True, line_width=3):
    # TODO: handle holes in polygons
    im = Image.new("RGB", (shape[1], shape[0]))
    im_px_access = im.load()
    draw = ImageDraw.Draw(im)

    vertex_list = []
    for coords in polygon:
        vertex = (coords[1], coords[0])
        if not np.isnan(vertex[0]) and not np.isnan(vertex[1]):
            vertex_list.append(vertex)
        else:
            break
    if edges:
        draw.line(vertex_list, fill=(0, 255, 0), width=line_width)
    if fill:
        draw.polygon(vertex_list, fill=(255, 0, 0))
    if vertices:
        draw.point(vertex_list, fill=(0, 0, 255))

    # Convert image to numpy array with the right number of channels
    array = np.array(im)
    selection = [fill, edges, vertices]
    selected_array = array[:, :, selection]
    return selected_array


def draw_polygons(polygons, shape, fill=True, edges=True, vertices=True, line_width=3):
    # TODO: handle holes in polygons

    # Channels
    fill_channel_index = 0  # Always first channel
    edges_channel_index = fill  # If fill == True, take second channel. If not then take first
    vertices_channel_index = fill + edges  # Same principle as above
    channel_count = fill + edges + vertices
    im_draw_list = []
    for channel_index in range(channel_count):
        im = Image.new("L", (shape[1], shape[0]))
        im_px_access = im.load()
        draw = ImageDraw.Draw(im)
        im_draw_list.append((im, draw))

    for polygon in polygons:
        vertex_list = []
        for coords in polygon:
            vertex = (coords[1], coords[0])
            if not np.isnan(vertex[0]) and not np.isnan(vertex[1]):
                vertex_list.append(vertex)
            else:
                break
        if fill:
            draw = im_draw_list[fill_channel_index][1]
            draw.polygon(vertex_list, fill=255)
        if edges:
            draw = im_draw_list[edges_channel_index][1]
            draw.line(vertex_list, fill=255, width=line_width)
        if vertices:
            draw = im_draw_list[vertices_channel_index][1]
            draw.point(vertex_list, fill=255)

    # Convert image to numpy array with the right number of channels
    array_list = [np.array(im_draw[0]) for im_draw in im_draw_list]
    array = np.stack(array_list, axis=-1)
    return array


def draw_polygon_map(polygons, shape, fill=True, edges=True, vertices=True, line_width=3):
    """
    Alias for draw_polygon function

    :param polygons:
    :param shape:
    :param fill:
    :param edges:
    :param vertices:
    :param line_width:
    :return:
    """
    return draw_polygons(polygons, shape, fill=fill, edges=edges, vertices=vertices, line_width=line_width)


def draw_polygon_maps(polygons_list, shape, fill=True, edges=True, vertices=True, line_width=3):
    polygon_maps_list = []
    for polygons in polygons_list:
        polygon_map = draw_polygon_map(polygons, shape, fill=fill, edges=edges, vertices=vertices,
                                       line_width=line_width)
        polygon_maps_list.append(polygon_map)
    disp_field_maps = np.stack(polygon_maps_list, axis=0)
    return disp_field_maps


def swap_coords(polygon):
    polygon_new = polygon.copy()
    polygon_new[..., 0] = polygon[..., 1]
    polygon_new[..., 1] = polygon[..., 0]
    return polygon_new


def prepare_polygons_for_tfrecord(gt_polygons, disp_polygons_list, boundingbox=None):
    assert len(gt_polygons)

    # print("Starting to crop polygons")
    # start = time.time()

    dtype = gt_polygons[0].dtype
    cropped_gt_polygons = []
    cropped_disp_polygons_list = [[] for i in range(len(disp_polygons_list))]
    polygon_length = 0
    for polygon_index, gt_polygon in enumerate(gt_polygons):
        if boundingbox is not None:
            cropped_gt_polygon = crop_polygon_to_patch_if_touch(gt_polygon, boundingbox)
        else:
            cropped_gt_polygon = gt_polygon
        if cropped_gt_polygon is not None:
            cropped_gt_polygons.append(cropped_gt_polygon)
            if polygon_length < cropped_gt_polygon.shape[0]:
                polygon_length = cropped_gt_polygon.shape[0]
            # Crop disp polygons
            for disp_index, disp_polygons in enumerate(disp_polygons_list):
                disp_polygon = disp_polygons[polygon_index]
                if boundingbox is not None:
                    cropped_disp_polygon = crop_polygon_to_patch(disp_polygon, boundingbox)
                else:
                    cropped_disp_polygon = disp_polygon
                cropped_disp_polygons_list[disp_index].append(cropped_disp_polygon)

    # end = time.time()
    # print("Finished cropping polygons in in {}s".format(end - start))
    #
    # print("Starting to pad polygons")
    # start = time.time()

    polygon_count = len(cropped_gt_polygons)
    if polygon_count:
        # Add +1 to both dimensions for end-of-item NaNs
        padded_gt_polygons = np.empty((polygon_count + 1, polygon_length + 1, 2), dtype=dtype)
        padded_gt_polygons[:, :, :] = np.nan
        padded_disp_polygons_array = np.empty((len(disp_polygons_list), polygon_count + 1, polygon_length + 1, 2),
                                              dtype=dtype)
        padded_disp_polygons_array[:, :, :] = np.nan
        for i, polygon in enumerate(cropped_gt_polygons):
            padded_gt_polygons[i, 0:polygon.shape[0], :] = polygon
        for j, polygons in enumerate(cropped_disp_polygons_list):
            for i, polygon in enumerate(polygons):
                padded_disp_polygons_array[j, i, 0:polygon.shape[0], :] = polygon
    else:
        padded_gt_polygons = padded_disp_polygons_array = None

    # end = time.time()
    # print("Finished padding polygons in in {}s".format(end - start))

    return padded_gt_polygons, padded_disp_polygons_array


def prepare_stages_polygons_for_tfrecord(gt_polygons, disp_polygons_list_list, boundingbox):
    assert len(gt_polygons)

    print(gt_polygons)
    print(disp_polygons_list_list)

    exit()

    # print("Starting to crop polygons")
    # start = time.time()

    dtype = gt_polygons[0].dtype
    cropped_gt_polygons = []
    cropped_disp_polygons_list_list = [[[] for i in range(len(disp_polygons_list))] for disp_polygons_list in
                                       disp_polygons_list_list]
    polygon_length = 0
    for polygon_index, gt_polygon in enumerate(gt_polygons):
        cropped_gt_polygon = crop_polygon_to_patch_if_touch(gt_polygon, boundingbox)
        if cropped_gt_polygon is not None:
            cropped_gt_polygons.append(cropped_gt_polygon)
            if polygon_length < cropped_gt_polygon.shape[0]:
                polygon_length = cropped_gt_polygon.shape[0]
            # Crop disp polygons
            for stage_index, disp_polygons_list in enumerate(disp_polygons_list_list):
                for disp_index, disp_polygons in enumerate(disp_polygons_list):
                    disp_polygon = disp_polygons[polygon_index]
                    cropped_disp_polygon = crop_polygon_to_patch(disp_polygon, boundingbox)
                    cropped_disp_polygons_list_list[stage_index][disp_index].append(cropped_disp_polygon)

    # end = time.time()
    # print("Finished cropping polygons in in {}s".format(end - start))
    #
    # print("Starting to pad polygons")
    # start = time.time()

    polygon_count = len(cropped_gt_polygons)
    if polygon_count:
        # Add +1 to both dimensions for end-of-item NaNs
        padded_gt_polygons = np.empty((polygon_count + 1, polygon_length + 1, 2), dtype=dtype)
        padded_gt_polygons[:, :, :] = np.nan
        padded_disp_polygons_array = np.empty(
            (len(disp_polygons_list_list), len(disp_polygons_list_list[0]), polygon_count + 1, polygon_length + 1, 2),
            dtype=dtype)
        padded_disp_polygons_array[:, :, :] = np.nan
        for i, polygon in enumerate(cropped_gt_polygons):
            padded_gt_polygons[i, 0:polygon.shape[0], :] = polygon
        for k, cropped_disp_polygons_list in enumerate(cropped_disp_polygons_list_list):
            for j, polygons in enumerate(cropped_disp_polygons_list):
                for i, polygon in enumerate(polygons):
                    padded_disp_polygons_array[k, j, i, 0:polygon.shape[0], :] = polygon
    else:
        padded_gt_polygons = padded_disp_polygons_array = None

    # end = time.time()
    # print("Finished padding polygons in in {}s".format(end - start))

    return padded_gt_polygons, padded_disp_polygons_array


def rescale_polygon(polygons, scaling_factor):
    """

    :param polygons:
    :return: scaling_factor
    """
    if len(polygons):
        rescaled_polygons = [polygon * scaling_factor for polygon in polygons]
        return rescaled_polygons
    else:
        return polygons


def get_edge_center(edge):
    return np.mean(edge, axis=0)


def get_edge_length(edge):
    return np.sqrt(np.sum(np.square(edge[0] - edge[1])))


def get_edges_angle(edge1, edge2):
    x1 = edge1[1, 0] - edge1[0, 0]
    y1 = edge1[1, 1] - edge1[0, 1]
    x2 = edge2[1, 0] - edge2[0, 0]
    y2 = edge2[1, 1] - edge2[0, 1]
    angle1 = compute_vector_angle(x1, y1)
    angle2 = compute_vector_angle(x2, y2)
    edges_angle = math.fabs(angle1 - angle2) % (2 * math.pi)
    if math.pi < edges_angle:
        edges_angle = 2 * math.pi - edges_angle
    return edges_angle


def compute_angle_two_points(point_source, point_target):
    vector = point_target - point_source
    angle = compute_vector_angle(vector[0], vector[1])
    return angle


def compute_angle_three_points(point_source, point_target1, point_target2):
    squared_dist_source_target1 = math.pow((point_source[0] - point_target1[0]), 2) + math.pow(
        (point_source[1] - point_target1[1]), 2)
    squared_dist_source_target2 = math.pow((point_source[0] - point_target2[0]), 2) + math.pow(
        (point_source[1] - point_target2[1]), 2)
    squared_dist_target1_target2 = math.pow((point_target1[0] - point_target2[0]), 2) + math.pow(
        (point_target1[1] - point_target2[1]), 2)
    dist_source_target1 = math.sqrt(squared_dist_source_target1)
    dist_source_target2 = math.sqrt(squared_dist_source_target2)
    try:
        cos = (squared_dist_source_target1 + squared_dist_source_target2 - squared_dist_target1_target2) / (
                    2 * dist_source_target1 * dist_source_target2)
    except ZeroDivisionError:
        return float('inf')
    cos = max(min(cos, 1),
              -1)  # Avoid some math domain error due to cos being slightly bigger than 1 (from floating point operations)
    angle = math.acos(cos)
    return angle


def are_edges_overlapping(edge1, edge2, threshold):
    """
    Checks if at least 2 different vertices of either edge lies on the other edge: it characterizes an overlap
    :param edge1:
    :param edge2:
    :param threshold:
    :return:
    """
    count_list = [
        is_vertex_on_edge(edge1[0], edge2, threshold),
        is_vertex_on_edge(edge1[1], edge2, threshold),
        is_vertex_on_edge(edge2[0], edge1, threshold),
        is_vertex_on_edge(edge2[1], edge1, threshold),
    ]
    # Count number of identical vertices
    identical_vertex_list = [
        np.array_equal(edge1[0], edge2[0]),
        np.array_equal(edge1[0], edge2[1]),
        np.array_equal(edge1[1], edge2[0]),
        np.array_equal(edge1[1], edge2[1]),
    ]
    adjusted_count = np.sum(count_list) - np.sum(identical_vertex_list)
    return 2 <= adjusted_count


# def are_edges_collinear(edge1, edge2, angle_threshold):
#     edges_angle = get_edges_angle(edge1, edge2)
#     return edges_angle < angle_threshold


def get_line_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return float('inf'), float('inf')
    return x / z, y / z


def are_edges_intersecting(edge1, edge2, epsilon=1e-6):
    """
    edge1 and edge2 should not have a common vertex between them
    :param edge1:
    :param edge2:
    :return:
    """
    intersect = get_line_intersect(edge1[0], edge1[1], edge2[0], edge2[1])
    # print("---")
    # print(edge1)
    # print(edge2)
    # print(intersect)
    if intersect[0] == float('inf') or intersect[1] == float('inf'):
        # Lines don't intersect
        return False
    else:
        # Lines intersect
        # Check if intersect point belongs to both edges
        angle1 = compute_angle_three_points(intersect, edge1[0], edge1[1])
        angle2 = compute_angle_three_points(intersect, edge2[0], edge2[1])
        intersect_belongs_to_edges = (math.pi - epsilon) < angle1 and (math.pi - epsilon) < angle2
        return intersect_belongs_to_edges


def shorten_edge(edge, length_to_cut1, length_to_cut2, min_length):
    center = get_edge_center(edge)
    total_length = get_edge_length(edge)
    new_length = total_length - length_to_cut1 - length_to_cut2
    if min_length <= new_length:
        scale = new_length / total_length
        new_edge = (edge.copy() - center) * scale + center
        return new_edge
    else:
        return None


def is_edge_in_triangle(edge, triangle):
    return edge[0] in triangle and edge[1] in triangle


def get_connectivity_of_edge(edge, triangles):
    connectivity = 0
    for triangle in triangles:
        connectivity += is_edge_in_triangle(edge, triangle)
    return connectivity


def get_connectivity_of_edges(edges, triangles):
    connectivity_of_edges = []
    for edge in edges:
        connectivity_of_edge = get_connectivity_of_edge(edge, triangles)
        connectivity_of_edges.append(connectivity_of_edge)
    return connectivity_of_edges


def polygon_to_closest_int(polygons):
    int_polygons = []
    for polygon in polygons:
        int_polygon = np.round(polygon)
        int_polygons.append(int_polygon)
    return int_polygons


def is_vertex_on_edge(vertex, edge, threshold):
    """
    :param vertex:
    :param edge:
    :param threshold:
    :return:
    """
    # Compare distances sum to edge length
    edge_length = get_edge_length(edge)
    dist1 = get_edge_length([vertex, edge[0]])
    dist2 = get_edge_length([vertex, edge[1]])
    vertex_on_edge = (dist1 + dist2) < (edge_length + threshold)
    return vertex_on_edge


def get_face_edges(face_vertices):
    edges = []
    prev_vertex = face_vertices[0]
    for vertex in face_vertices[1:]:
        edge = (prev_vertex, vertex)
        edges.append(edge)

        # For next iteration:
        prev_vertex = vertex
    return edges


def find_edge_in_face(edge, face_vertices):
    # Copy inputs list so that we don't modify it
    face_vertices = face_vertices[:]
    face_vertices.append(face_vertices[0])  # Close face (does not matter if it is already closed)
    edges = get_face_edges(face_vertices)
    index = edges.index(edge)
    return index


def clean_degenerate_face_edges(face_vertices):
    def recursive_clean_degenerate_face_edges(open_face_vertices):
        face_vertex_count = len(open_face_vertices)
        cleaned_open_face_vertices = []
        skip = False
        for index in range(face_vertex_count):
            if skip:
                skip = False
            else:
                prev_vertex = open_face_vertices[(index - 1) % face_vertex_count]
                vertex = open_face_vertices[index]
                next_vertex = open_face_vertices[(index + 1) % face_vertex_count]
                if prev_vertex != next_vertex:
                    cleaned_open_face_vertices.append(vertex)
                else:
                    skip = True
        if len(cleaned_open_face_vertices) < face_vertex_count:
            return recursive_clean_degenerate_face_edges(cleaned_open_face_vertices)
        else:
            return cleaned_open_face_vertices

    open_face_vertices = face_vertices[:-1]
    cleaned_face_vertices = recursive_clean_degenerate_face_edges(open_face_vertices)
    # Close cleaned_face_vertices
    cleaned_face_vertices.append(cleaned_face_vertices[0])
    return cleaned_face_vertices


def merge_vertices(main_face_vertices, extra_face_vertices, common_edge):
    sorted_common_edge = tuple(sorted(common_edge))
    open_face_vertices_pair = (main_face_vertices[:-1], extra_face_vertices[:-1])
    face_index = 0  # 0: current_face == main_face, 1: current_face == extra_face
    vertex_index = 0
    start_vertex = vertex = open_face_vertices_pair[face_index][vertex_index]
    merged_face_vertices = [start_vertex]
    faces_merged = False
    while not faces_merged:
        # Get next vertex
        next_vertex_index = (vertex_index + 1) % len(open_face_vertices_pair[face_index])
        next_vertex = open_face_vertices_pair[face_index][next_vertex_index]
        edge = (vertex, next_vertex)
        sorted_edge = tuple(sorted(edge))
        if sorted_edge == sorted_common_edge:
            # Switch current face
            face_index = 1 - face_index
            # Find vertex_index in new current face
            reverse_edge = (edge[1], edge[0])  # Because we are now on the other face
            edge_index = find_edge_in_face(reverse_edge, open_face_vertices_pair[face_index])
            vertex_index = edge_index + 1  # Index of the second vertex of edge
            # vertex_index = open_face_vertices_pair[face_index].index(vertex)
        vertex_index = (vertex_index + 1) % len(open_face_vertices_pair[face_index])
        vertex = open_face_vertices_pair[face_index][vertex_index]
        merged_face_vertices.append(vertex)
        faces_merged = vertex == start_vertex  # This also makes the merged_face closed
    # Remove degenerate face edges (edges where the face if on both sides of it)
    cleaned_merged_face_vertices = clean_degenerate_face_edges(merged_face_vertices)
    return cleaned_merged_face_vertices


if __name__ == "__main__":
    # polygon = np.array([
    #     [0, 0],
    #     [1, 0],
    #     [1, 1],
    #     [np.nan, np.nan],
    #     [0, 0],
    #     [1, 0],
    #     [1, 1],
    #     [np.nan, np.nan],
    # ], dtype=np.float32)
    # polygons = [
    #     polygon.copy(),
    #     polygon.copy(),
    #     polygon.copy(),
    #     polygon.copy() + 100,
    # ]
    #
    # bounding_box = [10, 10, 100, 100]  # Top left corner x, y, bottom right corner x, y
    #
    # cropped_polygons = crop_polygons_to_patch(polygons, bounding_box)
    # print(cropped_polygons)

    # # --- Check angle functions --- #
    # edge1 = np.array([
    #     [0, 0],
    #     [1, 0],
    # ])
    # edge2 = np.array([
    #     [1, 0],
    #     [2, 0],
    # ])
    # edge_radius = 0.1
    # edges_overlapping = are_edges_overlapping(edge1, edge2, edge_radius)
    # print("edges_overlapping:")
    # print(edges_overlapping)

    # --- clean_degenerate_face_edges --- #
    face_vertices = [215, 238, 220, 201, 193, 194, 195, 199, 213, 219, 235, 238, 215]
    # face_vertices = [1, 2, 3, 4, 5, 4, 3, 6, 1]
    print(face_vertices)
    cleaned_face_vertices = clean_degenerate_face_edges(face_vertices)
    print(cleaned_face_vertices)
