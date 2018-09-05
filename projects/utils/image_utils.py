from io import BytesIO
import math
import numpy as np
from PIL import Image
import skimage.draw

import python_utils

CV2 = False
if python_utils.module_exists("cv2"):
    import cv2
    CV2 = True

if python_utils.module_exists("matplotlib.pyplot"):
    import matplotlib.pyplot as plt


def load_image(image_filepath):
    image = Image.open(image_filepath)
    image.load()
    image_array = np.array(image, dtype=np.uint8)
    image.close()
    return image_array


def padded_boundingbox(boundingbox, padding):
    boundingbox_new = np.empty_like(boundingbox)
    boundingbox_new[0:2] = boundingbox[0:2] + padding
    boundingbox_new[2:4] = boundingbox[2:4] - padding
    return boundingbox_new


def bbox_add_margin(bbox, margin):
    bbox_new = bbox.copy()
    bbox_new[0:2] -= margin
    bbox_new[2:4] += margin
    return bbox_new


def bbox_to_int(bbox):
    bbox_new = [
        int(np.floor(bbox[0])),
        int(np.floor(bbox[1])),
        int(np.ceil(bbox[2])),
        int(np.ceil(bbox[3])),
    ]
    return bbox_new


def draw_line_aa_in_patch(edge, patch_bounds):
    rr, cc, prob = skimage.draw.line_aa(edge[0][0], edge[0][1], edge[1][0], edge[1][1])
    keep_mask = (patch_bounds[0] <= rr) & (rr < patch_bounds[2]) \
                & (patch_bounds[1] <= cc) & (cc < patch_bounds[3])
    rr = rr[keep_mask]
    cc = cc[keep_mask]
    prob = prob[keep_mask]
    return rr, cc, prob


def convert_array_to_jpg_bytes(image_array, mode=None):
    img = Image.fromarray(image_array, mode=mode)
    output = BytesIO()
    img.save(output, format="JPEG", quality=90)
    contents = output.getvalue()
    output.close()
    return contents


def displacement_map_to_transformation_maps(disp_field_map):
    disp_field_map = disp_field_map.astype(np.float32)
    i = np.arange(disp_field_map.shape[0], dtype=np.float32)
    j = np.arange(disp_field_map.shape[1], dtype=np.float32)
    iv, jv = np.meshgrid(i, j, indexing="ij")
    reverse_map_i = iv + disp_field_map[:, :, 1]
    reverse_map_j = jv + disp_field_map[:, :, 0]
    return reverse_map_i, reverse_map_j

if CV2:
    def apply_displacement_field_to_image(image, disp_field_map):
        trans_map_i, trans_map_j = displacement_map_to_transformation_maps(disp_field_map)
        misaligned_image = cv2.remap(image, trans_map_j, trans_map_i, cv2.INTER_CUBIC)
        return misaligned_image


    def apply_displacement_fields_to_image(image, disp_field_maps):
        disp_field_map_count = disp_field_maps.shape[0]
        misaligned_image_list = []
        for i in range(disp_field_map_count):
            misaligned_image = apply_displacement_field_to_image(image, disp_field_maps[i, :, :, :])
            misaligned_image_list.append(misaligned_image)
        return misaligned_image_list
else:
    def apply_displacement_fields_to_image(image, disp_field_map):
        print("cv2 is not available, the apply_displacement_fields_to_image(image, disp_field_map) function cannot work!")

    def apply_displacement_fields_to_image(image, disp_field_maps):
        print("cv2 is not available, the apply_displacement_fields_to_image(image, disp_field_maps) function cannot work!")


def compute_patch_boundingboxes(image_size, stride, patch_res):
    im_rows = image_size[0]
    im_cols = image_size[1]

    total_double_padding = patch_res - stride
    row_patch_count = max(1, int(math.ceil((im_rows - total_double_padding) / stride)))
    col_patch_count = max(1, int(math.ceil((im_cols - total_double_padding) / stride)))

    patch_boundingboxes = []
    for i in range(0, row_patch_count):
        if i < row_patch_count - 1:
            row_slice_begin = i * stride
            row_slice_end = row_slice_begin + patch_res
        else:
            row_slice_end = im_rows
            row_slice_begin = row_slice_end - patch_res
        for j in range(0, col_patch_count):
            if j < col_patch_count - 1:
                col_slice_begin = j*stride
                col_slice_end = col_slice_begin + patch_res
            else:
                col_slice_end = im_cols
                col_slice_begin = col_slice_end - patch_res

            patch_boundingbox = np.array([row_slice_begin, col_slice_begin, row_slice_end, col_slice_end])
            assert row_slice_end - row_slice_begin == col_slice_end - col_slice_begin == patch_res, "ERROR: patch does not have the requested shape"
            patch_boundingboxes.append(patch_boundingbox)

    return patch_boundingboxes


def clip_boundingbox(boundingbox, clip_list):
    assert len(boundingbox) == len(clip_list), "len(boundingbox) should be equal to len(clip_values)"
    clipped_boundingbox = []
    for bb_value, clip in zip(boundingbox[:2], clip_list[:2]):
        clipped_value = max(clip, bb_value)
        clipped_boundingbox.append(clipped_value)
    for bb_value, clip in zip(boundingbox[2:], clip_list[2:]):
        clipped_value = min(clip, bb_value)
        clipped_boundingbox.append(clipped_value)
    return clipped_boundingbox


def crop_or_pad_image_with_boundingbox(image, patch_boundingbox):
    im_rows = image.shape[0]
    im_cols = image.shape[1]

    row_padding_before = max(0, - patch_boundingbox[0])
    col_padding_before = max(0, - patch_boundingbox[1])
    row_padding_after = max(0, patch_boundingbox[2] - im_rows)
    col_padding_after = max(0, patch_boundingbox[3] - im_cols)

    # Center padding:
    row_padding = row_padding_before + row_padding_after
    col_padding = col_padding_before + col_padding_after
    row_padding_before = row_padding // 2
    col_padding_before = col_padding // 2
    row_padding_after = row_padding - row_padding // 2
    col_padding_after = col_padding - col_padding // 2

    clipped_patch_boundingbox = clip_boundingbox(patch_boundingbox, [0, 0, im_rows, im_cols])

    if len(image.shape) == 2:
        patch = image[clipped_patch_boundingbox[0]:clipped_patch_boundingbox[2], clipped_patch_boundingbox[1]:clipped_patch_boundingbox[3]]
        patch = np.pad(patch, [(row_padding_before, row_padding_after), (col_padding_before, col_padding_after)], mode="constant")
    elif len(image.shape) == 3:
        patch = image[clipped_patch_boundingbox[0]:clipped_patch_boundingbox[2], clipped_patch_boundingbox[1]:clipped_patch_boundingbox[3], :]
        patch = np.pad(patch, [(row_padding_before, row_padding_after), (col_padding_before, col_padding_after), (0, 0)], mode="constant")
    else:
        print("Image input does not have the right shape/")
        patch = None
    return patch


if __name__ == "__main__":
    im_rows = 5
    im_cols = 10
    stride = 1
    patch_res = 15

    image = np.random.randint(0, 256, size=(im_rows, im_cols, 3), dtype=np.uint8)
    image = Image.fromarray(image)
    image = np.array(image)
    plt.ion()
    plt.figure(1)
    plt.imshow(image)
    plt.show()

    # Cut patches
    patch_boundingboxes = compute_patch_boundingboxes(image.shape[0:2], stride, patch_res)

    plt.figure(2)

    for patch_boundingbox in patch_boundingboxes:
        patch = crop_or_pad_image_with_boundingbox(image, patch_boundingbox)
        plt.imshow(patch)
        plt.show()
        input("Press <Enter> to finish...")
