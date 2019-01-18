import numpy as np
import time
import sklearn.datasets
import skimage.transform

import python_utils
import image_utils

# if python_utils.module_exists("matplotlib.pyplot"):
#     import matplotlib.pyplot as plt

CV2 = False
if python_utils.module_exists("cv2"):
    import cv2
    CV2 = True

# import multiprocessing
#
# import python_utils
#
# if python_utils.module_exists("joblib"):
#     from joblib import Parallel, delayed
#     JOBLIB = True
# else:
#     JOBLIB = False


# def plot_field_map(field_map):
#     from mpl_toolkits.mplot3d import Axes3D
#
#     row = np.linspace(0, 1, field_map.shape[0])
#     col = np.linspace(0, 1, field_map.shape[1])
#     rr, cc = np.meshgrid(row, col, indexing='ij')
#
#     fig = plt.figure(figsize=(18, 9))
#     ax = fig.add_subplot(121, projection='3d')
#     ax.plot_surface(rr, cc, field_map[:, :, 0], rstride=3, cstride=3, linewidth=1, antialiased=True)
#
#     ax = fig.add_subplot(122, projection='3d')
#     ax.plot_surface(rr, cc, field_map[:, :, 1], rstride=3, cstride=3, linewidth=1, antialiased=True)
#
#     plt.show()

# --- Classes --- #

class DispFieldMapsPatchCreator:
    def __init__(self, global_shape, patch_res, map_count, modes, gauss_mu_range, gauss_sig_scaling):
        self.global_shape = global_shape
        self.patch_res = patch_res
        self.map_count = map_count
        self.modes = modes
        self.gauss_mu_range = gauss_mu_range
        self.gauss_sig_scaling = gauss_sig_scaling

        self.current_patch_index = -1
        self.patch_boundingboxes = image_utils.compute_patch_boundingboxes(self.global_shape, stride=self.patch_res, patch_res=self.patch_res)
        self.disp_maps = None
        self.create_new_disp_maps()

    def create_new_disp_maps(self):
        print("DispFieldMapsPatchCreator.create_new_disp_maps()")
        self.disp_maps = create_displacement_field_maps(self.global_shape, self.map_count, self.modes, self.gauss_mu_range, self.gauss_sig_scaling)

    def get_patch(self):
        self.current_patch_index += 1

        if len(self.patch_boundingboxes) <= self.current_patch_index:
            self.current_patch_index = 0
            self.create_new_disp_maps()

        patch_boundingbox = self.patch_boundingboxes[self.current_patch_index]
        patch_disp_maps = self.disp_maps[:, patch_boundingbox[0]:patch_boundingbox[2], patch_boundingbox[1]:patch_boundingbox[3], :]
        return patch_disp_maps

# --- --- #


def stretch(array):
    mini = np.min(array)
    maxi = np.max(array)
    if maxi - mini:
        array -= mini
        array *= 2 / (maxi - mini)
        array -= 1
    return array


def crop_center(array, out_shape):
    assert len(out_shape) == 2, "out_shape should be of length 2"
    in_shape = np.array(array.shape[:2])
    start = in_shape // 2 - (out_shape // 2)
    out_array = array[start[0]:start[0] + out_shape[0], start[1]:start[1] + out_shape[1], ...]
    return out_array


def multivariate_gaussian(pos, mu, sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """


    n = mu.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2 * np.pi) ** n * sigma_det)
    # This einsum call calculates (x-mu)T.sigma-1.(x-mu) in a vectorized
    # way across all the input variables.

    # print("\tStarting to create multivariate Gaussian")
    # start = time.time()

    # print((pos - mu).shape)
    # print(sigma_inv.shape)
    try:
        fac = np.einsum('...k,kl,...l->...', pos - mu, sigma_inv, pos - mu, optimize=True)
    except:
        fac = np.einsum('...k,kl,...l->...', pos - mu, sigma_inv, pos - mu)
    # print(fac.shape)

    # end = time.time()
    # print("\tFinished Gaussian in {}s".format(end - start))

    return np.exp(-fac / 2) / N


def create_multivariate_gaussian_mixture_map(shape, mode_count, mu_range, sig_scaling):
    shape = np.array(shape)
    # print("Starting to create multivariate Gaussian mixture")
    # main_start = time.time()

    dim_count = 2
    downsample_factor = 4
    dtype = np.float32

    mu_scale = mu_range[1] - mu_range[0]
    row = np.linspace(mu_range[0], mu_range[1], mu_scale*shape[0]/downsample_factor, dtype=dtype)
    col = np.linspace(mu_range[0], mu_range[1], mu_scale*shape[1]/downsample_factor, dtype=dtype)
    rr, cc = np.meshgrid(row, col, indexing='ij')
    grid = np.stack([rr, cc], axis=2)

    mus = np.random.uniform(mu_range[0], mu_range[1], (mode_count, dim_count, 2)).astype(dtype)
    # gams = np.random.rand(mode_count, dim_count, 2, 2).astype(dtype)
    signs = np.random.choice([1, -1], size=(mode_count, dim_count))

    # print("\tAdding gaussian mixtures one by one")
    # start = time.time()

    # if JOBLIB:
    #     # Parallel computing of multivariate gaussians
    #     inputs = range(8)
    #
    #     def processInput(i):
    #         size = 10 * i + 2000
    #         a = np.random.random_sample((size, size))
    #         b = np.random.random_sample((size, size))
    #         n = np.dot(a, b)
    #         return n
    #
    #     num_cores = multiprocessing.cpu_count()
    #     print("num_cores: {}".format(num_cores))
    #     # num_cores = 1
    #
    #     results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
    #     for result in results:
    #         print(result.shape)
    #
    #     gaussian_mixture = np.zeros_like(grid)
    # else:
    gaussian_mixture = np.zeros_like(grid)
    for mode_index in range(mode_count):
        for dim in range(dim_count):
            sig = (sig_scaling[1] - sig_scaling[0]) * sklearn.datasets.make_spd_matrix(2) + sig_scaling[0]
            # sig = (sig_scaling[1] - sig_scaling[0]) * np.dot(gams[mode_index, dim], np.transpose(gams[mode_index, dim])) + sig_scaling[0]
            sig = sig.astype(dtype)
            multivariate_gaussian_grid = signs[mode_index, dim] * multivariate_gaussian(grid, mus[mode_index, dim], sig)
            gaussian_mixture[:, :, dim] += multivariate_gaussian_grid

    # end = time.time()
    # print("\tFinished adding gaussian mixtures in {}s".format(end - start))

    # squared_gaussian_mixture = np.square(gaussian_mixture)
    # magnitude_disp_field_map = np.sqrt(squared_gaussian_mixture[:, :, 0] + squared_gaussian_mixture[:, :, 1])
    # max_magnitude = magnitude_disp_field_map.max()

    gaussian_mixture[:, :, 0] = stretch(gaussian_mixture[:, :, 0])
    gaussian_mixture[:, :, 1] = stretch(gaussian_mixture[:, :, 1])

    # Crop
    gaussian_mixture = crop_center(gaussian_mixture, shape//downsample_factor)

    # plot_field_map(gaussian_mixture)

    # Upsample mixture
    # gaussian_mixture = skimage.transform.rescale(gaussian_mixture, downsample_factor)
    gaussian_mixture = skimage.transform.resize(gaussian_mixture, shape)

    main_end = time.time()
    # print("Finished multivariate Gaussian mixture in {}s".format(main_end - main_start))

    return gaussian_mixture


def create_displacement_field_maps(shape, map_count, modes, gauss_mu_range, gauss_sig_scaling, seed=None):
    if seed is not None:
        np.random.seed(seed)
    disp_field_maps_list = []
    for disp_field_map_index in range(map_count):
        disp_field_map_normed = create_multivariate_gaussian_mixture_map(shape,
                                                                         modes,
                                                                         gauss_mu_range,
                                                                         gauss_sig_scaling)
        disp_field_maps_list.append(disp_field_map_normed)
    disp_field_maps = np.stack(disp_field_maps_list, axis=0)

    return disp_field_maps


if CV2:
    def find_homography_4pt(src, dst):
        """
        Estimates the homography that transofmrs src points into dst points.
        Then converts the matrix representation into the 4 points representation.

        :param src:
        :param dst:
        :return:
        """
        h_mat, _ = cv2.findHomography(src, dst)
        h_4pt = convert_h_mat_to_4pt(h_mat)
        return h_4pt


    def convert_h_mat_to_4pt(h_mat):
        src_4pt = np.array([[
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]], dtype=np.float64)
        h_4pt = cv2.perspectiveTransform(src_4pt, h_mat)
        return h_4pt


    def convert_h_4pt_to_mat(h_4pt):
        src_4pt = np.array([[
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]], dtype=np.float64)
        h_mat = cv2.getPerspectiveTransform(src_4pt, h_4pt)
        return h_mat


    def field_map_to_image(field_map):
        mag, ang = cv2.cartToPolar(field_map[..., 0], field_map[..., 1])
        hsv = np.zeros((field_map.shape[0], field_map.shape[1], 3))
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = hsv.astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return rgb
else:
    def find_homography_4pt(src, dst):
        print("cv2 is not available, the find_homography_4pt(src, dst) function cannot work!")


    def convert_h_mat_to_4pt(h_mat):
        print("cv2 is not available, the convert_h_mat_to_4pt(h_mat) function cannot work!")


    def convert_h_4pt_to_mat(h_4pt):
        print("cv2 is not available, the convert_h_4pt_to_mat(h_4pt) function cannot work!")


    def field_map_to_image(field_map):
        print("cv2 is not available, the field_map_to_image(field_map) function cannot work!")


def main():
    shape = (220, 220)
    mode_count = 30
    mu_range = [0, 1]
    sig_scaling = [0.0, 0.002]
    create_multivariate_gaussian_mixture_map(shape, mode_count, mu_range, sig_scaling)


if __name__ == "__main__":
    main()
