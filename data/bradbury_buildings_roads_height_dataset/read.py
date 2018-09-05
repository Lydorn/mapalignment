import os.path
import csv
import sys
import numpy as np

sys.path.append("../utils")
import visualization

import skimage.io

CITY_METADATA_DICT = {
    "Arlington": {
        "pixelsize": 0.3,
    },
    "Atlanta": {
        "pixelsize": 0.1524,
    },
    "Austin": {
        "pixelsize": 0.1524,
    },
    "DC": {
        "pixelsize": 0.16,
    },
    "NewHaven": {
        "pixelsize": 0.3,
    },
    "NewYork": {
        "pixelsize": 0.1524,
    },
    "Norfolk": {
        "pixelsize": 0.3048,
    },
    "SanFrancisco": {
        "pixelsize": 0.3,
    },
    "Seekonk": {
        "pixelsize": 0.3,
    },
}

DIRNAME_FORMAT = "{city}"  # City name
IMAGE_NAME_FORMAT = "{city}_{number:02d}"
IMAGE_FILENAME_FORMAT = IMAGE_NAME_FORMAT + ".tif"  # City name, number
POLYGONS_FILENAME_FORMAT = "{city}_{number:02d}_buildingCoord.csv"  # City name, number


def get_image_filepath(raw_dirpath, city, number):
    dirname = DIRNAME_FORMAT.format(city=city)
    filename = IMAGE_FILENAME_FORMAT.format(city=city, number=number)
    filepath = os.path.join(raw_dirpath, dirname, filename)
    return filepath


def get_polygons_filepath(raw_dirpath, city, number):
    dirname = DIRNAME_FORMAT.format(city=city)
    filename = POLYGONS_FILENAME_FORMAT.format(city=city, number=number)
    filepath = os.path.join(raw_dirpath, dirname, filename)
    return filepath


def load_image(raw_dirpath, city, number):
    filepath = get_image_filepath(raw_dirpath, city, number)
    image_array = skimage.io.imread(filepath)
    image_array = np.array(image_array, dtype=np.float64) / 255
    if image_array.shape[2] == 4:
        if city == "SanFrancisco":
            # San Francisco needs special treatment because its transparent pixels are white!
            alpha = image_array[:, :, 3:4]
            image_array = image_array[:, :, :3] * alpha  # Apply alpha in 4th channel (IR channel) if present
        else:
            image_array = image_array[:, :, :3]
    image_array = np.round(image_array * 255).astype(np.uint8)

    # The following is writen this way for future image-specific addition of metadata:
    image_metadata = {
        "filepath": filepath,
        "pixelsize": CITY_METADATA_DICT[city]["pixelsize"]
    }

    return image_array, image_metadata


def load_polygons(raw_dirpath, city, number):
    filepath = get_polygons_filepath(raw_dirpath, city, number)
    polygons_coords_list = []
    with open(filepath, 'r') as coords_csv:
        csv_reader = csv.reader(coords_csv, delimiter=',')
        for row_index, row in enumerate(csv_reader):
            if row_index != 0:  # Skip header
                polygon_coords = read_csv_row(row)
                polygons_coords_list.append(polygon_coords)
    return polygons_coords_list


def read_csv_row(row):
    # print("Polygon: {}".format(row[1]))
    coord_list = []
    for item in row[3:]:
        try:
            item_float = float(item)
            coord_list.append(item_float)
        except ValueError:
            pass
    coord_array = np.array(coord_list, dtype=np.float64)
    coord_array = np.reshape(coord_array, (-1, 2))
    # Switch from xy coordinates to ij:
    coord_array[:, 0], coord_array[:, 1] = coord_array[:, 1], coord_array[:, 0].copy()
    # polygon_utils.plot_polygon(gt_polygon_coords, color=None, draw_labels=False, label_direction=1)
    # gt_polygon_coords_no_nans = np.reshape(gt_polygon_coords[~np.isnan(gt_polygon_coords)], (-1, 2))
    return coord_array


def load_gt_data(raw_dirpath, city, number):
    # Load image
    image_array, image_metadata = load_image(raw_dirpath, city, number)

    # Load CSV data
    gt_polygons = load_polygons(raw_dirpath, city, number)

    # TODO: remove
    # gt_polygons_filepath = get_polygons_filepath(raw_dirpath, city, number)
    # visualization.save_plot_image_polygons(gt_polygons_filepath + ".polygons.png", image_array, [], gt_polygons, [])
    # TODO end

    return image_array, image_metadata, gt_polygons


def main():
    raw_dirpath = "raw"
    city = "Atlanta"
    number = 1
    image_array, image_metadata, gt_polygons = load_gt_data(raw_dirpath, city, number)

    print(image_array.shape)
    print(image_metadata)
    print(gt_polygons)


if __name__ == "__main__":
    main()
