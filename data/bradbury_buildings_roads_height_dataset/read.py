import os.path
import csv
import numpy as np

import skimage.io

CITY_METADATA_DICT = {
    "Arlington": {
        "pixelsize": 0.3,
        "numbers":  [1, 2, 3],
    },
    "Atlanta": {
        "pixelsize": 0.1524,
        "numbers":  [1, 2, 3],
    },
    "Austin": {
        "pixelsize": 0.1524,
        "numbers":  [1, 2, 3],
    },
    "DC": {
        "pixelsize": 0.16,
        "numbers":  [1, 2],
    },
    "NewHaven": {
        "pixelsize": 0.3,
        "numbers":  [1, 2],
    },
    "NewYork": {
        "pixelsize": 0.1524,
        "numbers":  [1, 2, 3],
    },
    "Norfolk": {
        "pixelsize": 0.3048,
        "numbers":  [1, 2, 3],
    },
    "SanFrancisco": {
        "pixelsize": 0.3,
        "numbers":  [1, 2, 3],
    },
    "Seekonk": {
        "pixelsize": 0.3,
        "numbers":  [1, 2, 3],
    },
}

DIRNAME_FORMAT = "{city}"  # City name
IMAGE_NAME_FORMAT = "{city}_{number:02d}"
IMAGE_FILENAME_EXTENSION = ".tif"
POLYGONS_FILENAME_EXTENSION = "_buildingCoord.csv"


def get_tile_info_list():
    tile_info_list = []
    for city, info in CITY_METADATA_DICT.items():
        for number in info["numbers"]:
            image_info = {
                "city": city,
                "number": number,
            }
            tile_info_list.append(image_info)
    return tile_info_list


def get_image_filepath(raw_dirpath, city, number):
    dirname = DIRNAME_FORMAT.format(city=city)
    image_name = IMAGE_NAME_FORMAT.format(city=city, number=number)
    filename = image_name + IMAGE_FILENAME_EXTENSION
    filepath = os.path.join(raw_dirpath, dirname, filename)
    return filepath


def get_polygons_filepath(raw_dirpath, city, number, polygons_filename_extension):
    dirname = DIRNAME_FORMAT.format(city=city)
    image_name = IMAGE_NAME_FORMAT.format(city=city, number=number)
    filename = image_name + polygons_filename_extension
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


def load_csv(filepath):
    polygons_coords_list = []
    with open(filepath, 'r') as coords_csv:
        csv_reader = csv.reader(coords_csv, delimiter=',')
        for row_index, row in enumerate(csv_reader):
            if row_index != 0:  # Skip header
                polygon_coords = read_csv_row(row)
                polygons_coords_list.append(polygon_coords)
    return polygons_coords_list


def load_polygons_from_npy(filepath):
    try:
        polygons = np.load(filepath)
    except FileNotFoundError:
        print("Filepath {} does not exist".format(filepath))
        polygons = None
    return polygons


def load_polygons(raw_dirpath, city, number, polygons_filename_extension):
    filepath = get_polygons_filepath(raw_dirpath, city, number, polygons_filename_extension)

    _, file_extension = os.path.splitext(filepath)
    if file_extension == ".csv":
        return load_csv(filepath)
    elif file_extension == ".npy":
        return load_polygons_from_npy(filepath)
    else:
        print("WARNING: file extension {} is not handled by this script. Use .csv or .npy.".format(file_extension))

    return None


def load_gt_data(raw_dirpath, city, number, overwrite_polygons_filename_extension=None):
    if overwrite_polygons_filename_extension is None:
        polygons_filename_extension = POLYGONS_FILENAME_EXTENSION
    else:
        polygons_filename_extension = overwrite_polygons_filename_extension

    # Load image
    image_array, image_metadata = load_image(raw_dirpath, city, number)

    # Load CSV data
    gt_polygons = load_polygons(raw_dirpath, city, number, polygons_filename_extension)

    # TODO: remove
    # sys.path.append("../utils")
    # import visualization
    # gt_polygons_filepath = get_polygons_filepath(raw_dirpath, POLYGONS_FILENAME_FORMAT, city, number)
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
