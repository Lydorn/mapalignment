import os.path
import urllib.request
import zipfile

BASE_URL = 'https://figshare.com/collections/Aerial_imagery_object_identification_dataset_for_building_and_road_detection_and_building_height_estimation/3290519'
FILE_URL_FORMAT = "https://ndownloader.figshare.com/articles/{}/versions/1"
FILE_METADATA_LIST = [
    {
        "dirname": "Arlington",
        "id": "3485204",
    },
    {
        "dirname": "Atlanta",
        "id": "3504308",
    },
    {
        "dirname": "Austin",
        "id": "3504317",
    },
    {
        "dirname": "DC",
        "id": "3504320",
    },
    {
        "dirname": "NewHaven",
        "id": "3504323",
    },
    {
        "dirname": "NewYork",
        "id": "3504326",
    },
    {
        "dirname": "Norfolk",
        "id": "3504347",
    },
    {
        "dirname": "SanFrancisco",
        "id": "3504350",
    },
    {
        "dirname": "Seekonk",
        "id": "3504359",
    },
    {
        "dirname": "Data_Description",
        "id": "3504413",
    }
]

DOWNLOAD_DIRPATH = "raw"


if not os.path.exists(DOWNLOAD_DIRPATH):
    os.makedirs(DOWNLOAD_DIRPATH)

for file_metadata in FILE_METADATA_LIST:
    dirname = file_metadata["dirname"]
    id = file_metadata["id"]
    download_dirpath = os.path.join(DOWNLOAD_DIRPATH, dirname)
    zip_download_dirpath = download_dirpath + ".zip"
    if not os.path.exists(download_dirpath):
        print("Downloading {}".format(dirname))
        urllib.request.urlretrieve(FILE_URL_FORMAT.format(id), zip_download_dirpath)
        zip_ref = zipfile.ZipFile(zip_download_dirpath, 'r')
        os.makedirs(download_dirpath)
        zip_ref.extractall(download_dirpath)
        zip_ref.close()
        os.remove(zip_download_dirpath)
    else:
        print("Directory {} already exists so skip download (remove directory if you want to download again)")
