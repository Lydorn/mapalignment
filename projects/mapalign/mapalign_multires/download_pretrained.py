import os.path
import urllib.request
import zipfile

# --- Params --- #

ressource_filename_list = ["runs.igarss2019.zip"]
ressource_dirpath_url = "https://www-sop.inria.fr/members/Nicolas.Girard/downloads/mapalignment"

script_filepath = os.path.realpath(__file__)
zip_download_dirpath = os.path.join(os.path.dirname(script_filepath), "runs.zip")
download_dirpath = os.path.join(os.path.dirname(script_filepath), "runs")

# --- --- #

for ressource_filename in ressource_filename_list:
    ressource_url = os.path.join(ressource_dirpath_url, ressource_filename)
    print("Downloading zip from {}, please wait... (approx. 406MB to download)".format(ressource_url))
    urllib.request.urlretrieve(ressource_url, zip_download_dirpath)
    print("Extracting zip...")
    zip_ref = zipfile.ZipFile(zip_download_dirpath, 'r')
    os.makedirs(download_dirpath)
    zip_ref.extractall(download_dirpath)
    zip_ref.close()
    os.remove(zip_download_dirpath)
