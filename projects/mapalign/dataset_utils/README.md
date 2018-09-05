# Introduction

This folder contains scripts to pre-process all three datasets used for training into TFRecords (Tensorflow's preferred format for datasets).
It also contains a script to parse TFRecords and perform online pre-processing of training examples.

# Datasets

## Aerial Image Dataset

The config_aerial_image_multires.py script contains all parameters for the pre-processing of the Aerial Image Dataset.
The preprocess_aerial_image_multires.py script performs the pre-processing with the following steps:
- Generate artificial displacement maps for every image
- Displace ground-truth polygons by these displacement maps
- Rasterize ground-truth and displaced polygons
- Rescale image with the specified downsampling factors (taking into account the REFERENCE_PIXEL_SIZE parameter)
- Split images and polygon rasters into small tiles ready for training

## Bradbury Buildings Dataset

The config_bradbuty_buildings_multires.py script contains all parameters for the pre-processing of the Bradbury Buildings  Dataset.
The preprocess_bradbuty_buildings_multires.py script performs the pre-processing with the following steps:
- Generate artificial displacement maps for every image
- Displace ground-truth polygons by these displacement maps
- Rasterize ground-truth and displaced polygons
- Rescale image with the specified downsampling factors (taking into account the REFERENCE_PIXEL_SIZE parameter)
- Split images and polygon rasters into small tiles ready for training

## Mapping Challenge Dataset

The config_mapping_challenge_multires.py script contains all parameters for the pre-processing of the Mapping Challenge Dataset.
The preprocess_mapping_challenge_multires.py script performs the pre-processing with the following steps:
- Generate artificial displacement maps for every image
- Displace ground-truth polygons by these displacement maps
- Rasterize ground-truth and displaced polygons
- Rescale image with the specified downsampling factors (taking into account the REFERENCE_PIXEL_SIZE parameter)
- Split images and polygon rasters into small tiles ready for training

# TFRecord parsing

All three datasets are pre-processed into TFRecords with the exact same format. Training can then be done seamlessly on an aggregation of these datasets.
TFRecords have to be parsed first and then online pre-processing operations can be applied to trainig examples such as augmentation (rotation, flips, etc.) and mini-batching.
These operations are performed by the read_and_decode() function defined in the dataset_multires.py script. The script can be executed to check if the TFRecords are well parsed.
