# Introduction

Aligning Cadaster Maps with Remote Sensing Images.

# Python environment

The code uses a few Python libraries such as Tensorflow, etc.
The docker image
[lydorn/anaconda-tensorflow-geo](docker/lydorn/anaconda-tensorflow-geo) has all the needed dependencies.
See the instructions in the [docker](docker) folder to install docker and build that image.

# Download datasets

Three datasets are used for training and have individual instructions for dowloading in their respective folder:
- [Inria Aerial Image Dataset](data/AerialImageDataset):
- [Aerial imagery object identification dataset for building and road detection, and building height estimation](data/bradbury_buildings_roads_height_dataset):
- [Mapping Challenge from CrowdAI Dataset](data/mapping_challenge_dataset):

# Pre-process datasets

All scripts for dataset handling relative to the alignment project are located in the [dataset_utils](projects/mapalign/dataset_utils) folder.
See the README in that folder for instructions on dataset pre-processing.

# Train and use models

The useful scripts for training and using the models are in the [mapalign_multires](projects/mapalign/mapalign_multires) folder.