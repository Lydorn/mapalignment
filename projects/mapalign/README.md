# Introduction

This folder contains all the scripts for the map alignment and updating project.
This README presents the steps to pre-process datasets, train networks and apply the method on new images.

# Python environment

The code uses a few Python libraries such as Tensorflow, etc.
The docker image 
[lydorn/anaconda-tensorflow-geo](../../../docker/lydorn/anaconda-tensorflow-geo) has all the needed dependencies.
See the instructions in the [docker](../../../docker) folder to install docker and build that image.

# Pre-process training datasets

## Download datasets

Three datasets are used for training and have individual instructions for dowloading in their respective folder:
- [Inria Aerial Image Dataset](../../../data/AerialImageDataset):
- [Aerial imagery object identification dataset for building and road detection, and building height estimation](../../../data/bradbury_buildings_roads_height_dataset):
- [Mapping Challenge from CrowdAI Dataset](../../../data/mapping_challenge_dataset):

## Pre-process datasets

All scripts for dataset handling relative to the alignment project are located in the [dataset_utils](../dataset_utils) folder.
See the README in that folder for instructions on dataset pre-processing.

## Train and use models

The useful scripts for training and using the models are in the [mapalign_multires](../dataset_utils) folder.