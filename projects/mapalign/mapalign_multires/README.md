# Inference

If you want to quickly perform inference on you own dataset, write a ```read.py``` script like this one: [read.py](../../../data/AerialImageDataset/read.py)
to implement a ```load_gt_data()``` function that reads the specific format of your dataset and outputs the image, its metadata and its ground truth data
in a consistent python format. Then duplicate and adapt these 2 scripts:
- [config_test_inria.py](config_test_inria.align_osm_gt.py)
- [2_test_aerial_image.align_osm_gt.py](2_test_aerial_image.align_osm_gt.py)
Instructions in the scripts will help you in changing relevant parts (comments starting with "CHANGE").

# Training

This project uses a multi-resolution approach.
To handle different image ground resolution well, a REFERENCE_PIXEL_SIZE parameter is set in [config.py] as well as in the config script of all dataset pre-processing.
All images are pre-processed by first virtually rescaling them so that they have a ground resolution of REFERENCE_PIXEL_SIZE and
then are downsampled with the specified downsampling factors. When training we then refer to a specific resolution by its downsampling factor (abbreviated ds_fac), for example ds_fac_1, ds_fac_2, etc.
Classically we train 4 different networks on 4 different resolutions: ds_fac_8, ds_fac_4, ds_fac_2 and ds_fac_1.

The training can be performed independently on 4 different GPUs.
It can also be done one resolution after the other,
the first resolution (ds_fac_8) training from scratch and
the following resolutions trained from the pre-trained weights of the previously trained resolution.

The [config.py](config.py) script contains all the hyper-parameters of the method.
The [1_train.py](1_train.py) is the script to be launched for training the networks.
It must be executed with proper arguments:
```
python 1_train.py --new_run --run_name=ds_fac_8 --batch_size=32 --ds_fac=8
```
To continue training of a previous run named "ds_fac_8":
```
python 1_train.py --run_name=ds_fac_8 --batch_size=32 --ds_fac=8
```

# Run management

A simple run management system has been implemented to handle different runs efficiently.
It is based on identifying a run by its name and timestamp. The argument ```--run_name``` refers to the last run by this name based on its timestamp.
If a new training session is started with the ```--new_run``` flag and a previously used name, the two runs will be differentiated by their timestamp.
More info on the run management system can be found in the script [run_utils.py](../../utils/run_utils.py)

# Testing

Several scripts are used to test and more generally use the trained models.

## Testing on Bradbury Buildings

All configuration parameters for these tests are in the 
[config_test_bradbury_buildings.py](config_test_bradbury_buildings.py) script.

### Building alignment
First generate several displacement maps for every test image by running 
[2_test_bradbury_buildings.1_generate_disps.py](2_test_bradbury_buildings.1_generate_disps.py).
Then align the displaced buildings with 
[2_test_bradbury_buildings.2_align.py](2_test_bradbury_buildings.2_align.py).
The point-distance to ground truth is measured and saved for later plotting.

### New building segmentation
Test the segmentation of new buildings by running
[2_test_bradbury_buildings.2_align.py](2_test_bradbury_buildings.3_detect_new_buildings.py). 
This script feeds the network only with the image, the polygon raster is empty (black). 
The network segments buildings that are not present in the input. 
The IoU (Intersection over Union) is measured and saved for later plotting.

## Testing on Inria Aerial Image Dataset

All configuration parameters for these tests are in the 
[config_test_aerial_image.py](config_test_aerial_image.py) script.

<!-- TODO: finish testing on this dataset -->

## Testing Stereo Dataset

The Stereo Dataset is composed of a few images from a company thus we cannot provide them.
Nonetheless the code to test the alignement on them is here.
Each area has 2 satellite images corresponding to 2 different views with a different angle.
Ground-Truth data is available on both images. This way true displacements can be retrieved:
The GT polygons for one view are displaced polygons for the other view. 
This type of displacement is very common in maps, it is due to drawing polygons on one image with an angle.
This dataset allows us to measure real-case scenario of alignment. 
It also provides a way to estimate the height of buildings, by aligning polygons on both images.

### Artificial building alignment

Apply the alignment method on generated artificial displaced polygons: 
[2_test_stereo.py](2_test_stereo.py).

### Real-case building alignment

Apply the alignment method on real-case displaced polygons (taking the other view's GT polygons as displaced polygons): 
[2_test_stereo_real_displacements.py](2_test_stereo_real_displacements.py).

## Plotting the results

### Alignment test plot

The script [3_test_plot.2_align.py](3_test_plot.2_align.py) plots alignment result curves.
Parameters can be set at the start of the script.

### Segmentation test plot

The script [3_test_plot.3_seg.py](3_test_plot.3_seg.py) plots segmentation result curves.
Parameters can be set at the start of the script.

## Building Height Estimation

Building polygons from 2 views of the same area can be aligned by a previous step:
[Real-case building alignment](#Real-case building alignment).
Then their height can by estimated with the script
[4_compute_building_heights.py](4_compute_building_heights.py).
Some parameters can be set at the start of the script.

## Model buildings in 3D

Now that we have building polygons and their estimated height, a 3D model can be built 
(it is a 2.5D model to be more precise). Open the file [5_model_buildings.blend](5_model_buildings.blend) 
from the command-line (this is needed for the script used inside the file to use the correct paths):
```
blender 5_model_buildings.blend
```
Blender is an open-source 3D program that has a Python API to easily build scripts using Blender's capabilities.
Once opened, you can see on the left a script window where the 
[5_model_buildings.py](5_model_buildings.py) script is loaded. With the mouse over that window,
press <Alt-P> to run the script within Blender. 
It uses the previously-computed building heights to extrude the building polygons

## Test the height estimation

Ground-Truth heights are available for our Stereo Dataset given to us by a company.
This script runs performs some measures to see how far we are from the Ground-Truth.

# Brief explanation of other scripts

All other scripts in this folder are not meant to be run on their own, they are used by the other scripts.

[loss_utils.py](loss_utils.py) implements loss functions built in the TF graph (used by model.py)

[model.py](model.py) implements the MapAlignModel Python class used to create the model, optimize it and run it. (used by 1_train.py)

[model_utils.py](model_utils.py) implements the neural network building functions (used by model.py)

[multires_pipeline.py](multires_pipeline.py) implements inference for full multi-resolution pipeline,
instantiating resolution-specific trained models and applying them in succession in a coarse-to-fine manner (used by test.py)

[test.py](test.py) implements several testing functions to test the accuracy of the method (used by all 2_test_*.py scripts)

# Glossary

Here is a list of words used throughout the project with their corresponding definition as some can be ambiguous.

| Word | Definition |
| ------ | ---------- |
| tile    | square portion of an image produced by pre-processing so that very big images are split into manageable pieces     |
| patch   | square crop of a tile produced by online-processing which is fed to the network (crop needed to fit the network's input and output parameters)    |
| layer   | regular NN layer    |
| level   | resolution level within a model/network. It is a set of layers whose inputs are from the same spatial resolution     |
