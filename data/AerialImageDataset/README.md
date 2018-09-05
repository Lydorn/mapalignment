## Download data
All 360 images are available on the Inria Aerial Image Labeling Dataset website:
https://project.inria.fr/aerialimagelabeling/

There are 180 "train" images and 180 "test" images.

The folder structure should look like this:

```
AerialImageDataset
|-- raw
|   |-- test
|   |   |-- gt
|   |   `-- images
|   `-- train
|       |-- gt
|       `-- images
|-- fetch_gt_polygons.py
|-- read.py
`-- README.md (this file)
```


## Images metadata

Pixel size (in meters): 0.3

## Fetch GT data from OSM online:

We pull GT polygons from OSM as they are in polygonal form (instead of the raster format of the Inria Aerial Image Labeling Dataset) and
also this way we also have GT data for the test images.

Go to AerialImageDataset/raw and execute to download the OSM GT:
```
python fetch_gt_polygons.py
```

## Analysis of alignement quality:

We manually inspected the OSM GT polygons to see how well it aligns with buildings.
Here is a qualitative assessment of the alignment:

bloomington: bad
bellingham: bad
innsbruck: average
sfo: bad
tyrol-e: bad
austin: good (however a minority of area is not from nadir and so the GT is not aligned with the building rooftops)
chicago: good
kitsap: bad
tyrol-w: average
vienna: average (mix of good and bad)
