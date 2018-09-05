## Download
Main URL of the dataset:
https://figshare.com/collections/Aerial_imagery_object_identification_dataset_for_building_and_road_detection_and_building_height_estimation/3290519

It is easier to execute the script:
```
python download.py
```

Some files may be named incorrectly. For example, "Atlanta/Atlanta_01buildingCoords.csv" must be renamed to "Atlanta/Atlanta_01_buildingCoords.csv"

The folder structure should look like this:

```
bradbury_buildings_road_height_dataset
|-- raw
|   |-- Arlington
|   |   |-- Arlington_01.tif
|   |   `-- ...
|   |-- Atlanta
|   |   |-- Atlanta_01.tif
|   |   `-- ...
|   `-- ...
|-- download.py
|-- read.py
`-- README.md (this file)
```

## Images metadata

Pixel sizes (in meters):
Arlington: 0.3
Atlanta: 0.1524 (0.5ft)
Austin: 0.1524 (0.5ft)
DC: 0.16
NewHaven: 0.3
NewYork: 0.1524 (0.5ft)
Norfolk: 0.3048 (1ft)
SanFransisco: 0.3
Seekonk: 0.3

## Analyse of alignement quality:

Arlington_01 is not well annotated (suburb)
Arlington_02 is not well annotated (suburb)
Arlington_03 is ok (suburb, some very well annotated, some not)
Atlanta_01 (but has angles) is well annotated (suburb)
Atlanta_02 (but has angles) is well annotated (suburb)
Atlanta_03 (but has angles in some places) is well annotated (suburb)
Austin_02 is very well annotated (and no angles) (suburb)
Austin_01 is very well annotated (and no angles) (suburb)
Austin_03 is very well annotated (and no angles) (suburb)
DC_01 is very well annotated (but has strong angles) (city center)
DC_02 is very well annotated (but has strong angles) (city center)
NewHaven_01 is well annotated (but has strong angles) (suburb)
NewHaven_02 is well annotated (but has strong angles) (suburb)
NewYork_01 is ok (but has strong angles) (suburb)
NewYork_02 is ok (but has angles) (suburb)
NewYork_03 is ok (but has strong angles) (suburb)
Norfolk_02 is very well annotated (and no angles) (suburb)
Norfolk_03 is very well annotated (and no angles) (has very few annotated buildings) (suburb)
Norfolk_01 is very well annotated (and no angles) (suburb)
SanFrancisco_01 is very well annotated (and no angles) (but noisy image) (suburb)
SanFrancisco_02 is very well annotated (and no angles) (but noisy image) (suburb)
SanFrancisco_03 is very well annotated (and no angles) (but noisy image) (suburb)
Seekonk_01 is not well annotated (suburb)
Seekonk_02 is not well annotated (suburb)
Seekonk_03 is not well annotated (suburb)