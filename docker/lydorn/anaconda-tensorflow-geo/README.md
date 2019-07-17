Docker container with the following installed:
- CUDA, cuDNN, Anaconda3 and Tensorflow 1.6 (from lyforn/anaconda-tensorflow image)
- OpenCV 3.1.0
- skimage
- pyproj
- gdal
- overpy

Build image:
```
sh build.sh
```

Or (does not take care of dependencies):
```
nvidia-docker build -t lydorn/anaconda-tensorflow-geo --rm .
```

Run container (change the path to the epitome-polygon-deep-learning folder if it is not in your home folder):
```
docker run --runtime=nvidia -it --rm -v ~/epitome-polygon-deep-learning:/workspace lydorn/anaconda-tensorflow-geo
```

On deepsat:
```
docker run --runtime=nvidia -it --rm -v /local/shared/epitome-polygon-deep-learning:/workspace lydorn/anaconda-tensorflow-geo
```