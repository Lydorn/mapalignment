Docker container with the following installed:
- CUDA, cuDNN and Anaconda3 (from lyforn/anaconda image)
- Tensorflow 1.6

Used in other Dockerfiles

Build image:
```
sh build.sh
```

Or (does not take care of dependencies):
```
nvidia-docker build -t lydorn/anaconda-tensorflow --rm .
```

Run container (change the path to the epitome-polygon-deep-learning folder if it is not in your home folder):
```
docker run --runtime=nvidia -it --rm -p 8888:8888 -p 6006:6006 -v ~/epitome-polygon-deep-learning:/workspace lydorn/anaconda-tensorflow
```

On deepsat:
```
docker run --runtime=nvidia -it --rm -p 8890:8888 -p 6008:6006 -v /local/shared/epitome-polygon-deep-learning:/workspace lydorn/anaconda-tensorflow
```