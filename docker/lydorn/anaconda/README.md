Docker container with the following installed:
- CUDA and cuDNN (from lydorn/dl-base)
- Anaconda3 5.1.0
- Python 3.6.8

Used in other Dockerfiles

Build image:
```
sh build.sh
```

Or (does not take care of dependencies):
```
nvidia-docker build -t lydorn/anaconda --rm .
```

Run container:
```
docker run --runtime=nvidia -it --rm -v ~/epitome-polygon-deep-learning:/workspace lydorn/anaconda
```