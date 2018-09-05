Docker container with the following installed:
- CUDA and cuDNN (from nvidia/nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04)
- Useful tools

Used in other Dockerfiles

Build image:
```
nvidia-docker build -t lydorn/dl-base --rm .
```

Run container:
```
docker run --runtime=nvidia -it --rm -v ~/epitome-polygon-deep-learning:/workspace lydorn/dl-base
```