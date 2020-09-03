#!/usr/bin/env bash

docker run --rm -it --init --gpus all --ipc=host --network=host -v ~/mapalignment:/workspace -v ~/data:/data -e NVIDIA_VISIBLE_DEVICES=0 lydorn/anaconda-tensorflow