#!/usr/bin/env bash

set -e

cd ../anaconda-tensorflow
sh build.sh

cd ../anaconda-tensorflow-geo
nvidia-docker build -t lydorn/anaconda-tensorflow-geo --rm .