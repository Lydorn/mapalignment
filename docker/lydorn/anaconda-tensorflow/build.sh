#!/usr/bin/env bash

set -e

cd ../anaconda
sh build.sh

cd ../anaconda-tensorflow
docker build -t lydorn/anaconda-tensorflow --rm .