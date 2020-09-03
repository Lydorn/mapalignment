#!/usr/bin/env bash

set -e

cd ../dl-base
sh build.sh

cd ../anaconda
docker build -t lydorn/anaconda --rm .