#!/usr/bin/env bash

set -e

nvidia-docker build -t lydorn/dl-base --rm .