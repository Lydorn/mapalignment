FROM nvidia/cuda:9.0-devel-ubuntu16.04

MAINTAINER Nicolas Girard <nicolas.jp.girard@gmail.com>

# Install useful tools
RUN apt-get update && apt-get install -y \
    wget \
    git \
    sudo \
    libcudnn7=7.0.5.15-1+cuda9.0 \
    libcudnn7-dev=7.0.5.15-1+cuda9.0 \
    fish

RUN rm -rf /var/lib/apt/lists/*

# Setup main volume
ENV WORKDIR=/workspace
VOLUME $WORKDIR
WORKDIR $WORKDIR

CMD fish