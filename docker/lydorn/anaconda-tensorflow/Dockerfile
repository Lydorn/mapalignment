FROM lydorn/anaconda

MAINTAINER Nicolas Girard <nicolas.jp.girard@gmail.com>

# Install Tensorflow
RUN pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp36-cp36m-linux_x86_64.whl
RUN pip install update

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Cleanup
RUN apt-get clean && \
    apt-get autoremove

COPY start_jupyter.sh $WORKDIR