FROM lydorn/dl-base

MAINTAINER Nicolas Girard <nicolas.jp.girard@gmail.com>

# Install Anaconda3
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH

# Install version 3.6.8 of Python
RUN conda install python=3.6.8

# Update conda
#RUN conda update -n base conda
