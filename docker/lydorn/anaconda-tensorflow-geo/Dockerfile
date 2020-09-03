FROM lydorn/anaconda-tensorflow

MAINTAINER Nicolas Girard <nicolas.jp.girard@gmail.com>

RUN apt-get update && apt-get install -y libgtk2.0

# Image manipulation
RUN conda install -c menpo opencv3=3.1.0

#RUN conda install -c conda-forge scikit-image -y
# Commented-out because it gets installed by pip later:

RUN conda install pyproj

## Install gdal
#RUN apt-get update
#RUN apt-get install -y software-properties-common
#RUN apt-add-repository ppa:ubuntugis/ubuntugis-unstable
#RUN apt-get update
#RUN apt-get install -y libgdal-dev
## See https://gist.github.com/cspanring/5680334:
#RUN pip install gdal --global-option=build_ext --global-option="-I/usr/include/gdal/"

# Install gdal
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-add-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update
RUN apt-get install -y libgdal-dev
# See https://gist.github.com/cspanring/5680334:
RUN pip install gdal==2.2.0 --global-option=build_ext --global-option="-I/usr/include/gdal/"

# Install overpy
RUN pip install overpy

# Install shapely
RUN conda install -c conda-forge shapely -y

# --- Install pycocotools
#RUN pip install git+https://github.com/crowdai/coco.git#subdirectory=PythonAPI
# Remove imageio files prior to the instalation of scikit-image requiring imageio to be upgraded:
RUN rm -rf /opt/conda/lib/python3.6/site-packages/imageio*

RUN pip install -U --no-cache-dir scikit-image
RUN pip install -U --no-cache-dir cython
RUN conda install numpy=1.19.1
RUN pip install --no-cache-dir "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
# ---

RUN conda install -c conda-forge jsmin

#RUN conda install -c anaconda joblib -y

# Cleanup
RUN apt-get clean && \
    apt-get autoremove
