FROM ubuntu:bionic

# Update base container install
RUN apt-get clean
RUN apt-get update
RUN apt-get upgrade -y

# Install GDAL dependencies
RUN apt-get install -y python3-pip 
RUN apt-get install -y libgdal-dev 
RUN apt-get install -y locales 

# Install dependencies for other packages
RUN apt-get install gcc g++
#RUN apt-get install jpeg-dev zlib-dev

# Ensure locales configured correctly
RUN locale-gen en_US.UTF-8
ENV LC_ALL='en_US.utf8'

# Set python aliases for python3
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# This will install latest version of GDAL
RUN apt-get -y install python3-gdal

# Build context
ADD test.py train.py evaluate.py src /

# Install dependencies for tiling
RUN pip3 install tensorflow && \
    pip3 install numpy && \
    pip3 install pillow && \
    pip3 install keras && \
    pip3 install requests && \
    pip3 install argparse && \
    pip3 install future

RUN pip3 install fiona && \
    pip3 install pyproj

RUN pip3 install shapely && \
    pip3 install pyshp

ENV PYTHONUNBUFFERED = '1'



