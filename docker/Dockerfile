# Base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Make sure the location is set!
WORKDIR /jetssl-lite

# Local and environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install \
        build-essential \
        git \
        wget \
        make \
        hdf5-tools \
        gcc \
        graphviz

# Update python pip
RUN python -m pip install --upgrade pip
RUN python --version
RUN python -m pip --version

# Need to install these before running requirements.txt
RUN python -m pip install packaging
RUN python -m pip install ninja

# Intall python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade -r requirements.txt
