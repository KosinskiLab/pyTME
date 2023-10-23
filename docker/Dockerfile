FROM ubuntu:22.04

ARG PYTHON_VERSION=3.11

RUN apt-get update --fix-missing && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        wget \
        time \
        git
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

RUN python3 -mpip install  \
    mrcfile \
    numpy \
    pyfftw \
    pytest \
    PyWavelets \
    PyYAML \
    scikit-image \
    scikit_learn \
    scipy \
    psutil

RUN mkdir src &&
    cd src && \
    git clone https://github.com/KosinskiLab/pyTME.git && \
    python3 -mpip install .
