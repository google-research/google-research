# TF3D Setup

[TOC]

## Requirements

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB?logo=python)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.3.0](https://img.shields.io/badge/TensorFlow-2.3-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.3.0)
[![Tensorflow Datasets](https://img.shields.io/badge/TensorFlow%20Datasets-4.1.0-FF6F00?logo=tensorflow)](https://github.com/tensorflow/datasets)
[![Numpy](https://img.shields.io/badge/Numpy-1.18.5-000000?&logo=numpy)](https://numpy.org)
[![Gin Config](https://img.shields.io/badge/Gin%20Config-0.4.0-000000?&logo=random)](https://github.com/google/gin-config)

## Install

The following steps are based on Ubuntu 18.04 on Google Cloud Platform, please
modify it based on your settings.

1.  Install the following packages:

    ```bash
    sudo apt update
    sudo apt install subversion git virtualenv
    ```

    Install
    [CUDA 10.1](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)
    and [cuDNN 7.6.5](https://developer.nvidia.com/rdp/cudnn-archive).

1.  Download the `tf3d` subdirectory:

    ```bash
    svn export https://github.com/google-research/google-research/trunk/tf3d
    # Or
    git clone https://github.com/google-research/google-research.git --depth=1
    ```

1.  Create and enter a virtual environment (optional but preferred):

    ```bash
    virtualenv -p python3 tf3d_env
    source ./tf3d_env/bin/activate
    ```

1.  Install the dependencies:

    ```bash
    pip install -r tf3d/requirements.txt
    ```

1.  Install the
    <a href='https://github.com/tensorflow/models/tree/master/research/object_detection#tensorflow-2x'>Tensorflow
    Object Detection API</a>.

    For Ubuntu 18.04, you may need these dependent packages:

    ```bash
    sudo apt update
    sudo apt install protobuf-compiler python3-dev
    ```

1.  Follow the instructions in `tf3d/ops` folder to install the `docker` and
    compile the custom ops.

## Prepare the data

Please follow the instructions in [TensorFlow 3D Datasets](tf3d_datasets.md).

