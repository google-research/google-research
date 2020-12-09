# TF3D Usage

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

Please follow the instructions in `tf3d/datasets` folder.

## Train

For each task and dataset, there is a corresponding `gin` config file and `bash`
script. You can edit the `gin` config file if you are interested in configuring
the network or loss, and edit its `bash` script for filling in the dataset path,
configuring the optimization and training.

Please note that the configuration in `bash` script will override the `gin`
config file.

For example, the following steps are for training semantic segmentation model on
Waymo Open Dataset:

```bash
# Edit the gin config file "tf3d/semantic_segmentation/configs/waymo_train.gin"
# Edit the script file "tf3d/semantic_segmentation/scripts/waymo/run_train_locally.sh"
# Run the train script (Optional: set ${CUDA_VISIBLE_DEVICES} before running)
bash tf3d/semantic_segmentation/scripts/waymo/run_train_locally.sh
```

The `${TRAIN_DIR}` (defined in the script) will contain a `train` folder with
Tensorboard `events` file, and a `model` folder with saved checkpoints.

## Evaluation

Similar to training, there is an eval `gin` config file, and a `bash` script for
for each task and dataset.

For example, the following steps are for evaluating semantic segmentation model
on Waymo Open Dataset:

```bash
# Edit the gin config file "tf3d/semantic_segmentation/configs/waymo_eval.gin"
# Edit the script file "tf3d/semantic_segmentation/scripts/waymo/run_eval_locally.sh"
# Run the evaluation script (Optional: set ${CUDA_VISIBLE_DEVICES} before running)
bash tf3d/semantic_segmentation/scripts/waymo/run_eval_locally.sh
```

The `${EVAL_DIR}` (defined in the script) will contain a
`eval_${EVAL_SPLIT}_mesh` folder and a `eval_${EVAL_SPLIT}` folder with
Tensorboard `events` file.

## Tensorboard

For monitoring the training progress and evaluation results, please start a
[Tensorboard](https://www.tensorflow.org/tensorboard) process:

```bash
tensorboard --logdir=${TRAIN_DIR or EVAL_DIR}
```

The training progress contains the loss values and learning rate.

The quantitative results include IoU (for semantic segmentation) or mAP (for
object detection) metrics. The qualitative results is in the `Mesh` tab of the
Tensorboard page, with the input point cloud, the ground truth and the
prediction.
