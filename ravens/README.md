# Ravens Environment

Ravens is a simple environment in PyBullet for benchmarking LfD (active work in progress) on various planar tasks.

For each task, all objects are randomly positioned and oriented in the workspace per episode.
Each task comes with an oracle that provides expert demonstrations.
To mimic the challenges of the real-world, the only input information that should be accessible to an agent as observations are raw RGB-D images and camera parameters (pose and intrinsics).

## Setup

#### Download model assets.
```shell
cd ravens
wget https://storage.googleapis.com/ravens-assets/ravens_assets_2020_09_30.zip
unzip ravens_assets_2020_09_30.zip
```

#### Quick: install Python libraries with pip (tested with Python 3.7):
```shell
./install_python_deps.sh
```

#### Recommended: virtualenv

To make it so you can have exact match of these Python packages, without affecting your other projects, we recommend using a Python virtual environment.

Step 1, make sure you have outside-of-Python dependencies installed:

- Install: CUDA, cuDNN (see "fresh Ubuntu install" script below if you need)

Step 2, set up a Python virtual environment:

```shell
pip install virtualenv
virtualenv ravens-env
source ravens-env/bin/activate
```

Step 3, install Python packages in this virtual environment:

```shell
./install_python_deps.sh
```

#### Full automatic setup: script for fresh Ubuntu installs (tested with Ubuntu 18.04):
```shell
bash install_dependencies.sh  # installs Miniconda, CUDA, cuDNN, TensorFlow, and other Python libraries
source ~/.bashrc
```

## Quick Start

Runs in headless mode.

```shell
python main.py --gpu=0
```

## Run with tensorboard

```
# run from root dir of project
python3 -m tensorboard.main --logdir=./
# open browser to where it tells you to
```

#### Plot results:

```shell
python plot.py
```
