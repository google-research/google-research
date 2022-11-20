# OpenScene

This is work in process and development.

## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `openscene`. For linux, you need to install `libopenexr-dev` before creating the environment.

```bash
sudo apt-get install libopenexr-dev
conda create -n openscene python=3.8
conda activate openscene
```

First install PyTorch (we tested on 1.7.1, but the following versions should also work):

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Next install MinkowskiNet:

```bash
sudo apt install build-essential python3-dev libopenblas-dev

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
                           --install-option="--blas=openblas"
```
Afterwards, install all the remaining dependencies:
```bash
pip install -r requirements.txt
```

## RUN
We conduct experiments on ScanNet, Matterport3D, and nuScenes. You can run our 3D distillation as following:

```bash
sh tool/train_disnet.sh EXP_NAME CONFIG.yaml NUM_THREADS
```

For inference, you can run as:

```bash
sh tool/test_disnet.sh EXP_NAME CONFIG.yaml "ensemble" NUM_THREADS
```
