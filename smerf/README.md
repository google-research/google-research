# SMERF: Real-Time View Synthesis for Large Scenes

This repository contains code for SMERF, a real-time approach for radiance
fields of large indoor & outdoor spaces.

## Installation

1. Install NVIDIA drivers, CUDA, and cuDNN. The instructions for this are
   platform-specific.

2. Install [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/).

3. Create a conda environment with Python 3.11.

```
conda create --name smerf-env python=3.11
conda activate smerf-env
```

4. Install JAX with GPU support.

```
python3 -m pip install --upgrade "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

5. Install `smerf` locally,

```
git clone https://github.com/smerf-3d/smerf.git
cd smerf
python3 -m pip install -e .
```

## Training

1. Download the mip-NeRF 360 and Zip-NeRF datasets. Unzip their contents like
   so,

    ```
    datasets/
      bicycle/
        images/     # Photos
        images_2/
        images_4/
        images_8/
        sparse/0/   # COLMAP camera parameters
      ...
    ```

2. Download teacher checkpoints. Unzip their contents like so,

    ```
    teachers/
      bicycle/
        checkpoint_50000/   # Model checkpoint
        config.gin          # Gin config
      ...
    ```

3. Run a training script

    ```
    ./scripts/demo.sh  # Train a small model on a single, local GPU. 
    ```
