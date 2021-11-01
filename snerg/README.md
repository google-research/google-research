# Baking Neural Radiance Fields for Real-Time View-Synthesis

This repository contains the public source code release for the paper
[Baking Neural Radiance Fields for Real-Time View-Synthesis (or SNeRG)](http://nerf.live).
This project is based on
[JAXNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf),
which is a [JAX](https://github.com/google/jax) implementation of
[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://www.matthewtancik.com/nerf).

This code is created and maintained by [Peter Hedman](https://phogzone.com).

*Please note that this is not an officially supported Google product.*

## Abstract

Neural volumetric representations such as Neural Radiance Fields (NeRF) have
emerged as a compelling technique for learning to represent 3D scenes from
images with the goal of rendering photorealistic images of the scene from
unobserved viewpoints. However, NeRF's computational requirements are
prohibitive for real-time applications: rendering views from a trained NeRF
requires querying a multilayer perceptron (MLP) hundreds of times per ray. We
present a method to train a NeRF, then precompute and store (i.e. "bake") it as
a novel representation called a Sparse Neural Radiance Grid (SNeRG) that enables
real-time rendering on commodity hardware. To achieve this, we introduce 1) a
reformulation of NeRF's architecture, and 2) a sparse voxel grid representation
with learned feature vectors. The resulting scene representation retains NeRF's
ability to render fine geometric details and view-dependent appearance, is
compact (averaging less than 90 MB per scene), and can be rendered in real-time
(higher than 30 frames per second on a laptop GPU). Actual screen captures are
shown in our video.

## Installation
We recommend using [Anaconda](https://www.anaconda.com/products/individual) to set
up the environment. Run the following commands:

```
# Clone the repo
svn export https://github.com/google-research/google-research/trunk/snerg
# Create a conda environment, note you can use python 3.6-3.8 as
# one of the dependencies (TensorFlow) hasn't supported python 3.9 yet.
conda create --name snerg python=3.6.13; conda activate snerg
# Prepare pip
conda install pip; pip install --upgrade pip
# Install requirements
pip install -r requirements.txt
```

[Optional] Install GPU and TPU support for Jax
```
# Remember to change cuda101 to your CUDA version, e.g. cuda110 for CUDA 11.0.
pip install --upgrade jax jaxlib==0.1.69+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

You can now test that everything installed correctly by training for a few
minibatches on the dummy data we provide in this repository:

```
python -m snerg.train \
  --data_dir=snerg/example_data \
  --train_dir=/tmp/snerg_test \
  --max_steps=5 \
  --factor=2 \
  --batch_size=512
```

## Data

Then, you'll need to download the datasets
from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download and unzip `nerf_synthetic.zip` and `nerf_llff_data.zip`.


## Running

To quickly try the pipeline out you can use the demo config (configs/demo),
however you will need the full configs (configs/blender or configs/llff) to
replicate our results.

The first step is to train a deferred NeRF network:

```
python -m snerg.train \
  --data_dir=/PATH/TO/YOUR/SCENE/DATA \
  --train_dir=/PATH/TO/THE/PLACE/YOU/WANT/TO/SAVE/CHECKPOINTS \
  --config=configs/CONFIG_YOU_LIKE
```

Then, you want, you can evaluate the performance of this trained network on
the test set:

```
python -m snerg.eval \
  --data_dir=/PATH/TO/YOUR/SCENE/DATA \
  --train_dir=/PATH/TO/THE/PLACE/YOU/SAVED/CHECKPOINTS \
  --config=configs/CONFIG_YOU_LIKE
```

Finally, to bake a trained deferred NeRF network into a SNeRG you can run:

```
python -m snerg.bake \
  --data_dir=/PATH/TO/YOUR/SCENE/DATA \
  --train_dir=/PATH/TO/THE/PLACE/YOU/SAVED/CHECKPOINTS \
  --config=configs/CONFIG_YOU_LIKE
```

You can also define your own configurations by passing command line flags.
Please refer to the `define_flags` function in `nerf/utils.py` for all the flags
and their meaning

## Running out of memory

Our baking pipeline consumes a lot of CPU ram: the 3D texture atlas takes
up a lot of space, and performing operations on it makes this issue worse.

The training pipeline should work with 64 GB or more of CPU RAM. However, for
the price of a small drop in quality, you can still run the pipeline on hardware
with less available RAM. For example:
```
voxel_resolution: 800
snerg_dtype: float16
```

## Viewer

You can run this viewer code by uploading it to your own web-server and pointing
it to a SNeRG output directory, e.g.
http://my.web.server.com/snerg/index.html?dir=scene_dir_on_server/baked/png

## Quality evaluation

We re-ran the quality evaluation from the paper, using the code published in
this reposority on both TPUs and GPUs (8x NVIDIA V100). The table below
summarizes the resulting quality (in terms of PSNR).



### Blender

| Scene                   |   Chair   |   Drums   |   Ficus   |   Hotdog  |    Lego   | Materials |    Mic    |    Ship   |    Mean   |
|-------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| JaxNeRF+ (on TPUs)      |   36.22   |   25.78   |   33.94   |   37.83   |   37.23   |   30.81   |   37.65   |   31.59   |   33.89   |
| Deferred NeRF (on TPUs) |   34.96   |   24.05   |   28.93   |   35.69   |   36.35   |   29.63   |   34.02   |   30.54   |   31.77   |
| Deferred NeRF (on GPUs) |   34.82   |   24.19   |   28.61   |   35.96   |   36.32   |   29.58   |   34.11   |   30.51   |   31.76   |
| SNeRG  (on TPUs)        |   34.25   |   24.47   |   29.30   |   34.91   |   34.66   |   28.44   |   32.79   |   29.19   |   31.00   |
| SNeRG  (on GPUs)        |   34.10   |   24.56   |   28.53   |   35.11   |   34.64   |   28.50   |   32.50   |   28.90   |   30.85   |



### LLFF

| Scene                   |    Room    |    Fern   |   Leaves   |  Fortress |  Orchids  |   Flower  |   T-Rex   |   Horns   |    Mean   |
|-------------------------|:----------:|:---------:|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| JaxNeRF+ (on TPUs)      |   33.85    |   23.98   |   20.61    |   31.12   |   19.83   |   28.42   |   27.31   |   29.07   |   26.77   |
| Deferred NeRF (on TPUs) |   32.36    |   24.69   |   20.50    |   31.41   |   19.66   |   27.56   |   27.92   |   28.44   |   26.57   |
| Deferred NeRF (on GPUs) |   32.85    |   24.80   |   20.94    |   31.67   |   19.39   |   27.73   |   28.17   |   28.43   |   26.75   |
| SNeRG  (on TPUs)        |   29.75    |   24.93   |   20.59    |   31.11   |   19.48   |   27.21   |   26.49   |   27.09   |   25.83   |
| SNeRG  (on GPUs)        |   30.07    |   24.63   |   20.76    |   30.92   |   19.26   |   27.47   |   26.72   |   27.09   |   25.87   |



## Citation

If you use this software package, please cite our paper:

```
@misc{hedman2021baking,
      title={Baking Neural Radiance Fields for Real-Time View Synthesis},
      author={Peter Hedman and Pratul P. Srinivasan and Ben Mildenhall and Jonathan T. Barron and Paul Debevec},
      year={2021},
      eprint={2103.14645},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Bug reports

This code repository is shared with all of Google Research, so it's not very
useful for reporting or tracking bugs. If you have any issues using this code,
please do not open an issue, and instead just email peter.j.hedman@gmail.com.




