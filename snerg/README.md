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
  --config=configs/CONFIG_YOU_LIKE \
  --chunk=4096
```

Finally, to bake a trained deferred NeRF network into a SNeRG you can run:

```
python -m snerg.bake \
  --data_dir=/PATH/TO/YOUR/SCENE/DATA \
  --train_dir=/PATH/TO/THE/PLACE/YOU/SAVED/CHECKPOINTS \
  --config=configs/CONFIG_YOU_LIKE
```

The `chunk` parameter defines how many rays are feed to the model in one go.
We recommend you to use the largest value that fits to your device's memory but
small values are fine, only a bit slow.


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




