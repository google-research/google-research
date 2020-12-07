# JaxNeRF

This is a [JAX](https://github.com/google/jax) implementation of
[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://www.matthewtancik.com/nerf).
This code is created and maintained by
[Boyang Deng](https://boyangdeng.com/),
[Jon Barron](https://jonbarron.info/),
and [Pratul Srinivasan](https://people.eecs.berkeley.edu/~pratul/).

<div align="center">
  <img width="95%" alt="NeRF Teaser" src="https://raw.githubusercontent.com/bmild/nerf/master/imgs/pipeline.jpg">
</div>

Our JAX implementation currently supports:

<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax"><span style="font-weight:bold">Platform</span></th>
    <th class="tg-0lax" colspan="2"><span style="font-weight:bold">Single-Host GPU</span></th>
    <th class="tg-0lax" colspan="2"><span style="font-weight:bold">Multi-Device TPU</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"><span style="font-weight:bold">Type</span></td>
    <td class="tg-0lax">Single-Device</td>
    <td class="tg-0lax">Multi-Device</td>
    <td class="tg-0lax">Single-Host</td>
    <td class="tg-0lax">Multi-Host</td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:bold">Training</span></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:bold">Evaluation</span></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
  </tr>
</tbody>
</table>

The training job on 128 TPUv2 cores can be done in **2.5 hours (v.s 3 days for TF
NeRF)** for 1 million optimization steps. In other words, JaxNeRF trains to the best while trains very fast.

As for inference speed, here are the statistics of rendering an image with
800x800 resolution (numbers are averaged over 50 rendering passes):

| Platform | 1 x NVIDIA V100 |                                                  8 x NVIDIA V100                                                  |                                                    128 x TPUv2                                                    |
|----------|:---------------:|:-----------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
| TF NeRF  |    27.74 secs   | <img src="http://storage.googleapis.com/gresearch/jaxnerf/cross.png"  alt="Not Supported" width=18px height=18px> | <img src="http://storage.googleapis.com/gresearch/jaxnerf/cross.png"  alt="Not Supported" width=18px height=18px> |
| JaxNeRF  |    20.77 secs   |                                                     2.65 secs                                                     |                                                     0.35 secs                                                     |


The code is tested and reviewed carefully to match the
[original TF NeRF implementation](https://github.com/bmild/nerf).
If you have any issues using this code, please do not open an issue as the repo
is shared by all projects under Google Research. Instead, just email
jaxnerf@google.com.

## Installation
We recommend using [Anaconda](https://www.anaconda.com/products/individual) to set
up the environment. Run the following commands:

```
# Clone the repo
svn export https://github.com/google-research/google-research/trunk/jaxnerf
# Create a conda environment, note you can use python 3.6-3.8 as
# one of the dependencies (TensorFlow) hasn't supported python 3.9 yet.
conda create --name jaxnerf python=3.6.12; conda activate jaxnerf
# Prepare pip
conda install pip; pip install --upgrade pip
# Install requirements
pip install -r jaxnerf/requirements.txt
# [Optional] Install GPU and TPU support for Jax
# Remember to change cuda101 to your CUDA version, e.g. cuda110 for CUDA 11.0.
pip install --upgrade jax jaxlib==0.1.57+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Then, you'll need to download the datasets
from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download the `nerf_synthetic.zip` and `nerf_llff_data.zip` and unzip them
in the place you like. Let's assume they are placed under `/tmp/jaxnerf/data/`.

That's it for installation. You're good to go. **Notice:** For the following instructions, you don't need to enter the jaxnerf folder. Just stay in the parent folder.

## Two Commands for Everything

```
bash jaxnerf/train.sh demo /tmp/jaxnerf/data
bash jaxnerf/eval.sh demo /tmp/jaxnerf/data
```

Once both jobs are done running (which may take a while if you only have 1 GPU
or CPU), you'll have a folder, `/tmp/jaxnerf/data/demo`, with:
  
  * Trained NeRF models for all scenes in the blender dataset.
  * Rendered images and depth maps for all test views.
  * The collected PSNRs of all scenes in a TXT file.
  
Note that we used the `demo` config here which is basically the `blender` config
in the paper except smaller batch size and much less train steps. Of course, you
can use other configs to replace `demo` and other data locations to replace
`/tmp/jaxnerf/data`.

We provide 2 configurations in the folder `configs` which match the original
configurations used in the paper for the blender dataset and the LLFF dataset.
Be careful when you use them. Their batch sizes are large so you may get OOM error if you have limited resources, for example, 1 GPU with small memory. Also, they have many many train steps so you may need days to finish training all scenes.

## Play with One Scene

You can also train NeRF on only one scene. The easiest way is to use given configs:

```
python -m jaxnerf.train \
  --data_dir=/PATH/TO/YOUR/SCENE/DATA \
  --train_dir=/PATH/TO/THE/PLACE/YOU/WANT/TO/SAVE/CHECKPOINTS \
  --config=configs/CONFIG_YOU_LIKE
```

Evaluating NeRF on one scene is similar:

```
python -m jaxnerf.eval \
  --data_dir=/PATH/TO/YOUR/SCENE/DATA \
  --train_dir=/PATH/TO/THE/PLACE/YOU/SAVED/CHECKPOINTS \
  --config=configs/CONFIG_YOU_LIKE \
  --chunk=4096
```

The `chunk` parameter defines how many rays are feed to the model in one go.
We recommend you to use the largest value that fits to your device's memory but
small values are fine, only a bit slow.

You can also define your own configurations by passing command line flags. Please refer to the `define_flags` function in `nerf/utils.py` for all the flags and their meanings.

## Pretrained Models

We provide a collection of pretrained NeRF models that match the numbers
reported in the [paper](https://arxiv.org/abs/2003.08934). Actually, ours are
slightly better overall because we trained for more iterations (while still
being much faster!). You can find our pretrained models
[here](http://storage.googleapis.com/gresearch/jaxnerf/jaxnerf_models.zip).
The performances (in PSNR) of our pretrained NeRF models are listed below:

### Blender


| Scene   |   Chair   |   Drums   |   Ficus   |   Hotdog  |    Lego   | Materials |    Mic    |    Ship   |    Mean   |
|---------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| TF NeRF |   33.00   | **25.01** | **30.13** |   36.18   |   32.54   |   29.62   |   32.91   |   28.65   |   31.01   |
| JaxNeRF | **33.81** |   24.82   |   29.83   | **36.64** | **32.73** | **29.65** | **34.28** | **28.84** | **32.33** |

### LLFF

| Scene   |    Room   |    Fern   |   Leaves  |  Fortress |  Orchids  |   Flower  |   T-Rex   |   Horns   |    Mean   |
|---------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| TF NeRF | **32.70** | **25.17** |   20.92   |   31.16   | **20.36** |   27.40   |   26.80   |   27.45   |   26.50   |
| JaxNeRF |   32.54   |   25.02   | **21.16** | **31.73** |   20.35   | **27.90** | **27.11** | **27.88** | **26.71** |

## Citation
If you use this software package, please cite it as:

```
@software{jaxnerf2020github,
  author = {Boyang Deng and Jonathan T. Barron and Pratul P. Srinivasan},
  title = {{JaxNeRF}: an efficient {JAX} implementation of {NeRF}},
  url = {https://github.com/google-research/google-research/tree/master/jaxnerf},
  version = {0.0},
  year = {2020},
}
```

and also cite the original NeRF paper:

```
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```

## Acknowledgement
We'd like to thank
[Daniel Duckworth](http://www.stronglyconvex.com/),
[Dan Gnanapragasam](https://research.google/people/DanGnanapragasam/),
and [James Bradbury](https://twitter.com/jekbradbury)
for their help on reviewing and optimizing this code.
We'd like to also thank the amazing [JAX](https://github.com/google/jax) team for
very insightful and helpful discussions on how to use JAX for NeRF.
