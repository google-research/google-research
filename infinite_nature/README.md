<br>
<video autoplay loop muted playsinline width="100%">
  <source src="https://infinite-nature.github.io/teaser_loop.mp4" type="video/mp4">
</video>

# Infinite Nature
[Project Page](https://infinite-nature.github.io/) |
[Paper](https://arxiv.org/abs/2012.09855)

This code accompanies the paper

**Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image** \
Andrew Liu, Richard Tucker, Varun Jumpani, Ameesh Makadia,
Noah Snavely, Angjoo Kanazawa

<div align="center">
  <video autoplay loop muted playsinline controls width="32%">
    <source src="https://infinite-nature.github.io/animation0.mp4" type="video/mp4">
  </video>
  <video autoplay loop muted playsinline controls width="32%">
    <source src="https://infinite-nature.github.io/animation1.mp4" type="video/mp4">
  </video>
  <video autoplay loop muted playsinline controls width="32%">
    <source src="https://infinite-nature.github.io/animation4.mp4" type="video/mp4">
  </video>
</div>

*Please note that this is not an officially supported Google product.*

## Instructions to setup the environment
### Python Environment

The following code base was successfully run with Python 3.8.7. We suggest
installing the library in a vitual environment as our code requires older
versions of libraries. We recommend creating a virtualenv (or conda).

To install libraries using pip, run: \
`pip3 install -r requirements.txt`

### BUILDING TF Mesh Renderer

We use the differentiable renderer from [here]
(https://github.com/google/tf_mesh_renderer). We use gcc to build
the library instead of their Bazel instructions to make it compatible with
Tensorflow 2.0. To download and build:\
`source download_tf_mesh_renderer.sh`

`tf_mesh_renderer` was originally built for Tensorflow \< 2.0.0, however we have
prepared a small patch which upgrades the functions we use to work in
Tensorflow 2.2.0. This means that the other parts of tf_mesh_renderer are still
version incompatible.

### Downloading data and pretrained checkpoint
We include a pretrained checkpoint that can be accessed by running:

```
wget https://storage.googleapis.com/gresearch/infinite_nature_public/ckpt.tar.gz
tar xvf ckpt.tar.gz
```

Sample autocruise inputs can be obtained by running: \
`wget https://storage.googleapis.com/gresearch/infinite_nature_public/autocruise_input1.pkl`

Inside this pickle file is a dictionary with the following entries:\
-`input_rgbd` A sample nature scene and its disparity predicted by
[MiDaS](https://github.com/intel-isl/MiDaS). Its shape is [160, 256, 4] and
values lie between [0, 1].\
-`input_intrinsics` Intrinsics matrix estimated for the input. These values are
not needed to run autocruise as we find assuming a 64&deg;  FOV is sufficient.

### Running autocruise Infinite Nature
We provide code for running our pretrained checkpoint on any pair of
RGB and disparity nature images. Note that if you modify the code to run on your own input images, you will need to run MiDaS first to compute disparity.

`python -m autocruise --output_folder=autocruise --num_steps=100`

will run 100 steps of Infinite Nature using autocruise to control the pose and save the frames to `autocruise/`.

### Bibtex
```
@InProceedings{infinite_nature_2020,
  author = {Liu, Andrew and Tucker, Richard and Jampani, Varun and
            Makadia, Ameesh and Snavely, Noah and Kanazawa, Angjoo},
  title = {Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image},
  booktitle = {arXiv},
  month = {December},
  year = {2020}
}
```
