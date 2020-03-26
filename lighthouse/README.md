# Lighthouse: Predicting Lighting Volumes for Spatially-Coherent Illumination
## Srinivasan et al., CVPR 2020

This release contains code for predicting incident illumination at any 3D
location within a scene. The algorithm takes a narrow-baseline stereo pair of
RGB images as input, and predicts a multiscale RGBA lighting volume.
Spatially-varying lighting within the volume can then be computed by standard
volume rendering.

## Installation

This code relies on some external codebases that do not use the Apache 2 license
used by Google. To complete this code release, follow the instructions at
the end of `nets.py` and `geometry/projector.py`. If you have any difficulty
with this, please contact barron@google.com and pratul@berkeley.edu.

## Running a pretrained model

``interiornet_test.py`` contains an example script for running a pretrained
model on the test set (formatted as .npz files). Please download and extract the
[pretrained model](https://drive.google.com/drive/folders/1VQjRpInmfspz0Rw0Dlm9RbdHX5ziFeDI?usp=sharing)
and
[testing examples](https://drive.google.com/a/berkeley.edu/file/d/121DHkPpbQlyedruI4huF36BF7T1LHcKX/view?usp=sharing)
files, and then include the corresponding file/directory names as command line
flags when running ``interiornet_test.py``.

Example usage (edit paths to match your directory structure):
``python -m lighthouse.interiornet_test --checkpoint_dir="lighthouse/model/" --data_dir="lighthouse/testset/" --output_dir="lighthouse/output/"``

## Training

Please refer to the ``train.py`` for code to use for training your own model.

This model was trained using the [InteriorNet](https://interiornet.org/)
dataset. It may be helpful to read ``data_loader.py`` to get an idea of how we
organized the InteriorNet dataset for training.

To train with the perceptual loss based on VGG features (as done in the paper),
please download the ``imagenet-vgg-verydeep-19.mat``
[pretrained VGG model](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models),
and include the corresponding path as a command line flag when running
``train.py``.

Example usage (edit paths to match your directory structure):
``python -m lighthouse.train --vgg_model_file="lighthouse/model/imagenet-vgg-verydeep-19.mat" --load_dir="" --data_dir="lighthouse/data/InteriorNet/" --experiment_dir=lighthouse/training/``

## Extra

This model is quite memory-hungry, and we used a NVIDIA Tesla V100 GPU for
training and testing with a single example per minibatch. You may run into
memory constraints when training on a GPU with less than 16 GB memory or testing
on a GPU with less than 12 GB memory. If you wish to train a model on a GPU with
<16 GB memory, you may want to try removing the finest volume in the multiscale
representation (see the model parameters in ``train.py``).

This code repository is shared with all of Google Research, so it's not very
useful for reporting or tracking bugs. If you have any issues using this code,
please do not open an issue, and instead just email pratul@berkeley.edu.

If you use this code, please cite it:
``
@article{Srinivasan2020,
  author    = {Pratul P. Srinivasan, Ben Mildenhall, Matthew Tancik, Jonathan T. Barron, Richard Tucker, Noah Snavely},
  title     = {Lighthouse: Predicting Lighting Volumes for Spatially-Coherent Illumination},
  journal   = {CVPR},
  year      = {2020},
}
``
