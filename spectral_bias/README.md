# Spectral Bias in Practice: The Role of Function Frequency in Generalization

Code for the paper: "[Spectral Bias in Practice: The Role of Function Frequency in Generalization] (https://arxiv.org/abs/2110.02424)"

Note that the model definitions and training code builds on the codebase for:
`AutoAugment: Learning Augmentation Policies from Data`.
The original code is publicly available here:
https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/autoaugment

## Label smoothing

The core functions used to add label smoothing noise at a target frequency are
`add_radial_noise` and `add_sinusoidal_noise` in `augmentation_transforms.py`. 
For sinusoidal noise in Fourier image directions, the directions are generated 
using `get_fourier_basis_image` in `freq_helpers.py`.

## Interpolation

The evaluation code for linear interpolation can be found in 
`evaluate_interpolation.py`. `aggregate_interp` selects the pairs of images to 
interpolate and then `batch_interpolation` evaluates the model along each path.

## Usage

Download cifar10 with
`curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
and put the unzipped data (you can unzip with `tar -xvzf cifar-10-binary.tar.gz`) in a `cifar10_data` folder inside `label_smoothing`.

This code requires tensorflow 1; an example `requirements.txt` is provided to 
set up a conda virtualenv with the necessary packages. Once you have an
environment and the codebase downloaded, `cd` into `label_smoothing`. 

You can run `python train_cifar.py` with any desired options to train a model and save
its performance and checkpoints, and then evaluate interpolations of a saved 
model using `python ../interpolation/evaluate_interpolation.py` with the desired
options. This will save the model logits (pre-softmax) on all sample points 
along all interpolation paths specified, as well as the total logit distance 
between the two images that define the path endpoints.

## Citing this work

```
@misc{fridovichkeil2021spectral,
      title={Spectral Bias in Practice: The Role of Function Frequency in Generalization},
      author={Sara Fridovich-Keil and Raphael Gontijo-Lopes and Rebecca Roelofs},
      year={2021},
      eprint={2110.02424},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
