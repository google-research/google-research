# Exploring the Uncertainty Properties of Neural Networks' Implicit Priors in the Infinite-Width Limit

[TOC]

## Overview
This directory contains the publicly available Colab Notebooks for JAX implementation of elliptical slice sampling and interacting with precomputed Myrtle kernels for the paper:


[**Exploring the Uncertainty Properties of Neural Networks' Implicit Priors in the Infinite-Width Limit**](https://arxiv.org/abs/2010.07355)

Ben Adlam\*, Jaehoon Lee\*, Lechao Xiao\*, Jeffrey Pennington, and Jasper Snoek

International Conference on Learning Representations (ICLR), 2021

\* denotes equal contribution.


## Gaussian process classification (GPC) using  elliptical slice sampling (ESS)

<a href="https://colab.research.google.com/github/google-research/google-research/blob/master/infinite_uncertainty/ess.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Here we provide JAX implementation of Gaussian process classification (GPC) using parallelized elliptical slice sampling (ESS). The algorithm is taken from Iain Murray, Ryan Prescott Adams, and David JC MacKay. `Elliptical Slice Sampling (2010)`.

We leverage recent theoretical advances that characterize the function-space prior of an ensemble of infinitely-wide NNs as a Gaussian process, termed the neural network Gaussian process (NNGP). We use the NNGP with a softmax link function to build a probabilistic model for multi-class classification and marginalize over the latent Gaussian outputs to sample from the posterior using ESS. This gives us a better understanding of the implicit prior NNs place on function space and allows a direct comparison of the calibration of the NNGP and its finite-width analogue.



## Loading precomputed Myrtle-10 kernels

<a href="https://colab.research.google.com/github/google-research/google-research/blob/master/infinite_uncertainty/kernel.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Here we provide downloadable precomputed Myrtle-10 neural kernels (NNGP and NTK) on [CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10) and [CIFAR-10 corruption](https://www.tensorflow.org/datasets/catalog/cifar10_corrupted) dataset.
The kernel is computed with the help of [Neural Tangents](https://github.com/google/neural-tangents) python library using V100 GPUs.
Note that the size of kernel is `(50k, 1M)` including clean test (10k) and 19 corruption types each of 5 strengths.

The kernels are located at GCS:

```
gs://neural-tangents-kernels/infinite-uncertainty/kernels/myrtle-10
```

Sub-directories are structured by corruption type and corruption strength. Kernel files are `numpy` arrays of size `(5000,  5000)` and `i`, `j` indicates block index (`0-9` for training data and `10,11` for test data).

```
myrtle-10/
  clean/
    nngp-i-j
    ntk-i-j
  brigthness_1/
    nngp-i-j
    ntk-i-j
  brigthness_2/
    nngp-i-j
    ntk-i-j
  ...
```

The [colab notebook](https://colab.research.google.com/github/google-research/google-research/blob/master/infinite_uncertainty/kernel.ipynb) contains simple code to interact with this precomputed kernels (NNGP an NTK).

Note: In order to load the full uncorrupted kernel (train + test = 60k), run the notebook on a machine with at least 32GB of RAM. Provided example loads 5k subset of training set.


## Citation

If you find this code, data or paper useful, please cite:

```
@article{adlam2020exploring,
  title={Exploring the Uncertainty Properties of Neural Networks' Implicit Priors in the Infinite-Width Limit},
  author={Adlam, Ben and Lee, Jaehoon and Xiao, Lechao and Pennington, Jeffrey and Snoek, Jasper},
  journal={International Conference on Learning Representations},
  year={2021}
}
```

## Contact

Please send pull requests and issues to Ben Adlam
([@bmeadlam](https://github.com/bmeadlam)), Jaehoon Lee
([@jaehlee](https://github.com/jaehlee)) or Lechao Xiao
([@SiuMath](https://github.com/SiuMath)).

