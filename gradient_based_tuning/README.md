# Simple and Effective Gradient-Based Tuning of Sequence-to-Sequence Models

This repository contains the code used in the paper "[Simple and Effective Gradient-Based Tuning of Sequence-to-Sequence Models](https://automl.cc/wp-content/uploads/2022/07/simple_and_effective_gradient_.pdf)"
by Jared Lichtarge, Chris Alberti, and Shankar Kumar, to be presented at AutoML 2022.


We describe a simple approach to tuning the hyper-parameters during training by
updating them with the gradient on the tuning-loss. This code implements this
approach in JAX. The code allows for the easy specification of any subset of
hyper-parameters to be learned during training. Specifically, it trains
Transformer models on translation data from WMT.

## Dependencies

All dependencies are listed in requirements.txt. Models are implemented in flax.
The WMT data is sourced from Tensorflow Datasets (https://www.tensorflow.org/datasets/api_docs/python/tfds).

### Python Environment
We suggest installing the library in a virtual environment as our code requires older versions of libraries. To do so, run the following on a path created to host this enviroment:

`python3 -m venv .
source bin/activate
`

To install libraries using pip, run: \
`pip3 install -r requirements.txt`

To run on GPU, additionally run: \
`pip3 install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.1.75+cuda11.cudnn82-cp37-none-manylinux2010_x86_64.whl`

## Usage

To download and format data, then train a model, run: \
`./run.sh`

By default this will use language pair Lithuanian-English (lt-en), and train a tiny model for demonstration purposes, as larger models will likely OOM unless run on GPU or TPU hardware.

## Demo

To view a demonstration of the code, which will train models with various settings and display the results via Tensorboard, see demo_colab.ipynb.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/google-research/blob/master/gradient_based_tuning/demo_colab.ipynb)


## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{
lichtarge2022simple,
title={Simple and Effective Gradient-Based Tuning of Sequence-to-Sequence Models},
author={Jared Lichtarge and Chris Alberti and Shankar Kumar},
booktitle={First Conference on Automated Machine Learning (Late-Breaking Workshop)},
year={2022},
url={https://openreview.net/forum?id=RBTUKLfQ_pc}
}
```
