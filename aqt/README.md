# Accurate Quantized Training

This directory contains libraries for running and analyzing
neural network quantization experiments in JAX and flax.

Contributors: Shivani Agrawal, Lisa Wang, Jonathan Malmaud, Lukasz Lew,
Pouya Dormiani, Phoenix Meadowlark, Oleg Rybakov.

## Installation
```
# Install SVN to only download the aqt directory of Google Research.
sudo apt install subversion

# Download this directory
svn export https://github.com/google-research/google-research/trunk/aqt

# Upgrade pip
pip install --user --upgrade pip

# Install the requirements from `requirements.txt`
pip install --user -r aqt/requirements.txt
```

## AQT Quantization Library

`Jax` and `Flax` quantization libraries provides `what you serve is what you train`
quantization for convolution and matmul. See [this README.md](./jax/README.md).

## Reporting Tool

After a training run has completed, the reporting tool in
`report_utils.py` allows to generate a concise experiment report with aggregated
metrics and metadata. See [this README.md](./utils/README.md).
