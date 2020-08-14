# Hierarchical Abstraction with Language

This package contains the code for training goal conditioned policy with
relabeling used in the Neurips 2019 paper
["Language as an Abstraction for Hierarchical Deep Reinforcement Learning"](https://arxiv.org/abs/1906.07343)

*Disclaimer: This is not an official Google product.*

## Pre-requisite
Besides those listed in `requirements.txt`, you must also get the [clevr-robot environment](https://github.com/google-research/clevr_robot_env)
in order to run this out of box. However, if you don't wish to run on the clevr-robot environment, you can
modify `experiment_setup.py` to use different environments.

## Usage
```
python -m hal.train
```
