# Reducing the variance in online optimization by transporting past gradients

This directory contains a TensorFlow implementation of an Implicit Gradient
Transport optimizer using Anytime Average. For details, see [Reducing the
variance in online optimization by transporting past gradients](https://arxiv.org/abs/1906.03532).

## Implementation details
The optimizer relies on a gradient extrapolation (the gradient is not computed
as the parameter values). The present implementation relies on the variables
containing the shifted parameters. The true parameters are instead contained in
associated slots. This is an important distinction, especially when considering
whether the learning curves are for the true or the shifted paramemeters.

## Code overview
The experimental framework is centered on a fork of the
[Cloud TPU resnet code](https://github.com/tensorflow/tpu/tree/master/models/official/resnet)
from May 2019.

`resnet_main.py` is the main executable. Important flags are:

* `mode`, which offers a special `eval_igt` mode for evaluating an IGT model at
the true parameters (vs shifted ones). This value should be used in conjunction
with the `igt_eval_mode` and `igt_eval_set` flags.
* `optimizer`, for setting the optimizer
* `igt_optimizer`, for setting the optimizer to use in conjunction with IGT
* `tail_fraction`, for setting IGT's any time average data window
* `lr_decay` and `lr_decay_step_fraction`

`dump_metrics_to_csv.py` is used to convert the learning curves from their
TensorFlow summary format to an easier to consume csv format.

## Citation
If you use this code for your publication, please cite the original paper:
```
@inproceedings{clark2019bam,
  title = {Reducing the variance in online optimization by transporting past
           gradients},
  author = {Sébastien Arnold and Pierre-Antoine Manzagol and Reza Harikandeh
            and Ioannis Mitliagkas and Nicolas Le Roux (Google Brain)},
  booktitle = {NeurIPS},
  year = {2019}
}
```
