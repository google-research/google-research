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
