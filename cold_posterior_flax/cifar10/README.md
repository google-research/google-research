# Flax implementation of Cold Posterior

This repository contains a Flax implementation of the SG-MCMC CIFAR experiment 
from the paper 
[How Good is the Bayes Posterior in Deep Neural Networks Really?](https://arxiv.org/pdf/2002.02405.pdf). It is based on the pre-linen Flax API.

## Current Status

***NOTE: This implementation is work in progress!***

## Basis
This codebase is based on the Flax example now located at:
https://github.com/google-research/google-research/tree/master/flax_models/cifar

The SGMCMC implementation is based on the original TF2.0 implementation at:
https://github.com/google-research/google-research/tree/master/cold_posterior_bnn

### For questions reach out to

Bas Veeling ([basveeling@gmail.com](mailto:basveeling@gmail.com))<br>
Tim Salimans ([salimans@google.com](mailto:salimans@google.com))

### Reference

> Florian Wenzel, Kevin Roth, Bastiaan S. Veeling, Jakub Swiatkowski, Linh Tran,
> Stephan Mandt, Jasper Snoek, Tim Salimans, Rodolphe Jenatton and Sebastian
> Nowozin (2020).
> [How Good is the Bayes Posterior in Deep Neural Networks Really?](https://arxiv.org/abs/2002.02405).
> In _International Conference of Machine Learning_.

