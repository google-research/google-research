# score_prior

[![Unittests](https://github.com/google-research/score_prior/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/score_prior/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/score_prior.svg)](https://badge.fury.io/py/score_prior)

*This is not an officially supported Google product.*

## About
**Project title:** "Score-based priors for inverse problems in imaging"

This project turns score-based diffusion models into explicit priors for Bayesian inverse problems in imaging. A "score-based prior" allows us to model complex, data-driven posterior distributions while using established sampling and optimization methods (e.g., MCMC, variational inference) that require an explicit log-density function. In our work, we propose a variational-inference approach to posterior sampling, in addition to empirically validating the accuracy and variance of a score prior.

**In this repo:** We include code for calculating log-probabilities given a pretrained score model, running our posterior sampling method, and training your own score model.

* **Probability flow ODE implementation** (`score_prior/probability_flow.py`). This is, to our knowledge, the first JIT-friendly implementation of the probability flow ODE used to compute log-densities under a score-based diffusion model [1]. It makes use of the Diffrax library to make choosing ODE solvers extremely simple.
* **Posterior sampling** (`score_prior/posterior_sampling`). We include code for training a RealNVP neural network to minimize the KL divergence to the target posterior. Once trained, this network can be efficiently sampled and treated as an approximate posterior. The posterior is defined by a log-likelihood function (see `score_prior/forward_models.py` for some predefined measurement operators and likelihoods) and a log-prior function (see `score_prior/posterior_sampling/losses.py` for some predefined priors, including L1/TV/TSV regularizers and score-based priors).
* **Score-model pretraining** (`score_prior/score_sde`). The [official repo](https://github.com/yang-song/score_sde) can be used to train your score model, but we provide training code that is compatible with Flax >= 0.6.0. The rest of this codebase references the official code whenever possible.

[1] Song et al. "Score-Based Generative Modeling Through SDEs." ICLR 2021.

## Setup
1. Clone the repo and do the following steps from the main repo directory.

2. Install `score_prior`.

```
pip install .
```

3. Install `score_sde`. This script downloads the official repo, installs it as a package, and modifies import statements accordingly.

```
sh setup.sh
```

NOTE: `score_prior/score_sde/losses.py` is a version of `score_sde/losses.py`
that's compatible with Flax after v0.6.0.

### Check installation
These tests exercise the essential modules.

```
python3 -m unittest score_prior/tests/posterior_sampling_test.py
python3 -m unittest score_prior/tests/score_sde_test.py
```
