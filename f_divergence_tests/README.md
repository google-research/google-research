# Regularized f-divergence kernel tests.

This directory contains source code to run experiments in the paper
Regularized f-divergence kernel tests.

- `test_two_samples.py` contains code for running two sample tests for several
distributions and f-divergences, that are adaptive to kernel bandwidth and
regularization parameter.

- `divergence/likelihood_ratio.py` contains code for likelihood ratio and
f-divergence estimation.

- `distributions.py` contains code for generating samples from distributions
used in the paper experiments.

- `hypothesis_test/` contains modules with testing functions (fuse,
aggregation).

- `run.sh` provides an example of running the main binary.
