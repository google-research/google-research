# Partial code for "Understanding and correcting pathologies in the training of learned optimizers"
Authors: Luke Metz, Niru Maheswaranathan, Jeremy Nixon, C. Daniel Freeman, Jascha Sohl-Dickstein

Paper: https://arxiv.org/abs/1810.10180

## What is in this folder
Due to the distributed nature of this project, there is a lot of coupling to internal Google infrastructure and releasing a fully running example is not possible at this time.
This folder contains the core components of our paper (learned optimizer architecture, evolutionary strategies + gradient based training algorithm) + some stubbed out code showing how the distributed training would work.

The architecture for the learned optimizer can be found in `fast_rolling_mlp.py`.

The evolutionary strategies + reparameterization gradient trainer can be found in `es_grad_inv_var.py`

The cluster is started with `run_chief.py` which is the chief worker that manages performing gradient updates on the learned optimizer. `run_worker.py` runs a worker.
These workers iteratively recreate a training graph, and push gradients to the parameter servers.

`run_single_eval.py` shows how one would use a learned optimizer.
This consists of sequential applications of the `Learner` defined in `fast_rolling_mlp.py`.

The remaining files are helpers / utilities.
