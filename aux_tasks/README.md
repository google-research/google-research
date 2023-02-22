# Aux Tasks

This paper contains code used used for experiments in
"A Novel Stochastic Gradient Descent Algorithm for Learning Principal Subspaces"
[link](https://arxiv.org/abs/2212.04025).

## Installation

To run experiments in this directory:

`git clone https://github.com/google-research/google_research.git`

`cd google_research`

We recommend doing the following in a virtual environment. (E.g. `python3 -m venv .venv && source .venv/bin/activate`)

`pip install --upgrade pip`

`pip install -r aux_tasks/requirements.txt`

## Running Experiments

The following sections give examples of how to run the experiments described in
our paper.

### synthetic

The synthetic directory allows running of experiments with synthetic
matrices, MNIST, and Puddle World.

Note: `run_synthetic.py` checkpoints as it goes, so if you change
the configuration between runs, make sure to use a new workdir!

**Synthetic Matrices**

To run an experiment using synthetic matrices, use the following command:

```
python -m aux_tasks.synthetic.run_synthetic \
  --workdir=/tmp/synthetic \
  --config.S=50 \
  --config.T=50 \
  --config.d=10 \
  --config.method=lissa \
  --config.main_batch_size=10 \
  --config.covariance_batch_size=10 \
  --config.weight_batch_size=10 \
  --config.rescale_psi=exp
```

For more information on what flags you can use, see
`aux_tasks/synthetic/run_synthetic.py`.

**MNIST**

To run an experiment using MNIST, use the following command:

```
python -m aux_tasks.synthetic.run_synthetic \
  --workdir=/tmp/synthetic \
  --config.use_mnist=true \
  --config.d=16 \
  --config.method=lissa \
  --config.main_batch_size=128 \
  --config.covariance_batch_size=128 \
  --config.weight_batch_size=128 \
  --config.optimizer=adam \
  --config.lr=5e-3 \
  --config.svd_path=mnist_svd.np
```

**Puddle World**

We supply the code for generating the Psi matrix for Puddle World
at `aux_tasks/puddle_world/compute_successor_representation.py`.
For example:

```
python -m aux_tasks.puddle_world.compute_successor_representation \
  --output_dir=/tmp/puddle_world \
  --arena_name=sutton \
  --num_bins=100
```

This will give the same setup as we used in our paper. To see more options,
see the flags in `compute_successor_representation.py`.

To run a Puddle World experiment, use the following command:

```
python -m aux_tasks.synthetic.run_synthetic \
  --workdir=/tmp/synthetic \
  --config.d=5 \
  --config.method=lissa \
  --config.main_batch_size=50 \
  --config.covariance_batch_size=50 \
  --config.weight_batch_size=50 \
  --config.suite=puddle_world \
  --config.puddle_world_path=puddle_world_data \
  --config.puddle_world_arena=sutton_20
```

## MNIST analysis

We include ipython notebooks for performing MNIST analyses under
`aux_tasks/mnist`.
