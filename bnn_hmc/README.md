# What Are Bayesian Neural Network Posteriors Really Like?

This repository contains the code to reproduce the experiments in the paper

[_What Are Bayesian Neural Network Posteriors Really Like?_](https://arxiv.org/abs/2104.14421)

by Pavel Izmailov, Sharad Vikram, Matthew D. Hoffman and Andrew Gordon Wilson.


## Introduction

In the paper, we use full-batch Hamiltonian Monte Carlo (HMC) to investigate
foundational questions in Bayesian deep learning.
We show that
- BNNs can achieve significant performance gains over standard training and
  deep ensembles;
- a single long HMC chain can provide a comparable representation of the
  posterior to multiple shorter chains;
- in contrast to recent studies, we find posterior tempering is not needed for
  near-optimal performance, with little evidence for a ``cold posterior''
  effect, which we show is largely an artifact of data augmentation;
- BMA performance is robust to the choice of prior scale, and relatively similar
  for diagonal Gaussian, mixture of Gaussian, and logistic priors;
- Bayesian neural networks show surprisingly poor generalization under domain
  shift;
- while cheaper alternatives such as deep ensembles and SGMCMC can provide good
  generalization, they provide distinct predictive
distributions from HMC. Notably, deep ensemble predictive distributions are
  similarly close to HMC as standard SGLD, and closer than standard variational
  inference.

In this repository we provide JAX code for reproducing results in the paper.

Please cite our work if you find it useful in your research:
```bibtex
@article{izmailov2021bayesian,
  title={What Are Bayesian Neural Network Posteriors Really Like?},
  author={Izmailov, Pavel and Vikram, Sharad and Hoffman, Matthew D and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:2104.14421},
  year={2021}
}
```

## Requirements

We use provide a `requirements.txt` file that can be used to create a conda
environment to run the code in this repo:
```bash
conda create --name <env> --file requirements.txt
```

Example set-up using `pip`:
```bash
pip install tensorflow

pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.65+cuda112 -f \
https://storage.googleapis.com/jax-releases/jax_releases.html

pip install git+https://github.com/deepmind/dm-haiku
pip install tensorflow_datasets
pip install tabulate
pip install optax
```
Please see the [_JAX repo_](https://github.com/google/jax) for the latest
instructions on how to install JAX on your hardware.

## File Structure

```
.
+-- core/
|   +-- hmc.py (The Hamiltonian Monte Carlo algorithm)
|   +-- sgmcmc.py (SGMCMC methods as optax optimizers)
|   +-- vi.py (Mean field variational inference)
+-- utils/ (Utility functions used by the training scripts)
|   +-- train_utils.py (The training epochs and update rules)
|   +-- models.py (Models used in the experiments)
|   +-- losses.py (Prior and likelihood functions)
|   +-- data_utils.py (Loading and pre-processing the data)
|   +-- optim_utils.py (Optimizers and learning rate schedules)
|   +-- ensemble_utils.py (Implementation of ensembling of predictions)
|   +-- metrics.py (Metrics used in evaluation)
|   +-- cmd_args_utils.py (Common command line arguments)
|   +-- script_utils.py (Common functionality of the training scripts)
|   +-- checkpoint_utils.py (Saving and loading checkpoints)
|   +-- logging_utils.py (Utilities for logging printing the results)
|   +-- precision_utils.py (Controlling the numerical precision)
|   +-- tree_utils.py (Common operations on pytree objects)
+-- run_hmc.py (HMC training script)
+-- run_sgd.py (SGD training script)
+-- run_sgmcmc.py (SGMCMC training script)
+-- run_vi.py (MFVI training script)
+-- make_posterior_surface_plot.py (script to visualize posterior density)
```

## Training Scripts

Common command line arguments:

* `seed` &mdash; random seed
* `dir` &mdash; training directory for saving the checkpoints and
tensorboard logs
* `dataset_name` &mdash; name of the dataset, e.g. `cifar10`, `cifar100`,
  `imdb`;
  for the UCI datasets, the name is specified as
  `<UCI dataset name>_<random seed>`, e.g. `yacht_2`, where the seed determines
  the train-test split
* `subset_train_to` &mdash; number of datapoints to use from the dataset;
  by default, the full dataset is used
* `model_name` &mdash; name of the neural network architecture, e.g. `lenet`,
  `resnet20_frn_swish`, `cnn_lstm`, `mlp_regression_small`
* `weight_decay` &mdash; weight decay; for Bayesian methods, weight decay
determines the prior variance (`prior_var = 1 / weight_decay`)
* `temperature` &mdash; posterior temperature (default: `1`)
* `init_checkpoint` &mdash; path to the checkpoint to use for initialization
  (optional)
* `tabulate_freq` &mdash; frequency of tabulate table header logging
* `use_float64` &mdash; use float64 precision (does not work on TPUs and some
  GPUs); by default, we use `float32` precision

### Running HMC

To run HMC, you can use the `run_hmc.py` training script. Arguments:

* `step_size` &mdash; HMC step size
* `trajectory_len` &mdash; HMC trajectory length
* `num_iterations` &mdash; Total number of HMC iterations
* `max_num_leapfrog_steps` &mdash; Maximum number of leapfrog steps allowed;
  meant as a sanity check and should be greater than
  `trajectory_len / step_size`
* `num_burn_in_iterations` &mdash; Number of burn-in iterations (default: `0`)

#### Examples
CNN-LSTM on IMDB:
```bash
# Temperature = 1
python3 run_hmc.py --seed=1 --weight_decay=40. --temperature=1. \
  --dir=runs/hmc/imdb/ --dataset_name=imdb --model_name=cnn_lstm \
  --use_float64 --step_size=1e-5 --trajectory_len=0.24 \
  --max_num_leapfrog_steps=30000

# Temperature = 0.3
python3 run_hmc.py --seed=1 --weight_decay=40. --temperature=0.3 \
  --dir=runs/hmc/imdb/ --dataset_name=imdb --model_name=cnn_lstm \
  --use_float64 --step_size=3e-6 --trajectory_len=0.136 \
  --max_num_leapfrog_steps=46000

# Temperature = 0.1
python3 run_hmc.py --seed=1 --weight_decay=40. --temperature=0.1 \
  --dir=runs/hmc/imdb/ --dataset_name=imdb --model_name=cnn_lstm \
  --use_float64 --step_size=1e-6 --trajectory_len=0.078 \
  --max_num_leapfrog_steps=90000

# Temperature = 0.03
python3 run_hmc.py --seed=1 --weight_decay=40. --temperature=0.03  \
  --dir=runs/hmc/imdb/ --dataset_name=imdb --model_name=cnn_lstm \
  --use_float64 --step_size=1e-6 --trajectory_len=0.043 \
  --max_num_leapfrog_steps=45000
```
We ran these commands on a machine with 8 NVIDIA Tesla V-100 GPUs.

MLP on a subset of 160 datapoints from MNIST:
```bash
python3 run_hmc.py --seed=0 --weight_decay=1. --temperature=1. \
  --dir=runs/hmc/mnist_subset160 --dataset_name=mnist \
  --model_name=mlp_classification --step_size=3.e-5 --trajectory_len=1.5 \
  --num_iterations=100 --max_num_leapfrog_steps=50000 \
  --num_burn_in_iterations=10 --subset_train_to=160
```
This script can be ran on a single GPU.

**Note**: we run HMC on CIFAR-10 on TPU pod with 512 TPU devices with a
modified version of the code that we will release soon.

### Running SGD and Deep Ensembles

To run SGD, you can use the `run_sgd.py` training script. Arguments:

* `init_step_size` &mdash; Initial SGD step size; we use a cosine schedule
* `num_epochs` &mdash; total number of SGD epochs iterations
* `batch_size` &mdash; batch size
* `eval_freq` &mdash; frequency of evaluation (epochs)
* `save_freq` &mdash; frequency of checkpointing (epochs)
* `momentum_decay` &mdash; momentum decay parameter for SGD

#### Examples

ResNet-20-FRN on CIFAR-10:
```bash
python3 run_sgd.py --seed=1 --weight_decay=10 --dir=runs/sgd/cifar10/ \
  --dataset_name=cifar10 --model_name=resnet20_frn_swish \
  --init_step_size=3e-7 --num_epochs=500 --eval_freq=10 --batch_size=80 \
  --save_freq=500 --subset_train_to=40960
```

ResNet-20-FRN on CIFAR-100:
```bash
python3 run_sgd.py --seed=1 --weight_decay=10 --dir=runs/sgd/cifar100/ \
  --dataset_name=cifar100 --model_name=resnet20_frn_swish \
  --init_step_size=1e-6 --num_epochs=500 --eval_freq=10 --batch_size=80 \
  --save_freq=500 --subset_train_to=40960
```

CNN-LSTM on IMDB:
```bash
python3 run_sgd.py --seed=1 --weight_decay=3. --dir=runs/sgd/imdb/ \
  --dataset_name=imdb --model_name=cnn_lstm --init_step_size=3e-7 \
  --num_epochs=500 --eval_freq=10 --batch_size=80 --save_freq=500
```

To train a deep ensemble, we simply train multiple copies of SGD with different
random seeds.

### Running SGMCMC

To run SGMCMC variations, you can use the `run_sgmcmc.py` training script.
It shares command line arguments with SGD, but also introduces the
following arguments:

* `preconditioner` &mdash; choice of preconditioner (`None` or `RMSprop`;
  default: `None`)
* `step_size_schedule` &mdash; choice step size schedule
  (`constant` or `cyclical`); constant sets the step size to `final_step_size`
  after a cosine burn-in for `num_burnin_epochs` epochs. `cyclical` uses a
  constant burn-in for `num_burnin_epochs` epochs and then a cosine cyclical
  schedule (default: `constant`)
* `num_burnin_epochs` &mdash; number of epochs before final lr is reached
* `final_step_size` &mdash; final step size (used only with constant schedule;
  default: `init_step_size`)
* `step_size_cycle_length_epochs` &mdash; cycle length
  (epochs; used only with cyclic schedule; default: `50`)

* `save_all_ensembled` &mdash; save all the networks that are ensembled
* `ensemble_freq` &mdash; frequency of ensembling the iterates
  (epochs; default: `10`)

#### Examples

ResNet-20-FRN on CIFAR-10:

```bash
# SGLD
python3 run_sgmcmc.py --seed=1 --weight_decay=5. --dir=runs/sgmcmc/cifar10/ \
  --dataset_name=cifar10 --model_name=resnet20_frn_swish --init_step_size=1e-6 \
  --final_step_size=1e-6 --num_epochs=10000 --num_burnin_epochs=1000 \
  --eval_freq=10 --batch_size=80 --save_freq=10 --momentum=0. \
  --subset_train_to=40960

# SGHMC
python3 run_sgmcmc.py --seed=1 --weight_decay=5 --dir=runs/sgmcmc/cifar10/ \
  --dataset_name=cifar10 --model_name=resnet20_frn_swish --init_step_size=3e-7 \
  --final_step_size=3e-7 --num_epochs=10000 --num_burnin_epochs=1000 \
  --eval_freq=10 --batch_size=80 --save_freq=10 --subset_train_to=40960 \
  --momentum=0.9

# SGHMC-CLR
python3 run_sgmcmc.py --seed=1 --weight_decay=5 --dir=runs/sgmcmc/cifar10/ \
  --dataset_name=cifar10 --model_name=resnet20_frn_swish --init_step_size=3e-7 \
  --num_epochs=10000 --num_burnin_epochs=1000 --step_size_schedule=cyclical \
  --step_size_cycle_length_epochs=50 --ensemble_freq=50 --eval_freq=10 \
  --batch_size=80 --save_freq=1000 --subset_train_to=40960 \
  --preconditioner=None --momentum=0.95 --eval_freq=10 --save_all_ensembled

# SGHMC-CLR-Prec
python3 run_sgmcmc.py --seed=1 --weight_decay=5 --dir=runs/sghmc/cifar10/ \
  --dataset_name=cifar10 --model_name=resnet20_frn_swish --init_step_size=3e-5 \
  --num_epochs=10000 --num_burnin_epochs=1000 --step_size_schedule=cyclical \
  --step_size_cycle_length_epochs=50 --ensemble_freq=50 --eval_freq=10 \
  --batch_size=80 --save_freq=50 --subset_train_to=40960 \
  --preconditioner=RMSprop --momentum=0.95 --eval_freq=10 --save_all_ensembled
```

### Running MFVI

To run mean field variational inference (MFVI), you can use the `run_mfvi.py`
training script. It shares command line arguments with SGD, but also introduces
the following arguments:

* `optimizer` &mdash; choice of optimizer (`SGD` or `Adam`; default: SGD)
* `vi_sigma_init` &mdash; initial value of the standard deviation over the
  weights in MFVI (default: 1e-3)
* `vi_ensemble_size` &mdash; size of the ensemble sampled in the VI evaluation
  (default: 20)
* `mean_init_checkpoint` &mdash; SGD checkpoint to use for initialization of
  the mean of the MFVI approximation

#### Examples

ResNet-20-FRN on CIFAR-10 or CIFAR-100:
```bash
python3 run_vi.py --seed=11 --weight_decay=5. --dir=runs/vi/cifar100/ \
  --dataset_name=[cifar10 | cifar100] --model_name=resnet20_frn_swish \
  --init_step_size=1e-4 --num_epochs=300 --eval_freq=10 --batch_size=80 \
  --save_freq=300 --subset_train_to=40960 --optimizer=Adam \
  --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=20 \
  --mean_init_checkpoint=<path-to-sgd-solution>
```

CNN-LSTM on IMDB:
```bash
python3 run_vi.py --seed=11 --weight_decay=5. --dir=runs/vi/imdb/ \
  --dataset_name=imdb --model_name=cnn_lstm --init_step_size=1e-4 \
  --num_epochs=500 --eval_freq=10 --batch_size=80 --save_freq=200 \
  --optimizer=Adam --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=20 \
  --mean_init_checkpoint=<path-to-sgd-solution>
```

### Visualizing Posterior Density

You can produce posterior density visualizations similar to the ones
presented in the paper using the `makemake_posterior_surface_plot.py`
script. Arguments:

* `limit_bottom` &mdash; limit of the loss surface visualization along the
  vertical direction at the bottom (defaul: `-0.25`)
* `limit_top` &mdash; limit of the loss surface visualization along the
  vertical direction at the top (defaul: `-0.25`)
* `limit_left` &mdash; limit of the loss surface visualization along the
  horizontal direction on the left (defaul: `1.25`)
* `limit_right` &mdash; limit of the loss surface visualization along the
  horizontal direction on the right (defaul: `1.25`)
* `grid_size` &mdash; number of grid points in each direction (default: `20`)
* `checkpoint1` &mdash; path to the first checkpoint
* `checkpoint2` &mdash; path to the second checkpoint
* `checkpoint3` &mdash; path to the third checkpoint

The script visualizes the posterior log-density, log-likelihood and log-prior
in the plane containing the three provided checkpoints.

### Example

CNN-LSTM on IMDB:
```bash
python3 make_posterior_surface_plot.py --weight_decay=40 --temperature=1. \
  --dir=runs/surface_plots/imdb/ --model_name=cnn_lstm --dataset_name=imdb \
  --checkpoint1=<ckpt1> --checkpoint2=<ckpt2> --checkpoint3=<ckpt3>
  --limit_bottom=-0.75 --limit_left=-0.75 --limit_right=1.75 --limit_top=1.75 \
  --grid_size=50
```
