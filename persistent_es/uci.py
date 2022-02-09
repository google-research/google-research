# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run ES and PES to tune regularization hyperparameters (e.g., L2 coefficient)

for UCI regression tasks.

Run the UCI experiments:
------------------------
for INIT_THETA in 5 3 1 -1 -3 -5 ; do
    CUDA_VISIBLE_DEVICES=-1 python uci.py \
        --estimate=pes \
        --K=1 \
        --outer_lr=0.003 \
        --lr=0.001 \
        --init_theta=$INIT_THETA \
        --save_dir=saves_uci &
done

for INIT_THETA in 5 3 1 -1 -3 -5 ; do
    CUDA_VISIBLE_DEVICES=-1 python uci.py \
        --estimate=es \
        --K=1 \
        --outer_lr=0.003 \
        --lr=0.001 \
        --init_theta=$INIT_THETA \
        --save_dir=saves_uci &
done

for INIT_THETA in 5 3 1 -1 -3 -5 ; do
    CUDA_VISIBLE_DEVICES=-1 python uci.py \
        --estimate=pes-a \
        --K=1 \
        --outer_lr=0.003 \
        --lr=0.001 \
        --init_theta=$INIT_THETA \
        --save_dir=saves_uci &
done


Plot the results:
-----------------
python plot_uci.py
"""
import os
import sys
import pdb
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from functools import partial
from typing import NamedTuple, Optional, Any, Union

# YAML setup
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import jax
import jax.numpy as jnp
from jax import flatten_util

import optax

# Local imports
import inner_optim
import gradient_estimators
from logger import CSVLogger

parser = argparse.ArgumentParser(description='UCI Regression Task')
parser.add_argument('--data', type=str, default='yacht', help='Dataset')

parser.add_argument(
    '--outer_iterations',
    type=int,
    default=50000,
    help='Number of meta-optimization iterations')
parser.add_argument(
    '--outer_optimizer', type=str, default='adam', help='Outer optimizer')
parser.add_argument(
    '--outer_lr', type=float, default=1e-3, help='Outer learning rate')
parser.add_argument(
    '--outer_b1',
    type=float,
    default=0.99,
    help='Outer optimizer Adam b1 hyperparameter')
parser.add_argument(
    '--outer_b2',
    type=float,
    default=0.9999,
    help='Outer optimizer Adam b2 hyperparameter')
parser.add_argument(
    '--outer_eps',
    type=float,
    default=1e-8,
    help='Outer optimizer Adam epsilon hyperparameter')
parser.add_argument(
    '--outer_clip',
    type=float,
    default=-1,
    help='Outer gradient clipping (-1 means no clipping)')

parser.add_argument(
    '--regularizer_type',
    type=str,
    default='l2',
    choices=['l1', 'l2'],
    help='Regularizer type (l1 or l2)')
parser.add_argument(
    '--init_theta',
    type=float,
    default=-1.0,
    help='The initial regularization coefficient for L1 or L2')

parser.add_argument(
    '--inner_optimizer', type=str, default='sgd', help='Inner optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

parser.add_argument(
    '--estimate',
    type=str,
    default='pes',
    choices=['tbptt', 'es', 'pes', 'pes-a'],
    help='Type of gradient estimate (es or pes)')
# parser.add_argument('--T', type=int, default=10000,
parser.add_argument(
    '--T',
    type=int,
    default=1000000,
    help='Maximum number of iterations of the inner loop')
parser.add_argument(
    '--K',
    type=int,
    default=10,
    help='Number of steps to unroll (== truncation length)')
parser.add_argument(
    '--N', type=int, default=4, help='Number of ES/PES particles')
parser.add_argument(
    '--sigma',
    type=float,
    default=0.01,
    help='Variance for ES/PES perturbations')

parser.add_argument(
    '--log_every',
    type=int,
    default=10,
    help='Log the full training and val losses to the CSV log'
    'every N iterations')
parser.add_argument(
    '--prefix', type=str, default='', help='Optional experiment name prefix')
parser.add_argument(
    '--save_dir', type=str, default='saves', help='Save directory')
parser.add_argument('--seed', type=int, default=3, help='Random seed')
args = parser.parse_args()

# Set random seed
np.random.seed(args.seed)

if args.prefix:
  exp_name = '{}-{}-{}-theta:{}-lr:{}-olr:{}-K:{}-N:{}-sig:{}-s:{}'.format(
      args.prefix, args.data, args.estimate, args.init_theta, args.lr,
      args.outer_lr, args.K, args.N, args.sigma, args.seed)
else:
  exp_name = '{}-{}-theta:{}-lr:{}-olr:{}-K:{}-N:{}-sig:{}-s:{}'.format(
      args.data, args.estimate, args.init_theta, args.lr, args.outer_lr, args.K,
      args.N, args.sigma, args.seed)

save_dir = os.path.join(args.save_dir, exp_name)

# Create experiment save directory
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

# Save command-line arguments
with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
  yaml.dump(vars(args), f)

iteration_logger = CSVLogger(
    fieldnames=[
        'outer_iteration', 'total_inner_iterations', 'val_loss', 'theta',
        'theta_grad'
    ],
    filename=os.path.join(save_dir, 'iteration.csv'))

# Based on https://github.com/stanfordmlgroup/ngboost/blob/master/examples/experiments/regression_exp.py
dataset_name_to_loader = {
    'housing':
        lambda: pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
            header=None,
            delim_whitespace=True,
        ),
    'concrete':
        lambda: pd.read_excel(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        ),
    'wine':
        lambda: pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
            delimiter=';',
        ),
    'kin8nm':
        lambda: pd.read_csv('data/uci/kin8nm.csv'),
    'naval':
        lambda: pd.read_csv(
            'data/uci/naval-propulsion.txt', delim_whitespace=True, header=None)
        .iloc[:, :-1],
    'power':
        lambda: pd.read_excel('data/uci/power-plant.xlsx'),
    'energy':
        lambda: pd.read_excel(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
        ).iloc[:, :-1],
    'protein':
        lambda: pd.read_csv('data/uci/protein.csv')
        [['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'RMSD']],
    'yacht':
        lambda: pd.read_csv(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data',
            header=None,
            delim_whitespace=True,
        ),
    'msd':
        lambda: pd.read_csv('data/uci/YearPredictionMSD.txt').iloc[:, ::-1],
}

data = dataset_name_to_loader['yacht']()
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# If we want to normalize the data with mean and standard deviation computed
# over all the data (including both the training and val data)
# ------------------------------------------------------------
X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
y_mean, y_std = np.mean(y, axis=0), np.std(y, axis=0)
X = (X - X_mean) / X_std
y = (y - y_mean) / y_std
# ------------------------------------------------------------

# Dataloading based on:
#   https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
n = X.shape[0]
np.random.seed(1)

permutation = np.random.choice(range(n), n, replace=False)
end_train = round(n * 80.0 / 100.0)  # Take 80% of the data as training data
end_vak = n
train_index = permutation[0:end_train]
val_index = permutation[end_train:n]

x_train, x_val = X[train_index], X[val_index]
y_train, y_val = y[train_index], y[val_index]

print(
    'x_train.shape: {} | y_train.shape: {} | x_val.shape: {} | y_val.shape: {}'
    .format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))

N_train = x_train.shape[0]
N_val = x_val.shape[0]
print('N_train: {} | N_val: {}'.format(N_train, N_val))


def model_forward(w, x):
  return jnp.dot(x, w)


@jax.jit
def L_t_reg(w, theta):
  output = model_forward(w, x_train)
  mse_loss = 0.5 * jnp.sum(jnp.square(output.reshape(-1) - y_train.reshape(-1)))
  w_flat, _ = flatten_util.ravel_pytree(w)
  if args.regularizer_type == 'l2':
    regularizer = 0.5 * jnp.sum(jnp.exp(theta) * w_flat**2)
  elif args.regularizer_type == 'l1':
    regularizer = jnp.sum(jnp.exp(theta) * jnp.abs(w_flat))
  return mse_loss + regularizer


grad_L_t_reg = jax.jit(jax.grad(L_t_reg, argnums=0))

# ====================================


@jax.jit
def L_v(w):
  output = model_forward(w, x_val)
  mse_loss = 0.5 * jnp.sum(jnp.square(output.reshape(-1) - y_val.reshape(-1)))
  return mse_loss


@partial(jax.jit, static_argnames=('T', 'K'))
def unroll(rng, theta, state, T, K):

  def update(loop_state, i):
    state, L = loop_state
    g = grad_L_t_reg(state.inner_state, theta)
    inner_state_updated, opt_state = opt_step(state.inner_state, g,
                                              state.inner_opt_state)

    loss_value = L_v(inner_state_updated)
    L += loss_value

    state_updated = state._replace(
        inner_state=inner_state_updated,
        inner_opt_state=opt_state,
        t=state.t + 1,
    )
    return (state_updated, L), loss_value

  (state_updated, L), loss_values = jax.lax.scan(update, (state, 0.0),
                                                 jnp.array(list(range(K))))

  result = jnp.sum(loss_values)
  return result, state_updated


@partial(jax.jit, static_argnums=3)
def unroll_and_L_v(w, theta, inner_optim_params, K):
  w_unrolled, _ = unroll(w, theta, inner_optim_params, K)
  return L_v(w_unrolled)


grad_unroll_and_L_v = jax.jit(
    jax.grad(unroll_and_L_v, argnums=1), static_argnums=3)

opt_funcs = inner_optim.init_optimizer('sgd')
reset_opt_params = opt_funcs['reset_opt_params']
opt_step = opt_funcs['opt_step']

init_opt_params = {'lr': 0.001, 'wd': 0.0}


class InnerState(NamedTuple):
  inner_state: jnp.ndarray
  inner_opt_state: Any
  t: jnp.ndarray
  pert_accums: Optional[jnp.ndarray] = None


def init_state_fn(rng):
  """Initialize the inner parameters."""
  w = jax.random.normal(key, (x_train.shape[1],))
  inner_opt_state = reset_opt_params(w, init_opt_params)

  inner_state = InnerState(
      t=jnp.array(0).astype(jnp.int32),
      inner_state=w,
      inner_opt_state=inner_opt_state)
  return inner_state


theta = jnp.array([args.init_theta])
theta_opt = optax.adam(args.outer_lr)
theta_opt_state = theta_opt.init(theta)

key = jax.random.PRNGKey(args.seed)
estimator = gradient_estimators.MultiParticleEstimator(
    key=key,
    theta_shape=theta.shape,
    n_chunks=1,
    n_particles_per_chunk=args.N,
    K=args.K,
    T=args.T,
    sigma=args.sigma,
    method='lockstep',
    estimator_type=args.estimate,
    init_state_fn=init_state_fn,
    unroll_fn=unroll,
)

iterations = []
lam_trajectory = []
eps_trajectory = []
total_inner_iterations = 0

# A particle that will be evolved using the mean theta
state_mean = init_state_fn(key)

key = jax.random.PRNGKey(3)
for outer_iteration in range(args.outer_iterations):
  key, skey = jax.random.split(key)
  theta_grad = estimator.grad_estimate(theta)

  if args.outer_clip > 0:
    theta_grad = jnp.clip(theta_grad, -args.outer_clip, args.outer_clip)

  theta_update, theta_opt_state = theta_opt.update(theta_grad, theta_opt_state)
  theta = optax.apply_updates(theta, theta_update)

  _, state_mean = unroll(key, theta, state_mean, args.T, args.K)
  total_inner_iterations += 2 * args.K
  total_inner_iterations_including_N = args.N * total_inner_iterations

  if outer_iteration % args.log_every == 0:
    val_loss = L_v(state_mean.inner_state)
    print('Iter: {} | Total iter: {} | Val Loss: {:6.4f} | Theta: {} | Grad: {}'
          .format(outer_iteration, total_inner_iterations_including_N, val_loss,
                  theta, theta_grad))

    iteration_stat_dict = {
        'outer_iteration': outer_iteration,
        'total_inner_iterations': total_inner_iterations_including_N,
        'val_loss': val_loss,
        'theta': theta,
        'theta_grad': theta_grad
    }
    iteration_logger.writerow(iteration_stat_dict)
