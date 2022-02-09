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

"""Script to evaluate a grid over hyperparameters for the 2D toy regresion

problem.

Example:
--------
python toy_regression_grid.py \
    --inner_optimizer=sgd \
    --tune_params=lr:linear \
    --save_dir=saves/toy_regression
"""
import os
import pdb
import argparse
import itertools
import pickle as pkl
from functools import partial

import numpy as onp

import jax
import jax.numpy as jnp

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Local imports
import schedule
import inner_optim
import hparam_utils

parser = argparse.ArgumentParser(description='Toy Regression Grid')
parser.add_argument(
    '--tune_params',
    type=str,
    default='lr:linear',
    help='A comma-separated string of hyperparameters to'
    'search over')
parser.add_argument(
    '--inner_optimizer',
    type=str,
    default='sgd',
    choices=['sgd', 'sgdm', 'adam'],
    help='Inner optimizer')
parser.add_argument(
    '--evaluation',
    type=str,
    default='sum',
    choices=['sum', 'final'],
    help='Whether to evaluate theta based on the sum of losses'
    'for the unroll or the final loss')
parser.add_argument(
    '--num_points', type=int, default=2000, help='Num points for an NxN grid')
parser.add_argument('--T', type=int, default=100,
                    help='Number of unroll steps for the inner optimization')
parser.add_argument('--save_dir', type=str, default='saves/toy_regression',
                    help='Save directory')
args = parser.parse_args()

args.tune_param_string = args.tune_params
args.tune_params = [{
    'param': p.split(':')[0],
    'sched': p.split(':')[1]
} for p in args.tune_params.split(',')]

@jax.jit
def loss(x):
  """Inner loss surface"""
  return jnp.sqrt(x[0]**2 + 5) - jnp.sqrt(5) + jnp.sin(x[1])**2 * \
         jnp.exp(-5*x[0]**2) + 0.25*jnp.abs(x[1] - 100)

loss_grad = jax.jit(jax.grad(loss))

opt_funcs = inner_optim.init_optimizer(args.inner_optimizer)
reset_opt_params = opt_funcs['reset_opt_params']
opt_step = opt_funcs['opt_step']

default_values = {
    'lr': 0.1,
    'mom': 0.9,
    'b1': 0.9,
    'b2': 0.9,
    'eps': 1e-8,
    'wd': 1e-9
}


def get_inner_optim_params(inner_optim_params, theta, t, T):
  for setting in args.tune_params:
    param = setting['param']
    sched = setting['sched']
    theta_subset = theta[jnp.array(idx_dict[param])]
    inner_optim_params[param] = schedule.schedule_funcs[sched](
        inner_optim_params, theta_subset, param, t, T)
  return inner_optim_params

@partial(jax.jit, static_argnums=(3,4))
def unroll(theta, x_init, t, T, K):
  inner_optim_params = reset_opt_params(x_init, default_values)
  x_current = x_init

  def update(state, t):
    x_current, inner_optim_params, theta, L = state
    g = loss_grad(x_current)
    inner_optim_params = get_inner_optim_params(inner_optim_params, theta, t, T)
    x_current, inner_optim_params = opt_step(x_current, g, inner_optim_params)
    loss_value = loss(x_current)
    L += loss_value
    return (x_current, inner_optim_params, theta, L), loss_value

  (x_current, inner_optim_params, theta,
   L), loss_values = jax.lax.scan(update,
                                  (x_current, inner_optim_params, theta, 0.0),
                                  jnp.array(list(range(K))))
  if args.evaluation == 'sum':
    return jnp.sum(loss_values)
  elif args.evaluation == 'final':
    return loss_values[-1]

L_grid = onp.zeros((args.num_points, args.num_points))

theta0_min, theta0_max = -5, 5
theta1_min, theta1_max = -5, 5

# Creating the theta
# =================================================================
theta_vals = []
idx_dict = {}
idx = 0
for setting in args.tune_params:
  param = setting['param']
  sched = setting['sched']
  default = hparam_utils.uncons_funcs[param](default_values[param])
  if sched == 'fixed':
    theta_vals += [default]
    idx_dict[param] = idx
    idx += 1
  elif sched == 'linear':
    theta_vals += [default, default]
    idx_dict[param] = [idx, idx + 1]
    idx += 2
  elif sched == 'inverse-time-decay':
    theta_vals += [default, default]
    idx_dict[param] = [idx, idx + 1]
    idx += 2

theta = jnp.array(theta_vals)
# =================================================================

theta0_vals = jnp.linspace(theta0_min, theta0_max, args.num_points)
theta1_vals = jnp.linspace(theta1_min, theta1_max, args.num_points)
theta_product = jnp.array(
    onp.stack(itertools.product(theta0_vals, theta1_vals)))

x_init = jnp.array([1.0, 1.0])
F = jax.vmap(
    unroll, in_axes=(0, None, None, None, None))(theta_product, x_init, 0,
                                                 args.T, args.T)
L_grid = F.reshape(args.num_points, args.num_points).T

best_idx = jnp.argmin(F)
best_F = F[best_idx]
best_theta = theta_product[best_idx]

print('Best theta: {}'.format(jnp.exp(best_theta)))
print('Best L: {:6.4f}'.format(float(best_F)))

X = theta0_vals
Y = theta1_vals
xv, yv = onp.meshgrid(X, Y)

if not os.path.exists(args.save_dir):
  os.makedirs(args.save_dir)

fname = '{}_{}_{}_T_{}_N_{}_grid'.format(args.inner_optimizer,
                                         args.tune_param_string,
                                         args.evaluation, args.T,
                                         args.num_points)

with open(os.path.join(args.save_dir, '{}.pkl'.format(fname)), 'wb') as f:
  pkl.dump({'xv': xv, 'yv': yv, 'L_grid': L_grid, 'best_theta': best_theta}, f)

plt.figure(figsize=(8,6))
cmesh = plt.pcolormesh(
    xv,
    yv,
    L_grid,
    norm=colors.LogNorm(),
    cmap=plt.cm.Purples_r,
    linewidth=0,
    rasterized=True)
cbar = plt.colorbar(cmesh)
cbar.ax.tick_params(labelsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'$\theta_0$', fontsize=20)
plt.ylabel(r'$\theta_1$', fontsize=20)
plt.title(
    '{}, T={}, {}'.format(args.inner_optimizer.title(), args.T,
                          args.evaluation.title()),
    fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, '{}.png'.format(fname)))
plt.savefig(os.path.join(args.save_dir, '{}.pdf'.format(fname)))
