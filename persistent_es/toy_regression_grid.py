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

"""Script to evaluate a grid over hyperparameters for the 2D toy regresion problem.

Example:
--------
python toy_regression_grid.py --T=100
"""
import os
import ipdb
import argparse
import pickle as pkl
from tqdm import tqdm
from functools import partial

import numpy as onp

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='JAX Toy Regression Grid')
parser.add_argument('--tune_params', type=str, default='lr:linear',
                    help='A comma-separated string of hyperparameters to search over')
parser.add_argument('--inner_optimizer', type=str, default='sgd', choices=['sgd', 'sgdm'],
                    help='Inner optimizer')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='Whether to use Nesterov momentum for the inner optimization')
parser.add_argument('--evaluation', type=str, default='sum', choices=['sum', 'final'],
                    help='Whether to evaluate theta based on the sum of losses for the unroll or the final loss')
parser.add_argument('--num_points', type=int, default=400,
                    help='Num points for an NxN grid')
parser.add_argument('--T', type=int, default=100,
                    help='Number of unroll steps for the inner optimization')
parser.add_argument('--save_dir', type=str, default='saves/toy_regression',
                    help='Save directory')
args = parser.parse_args()

args.tune_param_string = args.tune_params
args.tune_params = [{'param': p.split(':')[0], 'sched': p.split(':')[1]} for p in args.tune_params.split(',')]

@jax.jit
def loss(x):
    """Inner loss surface"""
    return jnp.sqrt(x[0]**2 + 5) - jnp.sqrt(5) + jnp.sin(x[1])**2 * jnp.exp(-5*x[0]**2) + 0.25*jnp.abs(x[1] - 100)

loss_grad = jax.jit(jax.grad(loss))


# =======================================================================
# Inner Optimization
# =======================================================================
if args.inner_optimizer == 'sgdm':
    def reset_inner_optim_params(params):
        return { 'lr': 10.0,
                 'momentum': 0.9,
                 'weight_decay': 0.0,
                 'buf': jax.tree_map(lambda x: jnp.zeros(x.shape), params)
               }

    def inner_optimizer_step(params, grads, optim_params, t):
        """This follows the PyTorch SGD + momentum implementation.
           From https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
        """
        # Weight decay
        d_p = jax.tree_multimap(lambda g, p: g + optim_params['weight_decay'] * p, grads, params)
        # Momentum
        optim_params['buf'] = jax.tree_multimap(lambda b, g: b * optim_params['momentum'] + g, optim_params['buf'], d_p)
        # Nesterov
        if args.nesterov:
            d_p = jax.tree_multimap(lambda g, b: g + optim_params['momentum'] * b, d_p, optim_params['buf'])
        else:
            d_p = optim_params['buf']
        updated_params = jax.tree_multimap(lambda p, g: p - optim_params['lr'] * g, params, d_p)
        return updated_params, optim_params

elif args.inner_optimizer == 'sgd':
    def reset_inner_optim_params(params):
        return { 'lr': 10.0 }

    def inner_optimizer_step(params, grads, optim_params, t):
        updated_params = params - optim_params['lr'] * grads
        return updated_params, optim_params
# =======================================================================

def get_linear_sched_value(theta, t, T):
    init_value = theta[0]
    final_value = theta[1]
    return init_value * (T - t) / T + final_value * t / T

def get_inverse_time_decay_sched_value(theta, t, T):
    init_value = theta[0]
    decay = theta[1]
    return init_value / ((1 + t / float(T))**decay)

schedule_funcs = { 'linear': get_linear_sched_value,
                   'inverse-time-decay': get_inverse_time_decay_sched_value }

def get_inner_optim_params(inner_optim_params, theta, t, T, K):
    for setting in args.tune_params:
        param = setting['param']
        sched = setting['sched']
        theta_subset = theta[idx_dict[param]]
        theta_constrained = cons_func_dict[param](theta_subset)
        inner_optim_params[param] = schedule_funcs[sched](theta_constrained, t, T)
    return inner_optim_params

@partial(jax.jit, static_argnums=(3,4))
def unroll(x_init, theta, t_current, T, K):
    inner_optim_params = reset_inner_optim_params(x_init)
    x_current = x_init

    def update(state, t_current):
        x_current, inner_optim_params, theta, L = state
        g = loss_grad(x_current)
        inner_optim_params = get_inner_optim_params(inner_optim_params, theta, t_current, T, K)
        x_current, inner_optim_params = inner_optimizer_step(x_current, g, inner_optim_params, t_current)
        loss_value = loss(x_current)
        L += loss_value
        return (x_current, inner_optim_params, theta, L), loss_value

    (x_current, inner_optim_params, theta, L), loss_values = jax.lax.scan(update, (x_current, inner_optim_params, theta, 0.0), jnp.array(list(range(K))))
    if args.evaluation == 'sum':
        return jnp.sum(loss_values)  # Should be equal to L right? So can we just return L?
    elif args.evaluation == 'final':
        return loss_values[-1]

L_grid = onp.zeros((args.num_points, args.num_points))

theta0_min, theta0_max = -5, 5
theta1_min, theta1_max = -5, 5

best_F = 1e10
best_theta = None

X = []
Y = []

default_value = { 'lr': 1.0,
                  'b1': 0.9,
                  'b2': 0.9,
                  'eps': 1e-8,
                  'momentum': 0.9,
                  'weight_decay': 1e-9  # Just so it's not equal to 0, which causes problems with jax.log
                }

cons_func_dict = { 'lr': jnp.exp,
                   'b1': jax.nn.sigmoid,
                   'b2': jax.nn.sigmoid,
                   'eps': jnp.exp,
                   'momentum': jax.nn.sigmoid,
                   'weight_decay': jnp.exp
                 }

uncons_func_dict = { 'lr': jnp.log,
                     'b1': jax.scipy.special.logit,
                     'b2': jax.scipy.special.logit,
                     'eps': jnp.log,
                     'momentum': jax.scipy.special.logit,
                     'weight_decay': jnp.log
                   }

theta_vals = []
idx_dict = {}
idx = 0
for setting in args.tune_params:
    param = setting['param']
    sched = setting['sched']
    default = uncons_func_dict[param](default_value[param])
    if sched == 'fixed':
        theta_vals += [default]
        idx_dict[param] = idx
        idx += 1
    elif sched == 'linear':
        theta_vals += [default, default]
        idx_dict[param] = [idx, idx+1]
        idx += 2
    elif sched == 'inverse-time-decay':
        theta_vals += [default, default]
        idx_dict[param] = [idx, idx+1]
        idx += 2

theta = jnp.array(theta_vals)

for (i, theta0) in tqdm(enumerate(jnp.linspace(theta0_min, theta0_max, args.num_points))):
    for (j, theta1) in enumerate(jnp.linspace(theta1_min, theta1_max, args.num_points)):

        theta_vals = [theta0, theta1]
        theta = jnp.array(theta_vals)

        x_init = jnp.array([1.0, 1.0])
        F = unroll(x_init, theta, 0, args.T, args.T)  # Unroll for the full T inner iterations
        L_grid[j,i] = F

        X.append(theta1)
        Y.append(theta0)

        if F < best_F:
            best_F = F
            best_theta = theta

print('Best theta: {}'.format(jnp.exp(best_theta)))
print('Best L: {:6.4f}'.format(float(best_F)))

X = onp.linspace(theta0_min, theta0_max, args.num_points)
Y = onp.linspace(theta1_min, theta1_max, args.num_points)
xv, yv = onp.meshgrid(X, Y)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

fname = '{}_{}_{}_T_{}_N_{}_grid'.format(args.inner_optimizer, args.tune_param_string,
                                         args.evaluation, args.T, args.num_points)

with open(os.path.join(args.save_dir, '{}.pkl'.format(fname)), 'wb') as f:
    pkl.dump({ 'xv': xv, 'yv': yv, 'L_grid': L_grid, 'best_theta': best_theta }, f)

plt.figure(figsize=(8,6))
plt.pcolormesh(xv, yv, onp.log(L_grid))
plt.plot(best_theta[0], best_theta[1], marker='X', color='red')
plt.colorbar()
plt.xlabel(r'$\theta_0$', fontsize=18)
plt.ylabel(r'$\theta_1$', fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, '{}.png'.format(fname)))
