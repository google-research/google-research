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

python uci.py \
    --estimate=pes \
    --k=1 \
    --lr=1e-4 \
    --outer_lr=1e-3 \
    --sigma=0.1 \
    --seed=3 \
    --log_every=10 \
    --save_dir=saves_uci
"""
import os
import sys
import ipdb
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from functools import partial

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import jax
from jax.config import config
# config.update('jax_disable_jit', True)
import jax.numpy as jnp
from jax import flatten_util

# Local imports
import general_utils

parser = argparse.ArgumentParser(description='UCI Regression Task')
parser.add_argument('--data', type=str, default='yacht',
                    help='Dataset')

parser.add_argument('--outer_iterations', type=int, default=50000,
                    help='Number of meta-optimization iterations')
parser.add_argument('--outer_optimizer', type=str, default='adam',
                    help='Outer optimizer')
parser.add_argument('--outer_lr', type=float, default=1e-3,
                    help='Outer learning rate')
parser.add_argument('--outer_b1', type=float, default=0.9,
                    help='Outer optimizer Adam b1 hyperparameter')
parser.add_argument('--outer_b2', type=float, default=0.99,
                    help='Outer optimizer Adam b2 hyperparameter')
parser.add_argument('--outer_eps', type=float, default=1e-8,
                    help='Outer optimizer Adam epsilon hyperparameter')
parser.add_argument('--outer_gamma', type=float, default=0.9,
                    help='Outer RMSprop gamma hyperparameter')
parser.add_argument('--outer_momentum', type=float, default=0.9,
                    help='Outer optimizer momentum')

parser.add_argument('--outer_clip', type=float, default=-1,
                    help='Outer gradient clipping (-1 means no clipping)')

parser.add_argument('--regularizer_type', type=str, default='l2', choices=['l1', 'l2'],
                    help='Regularizer type (l1 or l2)')
parser.add_argument('--initial_theta', type=float, default=-1.0,
                    help='The initial regularization coefficient for L1 or L2')

parser.add_argument('--inner_optimizer', type=str, default='sgd',
                    help='Inner optimizer')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--b1', type=float, default=0.99,
                    help='Adam b1 hyperparameter')
parser.add_argument('--b2', type=float, default=0.999,
                    help='Adam b2 hyperparameter')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='Adam epsilon hyperparameter')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum')
parser.add_argument('--weight_decay', type=float, default=1e-10,
                    help='Weight decay')
parser.add_argument('--l1', type=float, default=1e-10,
                    help='L1 regularization')
parser.add_argument('--l2', type=float, default=1e-10,
                    help='L2 regularization')
parser.add_argument('--random_hparam_init', action='store_true', default=False,
                    help='Whether to initialize the hyperparameters to random values')

parser.add_argument('--estimate', type=str, default='pes', choices=['tbptt', 'es', 'pes', 'pes-analytic'],
                    help='Type of gradient estimate (es or pes)')
parser.add_argument('--T', type=int, default=10000,
                    help='Maximum number of iterations of the inner loop')
parser.add_argument('--K', type=int, default=10,
                    help='Number of steps to unroll (== truncation length)')
parser.add_argument('--N', type=int, default=4,
                    help='Number of ES/PES particles')
parser.add_argument('--sigma', type=float, default=0.1,
                    help='Variance for ES/PES perturbations')

parser.add_argument('--log_every', type=int, default=10,
                    help='Log the full training and val losses to the CSV log every N iterations')

parser.add_argument('--prefix', type=str, default='',
                    help='Optional experiment name prefix')
parser.add_argument('--save_dir', type=str, default='saves',
                    help='Save directory')
parser.add_argument('--tb_logdir', type=str, default='runs_uci_test',
                    help='(Optional) Directory in which to save Tensorboard logs')
parser.add_argument('--seed', type=int, default=3,
                    help='Random seed')
args = parser.parse_args()

# Set random seed
np.random.seed(args.seed)

if args.prefix:
    exp_name = '{}-{}-{}-lr:{}-olr:{}-K:{}-N:{}-sig:{}-s:{}'.format(
                args.prefix, args.data, args.estimate, args.lr, args.outer_lr, args.K, args.N, args.sigma, args.seed)
else:
    exp_name = '{}-{}-lr:{}-olr:{}-K:{}-N:{}-sig:{}-s:{}'.format(
                args.data, args.estimate, args.lr, args.outer_lr, args.K, args.N, args.sigma, args.seed)

save_dir = os.path.join(args.save_dir, exp_name)

# Create experiment save directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save command-line arguments
with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
    yaml.dump(vars(args), f)

iteration_logger = general_utils.CSVLogger(fieldnames=['outer_iteration', 'total_inner_iterations', 'val_loss', 'theta', 'theta_grad'],
                                           filename=os.path.join(save_dir, 'iteration.csv'))

if args.data == 'irm':
    N_train = 1000
    sigma_train = 1.0
    x1_train = np.random.normal(loc=0.0, scale=sigma_train, size=(N_train,1))
    y_train = x1_train + np.random.normal(loc=0.0, scale=sigma_train, size=(N_train,1))
    x2_train = y_train + np.random.normal(loc=0.0, scale=1.0, size=(N_train,1))
    x_train = np.concatenate([x1_train, x2_train], axis=1)

    N_val = 1000
    sigma_val = 10.0
    x1_val = np.random.normal(loc=0.0, scale=sigma_val, size=(N_val,1))
    y_val = x1_val + np.random.normal(loc=0.0, scale=sigma_val, size=(N_val,1))
    x2_val = y_val + np.random.normal(loc=0.0, scale=1.0, size=(N_val,1))
    x_val = np.concatenate([x1_val, x2_val], axis=1)
elif args.data == 'yacht':
    # Based on https://github.com/stanfordmlgroup/ngboost/blob/master/examples/experiments/regression_exp.py
    dataset_name_to_loader = {
        'housing': lambda: pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
            header=None,
            delim_whitespace=True,
        ),
        'concrete': lambda: pd.read_excel(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        ),
        'wine': lambda: pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
            delimiter=';',
        ),
        'kin8nm': lambda: pd.read_csv('data/uci/kin8nm.csv'),
        'naval': lambda: pd.read_csv(
            'data/uci/naval-propulsion.txt', delim_whitespace=True, header=None
        ).iloc[:, :-1],
        'power': lambda: pd.read_excel('data/uci/power-plant.xlsx'),
        'energy': lambda: pd.read_excel(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
        ).iloc[:, :-1],
        'protein': lambda: pd.read_csv('data/uci/protein.csv')[
            ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'RMSD']
        ],
        'yacht': lambda: pd.read_csv(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data',
            header=None,
            delim_whitespace=True,
        ),
        'msd': lambda: pd.read_csv('data/uci/YearPredictionMSD.txt').iloc[:, ::-1],
    }

    data = dataset_name_to_loader['yacht']()
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values

    # If we want to normalize the data with mean and standard deviation computed
    # over all the data (including both the training and val data)
    # ------------------------------------------------------------
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    y_mean, y_std = np.mean(y, axis=0), np.std(y, axis=0)
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std
    # ------------------------------------------------------------

    # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
    n = X.shape[0]
    np.random.seed(1)

    permutation = np.random.choice(range(n), n, replace=False)
    end_train = round(n * 80.0 / 100.0)  # Take 80% of the data as training data
    end_vak = n
    train_index = permutation[0:end_train]
    val_index = permutation[end_train:n]

    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    print('x_train.shape: {} | y_train.shape: {} | x_val.shape: {} | y_val.shape: {}'.format(
           x_train.shape, y_train.shape, x_val.shape, y_val.shape))

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

@jax.jit
def L_t(w):
    output = model_forward(w, x_train)
    mse_loss = 0.5 * jnp.sum(jnp.square(output.reshape(-1) - y_train.reshape(-1)))
    return mse_loss

# ====================================

@jax.jit
def L_v(w):
    output = model_forward(w, x_val)
    mse_loss = 0.5 * jnp.sum(jnp.square(output.reshape(-1) - y_val.reshape(-1)))
    return mse_loss

@jax.jit
def update(state, i):
    (w, theta, inner_optim_params) = state
    w_grad = grad_L_t_reg(w, theta)
    w, inner_optim_params = inner_optimizer_step(w, w_grad, inner_optim_params)
    return (w, theta, inner_optim_params), w

@partial(jax.jit, static_argnums=3)
def unroll(w, theta, inner_optim_params, K):
    initial_state = (w, theta, inner_optim_params)
    iterations = jax.lax.iota(jnp.int32, K)
    state, outputs = jax.lax.scan(update, initial_state, None, length=K)
    (w_current, theta, inner_optim_params) = state
    return w_current, inner_optim_params

@partial(jax.jit, static_argnums=3)
def unroll_and_L_v(w, theta, inner_optim_params, K):
    w_unrolled, _ = unroll(w, theta, inner_optim_params, K)
    return L_v(w_unrolled)

grad_unroll_and_L_v = jax.jit(jax.grad(unroll_and_L_v, argnums=1), static_argnums=3)

if args.inner_optimizer == 'sgd':
    def reset_inner_optim_params(params):
      return { 'lr':  args.lr }

    in_axes_for_inner_optim = { 'lr': None }

    @jax.jit
    def inner_optimizer_step(params, grads, inner_optim_params):
        lr = inner_optim_params['lr']
        updated_params = params - lr * grads
        return updated_params, inner_optim_params

@partial(jax.jit, static_argnums=(4,5,6))
def es_grad_estimate(key, theta, w, inner_optim_params, K, N, sigma):
    pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
    neg_pert = -pos_pert
    perts = jnp.concatenate([pos_pert, neg_pert])
    val_losses = jax.vmap(unroll_and_L_v, in_axes=(None,0,None,None))(w, theta+perts, inner_optim_params, K)
    gradient_estimate = jnp.sum(val_losses.reshape(-1, 1) * perts, axis=0) / (N * sigma**2)
    return gradient_estimate

@partial(jax.jit, static_argnums=(5,6,7))
def pes_grad_estimate(key, theta, ws, inner_optim_params, perturbation_accums, K, N, sigma):
    pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
    neg_pert = -pos_pert
    perts = jnp.concatenate([pos_pert, neg_pert])
    ws, inner_optim_params = jax.vmap(unroll, in_axes=(0,0,in_axes_for_inner_optim,None))(ws, theta+perts, inner_optim_params, K)
    val_losses = jax.vmap(L_v, in_axes=0)(ws)
    perturbation_accums = perturbation_accums + perts
    gradient_estimate = jnp.sum(val_losses.reshape(-1, 1) * perturbation_accums, axis=0) / (N * sigma**2)
    return gradient_estimate, ws, inner_optim_params, perturbation_accums

@partial(jax.jit, static_argnums=(6,7,8))
def pes_grad_estimate_analytic(key, theta, w, ws, inner_optim_params, perturbation_accums, K, N, sigma):
    pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
    neg_pert = -pos_pert
    perts = jnp.concatenate([pos_pert, neg_pert])
    ws, inner_optim_params_updated = jax.vmap(unroll, in_axes=(0,0,in_axes_for_inner_optim,None))(ws, theta+perts, inner_optim_params, K)
    val_losses = jax.vmap(L_v, in_axes=0)(ws)
    analytic_gradient = grad_unroll_and_L_v(w, theta, inner_optim_params, K)
    things = val_losses - jnp.dot(perts, analytic_gradient)
    gradient_estimate = jnp.sum(things.reshape(-1, 1) * perturbation_accums, axis=0) / (N * sigma**2)
    perturbation_accums = perturbation_accums + perts
    gradient_estimate = gradient_estimate + analytic_gradient
    return gradient_estimate, ws, inner_optim_params_updated, perturbation_accums


if args.data == 'irm':
    # Generate and save grid data for heatmap plotting
    # ------------------------------------------------------------------
    r0 = [0,10]
    rd = [-5,5]
    M = 200
    x = np.linspace(r0[0], r0[1], M)
    y = np.linspace(rd[0], rd[1], M)
    xx, yy = np.meshgrid(x,y)
    lam_values = np.array((xx.ravel(), yy.ravel())).T

    L_v_values = jax.jit(jax.vmap(mp_and_L_v, in_axes=(0,)))(lam_values)
    L_v_grid = L_v_values.reshape(M, M)

    with open(os.path.join(save_dir, 'grid_data.pkl'), 'wb') as f:
        pkl.dump({'xx': xx, 'yy': yy, 'L_v_grid': L_v_grid}, f)
    # ------------------------------------------------------------------


iterations = []
lam_trajectory = []
eps_trajectory = []

theta = jnp.array([args.initial_theta])
key = jax.random.PRNGKey(args.seed)
w = jax.random.normal(key, (x_train.shape[1],))
inner_optim_params = reset_inner_optim_params(w)

# Initialize PES stuff
if args.estimate in ['pes', 'pes-analytic']:
    perturbation_accums = jnp.zeros((args.N, len(theta)))
    ws = jnp.stack([w] * args.N)


# Outer optimization
# =======================================================================
if args.outer_optimizer == 'adam':
    outer_optim_params = {
        'lr': args.outer_lr,
        'b1': args.outer_b1,
        'b2': args.outer_b2,
        'eps': args.outer_eps,
        'm': jnp.zeros(len(theta)),
        'v': jnp.zeros(len(theta)),
    }

    @jax.jit
    def outer_optimizer_step(params, grads, optim_params, t):
        lr = optim_params['lr']
        b1 = optim_params['b1']
        b2 = optim_params['b2']
        eps = optim_params['eps']

        optim_params['m'] = (1 - b1) * grads + b1 * optim_params['m']
        optim_params['v'] = (1 - b2) * (grads**2) + b2 * optim_params['v']
        mhat = optim_params['m'] / (1 - b1**(t+1))
        vhat = optim_params['v'] / (1 - b2**(t+1))

        updated_params = params - lr * mhat / (jnp.sqrt(vhat) + eps)
        return updated_params, optim_params
elif args.outer_optimizer == 'sgd':
    outer_optim_params = { 'lr': args.outer_lr }

    @jax.jit
    def outer_optimizer_step(params, grads, optim_params, t):
        updated_params = params - optim_params['lr'] * grads
        return updated_params, optim_params
# =======================================================================

total_inner_iterations = 0

key = jax.random.PRNGKey(3)
for outer_iteration in range(args.outer_iterations):
    if args.estimate == 'tbptt':
        theta_grad = grad_unroll_and_L_v(w, theta, inner_optim_params, args.K)
        w, inner_optim_params = unroll(w, theta, inner_optim_params, args.K)
    elif args.estimate == 'es':
        key, skey = jax.random.split(key)
        theta_grad = es_grad_estimate(skey,
                                      theta,
                                      w,
                                      inner_optim_params,
                                      args.K,
                                      args.N,
                                      args.sigma)
        w, inner_optim_params = unroll(w, theta, inner_optim_params, args.K)
    elif args.estimate == 'pes':
        key, skey = jax.random.split(key)
        theta_grad, ws, inner_optim_params, perturbation_accums = pes_grad_estimate(skey,
                                                                                    theta,
                                                                                    ws,
                                                                                    inner_optim_params,
                                                                                    perturbation_accums,
                                                                                    args.K,
                                                                                    args.N,
                                                                                    args.sigma)
        # inner_optim_params = reset_inner_optim_except_state(inner_optim_params, params)
        inner_optim_params = reset_inner_optim_params(ws)
        w, inner_optim_params = unroll(w, theta, inner_optim_params, args.K)
    elif args.estimate == 'pes-analytic':
        key, skey = jax.random.split(key)
        theta_grad, ws, inner_optim_params, perturbation_accums = pes_grad_estimate_analytic(skey,
                                                                                             theta,
                                                                                             w,
                                                                                             ws,
                                                                                             inner_optim_params,
                                                                                             perturbation_accums,
                                                                                             args.K,
                                                                                             args.N,
                                                                                             args.sigma)
        inner_optim_params = reset_inner_optim_params(ws)
        w, inner_optim_params = unroll(w, theta, inner_optim_params, args.K)

    total_inner_iterations += 2 * args.K  # Because we separately compute the gradient and then update the "base" w using the mean theta
    total_inner_iterations_including_N = args.N * total_inner_iterations

    if args.outer_clip > 0:
        theta_grad = jnp.clip(theta_grad, -args.outer_clip, args.outer_clip)

    theta, outer_optim_params = outer_optimizer_step(theta, theta_grad, outer_optim_params, outer_iteration)

    if outer_iteration % args.log_every == 0:
        val_loss = L_v(w)
        print('Iter: {} | Total inner iter: {} | Val Loss: {:6.4f} | Theta: {} | Grad: {}'.format(
               outer_iteration, total_inner_iterations_including_N, val_loss, theta, theta_grad))

        iteration_stat_dict = { 'outer_iteration': outer_iteration,
                                'total_inner_iterations': total_inner_iterations_including_N,
                                'val_loss': val_loss,
                                'theta': theta,
                                'theta_grad': theta_grad
                              }
        iteration_logger.writerow(iteration_stat_dict)
