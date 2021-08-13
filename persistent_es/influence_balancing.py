# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Influence balancing experiment based on the description in the UORO paper, https://arxiv.org/abs/1702.05043.

Examples:
---------
python influence_balancing.py --K=1 --estimate=tbptt
python influence_balancing.py --K=10 --estimate=tbptt
python influence_balancing.py --K=100 --estimate=tbptt

python influence_balancing.py --K=1 --estimate=es
python influence_balancing.py --K=1 --estimate=pes

python influence_balancing.py --K=1 --lr=1e-5 --estimate=uoro
python influence_balancing.py --K=1 --estimate=rtrl
"""
import os
import ipdb
import argparse
from functools import partial

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import jax
import jax.numpy as jnp

# Local imports
from logger import CSVLogger

parser = argparse.ArgumentParser(description='Influence Balancing Experiments')
parser.add_argument('--iterations', type=int, default=3000,
                    help='How many gradient steps to perform')
parser.add_argument('--estimate', type=str, default='tbptt', choices=['tbptt', 'es', 'pes', 'pes-analytic', 'uoro', 'rtrl'],
                    help='Which gradient estimate to use')
parser.add_argument('--K', type=int, default=1,
                    help='Unroll length')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--N', type=int, default=1000,
                    help='Number of particles for ES/PES')
parser.add_argument('--sigma', type=float, default=1e-1,
                    help='Perturbation scale for ES/PES')
parser.add_argument('--save_dir', type=str, default='saves/influence',
                    help='Save directory')
args = parser.parse_args()


exp_name = '{}-lr:{}-K:{}-sigma:{}-N:{}'.format(
            args.estimate, args.lr, args.K, args.sigma, args.N)

save_dir = os.path.join(args.save_dir, exp_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save command-line arguments
with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
    yaml.dump(vars(args), f)

iteration_logger = CSVLogger(fieldnames=['iteration', 'loss', 'long_unroll_loss', 'theta', 'gradient'],
                             filename=os.path.join(save_dir, 'iteration.csv'))

# Influence balancing problem setup
# -----------------------------------------------
n = 23
values = jnp.array([0.5]*n)
A = jnp.diag(values) + jnp.diag(values, 1)[:n,:n]
theta = jnp.array([0.5])
num_positive = 10
# -----------------------------------------------

@partial(jax.jit, static_argnums=2)
def unroll(theta, state, K):
    sign_vector = jnp.array([1] * num_positive + [-1] * (n - num_positive))
    theta_vec = jnp.repeat(theta, n) * sign_vector
    state_current = jnp.array(state)
    for i in range(K):
        state_current = jnp.matmul(A, state_current) + theta_vec
    loss = 0.5 * (state_current[0] - 1)**2
    return loss, state_current

unroll_grad = jax.jit(jax.value_and_grad(unroll, argnums=0, has_aux=True), static_argnums=(2,))

@partial(jax.jit, static_argnums=(3,4,5))
def es_grad_estimate(key, theta, state, K, N=10, sigma=0.1):
    pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
    neg_pert = -pos_pert
    perts = jnp.concatenate([pos_pert, neg_pert])
    losses, states = jax.vmap(unroll, in_axes=(0,None,None))(theta+perts, state, K)
    gradient_estimate = jnp.sum(losses.reshape(-1, 1) * perts, axis=0) / (N * sigma**2)
    return gradient_estimate

@partial(jax.jit, static_argnums=(4,5,6))
def pes_grad_estimate(key, theta, states, perturbation_accums, K, N=10, sigma=0.1):
    pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
    neg_pert = -pos_pert
    perts = jnp.concatenate([pos_pert, neg_pert])
    losses, states = jax.vmap(unroll, in_axes=(0,0,None))(theta+perts, states, K)
    perturbation_accums = perturbation_accums + perts
    gradient_estimate = jnp.sum(losses.reshape(-1, 1) * perturbation_accums, axis=0) / (N * sigma**2)
    return gradient_estimate, states, perturbation_accums

@partial(jax.jit, static_argnums=(5,6,7))
def pes_grad_estimate_analytic(key, theta, state, states, perturbation_accums, K, N=10, sigma=0.1):
    pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
    neg_pert = -pos_pert
    perts = jnp.concatenate([pos_pert, neg_pert])
    losses, states = jax.vmap(unroll, in_axes=(0,0,None))(theta+perts, states, K)
    (loss_mean_params, s), analytic_gradient = unroll_grad(theta, state, K)
    things = losses - jnp.dot(perts, analytic_gradient)
    gradient_estimate = jnp.sum(things.reshape(-1, 1) * perturbation_accums, axis=0) / (N * sigma**2)
    perturbation_accums = perturbation_accums + perts
    gradient_estimate = gradient_estimate + analytic_gradient
    return gradient_estimate, states, perturbation_accums

# ------------------------------
# Start of RTRL & UORO functions
# ------------------------------
@jax.jit
def f(theta, state):
    loss, state_next = unroll(theta, state, 1)
    return state_next

@jax.jit
def L(theta, state):
    loss, state_next = unroll(theta, state, 1)
    return loss

compute_d_state_new_d_theta_direct = jax.jit(jax.jacrev(f, argnums=0))
compute_d_state_new_d_state_old = jax.jit(jax.jacrev(f, argnums=1))
compute_dL_dstate_old = jax.jit(jax.grad(L, argnums=1))
compute_dL_dtheta_direct = jax.jit(jax.grad(L, argnums=0))
compute_loss_grad_bptt = jax.jit(jax.grad(L, argnums=0))

@jax.jit
def rtrl_grad(theta, state, dstate_dtheta):
    total_theta_grad = 0
    total_loss = 0.0

    if dstate_dtheta is None:
        dstate_dtheta = jnp.zeros((len(state), len(theta)))

    state_old = state
    state_new = f(theta, state_old)
    loss = L(theta, state_old)
    total_loss += loss

    dl_dstate_old = compute_dL_dstate_old(theta, state_old)        # (23,)
    dl_dtheta_direct = compute_dL_dtheta_direct(theta, state_old)  # (1,)

    d_state_new_d_state_old = compute_d_state_new_d_state_old(theta, state_old)        # (23, 23)
    d_state_new_d_theta_direct = compute_d_state_new_d_theta_direct(theta, state_old)  # (23, 1)

    theta_grad = jnp.dot(dl_dstate_old.reshape(1, -1), dstate_dtheta).reshape(-1) + dl_dtheta_direct
    total_theta_grad += theta_grad
    dstate_dtheta = jnp.dot(d_state_new_d_state_old, dstate_dtheta) + d_state_new_d_theta_direct  # (23, 1)

    return (total_loss, state_new, dstate_dtheta), total_theta_grad

@jax.jit
def uoro_grad(key, theta, state, s_tilde=None, theta_tilde=None):
    epsilon_perturbation = 1e-7
    epsilon_stability = 1e-7

    total_theta_grad = 0
    total_loss = 0.0

    if s_tilde is None:
        s_tilde = jnp.zeros(state.shape)

    if theta_tilde is None:
        theta_tilde = jnp.zeros(theta.shape)

    state_old = state                # (23,)
    state_new = f(theta, state_old)  # (23,)
    loss = L(theta, state_old)
    total_loss += loss

    dl_dstate_old = compute_dL_dstate_old(theta, state_old)           # (23,)
    dl_dtheta_direct = compute_dL_dtheta_direct(theta, state_old)     # (1,)

    indirect_grad = (dl_dstate_old * s_tilde).sum() * theta_tilde     # (1,)
    pseudograds = indirect_grad + dl_dtheta_direct                    # (1,)

    state_old_perturbed = state_old + s_tilde * epsilon_perturbation  # (23,)
    state_new_perturbed = f(theta, state_old_perturbed)               # (23,)

    state_deriv_in_direction_s_tilde = (state_new_perturbed - state_new) / epsilon_perturbation  # (23,)

    nus = jnp.round(jax.random.uniform(key, state_old.shape)) * 2 - 1  # (23,)

    # Tricky part is this first line
    custom_f = lambda param_vector: f(param_vector, state_old)
    primals, f_vjp = jax.vjp(custom_f, theta)
    direct_theta_tilde_contribution, = f_vjp(nus)  # (1,)

    rho_0 = jnp.sqrt((jnp.linalg.norm(theta_tilde) + epsilon_stability) / (jnp.linalg.norm(state_deriv_in_direction_s_tilde) + epsilon_stability))
    rho_1 = jnp.sqrt((jnp.linalg.norm(direct_theta_tilde_contribution) + epsilon_stability) / (jnp.linalg.norm(nus) + epsilon_stability))

    theta_grad = pseudograds
    total_theta_grad += theta_grad

    s_tilde = rho_0 * state_deriv_in_direction_s_tilde + rho_1 * nus
    theta_tilde = theta_tilde / rho_0 + direct_theta_tilde_contribution / rho_1

    return (total_loss, state_new, s_tilde, theta_tilde), total_theta_grad
# ------------------------------
# End of UORO & RTRL functions
# ------------------------------

state = jnp.ones(n)
if args.estimate in ['pes', 'pes-analytic']:
    perturbation_accums = jnp.zeros((args.N, len(theta)))
    states = jnp.ones((args.N,n))

# This is for RTRL
dstate_dtheta = None

# These two lines are for UORO
s_tilde = None
theta_tilde = None

key = jax.random.PRNGKey(3)
for i in range(args.iterations):
    if args.estimate == 'tbptt':
        (loss, state), gradient = unroll_grad(theta, state, args.K)
    elif args.estimate == 'es':
        key, skey = jax.random.split(key)
        gradient = es_grad_estimate(skey,
                                    theta,
                                    state,
                                    args.K,
                                    args.N,
                                    args.sigma)
        loss, state = unroll(theta, state, args.K)
    elif args.estimate == 'pes':
        key, skey = jax.random.split(key)
        gradient, states, perturbation_accums = pes_grad_estimate(skey,
                                                                  theta,
                                                                  states,
                                                                  perturbation_accums,
                                                                  args.K,
                                                                  args.N,
                                                                  args.sigma)
        loss, state = unroll(theta, state, args.K)
    elif args.estimate == 'pes-analytic':
        key, skey = jax.random.split(key)
        gradient, states, perturbation_accums = pes_grad_estimate_analytic(skey,
                                                                           theta,
                                                                           state,
                                                                           states,
                                                                           perturbation_accums,
                                                                           args.K,
                                                                           args.N,
                                                                           args.sigma)
        loss, state = unroll(theta, state, args.K)
    elif args.estimate == 'uoro':
        key, skey = jax.random.split(key)
        (loss, state, s_tilde, theta_tilde), gradient = uoro_grad(skey,
                                                                  theta,
                                                                  state,
                                                                  s_tilde,
                                                                  theta_tilde)
    elif args.estimate == 'rtrl':
        (loss, state, dstate_dtheta), gradient = rtrl_grad(theta, state, dstate_dtheta)

    lr = args.lr
    theta = theta - lr * gradient
    long_unroll_loss, _ = unroll(theta, jnp.ones(n), 500)
    print('Iter: {} | lr: {:6.4e} | Loss: {:6.4e} | Long unroll loss: {:6.4e} | Theta: {} | Grad: {}'.format(
           i, lr, loss, long_unroll_loss, theta, gradient))

    iteration_logger.writerow({ 'iteration': i,
                                'loss': loss,
                                'long_unroll_loss': long_unroll_loss,
                                'theta': theta,
                                'gradient': gradient
                              })
