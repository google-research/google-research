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

"""Meta-learning a learning rate schedule for a toy 2D regression task.

Examples:
---------
python toy_regression.py --estimate=tbptt
python toy_regression.py --estimate=rtrl
python toy_regression.py --estimate=uoro
python toy_regression.py --estimate=es
python toy_regression.py --estimate=pes
"""
import os
import sys
import ipdb
import time
import argparse
from functools import partial

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import numpy as onp

import jax
import jax.numpy as jnp

# Local imports
from logger import CSVLogger


parser = argparse.ArgumentParser(description='Hyperparameter Optimization for a 2D Toy Regression Task')
parser.add_argument('--estimate', type=str, default='es', choices=['tbptt', 'rtrl', 'uoro', 'es', 'pes'],
                    help='Method for estimating gradients')
parser.add_argument('--outer_iterations', type=int, default=50000,
                    help='Max number of outer iterations')
parser.add_argument('--schedule_type', type=str, default='linear', choices=['linear', 'inverse-time-decay'],
                    help='The type of schedule to parameterize with our theta0, theta1 hyperparameters')
parser.add_argument('--T', type=int, default=100,
                    help='Number of unroll steps for the inner optimization')
parser.add_argument('--K', type=int, default=10,
                    help='Truncation length')
parser.add_argument('--N', type=int, default=100,
                    help='Number of noise samples for ES/PES')
parser.add_argument('--sigma', type=float, default=1.0,
                    help='Perturbation scale for ES/PES')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                    help='Optimizer')
parser.add_argument('--outer_lr', type=float, default=1e-2,
                    help='Outer learning rate')
parser.add_argument('--outer_b1', type=float, default=0.99,
                    help='Outer optimizer beta1')
parser.add_argument('--outer_b2', type=float, default=0.999,
                    help='Outer optimizer beta2')
parser.add_argument('--outer_eps', type=float, default=1e-8,
                    help='Outer optimizer epsilon')
parser.add_argument('--outer_clip', type=float, default=-1,
                    help='Gradient clipping coefficient for the outer optimization (-1 means no clipping)')
parser.add_argument('--pes_analytic', action='store_true', default=False,
                    help='Whether to use the analytic gradient to reduce the variance of PES')
parser.add_argument('--shuffle_perts', action='store_true', default=False,
                    help='Whether to shuffle the orders of the PES perturbations for each unroll')
parser.add_argument('--theta0', type=float, default=0.01,
                    help='Initial value for the initial learning rate')
parser.add_argument('--theta1', type=float, default=0.01,
                    help='Initial value for the final learning rate')
parser.add_argument('--log_interval', type=int, default=100,
                    help='How often to print stats and save to log files')
parser.add_argument('--seed', type=int, default=1,
                    help='Seed for PRNG')
parser.add_argument('--save_dir', type=str, default='saves/toy_regression',
                    help='Base save directory')
args = parser.parse_args()


exp_name = '{}-s:{}-optim:{}-lr:{}-T:{}-K:{}-N:{}-sigma:{}-seed:{}'.format(
            args.estimate, args.schedule_type, args.optimizer, args.outer_lr,
            args.T, args.K, args.N, args.sigma, args.seed)

save_dir = os.path.join(args.save_dir, exp_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save command-line arguments
with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
    yaml.dump(vars(args), f)

iteration_logger = CSVLogger(fieldnames=['time_elapsed', 'iteration', 'inner_problem_steps', 'theta0', 'theta1',
                                         'theta0_grad', 'theta1_grad', 'L'],
                             filename=os.path.join(save_dir, 'iteration.csv'))

@jax.jit
def loss(x):
    """Inner loss surface"""
    return jnp.sqrt(x[0]**2 + 5) - jnp.sqrt(5) + jnp.sin(x[1])**2 * jnp.exp(-5*x[0]**2) + 0.25*jnp.abs(x[1] - 100)

loss_grad = jax.jit(jax.grad(loss))


@jax.jit
def update(state, i):
    (L_sum, x, theta, t_current, T, K) = state
    if args.schedule_type == 'linear':
        lr = jnp.exp(theta[0]) * (T - t_current) / T + jnp.exp(theta[1]) * t_current / T
    elif args.schedule_type == 'inverse-time-decay':
        lr = jnp.exp(theta[0]) / ((1 + t_current / T)**jnp.exp(theta[1]))
    g = loss_grad(x)
    x = x - lr * g
    L_sum += loss(x) * (t_current < T)
    t_current += 1
    return (L_sum, x, theta, t_current, T, K), x  # This last x returned will give us the trajectory in the end


@partial(jax.jit, static_argnums=(3,4))
def unroll(x_init, theta, t_current, T, K):
    L_sum = 0.0
    initial_state = (L_sum, x_init, theta, t_current, T, K)
    iterations = jax.lax.iota(jnp.int32, K)  # This is like jnp.arange(0, K), I don't think iota works for different starting points than 0...
    state, outputs = jax.lax.scan(update, initial_state, iterations)
    (L, x_current, theta, t_current, T, K) = state
    x_trajectory = jnp.stack(outputs).T
    return L, (x_current, x_trajectory, t_current)

grad_unroll = jax.jit(jax.grad(unroll, argnums=1, has_aux=True), static_argnums=(3,4))
loss_and_grad_unroll = jax.jit(jax.value_and_grad(unroll, argnums=1, has_aux=True), static_argnums=(3,4))


# Functions for RTRL/UORO
# ----------------------------------------------------------------------------------
@jax.jit
def single_step(theta, x, t_current, T):
    g = loss_grad(x)
    lr = jnp.exp(theta[0]) * (T - t_current) / T + jnp.exp(theta[1]) * t_current / T
    x_new = x - lr * g
    return x_new

@jax.jit
def single_step_loss(theta, x, t_current, T):
    L = 0
    g = loss_grad(x)
    lr = jnp.exp(theta[0]) * (T - t_current) / T + jnp.exp(theta[1]) * t_current / T
    x_new = x - lr * g
    L += loss(x_new) * (t_current < T)
    return L

compute_dL_dstate_old = jax.jit(jax.grad(single_step_loss, argnums=1))
compute_dL_dtheta_direct = jax.jit(jax.grad(single_step_loss, argnums=0))

compute_d_state_new_d_state_old = jax.jit(jax.jacrev(single_step, argnums=1))
compute_d_state_new_d_theta_direct = jax.jit(jax.jacrev(single_step, argnums=0))

def rtrl_grad(theta, x, t0, T, K, dstate_dtheta=None):
    t_current = t0
    mystate = x
    total_loss = 0.0

    if dstate_dtheta is None:
        dstate_dtheta = jnp.zeros((len(mystate), len(theta)))

    total_theta_grad = 0
    total_loss = 0.0

    for step in range(K):
        state_vec_old = mystate
        state_new = single_step(theta, mystate, t_current, T)
        total_loss += loss(state_new) * (t_current < T)

        dl_dstate_old = compute_dL_dstate_old(theta, state_vec_old, t_current, T)
        dl_dtheta_direct = compute_dL_dtheta_direct(theta, state_vec_old, t_current, T)

        d_state_new_d_state_old = compute_d_state_new_d_state_old(theta, state_vec_old, t_current, T)
        d_state_new_d_theta_direct = compute_d_state_new_d_theta_direct(theta, state_vec_old, t_current, T)

        theta_grad = jnp.dot(dl_dstate_old.reshape(1, -1), dstate_dtheta).reshape(-1) + dl_dtheta_direct
        total_theta_grad += theta_grad
        dstate_dtheta = jnp.dot(d_state_new_d_state_old, dstate_dtheta) + d_state_new_d_theta_direct

        mystate = state_new

        t_current += 1

    return (total_loss, mystate, dstate_dtheta), total_theta_grad

rtrl_grad = jax.jit(rtrl_grad, static_argnums=(3,4))


def uoro_grad(key, theta, x, t0, T, K, s_tilde=None, theta_tilde=None):
    epsilon_perturbation = 1e-7
    epsilon_stability = 1e-7

    t_current = t0
    mystate = x
    total_theta_grad = 0
    total_loss = 0.0

    if s_tilde is None:
        s_tilde = jnp.zeros(mystate.shape)

    if theta_tilde is None:
        theta_tilde = jnp.zeros(theta.shape)

    for i in range(K):
        state_vec_old = mystate

        state_new = single_step(theta, mystate, t_current, T)
        total_loss += loss(state_new) * (t_current < T)

        state_vec_new = state_new

        dl_dstate_old = compute_dL_dstate_old(theta, state_vec_old, t_current, T)
        dl_dtheta_direct = compute_dL_dtheta_direct(theta, state_vec_old, t_current, T)

        indirect_grad = (dl_dstate_old * s_tilde).sum() * theta_tilde
        pseudograds = indirect_grad + dl_dtheta_direct

        state_old_perturbed = state_vec_old + s_tilde * epsilon_perturbation
        state_vec_new_perturbed = single_step(theta, state_old_perturbed, t_current, T)

        state_deriv_in_direction_s_tilde = (state_vec_new_perturbed - state_vec_new) / epsilon_perturbation

        key, skey = jax.random.split(key)
        nus = jnp.round(jax.random.uniform(skey, state_vec_old.shape)) * 2 - 1

        custom_f = lambda param_vector: single_step(param_vector, state_vec_old, t_current, T)
        primals, f_vjp = jax.vjp(custom_f, theta)
        direct_theta_tilde_contribution, = f_vjp(nus)

        rho_0 = jnp.sqrt((jnp.linalg.norm(theta_tilde) + epsilon_stability) / (jnp.linalg.norm(state_deriv_in_direction_s_tilde) + epsilon_stability))
        rho_1 = jnp.sqrt((jnp.linalg.norm(direct_theta_tilde_contribution) + epsilon_stability) / (jnp.linalg.norm(nus) + epsilon_stability))

        theta_grad = pseudograds
        total_theta_grad += theta_grad

        s_tilde = rho_0 * state_deriv_in_direction_s_tilde + rho_1 * nus
        theta_tilde = theta_tilde / rho_0 + direct_theta_tilde_contribution / rho_1

        mystate = state_new
        t_current += 1

    return (key, total_loss, mystate, s_tilde, theta_tilde), total_theta_grad

uoro_grad = jax.jit(uoro_grad, static_argnums=(4,5))
# ----------------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(4,5,6,7))
def es_grad(key, x, theta, t0, T, K, sigma, N):
    pos_perturbations = jax.random.normal(key, (N//2, theta.shape[0])) * sigma
    neg_perturbations = -pos_perturbations
    perturbations = jnp.concatenate([pos_perturbations, neg_perturbations], axis=0)

    L, _ = jax.vmap(unroll, in_axes=(None,0,None,None,None))(x, theta + perturbations, t0, T, K)
    theta_grad = jnp.mean(perturbations * L.reshape(-1, 1) / (sigma**2), axis=0)
    return theta_grad


@partial(jax.jit, static_argnums=(5,6,7,8))
def pes_grad(key, xs, perturbation_accum, theta, t0, T, K, sigma, N):
    pos_perturbations = jax.random.normal(key, (N//2, theta.shape[0])) * sigma
    neg_perturbations = -pos_perturbations
    perturbations = jnp.concatenate([pos_perturbations, neg_perturbations], axis=0)

    L, aux = jax.vmap(unroll, in_axes=(0,0,None,None,None))(xs, theta + perturbations, t0, T, K)
    xs, x_trajectories, ts = aux
    perturbation_accum = perturbation_accum + perturbations
    theta_grad = jnp.mean(perturbation_accum * L.reshape(-1, 1) / (sigma**2), axis=0)
    return theta_grad, xs, perturbation_accum


# Outer optimization loop
# =======================================================================
theta = jnp.log(jnp.array([args.theta0, args.theta1]))

if args.estimate in ['tbptt', 'rtrl', 'uoro', 'es']:
    x = jnp.array([1.0, 1.0])
elif args.estimate == 'pes':
    x = jnp.array([1.0, 1.0])
    xs = jnp.ones((args.N, 2)) * jnp.array([1.0, 1.0])
    perturbation_accum = jnp.zeros((args.N, theta.shape[0]))

# =======================================================================
# Optimization
# =======================================================================
if args.optimizer == 'adam':
    optim_params = {
        'lr': args.outer_lr,
        'b1': args.outer_b1,
        'b2': args.outer_b2,
        'eps': args.outer_eps,
        'm': jnp.zeros(theta.shape[0]),
        'v': jnp.zeros(theta.shape[0]),
    }

    @jax.jit
    def optimizer_step(params, grads, optim_params, t):
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

elif args.optimizer == 'sgd':
    optim_params = {
        'lr': args.outer_lr,
    }

    @jax.jit
    def optimizer_step(params, grads, optim_params, t):
        updated_params = params - optim_params['lr'] * grads
        return updated_params, optim_params
# =======================================================================

t = 0
initial_point = jnp.array([1.0, 1.0])
key = jax.random.PRNGKey(args.seed)

start_time = time.time()  # This will never be reset, it will track time from the very start
log_start_time = time.time()  # This will be reset every time we print out stats

if args.estimate == 'tbptt':
    for i in range(args.outer_iterations):

        if t >= args.T:
            x = jnp.array(initial_point)  # Reset the inner parameters
            t = 0

        theta_grad, aux = grad_unroll(x, theta, t, args.T, args.K)
        x = aux[0]  # Update to be the x we get by unrolling the optimization with theta
        t = aux[2]  # Update to t_current from the unroll we used to get x

        # Gradient clipping
        if args.outer_clip > 0:
            theta_grad = theta_grad * jnp.minimum(1., args.outer_clip / (jnp.linalg.norm(theta_grad) + 1e-8))

        theta, optim_params = optimizer_step(theta, theta_grad, optim_params, i)

        if i % args.log_interval == 0:
            L, _ = unroll(jnp.array(initial_point), theta, 0, args.T, args.T)  # Evaluate on the full unroll
            iteration_logger.writerow({ 'time_elapsed': time.time() - start_time,
                                        'iteration': i,
                                        'inner_problem_steps': i * args.K,
                                        'theta0': float(theta[0]),
                                        'theta1': float(theta[1]),
                                        'theta0_grad': float(theta_grad[0]),
                                        'theta1_grad': float(theta_grad[1]),
                                        'L': float(L) })

            print('Time: {:6.3f} | Meta-iter: {} | theta: {} | theta_grad: {} | L: {:6.3f}'.format(
                   time.time() - log_start_time, i, jnp.exp(theta), theta_grad, float(L)))
            sys.stdout.flush()
            log_start_time = time.time()

elif args.estimate == 'rtrl':

    dstate_dtheta = None

    for i in range(args.outer_iterations):
        key, skey = jax.random.split(key)

        if t >= args.T:
            t = 0
            x = jnp.array(initial_point)  # Reset the inner parameters
            dstate_dtheta = None

        (total_loss, mystate, dstate_dtheta), theta_grad = rtrl_grad(theta, x, t, args.T, args.K, dstate_dtheta=dstate_dtheta)

        L, aux = unroll(x, theta, t, args.T, args.K)
        x = aux[0]  # Update to be the x we get by unrolling the optimization with theta
        t = aux[2]  # Update to t_current from the unroll we used to get x

        theta, optim_params = optimizer_step(theta, theta_grad, optim_params, i)

        if i % args.log_interval == 0:
            L, _ = unroll(jnp.array(initial_point), theta, 0, args.T, args.T)  # Evaluate on the full unroll

            iteration_logger.writerow({ 'time_elapsed': time.time() - start_time,
                                        'iteration': i,
                                        'inner_problem_steps': i * args.K,
                                        'theta0': float(theta[0]),
                                        'theta1': float(theta[1]),
                                        'theta0_grad': float(theta_grad[0]),
                                        'theta1_grad': float(theta_grad[1]),
                                        'L': float(L) })

            print('Time: {:6.3f} | Meta-iter: {} | theta: {} | theta_grad: {} | L: {:6.3f}'.format(
                   time.time() - log_start_time, i, jnp.exp(theta), theta_grad, float(L)))
            sys.stdout.flush()
            log_start_time = time.time()

elif args.estimate == 'uoro':
    s_tilde = None
    theta_tilde = None

    for i in range(args.outer_iterations):
        key, skey = jax.random.split(key)

        if t >= args.T:
            t = 0
            x = jnp.array(initial_point)  # Reset the inner parameters
            s_tilde = None
            theta_tilde = None

        (key, total_loss, mystate, s_tilde, theta_tilde), theta_grad = uoro_grad(skey, theta, x, t, args.T, args.K, s_tilde=None, theta_tilde=None)

        L, aux = unroll(x, theta, t, args.T, args.K)
        x = aux[0]  # Update to be the x we get by unrolling the optimization with theta
        t = aux[2]  # Update to t_current from the unroll we used to get x

        theta, optim_params = optimizer_step(theta, theta_grad, optim_params, i)

        if i % args.log_interval == 0:
            L, _ = unroll(jnp.array(initial_point), theta, 0, args.T, args.T)  # Here, the number of unroll steps is the same as the max number of iterations

            iteration_logger.writerow({ 'time_elapsed': time.time() - start_time,
                                        'iteration': i,
                                        'inner_problem_steps': i * args.K,
                                        'theta0': float(theta[0]),
                                        'theta1': float(theta[1]),
                                        'theta0_grad': float(theta_grad[0]),
                                        'theta1_grad': float(theta_grad[1]),
                                        'L': float(L) })

            print('Time: {:6.3f} | Meta-iter: {} | theta: {} | theta_grad: {} | L: {:6.3f}'.format(
                   time.time() - log_start_time, i, jnp.exp(theta), theta_grad, float(L)))
            sys.stdout.flush()
            log_start_time = time.time()

elif args.estimate == 'es':
    for i in range(args.outer_iterations):
        key, skey = jax.random.split(key)

        if t >= args.T:
            x = jnp.array(initial_point)  # Reset the inner parameters
            t = 0

        theta_grad = es_grad(skey, x, theta, t, args.T, args.K, args.sigma, args.N)
        L, aux = unroll(x, theta, t, args.T, args.K)
        x = aux[0]  # This is actually "updating" x to be the x we get by unrolling the optimization with theta
        t = aux[2]  # Update i to t_current from the unroll we used to update x

        theta, optim_params = optimizer_step(theta, theta_grad, optim_params, i)

        if i % args.log_interval == 0:
            L, _ = unroll(jnp.array(initial_point), theta, 0, args.T, args.T)  # Here, the number of unroll steps is the same as the max number of iterations

            iteration_logger.writerow({ 'time_elapsed': time.time() - start_time,
                                        'iteration': i,
                                        'inner_problem_steps': i * args.K,
                                        'theta0': float(theta[0]),
                                        'theta1': float(theta[1]),
                                        'theta0_grad': float(theta_grad[0]),
                                        'theta1_grad': float(theta_grad[1]),
                                        'L': float(L) })

            print('Time: {:6.3f} | Meta-iter: {} | theta: {} | theta_grad: {} | L: {:6.3f}'.format(
                   time.time() - log_start_time, i, jnp.exp(theta), theta_grad, float(L)))
            sys.stdout.flush()
            log_start_time = time.time()

elif args.estimate == 'pes':
    for i in range(args.outer_iterations):
        key, skey = jax.random.split(key)

        if t >= args.T:
            t = 0
            xs = jnp.ones((args.N, 2)) * jnp.array([1.0, 1.0])  # Reset the inner parameters
            x = jnp.array(initial_point)
            perturbation_accum = jnp.zeros((args.N, theta.shape[0]))

        if args.pes_analytic:
            theta_grad, xs, perturbation_accum = pes_grad_analytic(skey, x, xs, perturbation_accum, theta, t, args.T, args.K, args.sigma, args.N)
            _, (x, _, _) = unroll(x, theta, t, args.T, args.K)
        else:
            theta_grad, xs, perturbation_accum = pes_grad(skey, xs, perturbation_accum, theta, t, args.T, args.K, args.sigma, args.N)

        theta, optim_params = optimizer_step(theta, theta_grad, optim_params, i)
        t += args.K  # Because for PES we already updated the inner parameters with a K-step unroll in the pes_grad function

        if i % args.log_interval == 0:
            L, _ = unroll(jnp.array(initial_point), theta, 0, args.T, args.T)  # Here, the number of unroll steps is the same as the max number of iterations

            iteration_logger.writerow({ 'time_elapsed': time.time() - start_time,
                                        'iteration': i,
                                        'inner_problem_steps': i * args.K,
                                        'theta0': float(theta[0]),
                                        'theta1': float(theta[1]),
                                        'theta0_grad': float(theta_grad[0]),
                                        'theta1_grad': float(theta_grad[1]),
                                        'L': float(L) })

            print('Time: {:6.3f} | Meta-iter: {} | theta: {} | theta_grad: {} | L: {:6.3f}'.format(
                   time.time() - log_start_time, i, jnp.exp(theta), theta_grad, float(L)))
            sys.stdout.flush()
            log_start_time = time.time()
