"""Meta-learning a learning rate schedule for a toy 2D regression task.

Examples:
---------
CUDA_VISIBLE_DEVICES=-1 python toy_regression.py --estimate=tbptt
CUDA_VISIBLE_DEVICES=-1 python toy_regression.py --estimate=rtrl
CUDA_VISIBLE_DEVICES=-1 python toy_regression.py --estimate=uoro
CUDA_VISIBLE_DEVICES=-1 python toy_regression.py --estimate=es --sigma=1.0
CUDA_VISIBLE_DEVICES=-1 python toy_regression.py --estimate=pes
"""
import os
import csv
import sys
import pdb
import copy
import time
import argparse
from functools import partial
from typing import NamedTuple, Optional, Any

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import jax
import jax.numpy as jnp

import optax

# Local imports
import schedule
import inner_optim
import gradient_estimators
from logger import CSVLogger


parser = argparse.ArgumentParser(
    description='Hyperparameter Optimization for a 2D Toy Regression Task'
)
parser.add_argument('--outer_iterations', type=int, default=20000,
                    help='Max number of outer iterations')

# Meta-optimization hyperparameters
parser.add_argument('--estimate', type=str, default='es',
                    choices=['tbptt', 'rtrl', 'uoro', 'es', 'pes', 'pes-a'],
                    help='Method for estimating gradients')
parser.add_argument('--tune_params', type=str, default='lr:linear',
                    help='Comma-separated string of hparams to search over')
parser.add_argument('--objective', type=str, default='sum',
                    choices=['sum', 'final'],
                    help='Whether to evaluate theta based on the sum of losses'
                         'for the unroll or the final loss')
parser.add_argument('--T', type=int, default=100,
                    help='Number of unroll steps for the inner optimization')
parser.add_argument('--K', type=int, default=10,
                    help='Truncation length')
parser.add_argument('--sigma', type=float, default=0.3,
                    help='Perturbation scale for ES/PES')
parser.add_argument('--n_per_chunk', type=int, default=100,
                    help='Number of particles per chunk')
parser.add_argument('--n_chunks', type=int, default=1,
                    help='Number of ParticleChunks')

# Inner optimization hyperparameters
parser.add_argument('--inner_optimizer', type=str, default='sgd',
                    choices=['sgd', 'sgdm', 'adam'],
                    help='Inner optimizer')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Learning rate')
parser.add_argument('--b1', type=float, default=0.99,
                    help='Adam b1 hyperparameter')
parser.add_argument('--b2', type=float, default=0.999,
                    help='Adam b2 hyperparameter')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='Adam epsilon hyperparameter')
parser.add_argument('--mom', type=float, default=0.9,
                    help='Momentum')
parser.add_argument('--wd', type=float, default=1e-10,
                    help='Weight decay')

# Outer optimization hyperparameters
parser.add_argument('--outer_optimizer', type=str, default='adam',
                    choices=['adam', 'sgd'],
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
                    help='Gradient clipping for the outer optimization'
                         '(-1 means no clipping)')

parser.add_argument('--init_theta', type=str, default='-4.0,-4.0',
                    help='Initial value for the initial learning rate')

# Logging/plotting hyperparameters
parser.add_argument('--print_interval', type=int, default=10,
                    help='How often to print stats')
parser.add_argument('--log_interval', type=int, default=10,
                    help='How often to print stats and save to log files')
parser.add_argument('--seed', type=int, default=3,
                    help='Seed for PRNG')
parser.add_argument('--save_dir', type=str, default='saves/toy_regression',
                    help='Base save directory')
args = parser.parse_args()

args.tune_param_string = args.tune_params
args.tune_params = [{'param': p.split(':')[0], 'sched': p.split(':')[1]} for
                    p in args.tune_params.split(',')]

exp_name = '{}-{}-{}-{}-optim:{}-lr:{}-T:{}-K:{}-Nc:{}-Npc:{}-sigma:{}-seed:{}'.format(
            args.estimate, args.objective, args.tune_param_string,
            args.init_theta, args.outer_optimizer, args.outer_lr,
            args.T, args.K, args.n_chunks, args.n_per_chunk, args.sigma,
            args.seed)

args.init_theta = [float(p.strip()) for p in args.init_theta.split(',')]

save_dir = os.path.join(args.save_dir, exp_name)
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

# Save command-line arguments
with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
  yaml.dump(vars(args), f)

iteration_logger = CSVLogger(
    fieldnames=['time_elapsed', 'iteration', 'inner_problem_steps',
                'theta0', 'theta1', 'theta0_grad', 'theta1_grad', 'L'],
    filename=os.path.join(save_dir, 'iteration.csv')
)

@jax.jit
def loss(x):
  """Inner loss surface"""
  return jnp.sqrt(x[0]**2 + 5) - jnp.sqrt(5) + jnp.sin(x[1])**2 * \
         jnp.exp(-5*x[0]**2) + 0.25*jnp.abs(x[1] - 100)

loss_grad = jax.jit(jax.grad(loss))


# =======================================================================
# Inner Optimization
# =======================================================================
opt_funcs = inner_optim.init_optimizer(args.inner_optimizer)

reset_opt_params = opt_funcs['reset_opt_params']
opt_step = opt_funcs['opt_step']

if args.inner_optimizer == 'adam':
  init_opt_params = {
      'lr': args.lr,
      'b1': args.b1,
      'b2': args.b2,
      'wd': args.wd,
      'eps': args.eps
  }
elif args.inner_optimizer == 'sgdm':
  init_opt_params = {
      'lr': args.lr,
      'mom': args.momentum
  }
elif args.inner_optimizer == 'sgd':
  init_opt_params = {
      'lr': args.lr,
      'wd': args.wd
  }

def get_inner_opt_params(inner_opt_params, theta, t, T):
  updated_inner_opt_params = copy.deepcopy(inner_opt_params)
  for setting in args.tune_params:
    param = setting['param']
    sched = setting['sched']
    # Only deal with optimization hparams here
    if param not in ['lr', 'mom', 'b1', 'b2', 'eps', 'wd']:
      continue
    theta_subset = theta[jnp.array(idx_dict[param])]
    updated_inner_opt_params[param] = schedule.schedule_funcs[sched](
        updated_inner_opt_params, theta_subset, param, t, T
    )
  return updated_inner_opt_params


@partial(jax.jit, static_argnames=('T', 'K'))
def unroll(rng, theta, state, T, K):
  def update(loop_state, t):
    state, L = loop_state
    g = loss_grad(state.inner_state)
    inner_opt_params = get_inner_opt_params(
        state.inner_opt_state, theta, state.t, T
    )
    inner_state_updated, inner_opt_params = opt_step(
        state.inner_state, g, inner_opt_params
    )
    loss_value = loss(inner_state_updated)
    L += loss_value

    state_updated = state._replace(
      inner_state=inner_state_updated,
      inner_opt_state=inner_opt_params,
      t=state.t+1,
    )
    return (state_updated, L), loss_value

  (state_updated, L), loss_values = jax.lax.scan(
    update, (state, 0.0), jnp.array(list(range(K)))
  )

  result = jnp.sum(loss_values)
  return result, state_updated

grad_unroll = jax.jit(jax.grad(unroll, argnums=0, has_aux=True),
                      static_argnames=('T', 'K'))


theta_vals = []
idx_dict = {}
idx = 0
for (j, setting) in enumerate(args.tune_params):
  param = setting['param']
  sched = setting['sched']
  default = args.init_theta[j]
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


class InnerState(NamedTuple):
  inner_state: jnp.ndarray
  inner_opt_state: Any
  t: jnp.ndarray
  pert_accums: Optional[jnp.ndarray] = None


def init_state_fn(rng):
  """Initialize the inner parameters.
  """
  x = jnp.array([1.0, 1.0])
  inner_opt_state = reset_opt_params(x, init_opt_params)
  inner_state = InnerState(
      # Need to have t be floating point for jax.grad in RTRL and UORO
      t=jnp.array(0.0),
      inner_state=x,
      inner_opt_state=inner_opt_state
  )
  return inner_state

# =============================================================

theta_opt = optax.adam(args.outer_lr, b1=args.outer_b1, b2=args.outer_b2)
theta_opt_state = theta_opt.init(theta)

key = jax.random.PRNGKey(args.seed)
estimator = gradient_estimators.MultiParticleEstimator(
  key=key,
  theta_shape=theta.shape,
  n_chunks=args.n_chunks,
  n_particles_per_chunk=args.n_per_chunk,
  K=args.K,
  T=args.T,
  sigma=args.sigma,
  method='lockstep',
  estimator_type=args.estimate,
  init_state_fn=init_state_fn,
  unroll_fn=unroll,
)

# This will never be reset, it will track time from the very start
start_time = time.time()

# This will be reset every time we print out stats
log_start_time = time.time()

# Meta-optimization loop
for outer_iteration in range(args.outer_iterations):
  theta_grad = estimator.grad_estimate(theta)
  theta_update, theta_opt_state = theta_opt.update(theta_grad, theta_opt_state)
  theta = optax.apply_updates(theta, theta_update)

  if outer_iteration % args.print_interval == 0:
    print('Outer iter: {} | Outer params: {} | Exp(outer): {}'.format(
      outer_iteration, theta, jnp.exp(theta)
    ))
    sys.stdout.flush()

  if outer_iteration % args.log_interval == 0:
    key, skey = jax.random.split(key)

    # Evaluate on the full unroll
    fresh_inner_state = init_state_fn(skey)
    L, _ = unroll(skey, theta, fresh_inner_state, args.T, args.T)

    iteration_logger.writerow({
      'time_elapsed': time.time() - start_time,
      'iteration': outer_iteration,
      'inner_problem_steps': outer_iteration * args.K,
      'theta0': float(theta[0]),
      'theta1': float(theta[1]),
      'theta0_grad': float(theta_grad[0]),
      'theta1_grad': float(theta_grad[1]),
      'L': float(L)
    })

    print('Time: {:6.3f} | Meta-iter: {} | theta: {} | theta_grad: {} | L: {:6.3f}'.format(
           time.time() - log_start_time, outer_iteration, theta, theta_grad,
           float(L)))
    sys.stdout.flush()
    log_start_time = time.time()
