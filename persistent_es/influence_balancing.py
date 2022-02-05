"""Influence balancing experiment based on the description in the UORO paper,
https://arxiv.org/abs/1702.05043.

Examples:
---------
CUDA_VISIBLE_DEVICES=-1 python influence_balancing.py --K=1 --estimate=tbptt
CUDA_VISIBLE_DEVICES=-1 python influence_balancing.py --K=10 --estimate=tbptt
CUDA_VISIBLE_DEVICES=-1 python influence_balancing.py --K=100 --estimate=tbptt

CUDA_VISIBLE_DEVICES=-1 python influence_balancing.py --K=1 --estimate=es
CUDA_VISIBLE_DEVICES=-1 python influence_balancing.py --K=1 --estimate=pes

CUDA_VISIBLE_DEVICES=-1 python influence_balancing.py --K=1 --lr=1e-5 --estimate=uoro
CUDA_VISIBLE_DEVICES=-1 python influence_balancing.py --K=1 --estimate=rtrl
"""
import os
import sys
import pdb
import argparse
from functools import partial
from typing import NamedTuple, Optional, Union

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import jax
import jax.numpy as jnp

import optax
import haiku as hk

# Local imports
import gradient_estimators
from logger import CSVLogger


method_choices = ['tbptt', 'es', 'pes', 'pes-a', 'uoro', 'rtrl']

parser = argparse.ArgumentParser(description='Influence Balancing Task')
parser.add_argument('--iterations', type=int, default=3000,
                    help='How many gradient steps to perform')
parser.add_argument('--estimate', type=str, default='tbptt',
                    choices=method_choices,
                    help='Which gradient estimate to use')
parser.add_argument('--K', type=int, default=1,
                    help='Unroll length')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--N', type=int, default=1000,
                    help='Number of particles for ES/PES')
parser.add_argument('--sigma', type=float, default=1e-1,
                    help='Perturbation scale for ES/PES')
parser.add_argument('--outer_clip', type=float, default=-1,
                    help='Optional outer gradient clipping')
parser.add_argument('--seed', type=int, default=3,
                    help='Random seed')
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

iteration_logger = CSVLogger(
  fieldnames=['iteration', 'loss', 'theta', 'gradient'],
  filename=os.path.join(save_dir, 'iteration.csv')
)

# Influence balancing problem setup
# -----------------------------------------------
n = 23
values = jnp.array([0.5] * n)
A = jnp.diag(values) + jnp.diag(values, 1)[:n,:n]
theta = jnp.array([0.5])
num_positive = 10
sign_vector = jnp.array([1] * num_positive + [-1] * (n - num_positive))
# -----------------------------------------------

@partial(jax.jit, static_argnames=('T', 'K'))
def unroll(rng, theta, state, T, K):
  theta_vec = jnp.repeat(theta, n) * sign_vector
  state_current = jnp.array(state.inner_state)
  for i in range(K):
    state_current = jnp.matmul(A, state_current) + theta_vec
  loss = 0.5 * (state_current[0] - 1)**2
  updated_state = state._replace(inner_state=state_current,
                                 t=state.t + K)
  return loss, updated_state


class InnerState(NamedTuple):
  inner_state: Union[jnp.ndarray, hk.Params]
  t: jnp.ndarray
  pert_accums: Optional[jnp.ndarray] = None


def init_state_fn(rng):
  """Initialize the inner parameters.
  """
  inner_state = InnerState(t=jnp.zeros(1), inner_state=jnp.ones(n))
  return inner_state


key = jax.random.PRNGKey(args.seed)
estimator = gradient_estimators.MultiParticleEstimator(
  key=key,
  theta_shape=theta.shape,
  n_chunks=1,
  n_particles_per_chunk=args.N,
  K=args.K,
  T=None,
  sigma=args.sigma,
  method='lockstep',
  estimator_type=args.estimate,
  init_state_fn=init_state_fn,
  unroll_fn=unroll,
)

# Set up outer parameters and outer optimization
theta_opt = optax.sgd(args.lr)
theta_opt_state = theta_opt.init(theta)

state_from_scratch = InnerState(t=jnp.zeros(1), inner_state=jnp.ones(n))
for i in range(args.iterations):
  gradient = estimator.grad_estimate(theta)

  if args.outer_clip > 0:
    gradient = jnp.clip(gradient, a_min=-args.outer_clip, a_max=args.outer_clip)

  outer_update, theta_opt_state = jax.jit(theta_opt.update)(gradient, theta_opt_state)
  theta = jax.jit(optax.apply_updates)(theta, outer_update)

  loss, _ = unroll(None, theta, state_from_scratch, 500, 500)
  print('Iter: {} | Unroll loss: {:6.4e} | Theta: {} | Grad: {}'.format(
         i, loss, theta, gradient))
  sys.stdout.flush()

  iteration_logger.writerow({
    'iteration': i,
    'loss': loss,
    'theta': theta,
    'gradient': gradient
  })
