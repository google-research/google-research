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

r"""Implicit aux tasks training.

python -m aux_tasks.minigrid.trainbis \
  --env_name=classic_fourrooms \
  --base_dir=/tmp/minigrid

"""

import functools
import json
import os
import os.path as osp

from absl import app
from absl import flags
import gin
import gym
from gym_minigrid.wrappers import RGBImgObsWrapper
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from aux_tasks.minigrid import estimates
from aux_tasks.minigrid import random_mdp
from aux_tasks.minigrid import rl_basics
from aux_tasks.minigrid import utils
from minigrid_basics.custom_wrappers import coloring_wrapper
from minigrid_basics.custom_wrappers import mdp_wrapper
from minigrid_basics.envs import mon_minigrid



flags.DEFINE_string('base_dir', None,
                    'Base directory to store stats.',
                    required=True)
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')
flags.DEFINE_string(
    'env_name', 'classic_fourrooms',
    'Name of environment(s) to load/create. If None, will '
    'create a set of random MDPs.')
flags.DEFINE_multi_string('experiments', [], 'List of experiments to run.')
flags.DEFINE_integer('num_states', 10, 'Number of states in MDP.')
flags.DEFINE_integer('num_actions', 2, 'Number of actions in MDP.')
flags.DEFINE_bool('plot_singular_vectors', False,
                  'Whether to plot singular vectors.')
flags.DEFINE_bool('plot_grid', False,
                  'Plot environment grid.')
flags.DEFINE_bool('verbose', False, 'Whether to print verbose messages.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_float('lr', 0.1, 'learning rate')
flags.DEFINE_float('alpha', 0.9, 'defined in lissa')
flags.DEFINE_integer('j', 64, 'num of samples for lissa')
flags.DEFINE_integer('num_rows', 64, 'number of rows used for estimators')
flags.DEFINE_integer('d', 1, 'feature dimension')
flags.DEFINE_integer('epochs', 1001, 'num of epochs')
flags.DEFINE_integer('skipsize', 1, 'skipsize for metrics')
flags.DEFINE_integer('skipsize_train', 1000, 'skipsize for training')
flags.DEFINE_boolean('use_l2_reg', False, 'l2 reg.')
flags.DEFINE_boolean('use_penalty', False, 'lambda * Id penalty.')
flags.DEFINE_float('reg_coeff', 0.0, 'defined in lissa')
flags.DEFINE_string('estimator', 'lissa',
                    'lissa or russian_roulette')
flags.DEFINE_string('optimizer', 'sgd',
                    'optax optimizer')
flags.DEFINE_string(
    'custom_base_dir_from_hparams', None,
    'If not None, will set the base_directory prefixed with '
    'the value of this flag, and a subdirectory using game '
    'name and hparam settings. For example, if your game is '
    'Breakout with hparams epsilon=0.01 and horizon=6, '
    'the resulting directory would be:'
    'FLAGS.base_dir/Breakout/0.01_6/')


FLAGS = flags.FLAGS

_ENV_CONFIG_PATH = 'aux_tasks/minigrid/envs'



def train(  # pylint: disable=invalid-name
    Phi,
    Psi,
    num_epochs,
    learning_rate,
    key,
    estimator,
    alpha,
    optimizer,
    use_l2_reg,
    reg_coeff,
    use_penalty,
    j,
    num_rows,
    skipsize=1):
  """Training function."""
  Phis = [Phi]  # pylint: disable=invalid-name
  grads = []
  if optimizer == 'sgd':
    optim = optax.sgd(learning_rate)
  elif optimizer == 'adam':
    optim = optax.adam(learning_rate)
  opt_state = optim.init(Phi)
  for i in tqdm(range(num_epochs)):
    key, subkey = jax.random.split(key)
    Phi, opt_state, grad = estimates.nabla_phi_analytical(
        Phi, Psi, subkey, optim, opt_state, estimator, alpha, use_l2_reg,
        reg_coeff, use_penalty, j, num_rows)
    Phis.append(Phi)
    grads.append(grad)
    if i % skipsize == 0:
      Phis.append(Phi)
      grads.append(grad)
  return jnp.stack(Phis), jnp.stack(grads)


@functools.partial(jax.jit, static_argnums=(2))
def calc_gm_distances(phis, truth, skipsize=100):
  indices = jnp.arange(0, len(phis), skipsize)
  return jax.lax.map(lambda phi: utils.grassman_distance(phi, truth),
                     phis[indices])


@functools.partial(jax.jit, static_argnums=(2))
def calc_dot_products(phis, truth, skipsize=100):
  def _dot(phi):
    return jnp.squeeze(phi.T @ truth /
                       (jnp.linalg.norm(phi) * jnp.linalg.norm(truth)))
  indices = jnp.arange(0, len(phis), skipsize)
  return jax.lax.map(_dot, phis[indices])


@functools.partial(jax.jit, static_argnums=(2))
def calc_frob_norms(phis, Psi, skipsize=100):  # pylint: disable=invalid-name
  indices = jnp.arange(0, len(phis), skipsize)
  return jax.lax.map(lambda phi: utils.outer_objective_mc(phi, Psi),
                     phis[indices])


@functools.partial(jax.jit, static_argnums=(1))
def calc_grad_norms(grads, skipsize=100):
  indices = jnp.arange(0, len(grads), skipsize)
  return jax.lax.map(lambda grad: jnp.sum(jnp.square(grad)), grads[indices])


@functools.partial(jax.jit, static_argnums=(1))
def calc_Phi_norm(phis, skipsize=100):  # pylint: disable=invalid-name
  indices = jnp.arange(0, len(phis), skipsize)
  return jax.lax.map(lambda phi: jnp.sum(jnp.square(phi)), phis[indices])


@functools.partial(jax.jit, static_argnums=(1))
def calc_sranks(phis, skipsize=100):
  def _srank(phi):
    # return jnp.linalg.norm(phi, ord='fro')**2
    #/ jnp.linalg.norm(phi, ord=2)**2
    # return jnp.linalg.norm(phi.T @ phi, ord=2)
    return jnp.linalg.cond(phi)
  indices = jnp.arange(0, len(phis), skipsize)
  return jax.lax.map(_srank, phis[indices])


def main(_):
  flags.mark_flags_as_required(['base_dir'])
  if FLAGS.custom_base_dir_from_hparams is not None:
    FLAGS.base_dir = os.path.join(FLAGS.base_dir,
                                  FLAGS.custom_base_dir_from_hparams)
  else:
    # Add Work unit to base directory path, if it exists.
    if 'xm_wid' in FLAGS and FLAGS.xm_wid > 0:
      FLAGS.base_dir = os.path.join(FLAGS.base_dir, str(FLAGS.xm_wid))
  xm_parameters = (None
                   if 'xm_parameters' not in FLAGS else FLAGS.xm_parameters)
  if xm_parameters:
    xm_params = json.loads(xm_parameters)
    if 'env_name' in xm_params:
      FLAGS.env_name = xm_params['env_name']
  if FLAGS.env_name is None:
    base_dir = os.path.join(
        FLAGS.base_dir, '{}_{}'.format(FLAGS.num_states, FLAGS.num_actions))
  else:
    base_dir = os.path.join(FLAGS.base_dir, FLAGS.env_name)
  base_dir = os.path.join(base_dir, 'PVF', FLAGS.estimator, f'lr_{FLAGS.lr}')
  if not tf.io.gfile.exists(base_dir):
    tf.io.gfile.makedirs(base_dir)
  if FLAGS.env_name is not None:
    gin.add_config_file_search_path(_ENV_CONFIG_PATH)
    gin.parse_config_files_and_bindings(
        config_files=[f'{FLAGS.env_name}.gin'],
        bindings=FLAGS.gin_bindings,
        skip_unknown=False)
    env_id = mon_minigrid.register_environment()
    env = gym.make(env_id)
    env = RGBImgObsWrapper(env)  # Get pixel observations
    # Get tabular observation and drop the 'mission' field:
    env = mdp_wrapper.MDPWrapper(env, get_rgb=False)
    env = coloring_wrapper.ColoringWrapper(env)
  if FLAGS.env_name is None:
    env = random_mdp.RandomMDP(FLAGS.num_states, FLAGS.num_actions)
    # We add the discount factor to the environment.
  env.gamma = FLAGS.gamma
  P = utils.transition_matrix(env, rl_basics.policy_random(env))  # pylint: disable=invalid-name
  S = P.shape[0]  # pylint: disable=invalid-name
  Psi = jnp.linalg.solve(jnp.eye(S) - env.gamma * P, jnp.eye(S))  # pylint: disable=invalid-name
  # Normalize tasks so that they have maximum value 1.
  max_task_value = np.max(Psi, axis=0)
  Psi /= max_task_value  # pylint: disable=invalid-name

  left_vectors, _, _ = jnp.linalg.svd(Psi)  # pylint: disable=invalid-names
  approx_error = utils.approx_error(left_vectors, FLAGS.d, Psi)

  #   Initialization of Phi
  representation_init = jax.random.normal(  # pylint: disable=invalid-names
      jax.random.PRNGKey(0),
      (S, FLAGS.d),  # pylint: disable=invalid-name
      dtype=jnp.float64)
  representations, grads = train(representation_init,
                                 Psi, FLAGS.epochs, FLAGS.lr,
                                 jax.random.PRNGKey(0), FLAGS.estimator,
                                 FLAGS.alpha, FLAGS.optimizer, FLAGS.use_l2_reg,
                                 FLAGS.reg_coeff, FLAGS.use_penalty, FLAGS.j,
                                 FLAGS.num_rows, FLAGS.skipsize_train)

  gm_distances = calc_gm_distances(representations, left_vectors[:, :FLAGS.d],
                                   FLAGS.skipsize)
  x_len = len(gm_distances)
  frob_norms = calc_frob_norms(representations, Psi, FLAGS.skipsize)
  if FLAGS.d == 1:
    dot_products = calc_dot_products(representations, left_vectors[:, :FLAGS.d],
                                     FLAGS.skipsize)
  else:
    dot_products = np.zeros((x_len,))
  grad_norms = calc_grad_norms(grads, FLAGS.skipsize)
  phi_norms = calc_Phi_norm(representations, FLAGS.skipsize)
  phi_ranks = calc_sranks(representations, FLAGS.skipsize)

  prefix = f'alpha{FLAGS.alpha}_j{FLAGS.j}_d{FLAGS.d}_regcoeff{FLAGS.reg_coeff}'

  with tf.io.gfile.GFile(osp.join(base_dir, f'{prefix}.npy'), 'wb') as f:
    np.save(
        f, {
            'gm_distances': gm_distances,
            'dot_products': dot_products,
            'frob_norms': frob_norms,
            'approx_error': approx_error,
            'grad_norms': grad_norms,
            'representations': representations,
            'phi_norms': phi_norms,
            'phi_ranks': phi_ranks
        },
        allow_pickle=True)

if __name__ == '__main__':
  app.run(main)
