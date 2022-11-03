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

python -m aux_tasks.minigrid.train \
  --env_name=classic_fourrooms \
  --base_dir=/tmp/minigrid

"""
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger
import gin
import gym
from gym_minigrid.wrappers import RGBImgObsWrapper
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

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
flags.DEFINE_float('lr', 0.01, 'learning rate')
flags.DEFINE_float('log_every', 1000000, 'logging frequence.')
flags.DEFINE_float('summary_writer_frequency', 1000000, 'logging frequence.')
flags.DEFINE_float('alpha', 0.1, 'defined in lissa')
flags.DEFINE_integer('j', 100, 'num of samples for lissa')
flags.DEFINE_integer('d', 1, 'feature dimension')
flags.DEFINE_integer('num_rows', 10, 'number of rows used for estimators')
flags.DEFINE_integer('epochs', 1000, 'num of epochs')
flags.DEFINE_boolean('use_l2_reg', True, 'l2 reg.')
flags.DEFINE_boolean('use_penalty', False, 'lambda * Id penalty.')
flags.DEFINE_float('reg_coeff', 0.01, 'defined in lissa')
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



class TrainRunner():
  """Object that handles running experiments."""

  def __init__(self,
               base_dir,
               env,
               training_steps,
               learning_rate,
               estimator,
               alpha,
               optimizer,
               use_l2_reg,
               reg_coeff,
               use_penalty,
               j,
               num_rows,
               key,
               log_every_n = 1,
               summary_writer_frequency = 1,
               checkpoint_file_prefix = 'ckpt',
               logging_file_prefix = 'log'):
    self._base_dir = base_dir
    self._summary_writer = tf.summary.create_file_writer(self._base_dir)
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._summary_writer_frequency = summary_writer_frequency
    self._training_steps = training_steps
    self._env = env
    self._learning_rate = learning_rate
    self._estimator = estimator
    self._alpha = alpha
    self._optimizer = optimizer
    self._use_l2_reg = use_l2_reg
    self._reg_coeff = reg_coeff
    self._use_penalty = use_penalty,
    self._num_rows = num_rows
    self._j = j
    self._key = key
    self._create_directories()
    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

  def _create_directories(self):
    """Create necessary sub-directories."""
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        if experiment_data is not None:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
        logging.info('Reloaded checkpoint and will start from iteration %d',
                     self._start_iteration)

  def _compute_pvf(self):
    """Initialize targets ie the PVF.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    P = utils.transition_matrix(self._env, rl_basics.policy_random(self._env))  # pylint: disable=invalid-name
    S = P.shape[0]  # pylint: disable=invalid-name
    Psi = jnp.linalg.solve(jnp.eye(S) - self._env.gamma * P, jnp.eye(S))  # pylint: disable=invalid-name
    left_vectors, _, _ = jnp.linalg.svd(Psi)  # pylint: disable=invalid-names
    #   Initialization of Phi
    representation_init = jax.random.normal(  # pylint: disable=invalid-names
        jax.random.PRNGKey(0),
        (S, FLAGS.d),  # pylint: disable=invalid-name
        dtype=jnp.float64)
    return Psi, left_vectors, representation_init

  def _train_one_step(self, epoch, Phi, Psi, left_vec, key, optim, opt_state):  # pylint: disable=invalid-name
    """Training function."""
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting epoch %d', epoch)
    start_time = time.time()
    Phi, opt_state, grads = estimates.nabla_phi_analytical(
        Phi,
        Psi,
        key,
        optim,  # pylint: disable=invalid-name
        opt_state,
        self._estimator,
        self._alpha,
        self._use_l2_reg,
        self._reg_coeff,
        self._use_penalty,
        self._j,
        self._num_rows)
    time_delta = time.time() - start_time
    statistics.append({'Time/epoch': time_delta})
    statistics.append({'representation': Phi})
    gm_distances = utils.grassman_distance(Phi, left_vec[:, :FLAGS.d])
    statistics.append({'GM_distances': gm_distances})
    frob_norms = utils.outer_objective_mc(Phi, Psi)
    statistics.append({'Frob_norms': frob_norms})
    phi_norms = jnp.sum(jnp.square(Phi))
    statistics.append({'phi_norms': phi_norms})
    grad_norms = jnp.sum(jnp.square(grads))
    phi_ranks = jnp.linalg.matrix_rank(Phi)
    statistics.append({'phi_ranks': phi_ranks})
    statistics.append({'grad_norms': grad_norms})
    if FLAGS.d == 1:
      dot_products = (Phi.T @ left_vec[:, :FLAGS.d] /
                      (jnp.linalg.norm(Phi) *
                       jnp.linalg.norm(left_vec[:, :FLAGS.d])))[0][0]
      statistics.append({'Dot_products': dot_products})
    else:
      dot_products = 0.

    # if epoch % self._summary_writer_frequency == 0:
    self._save_tensorboard_summaries(epoch, frob_norms, gm_distances,
                                     dot_products, phi_norms, grad_norms,
                                     phi_ranks)
    return statistics.data_lists, Phi, opt_state

  def _save_tensorboard_summaries(self, epoch, frob_norms, gm_distances,
                                  dot_products, phi_norms, grad_norms,
                                  phi_ranks):
    """Save statistics as tensorboard summaries."""
    with self._summary_writer.as_default():
      tf.summary.scalar('Losses/FrobNorms', frob_norms, step=epoch)
      tf.summary.scalar('Losses/DotProducts', dot_products, step=epoch)
      tf.summary.scalar('Losses/GM_distances', gm_distances, step=epoch)
      tf.summary.scalar('Losses/FeatureNorms', phi_norms, step=epoch)
      tf.summary.scalar('Losses/GradNorms', grad_norms, step=epoch)
      tf.summary.scalar('Losses/PhiRanks', phi_ranks, step=epoch)

  def _log_experiment(self, iteration, statistics):
    """Records the results of the current iteration.

    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration)

  def train(self):
    """Runs a full experiment."""
    logging.info('Beginning training...')
    Psi, left_vectors, Phi = self._compute_pvf()  # pylint: disable=invalid-name
    if self._optimizer == 'sgd':
      optim = optax.sgd(self._learning_rate)
    if self._optimizer == 'adam':
      optim = optax.adam(self._learning_rate)
    opt_state = optim.init(Phi)
    for epoch in range(self._training_steps):
      self._key, subkey = jax.random.split(self._key)
      statistics, Phi, opt_state = self._train_one_step(  # pylint: disable=invalid-name
          epoch,
          Phi,
          Psi,
          left_vectors,  # pylint: disable=invalid-name
          subkey,
          optim,
          opt_state)
      self._log_experiment(epoch, statistics)
    self._summary_writer.flush()


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
  base_dir = os.path.join(base_dir, 'PVF', FLAGS.estimator)
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

  logging.set_verbosity(logging.INFO)
  gin_files = []
  gin_bindings = FLAGS.gin_bindings


  runner = TrainRunner(base_dir, env, FLAGS.epochs, FLAGS.lr, FLAGS.estimator,
                       FLAGS.alpha, FLAGS.optimizer, FLAGS.use_l2_reg,
                       FLAGS.reg_coeff, FLAGS.use_penalty, FLAGS.j,
                       FLAGS.num_rows, jax.random.PRNGKey(0), FLAGS.epochs - 1,
                       FLAGS.epochs - 1)
  runner.train()


if __name__ == '__main__':
  app.run(main)
