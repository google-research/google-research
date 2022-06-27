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

r"""coherence computation.
"""
import json
import os
import os.path as osp
import pathlib
import pickle

from absl import app
from absl import flags
from absl import logging
from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from dopamine.discrete_domains import atari_lib
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent
from dopamine.jax.agents.rainbow import rainbow_agent
from flax import core
from flax.training import checkpoints as flax_checkpoints
import gin
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v1 as tf

from generalization_representations_rl_aistats22.coherence import networks


gfile = tf.compat.v1.gfile
AGENTS = [
    'jax_dqn', 'jax_rainbow', 'jax_implicit_quantile', 'mimplicit_quantile'
]
flags.DEFINE_enum('agent_name', None, AGENTS, 'Name of the agent.')
flags.DEFINE_string('checkpoint_dir', None, 'Checkpoint path to use')
flags.DEFINE_string('username', None, 'Username')

flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')

flags.DEFINE_string('game', 'Seaquest', 'Name of game')
flags.DEFINE_integer('checkpoint_every', 1,
                     'Compute ranks for `checkpoint_every` checkpoints apart.')
flags.DEFINE_integer('replay_capacity', 50000,
                     'Capacity of replay buffer to load.')
flags.DEFINE_integer('batch_size', 2048, 'Batch Size for feature matrix.')
flags.DEFINE_integer('num_buffers', 20, 'Number of buffers.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_boolean('debug_mode', False,
                     'Debug mode for fast runs for testing.')
flags.DEFINE_integer('residual_td', 0, 'Residual TD or not.')

FLAGS = flags.FLAGS

NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = atari_lib.NATURE_DQN_DTYPE
NATURE_DQN_STACK_SIZE = atari_lib.NATURE_DQN_STACK_SIZE


class PretrainedDQN(dqn_agent.JaxDQNAgent):

  def _build_replay_buffer(self):
    pass  # The Teacher doesn't use a replay buffer.


class PretrainedRainbow(rainbow_agent.JaxRainbowAgent):

  def _build_replay_buffer(self):
    pass  # The Teacher doesn't use a replay buffer.


class PretrainedIQN(implicit_quantile_agent.JaxImplicitQuantileAgent):

  def _build_replay_buffer(self):
    pass  # The Teacher doesn't use a replay buffer.


def reload_checkpoint(agent, checkpoint_path):
  """Reload variables from a fully specified checkpoint."""
  assert checkpoint_path is not None
  with tf.io.gfile.GFile(checkpoint_path, 'rb') as fin:
    bundle_dictionary = pickle.load(fin)
  reload_jax_checkpoint(agent, bundle_dictionary)


def reload_jax_checkpoint(agent, bundle_dictionary):
  """Reload variables from a fully specified checkpoint."""
  if bundle_dictionary is not None:
    agent.state = bundle_dictionary['state']
    if isinstance(bundle_dictionary['online_params'], core.FrozenDict):
      agent.online_params = bundle_dictionary['online_params']
    else:  # Load pre-linen checkpoint.
      agent.online_params = core.FrozenDict({
          'params': flax_checkpoints.convert_pre_linen(
              bundle_dictionary['online_params']).unfreeze()
      })
    # We recreate the optimizer with the new online weights.
    # pylint: disable=protected-access
    agent.optimizer = dqn_agent.create_optimizer(agent._optimizer_name)
    # pylint: enable=protected-access
    if 'optimizer_state' in bundle_dictionary:
      agent.optimizer_state = bundle_dictionary['optimizer_state']
    else:
      agent.optimizer_state = agent.optimizer.init(agent.online_params)
    logging.info('Done restoring!')


def get_checkpoints(ckpt_dir, max_checkpoints=200):
  """Get the full path of checkpoints in `ckpt_dir`."""
  return [
      os.path.join(ckpt_dir, f'ckpt.{idx}') for idx in range(max_checkpoints)
  ]


def get_features(agent, states):
  def feature_fn(state):
    return agent.network_def.apply(agent.online_params, state)
  compute_features = jax.vmap(feature_fn)
  features = []
  for state in states:
    features.append(jnp.squeeze(compute_features(state)))
  return np.concatenate(features, axis=0)


def compute_singular_values(matrix):
  """Compute srank(matrix) and other values."""
  ret_dict = dict()
  singular_vals = np.linalg.svd(
      matrix, full_matrices=False, compute_uv=False)
  nuclear_norm = np.sum(singular_vals)
  condition_number = singular_vals[0] / (singular_vals[-1] + 1e-8)
  ret_dict['singular_vals'] = singular_vals
  ret_dict['nuclear_norm'] = nuclear_norm
  ret_dict['condition_number'] = condition_number
  return ret_dict


def compute_eigenvalues(matrix):
  """Compute eig(matrix) and other values."""
  ret_dict = dict()
  eig_vals, _ = np.linalg.eig(matrix)
  ret_dict['eig_vals'] = eig_vals
  return ret_dict


def calculate_coherence(matrix, thresh=1e-5):
  num_rows, _ = matrix.shape
  u, s, _ = np.linalg.svd(matrix, full_matrices=False, compute_uv=True)
  rank = max(np.sum(s >= thresh), 1)
  u1 = u[:, :rank]
  projected_basis = np.matmul(u1, np.transpose(u1))
  norms = np.linalg.norm(projected_basis, axis=0, ord=2) ** 2
  eff_dim = num_rows * np.max(norms)
  coherence = eff_dim/rank
  return coherence, norms


def create_agent(environment, summary_writer=None):
  """Creates an online agent.

  Args:
    environment: An Atari 2600 environment.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    A DQN agent with metrics.
  """
  if FLAGS.agent_name == 'jax_dqn':
    agent = PretrainedDQN
    network = networks.NatureDQNNetworkWithFeatures
  elif FLAGS.agent_name == 'jax_rainbow':
    agent = PretrainedRainbow
    network = networks.RainbowNetworkWithFeatures
  elif FLAGS.agent_name == 'jax_implicit_quantile':
    agent = PretrainedIQN
    network = networks.ImplicitQuantileNetworkWithFeatures
  elif FLAGS.agent_name == 'mimplicit_quantile':
    agent = PretrainedIQN
    network = networks.ImplicitQuantileNetworkWithFeatures
  else:
    raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))

  return agent(
      num_actions=environment.action_space.n,
      summary_writer=summary_writer,
      network=network)


def create_game_replay_dir(game_name, run_number):
  return osp.join(ATARI_DATA_DIR, game_name, f'{run_number}/replay_logs')




def main(unused_argv):
  _ = unused_argv
  tf.disable_eager_execution()
  logging.set_verbosity(logging.INFO)
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings
  gin.parse_config_files_and_bindings(
      gin_files, bindings=gin_bindings, skip_unknown=False)

  paths = list(pathlib.Path(FLAGS.checkpoint_dir).parts)
  run_number = paths[-1].split('_')[-1]
  save_dir = osp.join(
      pathlib.Path(*paths), 'coherence', f'batch_size_{FLAGS.batch_size}')
  ckpt_dir = osp.join(FLAGS.checkpoint_dir, 'checkpoints')
  if gfile.Exists(save_dir):
    gfile.DeleteRecursively(save_dir)
  gfile.MakeDirs(save_dir)
  logging.info('Checkpoint directory: %s', ckpt_dir)
  logging.info('Save coherence computation in directory: %s', save_dir)

  logging.info('Game: %s', FLAGS.game)
  environment = atari_lib.create_atari_environment(
      game_name=FLAGS.game, sticky_actions=True)

  agent = create_agent(environment)

  checkpoints = get_checkpoints(ckpt_dir)

  replay_dir = create_game_replay_dir(FLAGS.game, run_number)
  logging.info('Replay dir: %s', replay_dir)

  replay_batch_size = 256
  num_batches = max(FLAGS.batch_size // replay_batch_size, 1)
  if FLAGS.debug_mode:
    states = [np.random.rand(replay_batch_size, 84, 84, 4)] * num_batches
    next_states = [np.random.rand(replay_batch_size, 84, 84, 4)] * num_batches
  else:
    data_replay = fixed_replay_buffer.FixedReplayBuffer(
        data_dir=replay_dir,
        replay_suffix=None,  # To load a specific buffer among the 50 buffers
        observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
        stack_size=NATURE_DQN_STACK_SIZE,
        update_horizon=1,
        replay_capacity=FLAGS.replay_capacity,
        batch_size=replay_batch_size,
        gamma=FLAGS.gamma,
        observation_dtype=NATURE_DQN_DTYPE.as_numpy_dtype)
    data_replay.reload_buffer(FLAGS.num_buffers)

    states = []
    next_states = []
    for _ in range(num_batches):
      transitions = data_replay.sample_transition_batch()
      states.append(transitions[0])
      next_states.append(transitions[3])

  checkpoint_every = FLAGS.checkpoint_every
  max_checkpoints = int(len(checkpoints) // checkpoint_every)

  coherences = []
  all_norms = []
  feature_matrices = []
  for mdx in range(max_checkpoints+1):
    checkpoint_num = mdx * checkpoint_every - 1
    logging.info('Checkpoint %d', checkpoint_num)
    # Checkpoint -1 corresponds to a random agent.
    if checkpoint_num >= 0:
      reload_checkpoint(agent, checkpoints[checkpoint_num])

    if FLAGS.residual_td:
      feature_matrix = get_features(agent, states)
      feature_matrix -= FLAGS.gamma * get_features(agent, next_states)
    else:
      feature_matrix = get_features(agent, states)
    if mdx <= 5:
      feature_matrices.append(feature_matrix)
    try:
      coherence, norms = calculate_coherence(feature_matrix)
      logging.info('Coherence: %0.2f', coherence)
      all_norms.append(norms)
      coherences.append(coherence)
    except Exception as e:  # pylint:disable=broad-except
      logging.info('Exception %s for checkpoint %d', e, checkpoint_num)
      continue

  prefix = 'residual_' if FLAGS.residual_td else ''

  logging.info('Number of checkpoints: %d', len(checkpoints))
  with gfile.Open(osp.join(save_dir, f'{prefix}coherence.npy'), 'wb') as f:
    np.save(f, coherences, allow_pickle=True)

  with gfile.Open(osp.join(save_dir, f'{prefix}norms.npy'), 'wb') as f:
    np.save(f, all_norms, allow_pickle=True)

  with gfile.Open(osp.join(save_dir, f'{prefix}features.npy'), 'wb') as f:
    np.save(f, feature_matrices, allow_pickle=True)

if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  app.run(main)
