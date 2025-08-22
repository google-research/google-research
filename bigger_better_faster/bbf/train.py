# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

r"""Entry point for Atari 100k experiments.

To train a BBF agent locally, run:

python -m bbf.train \
    --agent=BBF \
    --gin_files=bbf/configs/BBF.gin \
    --base_dir=/tmp/online_rl/bbf \
    --run_number=1

"""
import functools
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import train as base_train
from etils import epath
import gin
import jax.profiler
import numpy as np
import tensorflow.compat.v2 as tf

from bigger_better_faster.bbf import eval_run_experiment
from bigger_better_faster.bbf.agents import spr_agent

FLAGS = flags.FLAGS
CONFIGS_DIR = './configs'
AGENTS = [
    'rainbow',
    'der',
    'dopamine_der',
    'DrQ',
    'OTRainbow',
    'SPR',
    'SR-SPR',
    'BBF',
]

# flags are defined when importing run_xm_preprocessing
flags.DEFINE_enum('agent', 'SPR', AGENTS, 'Name of the agent.')
flags.DEFINE_integer('run_number', 1, 'Run number.')
flags.DEFINE_integer('agent_seed', None, 'If None, use the run_number.')
flags.DEFINE_boolean('no_seeding', True, 'If True, choose a seed at random.')
flags.DEFINE_string(
    'load_replay_dir', None, 'Directory to load the initial replay buffer from '
    'a fixed dataset. If None, no transitions are loaded. ')
flags.DEFINE_string('tag', None, 'Tag for this run')
flags.DEFINE_boolean(
    'save_replay', False, 'Whether to save agent\'s final replay buffer '
    'as a fixed dataset to ${base_dir}/replay_logs.')
flags.DEFINE_integer(
    'load_replay_number', None, 'Index of the replay run to load the initial '
    'replay buffer from a fixed dataset. If None, uses the `run_number`.')
flags.DEFINE_boolean('data_logging', False,
                     'Whether to use agent to log the replay buffer or not.')
flags.DEFINE_boolean('max_episode_eval', True,
                     'Whether to use `MaxEpisodeEvalRunner` or not.')


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in the
      config files.
  """
  gin.parse_config_files_and_bindings(
      gin_files, bindings=gin_bindings, skip_unknown=False)


def to_str(dictionary):
  new_dict = {}
  for k, v in dictionary.items():
    if isinstance(v, dict):
      new_dict[k] = to_str(v)
    else:
      new_dict[k] = str(v)
  return new_dict


def write_config(base_dir, seed, tag, agent):
  """Writes the configuration of the current training run to disk.

  Args:
    base_dir: Base directory of the training run.
    seed: Seed assigned to this run.
    tag: Tag assigned to this run.
    agent: Agent name for this run.

  Returns:
    The clean config that was written to disk.
  """
  clean_cfg = {k[1]: v for k, v in gin.config._CONFIG.items()}  # pylint: disable=protected-access
  clean_cfg['seed'] = seed
  clean_cfg['tag'] = tag
  clean_cfg['agent'] = agent

  for _ in range(10):
    config_dir = epath.Path(base_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / 'config.json'
    try:
      config_path.write_text(json.dumps(to_str(clean_cfg)))
      break
    except OSError as e:
      print(e)

  return clean_cfg


def create_load_replay_dir(xm_params):
  """Creates the directory for loading fixed replay data."""
  problem_name, run_number = '', ''
  for param, value in xm_params.items():
    if param.endswith('game_name'):
      problem_name = value
    elif param.endswith('run_number'):
      run_number = str(value)
  replay_dir = FLAGS.load_replay_dir
  if replay_dir:
    if FLAGS.load_replay_number:
      replay_number = str(FLAGS.load_replay_number)
    else:
      replay_number = run_number
    replay_dir = os.path.join(replay_dir, problem_name, replay_number,
                              'replay_logs')
  return replay_dir


def create_agent(
    sess,  # pylint: disable=unused-argument
    environment,
    seed,
    data_logging=False,
    summary_writer=None,
):
  """Helper function for creating agent."""
  if data_logging:
    raise NotImplementedError()
  return spr_agent.BBFAgent(
      num_actions=environment.action_space.n,
      seed=seed,
      summary_writer=summary_writer,
  )


def set_random_seed(seed):
  """Set random seed for reproducibility."""
  logging.info('Setting random seed: %d', seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  tf.random.set_seed(seed)
  np.random.seed(seed)


def main(unused_argv):
  """Main method.

    Args:
        unused_argv: Arguments (unused).
  """
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()

  if hasattr(base_train, 'run_xm_preprocessing'):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
      except RuntimeError as exception:
        # Memory growth must be set before GPUs have been initialized
        print(exception)
  else:
    try:
      tf.config.experimental.set_visible_devices([], 'GPU')
    except tf.errors.NotFoundError as tferror:
      print(tferror)

    base_dir = FLAGS.base_dir
    gin_files = FLAGS.gin_files
    gin_bindings = FLAGS.gin_bindings
    print('Got gin bindings:')
    print(gin_bindings)
    gin_bindings = [b.replace("'", '') for b in gin_bindings]
    print('Sanitized gin bindings to:')
    print(gin_bindings)

  # Add code for setting random seed using the run_number
  if FLAGS.no_seeding:
    seed = int(time.time() * 10000000) % 2**31
  else:
    seed = FLAGS.run_number if not FLAGS.agent_seed else FLAGS.agent_seed
  set_random_seed(seed)
  run_experiment.load_gin_configs(gin_files, gin_bindings)

  write_config(base_dir, seed, FLAGS.tag, FLAGS.agent)

  # Set the Jax agent seed
  create_agent_fn = functools.partial(
      create_agent, seed=seed, data_logging=FLAGS.data_logging)
  if FLAGS.max_episode_eval:
    kwargs = dict(
        load_replay_dir=FLAGS.load_replay_dir, save_replay=FLAGS.save_replay)
    runner_fn = eval_run_experiment.DataEfficientAtariRunner
    logging.info('Using MaxEpisodeEvalRunner for evaluation.')
    kwargs = {}  # No additional flags should be passed.
    runner = runner_fn(base_dir, create_agent_fn, **kwargs)
  else:
    runner = run_experiment.Runner(base_dir, create_agent_fn)

  jax.profiler.start_server(9999)
  print(f'Found devices {jax.local_devices()}')

  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
