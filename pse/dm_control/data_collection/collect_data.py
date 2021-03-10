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

# Lint as: python3
r"""Collect episode and trajectory data from a trained agent.

"""

import os

from absl import app
from absl import flags
from absl import logging
import gin
from tf_agents.utils import example_encoding_dataset

from pse.dm_control import collect_utils as utils
from pse.dm_control import process_data


flags.DEFINE_integer('total_episodes', 500, 'Number of steps in an episode.')
flags.DEFINE_integer('episodes_per_seed', 10,
                     'Number of episode per random seed.')
flags.DEFINE_string('model_dir', None,
                    'Model directory for reading logs/summaries/checkpoints.')

FLAGS = flags.FLAGS
INIT_DATA_SEED = 42  # Answer to life, the universe and everything


@gin.configurable
def collect_and_save_data(env_name,
                          model_dir,
                          trial_suffix,
                          max_episode_len,
                          root_dir,
                          total_episodes,
                          episodes_per_seed):

  saved_model_dir = utils.get_expanded_dir(
      model_dir, env_name, trial_suffix)
  saved_model_dir = os.path.join(saved_model_dir, 'policies/greedy_policy')
  max_train_step = 500 * max_episode_len  # 500k / action_repeat
  policy = utils.load_policy(saved_model_dir, max_train_step)
  trajectory_spec, episode_spec = utils.create_tensor_specs(
      policy.collect_data_spec, max_episode_len)
  episode2_spec = process_data.get_episode_spec(
      trajectory_spec, max_episode_len)

  root_dir = utils.get_expanded_dir(
      root_dir, env_name, trial_suffix, check=False)
  tf_episode_observer = example_encoding_dataset.TFRecordObserver(
      os.path.join(root_dir, 'episodes'), episode_spec, py_mode=True)
  tf_episode2_observer = example_encoding_dataset.TFRecordObserver(
      os.path.join(root_dir, 'episodes2'), episode2_spec, py_mode=True)

  num_seeds = total_episodes // episodes_per_seed
  max_steps = (max_episode_len + 1) * episodes_per_seed
  for seed in range(INIT_DATA_SEED, INIT_DATA_SEED + num_seeds):
    logging.info('Collection and saving for seed %d ..', seed)
    episodes, paired_episodes = utils.collect_pair_episodes(
        policy, env_name, random_seed=seed, max_steps=max_steps,
        max_episodes=episodes_per_seed)
    for episode_tuple in zip(episodes, paired_episodes):
      tf_episode_observer.write(episode_tuple)
      # Write (obs1, obs2, metric) tuples
      processed_episode_tuple = process_data.process_episode(
          episode_tuple[0], episode_tuple[1], gamma=process_data.GAMMA)
      tf_episode2_observer.write(processed_episode_tuple)
  tf_episode_observer.close()


def main(_):
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)

  if FLAGS.seed is not None:
    trial_suffix = f'{FLAGS.trial_id}/seed_{FLAGS.seed}'
  else:
    trial_suffix = str(FLAGS.trial_id)

  collect_and_save_data(
      env_name=FLAGS.env_name,
      model_dir=FLAGS.model_dir,
      trial_suffix=trial_suffix,
      max_episode_len=FLAGS.max_episode_len,
      root_dir=FLAGS.root_dir,
      total_episodes=FLAGS.total_episodes,
      episodes_per_seed=FLAGS.episodes_per_seed)


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('root_dir')
  app.run(main)
