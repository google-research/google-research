# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Main evaluation loop."""

# pylint: disable=unused-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import pprint
import re

from absl import app
from absl import flags
from absl import logging
import gym
import numpy as np
from policies.exponential_family import ExponentialFamilyPolicy
from policies.l_norm import LNormPolicy
import replay
import tensorflow.compat.v2 as tf
from tensorflow.python.ops import summary_ops_v2 as summary
import util
import yaml


FLAGS = flags.FLAGS
flags.DEFINE_string('experiment_path', None,
                    'Where to load experiment info from.')
flags.DEFINE_integer('num_episodes', -1, 'How many episodes to sample.')


def sample_episode(env, policy, memory, max_episode_length=1000):
  """Collect episodes from policy."""
  obs = env.reset()
  memory.log_init(obs)

  for step in range(max_episode_length):
    act = policy.argmax(np.expand_dims(obs, 0)).numpy()[0]
    next_obs, reward, term, _ = env.step(act)
    memory.log_experience(obs, act, reward, next_obs)
    if term:
      logging.info('Episode terminated early, step=%d', step)
      break
    obs = next_obs

  return memory


def most_recent_file(directory, regex_string):
  """Returns the path of the most recently modified file matching the regex."""
  file_times = [
      (f, os.path.getmtime(os.path.join(directory, f)))
      for f in os.listdir(directory)
      if re.search(regex_string, f)]
  if not file_times:
    return
  most_recent, _ = max(file_times, key=lambda x: x[1])
  return os.path.join(directory, most_recent)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  config_file = most_recent_file(FLAGS.experiment_path, r'config.yaml')
  assert config_file
  with open(config_file, 'r') as f:
    config = util.AttrDict(**yaml.load(f.read()))
  logging.info('Config:\n%s', pprint.pformat(config))
  env = gym.make(config.env)
  cls = globals()[config.policy]
  policy = cls(config)
  # Initialize policy
  policy.argmax(np.expand_dims(env.reset(), 0))

  # Load checkpoint.
  # Assuming policy is a keras.Model instance.
  logging.info('policy variables: %s',
               [v.name for v in policy.trainable_variables])
  ckpt = tf.train.Checkpoint(policy=policy)
  ckpt_file = most_recent_file(FLAGS.experiment_path, r'model.ckpt-[0-9]+')
  if ckpt_file:
    ckpt_file = re.findall('^(.*/model.ckpt-[0-9]+)', ckpt_file)[0]
    logging.info('Checkpoint file: %s', ckpt_file)
    ckpt.restore(ckpt_file).assert_consumed()
  else:
    raise RuntimeError('No checkpoint found')

  summary_writer = tf.summary.create_file_writer(FLAGS.experiment_path,
                                                 flush_millis=10000)

  logging.info('Starting Evaluation')
  it = (
      range(FLAGS.num_episodes) if FLAGS.num_episodes >= 0
      else itertools.count())
  for ep in it:
    memory = replay.Memory()
    sample_episode(env, policy, memory, max_episode_length=200)
    logging.info(ep)
    with summary_writer.as_default(), summary.always_record_summaries():
      summary.scalar('return', memory.observed_rewards().sum(), step=ep)
      summary.scalar('length', memory.observed_rewards().shape[-1], step=ep)

  logging.info('DONE')


if __name__ == '__main__':
  app.run(main)
