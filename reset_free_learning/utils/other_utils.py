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

"""Utilities used in main scripts."""

import numpy as np
import tensorflow as tf

from tf_agents.utils import nest_utils


def soft_std_clip_transform(log_std):
  scale_min, scale_max = (-20, 10)
  log_std = tf.keras.activations.tanh(log_std)
  log_std = scale_min + 0.5 * (scale_max - scale_min) * (log_std + 1)
  return tf.exp(log_std)


def std_clip_transform(stddevs):
  stddevs = tf.nest.map_structure(lambda t: tf.clip_by_value(t, -20, 10),
                                  stddevs)
  return tf.exp(stddevs)


def np_custom_save(fname, content):
  with tf.io.gfile.GFile(fname, 'w') as f:
    np.save(f, content, allow_pickle=False)


def np_custom_load(fname):
  with tf.io.gfile.GFile(fname, 'rb') as f:
    load_file = np.load(f)
  return load_file


# records video of one episode
def record_video(env_loader, video_filepath, policy, max_episode_length=None):
  print('unable to record video')


def copy_replay_buffer(small_buffer, big_buffer):
  """Copy small buffer into the big buffer."""
  all_data = nest_utils.unbatch_nested_tensors(small_buffer.gather_all())
  for trajectory in nest_utils.unstack_nested_tensors(all_data,
                                                      big_buffer.data_spec):
    big_buffer.add_batch(nest_utils.batch_nested_tensors(trajectory))
