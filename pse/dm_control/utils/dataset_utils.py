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

"""Dataset utilities."""

import os

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.utils import example_encoding_dataset

from pse.dm_control.utils import helper_utils


def image_aug(traj, img_pad=4):
  """Padding and cropping."""
  paddings = tf.constant([[img_pad, img_pad], [img_pad, img_pad], [0, 0]])
  obs = traj.observation['pixels']
  traj.observation['pixels'] = get_random_crop(obs, paddings)
  return traj


def get_random_crop(obs, paddings, seed=None):
  cropped_shape = obs.shape
  # The reference uses ReplicationPad2d in pytorch, but it is not available
  # in tf. Use 'SYMMETRIC' instead.
  padded_obs = tf.pad(obs, paddings, 'SYMMETRIC')
  return tf.image.random_crop(padded_obs, cropped_shape, seed)


def batch_image_aug(obs, img_pad=4, seed=None):
  """Padding and cropping."""
  paddings = tf.constant(
      [[0, 0], [img_pad, img_pad], [img_pad, img_pad], [0, 0]])
  return get_random_crop(obs, paddings, seed=seed)


def transform_episodes(x, y, z, img_pad):
  seed = np.random.randint(0, 1e15)
  return (batch_image_aug(x, img_pad=img_pad, seed=seed),
          batch_image_aug(y, img_pad=img_pad, seed=seed), z)


def load_trajs(dataset_path, batch_size, img_pad):
  traj_dataset = example_encoding_dataset.load_tfrecord_dataset(
      [dataset_path],
      buffer_size=int(1e6),
      as_experience=False,
      as_trajectories=True,
      add_batch_dim=False)
  logging.info('Traj dataset loaded from %s', dataset_path)

  traj_dataset = traj_dataset.shuffle(10000).repeat()

  image_aug_fn = lambda img: image_aug(img, img_pad)
  traj_dataset = traj_dataset.map(
      image_aug_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return traj_dataset.batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)


def process_episode(episode1, episode2, gamma):
  obs1 = tf.stack([y.observation['pixels'] for y in episode1], axis=0)
  obs2 = tf.stack([y.observation['pixels'] for y in episode2], axis=0)
  metric = helper_utils.compute_metric(episode1, episode2, gamma)
  return (obs1, obs2, metric)


def load_episodes(dataset_path, img_pad, gamma=0.99):
  """Load episode data from a fixed dataset."""
  episode_dataset = example_encoding_dataset.load_tfrecord_dataset(
      [dataset_path],
      buffer_size=None,
      as_experience=False,
      add_batch_dim=False)
  logging.info('Episode dataset loaded from %s', dataset_path)

  if dataset_path.endswith('episodes'):
    process_episode_fn = lambda x, y: process_episode(x, y, gamma)
    episode_dataset = episode_dataset.map(
        process_episode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  episode_dataset = episode_dataset.shuffle(50).repeat()
  transform_episodes_fn = lambda x, y, z: transform_episodes(x, y, z, img_pad)
  episode_dataset = episode_dataset.map(transform_episodes_fn)
  episode_dataset = episode_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  logging.info('Episode dataset processed.')
  return episode_dataset


def load_dataset(data_dir, batch_size, img_pad=4):
  logging.info('Loading training dataset from %s', data_dir)
  trajs = load_trajs(os.path.join(data_dir, 'traj'), batch_size, img_pad)
  episodes = load_episodes(os.path.join(data_dir, 'episodes2'), img_pad)
  return tf.data.Dataset.zip(
      (trajs, episodes)).prefetch(tf.data.experimental.AUTOTUNE)
