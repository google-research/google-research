# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Constellation dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow.compat.v1 as tf

ConstellationTuple = collections.namedtuple('ConstellationTuple',
                                            'presence corners pattern_presence '
                                            'pattern_id')


def create_numpy(
    size_n=1,
    shuffle_corners=True,
    gaussian_noise=0.,
    max_translation=1.,
    rotation_percent=0.0,
    max_upscale=0.0,
    which_patterns='basic',
    drop_prob=0.0,
):
  """Creates a batch of data using numpy."""

  sm = 1
  if which_patterns == 'basic':
    which_patterns = [[0], [1]]

  elif which_patterns == 'all':
    which_patterns = [[0], [1], [2], [3]]

  elif isinstance(which_patterns, str):
    raise ValueError('Pattern "{}" has not been '
                     'implemented.'.format(which_patterns))

  pattern = [
      np.array([[1 + 2 * sm, 1 + 2 * sm, 1], [1, 1, 1], [1 + 2 * sm, 1, 1],
                [1, 1 + 2 * sm, 1]]),  # square
      np.array([[1, 1 + sm, 1], [1 + 2 * sm, 1, 1], [1 + 2 * sm, 1 + 2 * sm,
                                                     1]]),  # triangle
      np.array([[1, 1, 1], [1 + sm, 1 - 2 * sm, 1], [1 + 2 * sm, 1 - sm, 1],
                [1 + 2 * sm, 1 + sm, 1], [1 + sm, 1 + 2 * sm, 1]]),  # pentagon
      np.array([[1, 1, 1], [1 + sm, 1, 1], [1 + 2 * sm, 1, 1],
                [1 + 2 * sm, 1 + sm, 1], [1 + 2 * sm, 1 + 2 * sm, 1]]),  # L
  ]

  caps_dim = pattern[0].shape[1]
  transformations = []
  all_corners = []
  all_corner_presence = []
  all_pattern_presence = []

  centers = np.array([0, 0, 1])
  for i in range(len(which_patterns)):
    corners = centers.copy()

    for j in range(len(which_patterns[i])):
      corner_trans = np.zeros((pattern[which_patterns[i][j]].shape[0], caps_dim,
                               caps_dim))

      corner_trans[:, -1, :] = pattern[which_patterns[i][j]] * (j + 1)
      corner_trans[:, :-1, :-1] = np.eye(caps_dim - 1)
      corners = np.matmul(corners, corner_trans)
      corners = corners.reshape(-1, caps_dim)

    transformation = np.zeros((size_n, caps_dim, caps_dim))
    transformation[:, :, -1] = [0, 0, 1]

    # [pi/2, pi]
    degree = (np.random.random((size_n)) - .5) * 2. * np.pi * rotation_percent
    scale = 1. + np.random.random((size_n)) * max_upscale
    translation = np.random.random((size_n, 2)) * 24. * max_translation
    transformation[:, 0, 0] = np.cos(degree) * scale
    transformation[:, 1, 1] = np.cos(degree) * scale
    transformation[:, 0, 1] = np.sin(degree) * scale
    transformation[:, 1, 0] = -np.sin(degree) * scale
    transformation[:, -1, :-1] = translation / scale[Ellipsis, np.newaxis]

    corners = np.matmul(corners, transformation)

    random_pattern_choice = np.random.binomial(1, 1. - drop_prob,
                                               (corners.shape[0], 1))

    random_corer_choice = np.tile(random_pattern_choice, (1, corners.shape[1]))

    all_corner_presence.append(random_corer_choice)
    all_pattern_presence.append(random_pattern_choice)
    transformations.append(transformation)
    all_corners.append(corners)

  capsules = np.concatenate(all_corners, axis=1)[Ellipsis, :2]
  all_corner_presence = np.concatenate(all_corner_presence, axis=1)
  all_pattern_presence = np.concatenate(all_pattern_presence, axis=1)

  pattern_ids = []
  current_pattern_id = 0
  for i in range(len(which_patterns)):
    for j in which_patterns[i]:
      corner_ids = [current_pattern_id] * len(pattern[j])
      pattern_ids.extend(corner_ids)
      current_pattern_id += 1

  pattern_ids = np.asarray(pattern_ids)[np.newaxis]
  pattern_ids = np.tile(pattern_ids, [capsules.shape[0], 1])

  capsules, all_corner_presence, all_pattern_presence = [
      i.astype(np.float32)
      for i in (capsules, all_corner_presence, all_pattern_presence)
  ]

  if shuffle_corners:
    for i in range(capsules.shape[0]):
      p = np.random.permutation(len(capsules[i]))
      capsules[i] = capsules[i][p]
      all_corner_presence[i] = all_corner_presence[i][p]
      pattern_ids[i] = pattern_ids[i][p]

  if gaussian_noise > 0.:
    capsules += np.random.normal(scale=gaussian_noise, size=capsules.shape)

  # normalize corners
  min_d, max_d = capsules.min(), capsules.max()
  capsules = (capsules - min_d) / (max_d - min_d + 1e-8) * 2 - 1.

  minibatch = dict(corners=capsules, presence=all_corner_presence,
                   pattern_presence=all_pattern_presence,
                   pattern_id=pattern_ids)

  return minibatch


def create(
    batch_size=32,
    subset=None,
    shuffle_corners=True,
    gaussian_noise=0.,
    max_translation=1.,
    rotation_percent=0.0,
    which_patterns='basic',
    drop_prob=0.0,
    max_scale=3.,
    min_scale=.1,
    use_scale_schedule=False,
    schedule_steps=100000.,
):
  """Creates the dataset loader."""

  del subset

  def generator(n_batches=100):
    """Generates `n_batches` of data."""

    step = 0
    n = n_batches * batch_size
    while True:

      if use_scale_schedule:
        scale = (max_scale - min_scale) * float(step) / schedule_steps
        scale = min(scale + min_scale, max_scale)
        step += n_batches

      else:
        scale = max_scale

      batch = create_numpy(n, shuffle_corners, gaussian_noise, max_translation,
                           rotation_percent, scale, which_patterns, drop_prob)

      for i in range(n_batches):
        st = i * batch_size
        ed = st + batch_size
        b = {k: v[st:ed] for k, v in batch.items()}

        yield b

  minibatch = next(generator(n_batches=1))
  dtypes = {k: v.dtype for k, v in minibatch.items()}
  shapes = {k: v.shape for k, v in minibatch.items()}

  dataset = tf.data.Dataset.from_generator(generator, dtypes, shapes)

  iter_data = dataset.make_one_shot_iterator()

  input_batch = iter_data.get_next()
  for _, v in input_batch.items():
    v.set_shape([batch_size] + v.shape[1:].as_list())

  return ConstellationTuple(**input_batch)
