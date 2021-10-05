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
"""DrQ Utils."""

import tensorflow as tf

IMG_PAD = 4


def image_aug(batch, meta):
  """Padding and cropping."""
  obs, action, reward, discount, next_obs = batch

  paddings = tf.constant(
      [[0, 0], [IMG_PAD, IMG_PAD], [IMG_PAD, IMG_PAD], [0, 0]])
  cropped_shape = obs.shape
  # The reference uses ReplicationPad2d in pytorch, but it is not available
  # in tf. Use 'SYMMETRIC' instead.
  obs = tf.pad(obs, paddings, 'SYMMETRIC')
  next_obs = tf.pad(next_obs, paddings, 'SYMMETRIC')

  def get_random_crop(padded_obs):
    return tf.image.random_crop(padded_obs, cropped_shape[1:])

  # Note: tf.image.random_crop called on a batched image applies the same
  # random crop to every image in the batch. Parallelize using map_fn to
  # diversify.
  aug_obs = tf.map_fn(get_random_crop, obs)
  aug_next_obs = tf.map_fn(get_random_crop, next_obs)

  return (aug_obs, action, reward, discount, aug_next_obs), meta
