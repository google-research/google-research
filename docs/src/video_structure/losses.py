# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Losses for the video representation model."""

import tensorflow.compat.v1 as tf


def temporal_separation_loss(cfg, coords):
  """Encourages keypoint to have different temporal trajectories.

  If two keypoints move along trajectories that are identical up to a time-
  invariant translation (offset), this suggest that they both represent the same
  object and are redundant, which we want to avoid.

  To measure this similarity of trajectories, we first center each trajectory by
  subtracting its mean. Then, we compute the pairwise distance between all
  trajectories at each timepoint. These distances are higher for trajectories
  that are less similar. To compute the loss, the distances are transformed by
  a Gaussian and averaged across time and across trajectories.

  Args:
    cfg: ConfigDict.
    coords: [batch, time, num_landmarks, 3] coordinate tensor.

  Returns:
    Separation loss.
  """
  x = coords[Ellipsis, 0]
  y = coords[Ellipsis, 1]

  # Center trajectories:
  x -= tf.reduce_mean(x, axis=1, keepdims=True)
  y -= tf.reduce_mean(y, axis=1, keepdims=True)

  # Compute pairwise distance matrix:
  d = ((x[:, :, :, tf.newaxis] - x[:, :, tf.newaxis, :]) ** 2.0 +
       (y[:, :, :, tf.newaxis] - y[:, :, tf.newaxis, :]) ** 2.0)

  # Temporal mean:
  d = tf.reduce_mean(d, axis=1)

  # Apply Gaussian function such that loss falls off with distance:
  loss_matrix = tf.exp(-d / (2.0 * cfg.separation_loss_sigma ** 2.0))
  loss_matrix = tf.reduce_mean(loss_matrix, axis=0)  # Mean across batch.
  loss = tf.reduce_sum(loss_matrix)  # Sum matrix elements.

  # Subtract sum of values on diagonal, which are always 1:
  loss -= cfg.num_keypoints

  # Normalize by maximal possible value. The loss is now scaled between 0 (all
  # keypoints are infinitely far apart) and 1 (all keypoints are at the same
  # location):
  loss /= cfg.num_keypoints * (cfg.num_keypoints - 1)

  return loss
