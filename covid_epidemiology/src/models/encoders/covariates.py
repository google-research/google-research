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

"""Opperations handling covariates in encoders."""

import tensorflow as tf


def mask_covariate_weights_for_timestep(covariate_weights,
                                        timestep,
                                        num_known_steps,
                                        is_training,
                                        covariate_feature_time_offset,
                                        active_window_size,
                                        use_fixed_covariate_mask=True,
                                        seed=None):
  """Returns a tensor with masked weights for the covariates at a given time.

  Args:
    covariate_weights: A tensor of shape [forecast_window_size, num_covariates]
    timestep: An integer timestep indicating what time we are currently
      predicting for.
    num_known_steps: The number of steps we actually have observations for.
    is_training: Whether this prediction will be used for training or inference
    covariate_feature_time_offset: An integer offset used to specify that even
      if data is available for more recent covariate timesteps, to not read
      anything more recent than this offset.
    active_window_size: The number of weights to leave unmasked.
    use_fixed_covariate_mask: If true will default to a mask of
      active_window_size starting at covariate_feature_time_offset for all
      features.
    seed: A random seed used to sample the covariate masks in training.
  """
  num_temporal_weights, num_covariates = covariate_weights.shape
  forecast_window_size = num_temporal_weights - active_window_size

  if use_fixed_covariate_mask:
    desired_offset = tf.constant(covariate_feature_time_offset, dtype=tf.int64)
  else:
    # Offset should not be less than:
    # - Covariate_feature_time_offset
    # - The number of steps we beyond the last observed value we are predicting.
    minval = tf.maximum(
        tf.constant(covariate_feature_time_offset, dtype=tf.int64),
        tf.constant(timestep + 1 - num_known_steps, dtype=tf.int64))
    # Offset should not be more than:
    # - forecast_window_size since we don't have weights beyond there
    # - current timestep minus extra window entries (beyond the first) to avoid
    #   querying negative timesteps.
    maxval = tf.minimum(
        tf.cast(forecast_window_size, dtype=tf.int64),
        tf.constant(timestep - (active_window_size - 1), dtype=tf.int64))
    if minval > maxval:
      desired_offset = tf.constant(-1, dtype=tf.int64)  # Mask out all weights.
    else:
      if is_training:
        desired_offset = tf.random.uniform(
            shape=(),
            minval=minval,
            maxval=maxval + 1,
            dtype=tf.int64,
            seed=seed)
      else:
        desired_offset = minval

  if desired_offset >= 0:  # Non-negative desired offsets indicate a valid mask.
    active_indices = tf.range(desired_offset,
                              desired_offset + active_window_size)
    one_d_mask = tf.scatter_nd(
        indices=tf.expand_dims(active_indices, axis=1),
        updates=tf.ones_like(active_indices),
        shape=tf.constant((num_temporal_weights,), dtype=tf.int64))
    mask = tf.expand_dims(one_d_mask, axis=1)
    mask = tf.tile(mask, (1, num_covariates))
  else:
    mask = tf.zeros((num_temporal_weights, num_covariates), dtype=tf.int64)

  return tf.multiply(covariate_weights, tf.cast(mask, dtype=tf.float32))
