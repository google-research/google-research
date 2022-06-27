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

"""Defines custom layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf


def soft_hot_layer(offsets, scales):
  """Embedding layer for scalar observations.

  For each input scalar, produces an embedding vector.

  Args:
    offsets: Matrix of shape `[input_dim, soft_hot_dim]` which contains the
        soft-hot "offset" for the i-th scalar input and j-th soft-hot feature.
        `input_dim` is the number of scalar inputs, and `soft_hot_dim` is the
        number of soft-hot features (output dim).
    scales: Array of shape `[input_dim]` which contains the soft-hot scales, one
        for each input scalar.

  Returns:
    tf.keras.layers.Layer object. This layer converts a tensor of shape
    `[..., input_dim]` to a tensor of shape `[..., input_dim*soft_hot_dim]`,
    which is a concat of the soft-hot embeddings for each input scalar.
  """
  offsets = np.asarray(offsets)
  scales = np.asarray(scales)
  assert offsets.ndim == 2
  assert scales.ndim == 1
  length = offsets.size
  offsets = offsets[np.newaxis, Ellipsis]
  scales = scales[np.newaxis, :, np.newaxis]

  def _f(x):
    return tf.reshape(
        tf.exp(-(tf.expand_dims(x, -1) - offsets)**2/scales),
        (tf.shape(x)[0], length))

  return tf.keras.layers.Lambda(_f)


def stretch(low, high, fraction):
  """Stretch out interval by the given fraction."""
  delta = abs(high - low)*fraction
  return (low - delta, high + delta)


def obs_embedding_kwargs(num_features, batch=None, bounds=None, variance=None,
                         spillover=0.05):
  """Helper function that generates arguments to `soft_hot_layer`.

  Usages
  1) obs_embedding_kwargs(num_features=N, batch=<2D tensor>)
  2) obs_embedding_kwargs(
         num_features=N,
         bounds=[(low_0, hi_0), (low_1, hi_1), ..., (low_K, hi_K)],
         variance=[v_0, v_1, ..., v_K])

  Args:
    num_features: Number of features in the generated embedding. Output size of
        the layer returned by this function.
    batch: (usage 1) A batch of data, with which to extract the input feature
        statistics. Automatically calculates `bounds` and `variance` arguments
        under the hood from `batch`.
    bounds: (usage 2) List of 2-tuples, giving the min and max values expected
        for each input feature. Values out of bounds are allowed, but may be
        difficult to distinguish from min and max values. Set higher `spillover`
        to accomidate out-of-bounds values.
    variance: (usage 2) List of variances of each input feature. This sets the
        width of the Gaussian kernels used by `soft_hot_layer`.
    spillover: Widens the sensitivity range.

  Returns:
    Keyword arguments for `soft_hot_layer`.

  """
  if batch is not None:
    assert bounds is None
    assert variance is None
    assert batch.ndim == 2
    bounds = list(zip(np.min(batch, axis=0), np.max(batch, axis=0)))
    variance = np.var(batch, axis=0)
  assert bounds is not None
  assert variance is not None
  dim = len(list(zip(*bounds))[0])  # Shape of the input (number of scalars).
  assert len(variance) == dim
  nf = num_features  # Number of soft-hot features.
  return {
      'offsets': np.stack(
          [np.linspace(*stretch(bounds[i][0], bounds[i][1], spillover), nf)
           for i in range(dim)],
          axis=0),  # shape: [dim, nF]
      'scales': variance,  # shape: [dim]
  }

