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

# Lint as: python2, python3
"""Common utility functions for cost model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
from typing import Any, Optional, Text, TypeVar

import numpy as np
import tensorflow.compat.v1 as tf

from tunas import cost_model_data
from tunas import schema


def get_mask(oneof):
  if len(oneof.choices) > 1 and oneof.mask is None:
    raise ValueError('OneOf must have a mask if it has more than one choice: {}'
                     .format(oneof))
  return oneof.mask


_T = TypeVar('_T')


def kron(x, y):
  """TF version of tensor product between two vectors (similar to np.kron)."""
  return tf.reshape(tf.expand_dims(x, 1) * tf.expand_dims(y, 0), [-1])


def estimate_cost(features, ssd):
  """Generate a TensorFlow subgraph to estimate the cost of an architecture.

  Args:
    features: A 1D float tensor containing features for a single network
        architecture.
    ssd: The name of the search space definition to use for the cost model.

  Returns:
    A scalar float tensor containing the estimated cost for the specified
    network architecture
  """
  kernel_data = cost_model_data.KERNEL_DATA[ssd]
  kernel_data = base64.decodebytes(kernel_data)
  kernel = np.frombuffer(kernel_data, cost_model_data.SERIALIZATION_DTYPE)
  kernel = kernel.reshape([-1, 1]).astype(np.float32)

  bias_data = cost_model_data.BIAS_DATA[ssd]
  bias_data = base64.decodebytes(bias_data)
  bias = np.frombuffer(bias_data, cost_model_data.SERIALIZATION_DTYPE)
  bias = bias.reshape([1]).astype(np.float32)

  with tf.name_scope('estimate_cost'):
    batch_features = tf.expand_dims(features, axis=0)
    batch_prediction = tf.linalg.matmul(batch_features, kernel)
    batch_prediction = tf.nn.bias_add(batch_prediction, bias)
    return tf.squeeze(batch_prediction, axis=[0, 1])
