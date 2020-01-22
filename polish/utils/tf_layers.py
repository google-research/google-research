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

"""Wrappers for TF layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
from polish.utils import tf_initializers


def fc(tensor_in, num_hidden, scope_name, init_scale=1.0, init_bias=0.0):
  """Implementation of a fully-connected layer.

  Args:
    tensor_in: Input to the layer.
    num_hidden: Number of neurons.
    scope_name: Scope name.
    init_scale: Used for initializer.
    init_bias: Bias initializer (constant initializer).

  Returns:
    tensor_in * w + b.
  """

  with tf.variable_scope(scope_name):
    nin = tensor_in.get_shape()[1].value
    # TODO(ayazdan): Add support for other initializers such as Xavier.
    w = tf.get_variable(
        'weight', [nin, num_hidden],
        initializer=tf_initializers.Orthogonal(init_scale))
    # Performs broadcasting
    b = tf.get_variable(
        'bias', [num_hidden], initializer=tf.constant_initializer(init_bias))
  return tf.matmul(tensor_in, w) + b
