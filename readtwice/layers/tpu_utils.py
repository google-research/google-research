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

"""TPU specific Tensorflow operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla


def cross_replica_concat(tensor,
                         num_replicas,
                         name = None):
  """Reduce a concatenation of the `tensor` across tpu cores.

  Branched from //audio/ears/nnfp/tensorflow/tpu_ops.py

  Args:
    tensor: tensor to concatenate.
    num_replicas: Number of TPU cores.
    name: A name for the op.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  replica_id = xla.replica_id()

  with tf.compat.v1.name_scope(name, 'tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[replica_id]],
        updates=[tensor],
        shape=[num_replicas] + tensor.shape.as_list())

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = tf.compat.v1.tpu.cross_replica_sum(ext_tensor)

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])
