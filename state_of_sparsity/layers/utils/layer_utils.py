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

"""Utilities for implementing sparse layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.contrib.eager.python import tfe as contrib_eager

from tensorflow.contrib.layers.python.layers import utils as layer_utils
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import variables as tf_variables  # pylint: disable=g-direct-tensorflow-import


# The following functions were taken from tensor2tensor. Needed for fast
# embedding lookups on TPUs.
#
# TODO(tgale): Remove these once this issue is resolved.
def is_xla_compiled():
  """Whether we are building graph that will be compiled by XLA.

  This checks whether the code is executing within an XLA context.

  If True, model authors should ensure the graph they build is compilable by
  XLA. Specifically, they should ensure that all ops have XLA implementations
  and that all shapes are statically known.

  Returns:
    bool, whether the current graph will be compiled for XLA.
  """
  ctxt = tf.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
  return control_flow_util.GetContainingXLAContext(ctxt) is not None


def reshape_like(a, b):
  """Reshapes a to match the shape of b in all but the last dimension."""
  ret = tf.reshape(a, tf.concat([tf.shape(b)[:-1], tf.shape(a)[-1:]], 0))
  if not contrib_eager.in_eager_mode():
    ret.set_shape(b.get_shape().as_list()[:-1] + a.get_shape().as_list()[-1:])
  return ret


def gather(params, indices, dtype=tf.float32):
  """Version of tf.gather that works faster on tpu."""
  if not is_xla_compiled():
    return tf.gather(params, indices)
  vocab_size = params.get_shape().as_list()[0]
  indices_flat = tf.reshape(indices, [-1])
  out = tf.matmul(tf.one_hot(indices_flat, vocab_size, dtype=dtype), params)
  out = reshape_like(out, tf.expand_dims(indices, -1))
  return out


def add_variable_to_collection(var, var_set, name):
  """Add provided variable to a given collection, with some checks."""
  collections = layer_utils.get_variable_collections(var_set, name) or []
  var_list = [var]
  if isinstance(var, tf_variables.PartitionedVariable):
    var_list = [v for v in var]
  for collection in collections:
    for var in var_list:
      if var not in tf.get_collection(collection):
        tf.add_to_collection(collection, var)


def standardize_data_format(data_format):
  if data_format == "channels_last":
    data_format = "NHWC"
  elif data_format == "channels_first":
    data_format = "NCHW"
  else:
    data_format = data_format
  return data_format
