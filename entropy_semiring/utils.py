# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Common Utility Functions."""
from typing import Union
from lingvo import compat as tf
import tensorflow_probability as tfp

SeqLen = Union[list[int], tf.Tensor]
TensorTuple = tuple[tf.Tensor, Ellipsis]


def logsumexp_list(tensor_list):
  """LogSumExp of a python list of identically shaped tensors."""
  return tf.reduce_logsumexp(tf.stack(tensor_list, axis=0), axis=0)


def weightedlogsumexp_list(tensor_list,
                           weights_list):
  """WeightedLogSumExp of a python list of identically shaped tensors."""
  ones = tf.ones_like(tensor_list[0])
  stack = tf.stack(tensor_list, axis=0)
  weights_list = [w * ones for w in weights_list]
  weights = tf.stack(weights_list, axis=0)
  return tfp.math.reduce_weighted_logsumexp(stack, weights, axis=0)


def logcrossmultiply(a, b, c,
                     d):
  """LogSumExp(a + d, b + c)."""
  return logsumexp_list([safe_result(a + d), safe_result(b + c)])


def logminus(logx, logy):
  """Calculates log(-xlog(y)) from log(x) and log(y)."""
  zero = tf.zeros_like(logx)
  neg_inf = tf.math.log(tf.zeros_like(logx))
  logminusxlogy = logx + tf.math.log(
      tf.where(tf.math.greater_equal(logy, 0.0), zero, -logy))
  return tf.where(tf.math.is_inf(logminusxlogy), neg_inf, logminusxlogy)


def logzero(shape = (1,),
            dtype = tf.float32):
  """Returns a negative infinity constant."""
  return tf.math.log(tf.zeros(shape=shape, dtype=dtype))


def safe_result(result):
  """Replaces the result with a negative infinity constant if it is infinite."""
  invalid = tf.math.is_inf(result)
  neg_inf = tf.math.log(tf.zeros_like(result))
  return tf.where(invalid, neg_inf, result)


def tuple_to_list(x):
  """Converts a tuple of lists into a list of tuples."""
  return list(zip(*x))
