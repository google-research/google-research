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

"""A collection of retrieval functions for negative mining.

Retrieval functions take in a matrix of scores and return a batch x `k` set of
indices indicating the `k` items retrieved.
"""
import abc
import tensorflow.compat.v2 as tf


class AbstractRetrievalFn(tf.Module, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def __call__(self, scores):
    pass


class MaxScoreRetrievalFn(AbstractRetrievalFn):

  def __call__(self, scores):
    indices = tf.argmax(scores, axis=1)
    return tf.expand_dims(indices, 1)


def _sample_gumbel(shape):
  uniform_vals = tf.random.uniform(shape)
  gumbel_vals = -tf.math.log(-tf.math.log(uniform_vals))
  return gumbel_vals


class GumbelMaxRetrievalFn(AbstractRetrievalFn):
  """Creates a retrieval function that uses Gumbel-max sampling.

  Gumbel-max sampling is an approach to sample from the softmax distribution of
  a set of scores by perturbing the scores then taking the argmax. The scores
  are first scaled by `inv_temp` then perturbed by adding Gumbel noise.
  """

  def __init__(self, inv_temp=1.0):
    super(GumbelMaxRetrievalFn, self).__init__()
    self.inv_temp = inv_temp

  def __call__(self, scores):
    gumbel_vals = _sample_gumbel(tf.shape(scores))
    perturbed_scores = self.inv_temp * scores + gumbel_vals
    indices = tf.argmax(perturbed_scores, axis=1)
    return tf.expand_dims(indices, 1)
