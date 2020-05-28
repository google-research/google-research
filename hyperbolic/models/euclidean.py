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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Euclidean embedding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v2 as tf

from hyperbolic.models.base import CFModel
from hyperbolic.utils import euclidean as euc_utils


class BaseE(CFModel):
  """Base model class for Euclidean embeddings."""

  def get_rhs(self, input_tensor):
    rhs = self.item(input_tensor[:, 1])
    return rhs

  def get_candidates(self, input_tensor):
    cands = self.item.embeddings
    return cands

  def similarity_score(self, lhs, rhs, eval_mode):
    if self.sim == 'dot':
      if eval_mode:
        if self.rhs_dep_lhs:
          score = tf.squeeze(
              tf.matmul(
                  tf.expand_dims(lhs, 1), tf.transpose(rhs, perm=[0, 2, 1])))
        else:
          score = tf.matmul(lhs, tf.transpose(rhs))
      else:
        score = tf.reduce_sum(lhs * rhs, axis=-1, keepdims=True)
    elif self.sim == 'dist':
      score = -euc_utils.euc_sq_distance(lhs, rhs, eval_mode, self.rhs_dep_lhs)
    else:
      raise AttributeError('Similarity function {} not recognized'.format(
          self.sim))
    return score


class SMFactor(BaseE):
  """Simple Matrix Factorization model."""

  def __init__(self, sizes, args):
    super(SMFactor, self).__init__(sizes, args)
    self.sim = 'dot'

  def get_queries(self, input_tensor):
    return self.user(input_tensor[:, 0])


class DistE(BaseE):
  """Simple Collaborative Metric Learning model."""

  def __init__(self, sizes, args):
    super(DistE, self).__init__(sizes, args)
    self.sim = 'dist'

  def get_queries(self, input_tensor):
    return self.user(input_tensor[:, 0])


class OperEmbE(BaseE):
  """Operator Embedding Euclidean model for collaborative filtering."""

  def __init__(self, sizes, args):
    super(OperEmbE, self).__init__(sizes, args)
    self.sim = 'dist'
    self.userpp = tf.keras.layers.Embedding(
        input_dim=sizes[0],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.user_regularizer,
        name='user_push_pull_embeddings')
    self.rhs_dep_lhs = True

  def get_rhs(self, input_tensor):
    rhs = tf.math.multiply(
        self.userpp(input_tensor[:, 0]), self.item(input_tensor[:, 1]))
    return rhs

  def get_candidates(self, input_tensor):
    items = tf.expand_dims(self.item.embeddings, 0)  # (1, n_item, rank)
    pp = tf.expand_dims(self.userpp(input_tensor[:, 0]), 1)  # (batch, 1, rank)
    cands = tf.math.multiply(pp, items)  # (batch, n_item, rank)
    return cands

  def get_queries(self, input_tensor):
    return self.user(input_tensor[:, 0])
