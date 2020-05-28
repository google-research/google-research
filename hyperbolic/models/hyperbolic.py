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
"""Hyperbolic embedding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v2 as tf


from hyperbolic.models.base import CFModel
from hyperbolic.utils import hyperbolic as hyp_utils


class BaseH(CFModel):
  """Base model class for hyperbolic embeddings."""

  def __init__(self, sizes, args):
    """Initialize Hyperbolic CF embedding model.

    Args:
      sizes: Tuple of size 2 containing (n_users, n_items).
      args: Namespace with config arguments (see config.py for detailed overview
        of arguments supported).
    """
    super(BaseH, self).__init__(sizes, args)
    self.c = tf.Variable(
        initial_value=tf.keras.backend.ones(1), trainable=args.train_c)

  def get_rhs(self, input_tensor):
    c = tf.math.softplus(self.c)
    return hyp_utils.expmap0(self.item(input_tensor[:, 1]), c)

  def get_candidates(self, input_tensor):
    c = tf.math.softplus(self.c)
    temp = self.item.embeddings
    return hyp_utils.expmap0(temp, c)

  def similarity_score(self, lhs, rhs, eval_mode):
    c = tf.math.softplus(self.c)
    if eval_mode and self.rhs_dep_lhs:
      return -hyp_utils.hyp_distance_batch_rhs(lhs, rhs, c)**2
    elif eval_mode and not self.rhs_dep_lhs:
      return -hyp_utils.hyp_distance_all_pairs(lhs, rhs, c)**2
    return -hyp_utils.hyp_distance(lhs, rhs, c)**2


class DistH(BaseH):
  """Hyperbolic translation with parameters defined in tangent space."""

  def get_queries(self, input_tensor):
    c = tf.math.softplus(self.c)
    lhs = hyp_utils.expmap0(self.user(input_tensor[:, 0]), c)
    return lhs


class OperEmbH(BaseH):
  """Operator Embedding Hyperbolic model for collaborative filtering."""

  def __init__(self, sizes, args):
    super(OperEmbH, self).__init__(sizes, args)
    self.userpp = tf.keras.layers.Embedding(
        input_dim=sizes[0],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.user_regularizer,
        name='user_push_pull_embeddings')
    self.rhs_dep_lhs = True

  def get_rhs(self, input_tensor):
    rhs_ts = tf.math.multiply(
        self.userpp(input_tensor[:, 0]), self.item(input_tensor[:, 1]))
    c = tf.math.softplus(self.c)
    return hyp_utils.expmap0(rhs_ts, c)

  def get_candidates(self, input_tensor):
    c = tf.math.softplus(self.c)
    items = tf.expand_dims(self.item.embeddings, 0)  # (1, n_item, rank)
    pp = tf.expand_dims(self.userpp(input_tensor[:, 0]), 1)  # (batch, 1, rank)
    cands_ts = tf.math.multiply(pp, items)  # (batch, n_item, rank)
    return hyp_utils.expmap0(cands_ts, c, self.rhs_dep_lhs)

  def get_queries(self, input_tensor):
    c = tf.math.softplus(self.c)
    lhs = hyp_utils.expmap0(self.user(input_tensor[:, 0]), c)
    return lhs
