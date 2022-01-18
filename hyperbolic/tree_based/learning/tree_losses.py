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
"""Loss functions for CF tree based learning."""

import numpy as np
import tensorflow.compat.v2 as tf
from hyperbolic.utils import hyperbolic as hyp_utils
from hyperbolic.utils.learn import softmax_cross_entropy


class TreeBasedLossFn(object):
  """loss function class for CF tree based embeddings.

  Attributes:
    n_items: int, total number of items.
    nodes_per_level: list of ints, holding the total number of nodes per level,
      not including the root nor the leaves' level.
    node_batch_per_level: list of ints, holding the batch size of nodes per
      level, not including the root nor the leaves' level.
    margin: non trainable tf Variable holding the margin for a pairwise hinge
      loss.
    sep_w: weight for seperation loss
  """

  def __init__(self, sizes, args):
    self.n_items = sizes[1]
    self.nodes_per_level = args.nodes_per_level
    self.node_batch_per_level = args.node_batch_per_level
    self.margin = tf.Variable(
        args.m * tf.keras.backend.ones(1),
        trainable=False)
    self.sep_w = args.sep_w

  def node_and_negative_sampling(self, input_batch):
    """Samples positive nodes and negative nodes and items based on node batch size."""
    batch_size = tf.shape(input_batch)[0]
    # negative items
    random_items = tf.random.uniform(
        shape=[batch_size, 1], minval=0, maxval=self.n_items, dtype=tf.int64)
    input_tensor = tf.concat([input_batch, random_items], axis=1)
    # positive and negative nodes batches by level
    node_batch_by_l = []  # holds batch node embeddings indices
    neg_node_in_batch_by_l = []  # holds indices within batch
    for l, node_batch in enumerate(self.node_batch_per_level):
      node_batch_by_l.append(
          tf.random.uniform(
              shape=[batch_size, node_batch],
              minval=np.sum(self.nodes_per_level[:l]),
              maxval=np.sum(
                  self.nodes_per_level[:l]) + self.nodes_per_level[l],
              dtype=tf.int64))
      neg_node_in_batch_by_l.append(
          tf.random.uniform(
              shape=[batch_size, node_batch],
              minval=np.sum(self.node_batch_per_level[:l]),
              maxval=np.sum(
                  self.node_batch_per_level[:l]) + self.node_batch_per_level[l],
              dtype=tf.int64))
    # positive nodes
    node_tensor = tf.concat(node_batch_by_l, axis=1)
    # negative nodes within batch
    neg_node_ind = tf.concat(neg_node_in_batch_by_l, axis=1)
    # rewrite the indices in tf.gather_nd format
    sizes = tf.shape(neg_node_ind)
    row_ind = tf.repeat(
        tf.reshape(tf.range(0, sizes[0], dtype=tf.int64), [sizes[0], 1]),
        repeats=sizes[1],
        axis=1)
    neg_node_ind = tf.stack([row_ind, neg_node_ind], axis=2)
    return node_tensor, neg_node_ind, input_tensor

  def pos_and_neg_loss(self, pos, neg):
    """Computes loss given a positive and negative sample.

    Args:
      pos: Tensor.
      neg: Tensor of the same shape of pos.

    Returns:
      pointwise loss of the same shape as pos.
    """
    raise NotImplementedError

  def user_item_node_interaction_loss(self, probs, user_node_distance,
                                      item_node_distance, user_item_distance,
                                      neg_node_ind):
    """Computes pairwise hinge based loss, as in the reference below.

    Args:
      probs: Tensor of size batch_size x tot_node_batch containing the
        probability a node is the ancestor of the positive item.
      user_node_distance: Tensor of size batch_size x tot_node_batch containing
        square of the distances between the nodes and the user.
      item_node_distance: Tensor of size batch_size x tot_node_batch containing
        square of the distances between the nodes and the positive item.
      user_item_distance: Tensor of size batch_size x 2 containing
        square of the distances between the user and the positive and negative
        items.
      neg_node_ind: Tensor of size batch_size x tot_node_batch x 2 containing
        indices of negative nodes (within the sampled batch, from the relevant
        level), in tf.gather_nd format.

    Returns:
      loss within the input_batch.
    """
    # TODO(advaw): change Idea 2 above to a real reference when possible.
    user_to_node = self.pos_and_neg_loss(
        user_node_distance, tf.gather_nd(user_node_distance, neg_node_ind))
    item_to_node = self.pos_and_neg_loss(
        item_node_distance, tf.gather_nd(item_node_distance, neg_node_ind))
    nodes_loss = tf.reduce_sum(probs * (user_to_node + item_to_node), axis=1)
    user_to_item = self.pos_and_neg_loss(user_item_distance[:, 0],
                                         user_item_distance[:, 1])
    loss = tf.reduce_mean(user_to_item + nodes_loss)
    return loss

  def calculate_loss(self, model, input_batch):
    """Computes loss with node batch sampling and negative sampling.

    Args:
      model: tf.keras.Model CF tree based embedding model.
      input_batch: Tensor of size batch_size x 2 containing input pairs
        representing (user, positive item).

    Returns:
      loss within the input_batch.
    """
    node_tensor, neg_node_ind, input_tensor = self.node_and_negative_sampling(
        input_batch)
    inputs = tf.concat([input_tensor, node_tensor], 1)
    probs, user_node_distance, item_node_distance, user_item_distance = model(
        inputs)
    return self.user_item_node_interaction_loss(probs, user_node_distance,
                                                item_node_distance,
                                                user_item_distance,
                                                neg_node_ind)


class HingeTreeLossFn(TreeBasedLossFn):
  """hinge loss function class for CF tree based embeddings."""

  def pos_and_neg_loss(self, pos, neg):
    return tf.math.maximum(pos - neg + self.margin, 0)


class CrossEntrpyTreeLossFn(TreeBasedLossFn):
  """Cross Entrpy loss function class for CF tree based embeddings."""

  def pos_and_neg_loss(self, pos, neg):
    """softmax cross entropy loss."""
    return softmax_cross_entropy(pos, neg)


class SeperationTreeLossFn(TreeBasedLossFn):
  """Loss function class for CF tree based embeddings, with seperation."""

  def seperation_loss(self, model, node_tensor, neg_node_ind):
    """Calculates -d(n,n')^2."""
    neg_nodes_actual_ind = tf.gather_nd(node_tensor, neg_node_ind)
    nodes = model.get_batch_nodes(node_tensor)
    neg_nodes = model.get_batch_nodes(neg_nodes_actual_ind)
    node_neg_node_dist = hyp_utils.hyp_distance(nodes, neg_nodes,
                                                tf.math.softplus(model.c))
    seperation = tf.reduce_mean(-model.square_distance(node_neg_node_dist))
    return seperation

  def calculate_loss(self, model, input_batch):
    node_tensor, neg_node_ind, input_tensor = self.node_and_negative_sampling(
        input_batch)
    inputs = tf.concat([input_tensor, node_tensor], 1)
    probs, user_node_distance, item_node_distance, user_item_distance = model(
        inputs)
    return self.user_item_node_interaction_loss(
        probs, user_node_distance, item_node_distance, user_item_distance,
        neg_node_ind) + self.sep_w * self.seperation_loss(
            model, node_tensor, neg_node_ind)


class SeperationHingeTreeLossFn(SeperationTreeLossFn):
  """hinge loss function class for CF tree based embeddings, with seperation."""

  def pos_and_neg_loss(self, pos, neg):
    return tf.math.maximum(pos - neg + self.margin, 0)


class SeperationCrossEntrpyTreeLossFn(SeperationTreeLossFn):
  """Cross Entrpy loss function class for CF tree based embeddings, with seperation."""

  def pos_and_neg_loss(self, pos, neg):
    """softmax cross entropy loss."""
    return softmax_cross_entropy(pos, neg)
