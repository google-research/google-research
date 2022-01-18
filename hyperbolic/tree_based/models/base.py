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

# Lint as: python3
"""class for Collaborative Filtering tree based model."""


import numpy as np
import tensorflow.compat.v2 as tf

from hyperbolic.utils import hyperbolic as hyp_utils
from hyperbolic.utils import tree as tree_utils


class CFTreeModel(tf.keras.Model):
  """CF tree based embedding model class.

  Hyperbolic model, with parameters being held on spheres in the Poincare ball.

  Attributes:
    rank: int, embeddings dimension.
    n_users: int, total number of users.
    n_items: int, total number of items.
    nodes_per_level: list of ints, holding the total number of nodes per level,
      not including the root nor the leaves' level.
    radii: list of floats, holding the radius of nodes per level including
      the leaves' level, not including the root.
    initializer: tf.keras.initializers class indicating which initializer
      to use.
    user: Tensorflow tf.keras.layers.Embedding class, holding user
      embeddings.
    item: Tensorflow tf.keras.layers.Embedding class, holding item
      embeddings.
    node: Tensorflow tf.keras.layers.Embedding class, holding node
      embeddings.
    c: Tensorflow Variable of shape () holding the curvature.
    node_batch_per_level: list of ints, holding the batch size of nodes per
      level, not including the root nor the leaves' level.
    radius_by_batch: Tensor of size sum(node_batch_per_level), holding
      radii of nodes based on their levels.
    stop_grad: Bool indicating whether to stop gradients w.r.t. user and item
      embeddings in the node interaction part.
    tot_levels: Int total number of additional levels in the tree, excluding
      the root and the leaves.
    k_threshold: Int - threshold for top_k predictions: below this number,
    the number of seen items (n_seen) per user is added to top_k eval and
    top (k+n_seen) nodes are explored.

  """

  def __init__(self, sizes, args):
    """Initialize CF tree embedding model."""
    super(CFTreeModel, self).__init__()
    self.rank = args.rank
    self.n_users = sizes[0]
    self.n_items = sizes[1]
    self.nodes_per_level = args.nodes_per_level
    self.radii = args.radii
    self.initializer = getattr(tf.keras.initializers, args.initializer)
    self.user = tf.keras.layers.Embedding(
        input_dim=self.n_users,
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        name='user_embeddings')
    self.node = tf.keras.layers.Embedding(
        input_dim=np.sum(self.nodes_per_level),
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        name='node_embeddings')
    self.item = tf.keras.layers.Embedding(
        input_dim=self.n_items,
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        name='item_embeddings')
    self.c = tf.Variable(
        initial_value=tf.keras.backend.ones(1), trainable=args.train_c)
    self.node_batch_per_level = args.node_batch_per_level
    self.radius_by_batch = tf.constant(
        np.concatenate([[self.radii[l]] * batch
                        for l, batch in enumerate(self.node_batch_per_level)]))
    self.stop_grad = args.stop_grad
    self.tot_levels = len(self.nodes_per_level)
    self.k_threshold = args.k_threshold

  def get_hyperbolic_points(self, radius, embeddings):
    return tf.cast(hyp_utils.tanh(radius),
                   tf.float64) * self.unit_norm(embeddings)

  def get_users(self, indices):
    return self.get_hyperbolic_points(self.radii[-1], self.user(indices))

  def get_all_users(self):
    return self.get_hyperbolic_points(self.radii[-1], self.user.embeddings)

  def get_items(self, indices):
    return self.get_hyperbolic_points(self.radii[-1], self.item(indices))

  def get_all_items(self):
    return self.get_hyperbolic_points(self.radii[-1], self.item.embeddings)

  def get_batch_nodes(self, indices):
    radius_batch = tf.reshape(self.radius_by_batch, [1, -1, 1])
    return self.get_hyperbolic_points(radius_batch, self.node(indices))

  def get_nodes_level(self, indices, l):
    return self.get_hyperbolic_points(self.radii[l], self.node(indices))

  def get_all_nodes_level(self, l):
    first_ind = sum(self.nodes_per_level[:l])
    indices = tf.range(first_ind, first_ind + self.nodes_per_level[l])
    return self.get_hyperbolic_points(self.radii[l], self.node(indices))

  def as_hyperbolic_points(self, input_tensor, nodes_ind):
    """hyperbolic embeddings for tree based model training.

    Args:
      input_tensor: Tensor of size batch_size x 3 containing user, positive
        item, negative item indices.
      nodes_ind: Tensor of size batch_size x tot_node_batch containing nodes
        indices, where tot_node_batch equals to sum(self.node_batch_per_level).
    Returns:
      users: Tensor of size batch_size x rank containing the user embeddings.
      items: Tensor of size batch_size x 2 x rank containing positive and
      negative items embeddings on the hyperbolic sphere (centered at the
      origin) of hyperbolic radius of radii[-1].
      nodes: Tensor of size batch_size x tot_batch_node x rank containing
      nodes embeddings on the hyperbolic spheres (centered at the origin)
      of hyperbolic radii radii[l], where l is the level the node belongs to.
    """
    users = self.get_users(input_tensor[:, 0])
    items = self.get_items(input_tensor[:, 1:])
    nodes = self.get_batch_nodes(nodes_ind)
    return users, items, nodes

  def all_distances(self, input_tensor, nodes_ind):
    """distance calculations for tree based model training.

    Args:
      input_tensor: Tensor of size batch_size x 3 containing user, positive
        item, negative item indices.
      nodes_ind: Tensor of size batch_size x tot_node_batch containing nodes
        indices, where tot_node_batch equals to sum(self.node_batch_per_level).
    Returns:
      user_node_distance: Tensor of size batch_size x tot_node_batch containing
        the distances between the nodes and the user.
      item_node_distance: Tensor of size batch_size x tot_node_batch containing
        the distances between the nodes and the positive item.
      user_item_distance: Tensor of size batch_size x 2 containing
        the distances between the user and the positive and negative items.
    """
    c = tf.math.softplus(self.c)
    users, items, nodes = self.as_hyperbolic_points(input_tensor, nodes_ind)
    user_node_distance = hyp_utils.hyp_distance_batch_rhs(
        users, nodes, c)
    pos_item_node_distance = hyp_utils.hyp_distance_batch_rhs(
        items[:, 0, :], nodes, c)
    user_item_distance = hyp_utils.hyp_distance_batch_rhs(
        users, items, c)
    return user_node_distance, pos_item_node_distance, user_item_distance

  def all_distances_sg(self, input_tensor, nodes_ind):
    """distance calculations for tree based model training, with gradient stops.

    Args:
      input_tensor: Tensor of size batch_size x 3 containing user, positive
        item, negative item indices.
      nodes_ind: Tensor of size batch_size x tot_node_batch containing nodes
        indices, where tot_node_batch equals to sum(self.node_batch_per_level).
    Returns:
      user_node_distance: Tensor of size batch_size x tot_node_batch containing
        the distances between the nodes and the user.
      item_node_distance: Tensor of size batch_size x tot_node_batch containing
        the distances between the nodes and the positive item.
      user_item_distance: Tensor of size batch_size x 2 containing
        the distances between the user and the positive and negative items.
    """
    c = tf.math.softplus(self.c)
    users, items, nodes = self.as_hyperbolic_points(input_tensor, nodes_ind)
    user_node_distance = hyp_utils.hyp_distance_batch_rhs(
        tf.stop_gradient(users), nodes, c)
    pos_item_node_distance = hyp_utils.hyp_distance_batch_rhs(
        tf.stop_gradient(items[:, 0, :]), nodes, c)
    user_item_distance = hyp_utils.hyp_distance_batch_rhs(
        users, items, c)
    return user_node_distance, pos_item_node_distance, user_item_distance

  def distance_to_probability(self, distance):
    return tf.math.exp(-distance)

  def square_distance(self, distance):
    return distance**2

  def unit_norm(self, tensor):
    return tf.math.l2_normalize(tensor, axis=-1)

  def call(self, inputs):
    """distance and probabilities calculations for tree based model training.

    Args:
      inputs: Tensor of size batch_size x (3 + tot_node_batch) containing user,
        positive item, negative item and nodes indices, where tot_node_batch
        equals to sum(self.node_batch_per_level).
    Returns:
      probs: Tensor of size batch_size x tot_node_batch containing the
        probability a node is the ancestor of the positive item.
      user_node_distance: Tensor of size batch_size x tot_node_batch containing
        square of the distances between the nodes and the user.
      item_node_distance: Tensor of size batch_size x tot_node_batch containing
        square of the distances between the nodes and the positive item.
      user_item_distance: Tensor of size batch_size x 2 containing
        square of the distances between the user and the positive and negative
        items.
    """
    input_tensor, node_tensor = tf.split(inputs, [3, tf.shape(inputs)[1] -3], 1)
    distances_func = self.all_distances_sg if self.stop_grad else self.all_distances
    distances = distances_func(input_tensor, node_tensor)
    user_node_distance = self.square_distance(distances[0])
    item_node_distance = self.square_distance(distances[1])
    user_item_distance = self.square_distance(distances[2])
    probs = tf.stop_gradient(self.distance_to_probability(item_node_distance))
    return probs, user_node_distance, item_node_distance, user_item_distance

  def get_scores_targets(self, input_tensor):
    """Computes pairs' scores as well as scores againts all possible items.

    Args:
      input_tensor: Tensor of size batch_size x 2 containing pairs' indices.

    Returns:
      scores: Numpy array of size batch_size x n_items containing users'
              scores against all possible items in the CF.
      targets: Numpy array of size batch_size x 1 containing pairs' scores.
    """
    c = tf.math.softplus(self.c)
    users = self.get_users(input_tensor[:, 0])
    items = self.get_all_items()
    user_item_distance = hyp_utils.hyp_distance_all_pairs(
        users, items, c)
    scores = (-user_item_distance).numpy()
    targets = scores[np.arange(scores.shape[0]), input_tensor[:, 1]].reshape(
        (-1, 1))
    return scores, targets

  def eval(self, examples, filters, batch_size=500):
    """Compute ranking-based evaluation metrics in full setting.

    Note: this evaluation is flat (doesn't use the learnt tree structure and
    ranks based on scoring a user against all items). For recommending by
    top-k tree search using the learnt tree, see self.build_tree and
    self.top_k_tree_eval.

    Args:
      examples: Tensor of size n_examples x 2 containing pairs' indices.
      filters: Dict representing items to skip per user for evaluation in
        the filtered setting.
      batch_size: batch size to use to compute scores.

    Returns:
      Numpy array of shape (n_examples, ) containing the rank of each example.
    """
    total_examples = tf.data.experimental.cardinality(examples).numpy()
    batch_size = min(batch_size, total_examples)
    ranks = np.ones(total_examples)
    for counter, input_tensor in enumerate(examples.batch(batch_size)):
      if batch_size * counter >= total_examples:
        break
      scores, targets = self.get_scores_targets(input_tensor)
      for i, query in enumerate(input_tensor):
        query = query.numpy()
        filter_out = filters[query[0]]
        scores[i, filter_out] = -1e6
      ranks[counter * batch_size:(counter + 1) * batch_size] += np.sum(
          (scores >= targets), axis=1)
    return ranks

  def build_tree(self):
    """Updates self.tree (the tree structure) based on closest nodes, bottom up."""
    c = tf.math.softplus(self.c)
    items = self.get_all_items()
    tot_levels = self.tot_levels
    nodes = [self.get_all_nodes_level(l) for l in range(tot_levels)]
    # closest_node_to_items is a numpy array of size (n_items, )
    # holding the parent node index from one level up.
    closest_node_to_items = (sum(
        self.nodes_per_level[:tot_levels - 1]) + tf.math.argmin(
            hyp_utils.hyp_distance_all_pairs(items, nodes[tot_levels - 1], c),
            axis=1)).numpy()
    # closest_node_to_nodes is a numpy array of size (n_tot_nodes, )
    # holding the parent node index from one level up. Root index is -1.
    closest_node_to_nodes = -np.ones(sum(self.nodes_per_level))
    for l in range(1, tot_levels):
      first_ind = sum(self.nodes_per_level[:l])
      last_ind = sum(self.nodes_per_level[:l + 1])
      closest_node_to_nodes[first_ind:last_ind] = (
          sum(self.nodes_per_level[:l - 1]) + tf.math.argmin(
              hyp_utils.hyp_distance_all_pairs(nodes[l], nodes[l - 1], c),
              axis=1)).numpy()
    self.tree = tree_utils.build_tree(closest_node_to_items,
                                      closest_node_to_nodes,
                                      self.nodes_per_level)

  def get_top_k_items(self, user, k):
    """Finds the top k items for given user using top-down top k search.

    Note: currently supports making top k items predictions for
    a single user at a time.

    Args:
      user: Tensor of size d containing the user embeddings.
      k: Int for top k.

    Returns:
      Numpy array of shape (k, ) containing the indices of the top k items.
    """
    ragged_childs = self.tree.as_ragged(sum(self.nodes_per_level))
    # get the child nodes of the root first
    current_nodes_ind = ragged_childs[-1]
    for l in range(self.tot_levels):
      current_nodes_ind = self.next_level_indices(current_nodes_ind,
                                                  ragged_childs, user, l, k)
    item_ind = current_nodes_ind - sum(self.nodes_per_level)
    items = self.get_items(item_ind)
    return tf.gather(item_ind, self.top_k_from_dist(user, items, k)).numpy()

  def top_k_from_dist(self, user, embeddings, k):
    c = tf.math.softplus(self.c)
    user_emb_distance = tf.reshape(
        hyp_utils.hyp_distance_all_pairs(
            tf.reshape(user, [1, -1]), embeddings, c), [-1])
    return tf.math.top_k(-user_emb_distance, k=k)[1]

  def next_level_indices(self, current_nodes_ind, ragged_childs, user, l, k):
    if len(current_nodes_ind.numpy()) <= k:
      first_node = sum(self.nodes_per_level[:l])
      last = sum(self.nodes_per_level[:l + 1])
      return tf.gather(ragged_childs, tf.range(first_node, last)).flat_values
    else:
      nodes = self.get_nodes_level(current_nodes_ind, l)
      top_k_nodes = tf.gather(current_nodes_ind,
                              self.top_k_from_dist(user, nodes, k))
      return tf.gather(ragged_childs, top_k_nodes).flat_values

  def top_k_tree_eval(self, examples, filters, k):
    """Compute ranking-based evaluation metrics in tree setting.

    Args:
      examples: Tensor of size n_examples x 2 containing pairs' indices.
      filters: Dict representing items to skip per user for evaluation in
        the filtered setting.
      k: number of nodes to explore in a top-k tree search.

    Returns:
      Numpy array of shape (n_examples, ) containing the rank of each example.
    """
    total_examples = tf.data.experimental.cardinality(examples).numpy()
    ranks = np.ones(total_examples)
    for i, pair in enumerate(examples):
      user = self.get_users(pair[0])
      filter_out = filters[pair[0].numpy()]
      if k <= self.k_threshold:
        scores = tree_utils.top_k_to_scores(
            self.get_top_k_items(user, k + len(filter_out)), self.n_items)
      else:
        scores = tree_utils.top_k_to_scores(
            self.get_top_k_items(user, k), self.n_items)
      target = scores[pair[1]]
      scores[filter_out] = -1e6
      ranks[i] += np.sum((scores >= target))
    return ranks
