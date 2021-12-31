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

# Lint as: python3
r"""DDGK: Learning Graph Representations for Deep Divergence Graph Kernels.

===============================
This is the implementation of the WWW 2019 paper,
[DDGK: Learning Graph Representations for Deep Divergence Graph Kernels]
(https://ai.google/research/pubs/pub47867).

The included code creates a Deep Divergence Graph Kernel as introduced in the
paper. The implementation makes use of the data sets collected here.

A distributed version of this implementation is necessary for large data sets.
------
If you find Deep Divergence Graph Kernels useful in your research, we ask that
you cite the following paper:
> Al-Rfou, R., Zelle, D., Perozzi, B., (2019).
> DDGK: Learning Graph Representations for Deep Divergence Graph Kernels.
> In _The Web Conference_.
Example execution
------
# From google-research/
python3 -m graph_embedding.ddgk.main --data_set=${ds} --working_dir=~/tmp

Where ${ds} is a data set as formatted [here]
(https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)
"""
import networkx as nx
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import training as contrib_training


def MutagHParams():
  return contrib_training.HParams(
      # The size of node embeddings.
      embedding_size=4,
      # The number of layers in the DNN.
      num_dnn_layers=4,
      # The window to average for scoring loss and accuracy calculation.
      score_window=10,
      # The adam learning rate
      learning_rate=.01,
      # The steps for training.
      train_num_epochs=600,
      # The steps for node mapping and scoring.
      score_num_epochs=600,
      # The label preserving loss for node mapping.
      node_label_loss_coefficient=.0,
      # The number of node labels.
      num_node_labels=7,
      # The label preserving loss for indicent edges.
      incident_label_loss_coefficient=.0,
      # The number of edge labels.
      num_edge_labels=4)


def AdjMatrixAccuracy(logits, labels):
  predictions = tf.cast(tf.greater(tf.sigmoid(logits), .5), tf.float64)
  accuracies = tf.cast(tf.equal(predictions, labels), tf.float64)

  return tf.reduce_mean(accuracies)  # Report accuracy per edge


def LogitsFromProb(prob):
  return tf.log(tf.clip_by_value(prob, 1e-12, 1.0))


def ProbFromCounts(counts):
  return counts / tf.clip_by_value(
      tf.reduce_sum(counts, axis=1, keepdims=True), 1e-9, 1e9)


def AdjMatrixLoss(logits, labels):
  losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  return tf.reduce_mean(losses)  # Report loss per edge


def NodesLabels(graph, num_labels):
  # labels size is (graph_num_node, 1)
  labels = [graph.nodes[n]['label'] for n in graph.nodes()]
  # labels size is (graph_num_node, num_labels)
  labels = tf.one_hot(labels, num_labels, dtype=tf.float64)
  return ProbFromCounts(labels)


def NeighborNodesLabels(graph, num_labels):
  neighbors_labels = np.zeros((graph.number_of_nodes(), num_labels))

  for v in graph.nodes():
    for u in graph.neighbors(v):
      neighbors_labels[v, graph.nodes[u]['label']] += 1.0

  return ProbFromCounts(neighbors_labels)


def EdgesLabels(graph, num_labels):
  labels = np.zeros((graph.number_of_nodes(), num_labels))

  for i, n in enumerate(graph.nodes()):
    for u, v in graph.edges(n):
      labels[i, graph[u][v]['label']] += 1.0

  return ProbFromCounts(labels)


def NeighborEdgesLabels(graph, num_labels):
  labels = np.zeros((graph.number_of_nodes(), num_labels))

  for i, v in enumerate(graph.nodes()):
    for u in graph.neighbors(v):
      for v1, v2, d in graph.edges(u, data=True):
        if v not in (v1, v2):
          labels[i, d['label']] += 1.0

  return ProbFromCounts(labels)


def NodeLabelLoss(source, source_node_prob, target, num_labels):
  # source_labels size is (source_num_node, num_labels)
  source_labels = NodesLabels(source, num_labels)

  # target_labels size is (target_num_node, num_labels)
  target_labels = NodesLabels(target, num_labels)
  # We take the log because the result of the multiplication is already a
  # probability.
  logits = LogitsFromProb(tf.matmul(source_node_prob, source_labels))
  losses = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=target_labels, logits=logits)

  # Report error per node.
  return tf.reduce_mean(losses)


def NeighborNodesLabelLoss(target_neighbors_prob, target, num_labels):
  target_labels = NodesLabels(target, num_labels)

  target_neighbors_labels = NeighborNodesLabels(target, num_labels)

  logits = LogitsFromProb(tf.matmul(target_neighbors_prob, target_labels))
  losses = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=target_neighbors_labels, logits=logits)

  # Report error per node.
  return tf.reduce_mean(losses)


def EdgeLabelLoss(source, source_node_prob, target, num_labels):
  source_labels = EdgesLabels(source, num_labels)
  target_labels = EdgesLabels(target, num_labels)

  logits = LogitsFromProb(tf.matmul(source_node_prob, source_labels))
  losses = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=target_labels, logits=logits)

  return tf.reduce_mean(losses)


def NeighborEdgesLabelsLoss(target_neighbors_prob, target, num_labels):
  target_labels = EdgesLabels(target, num_labels)
  target_neighbors_labels = NeighborEdgesLabels(target, num_labels)

  logits = LogitsFromProb(tf.matmul(target_neighbors_prob, target_labels))
  losses = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=target_neighbors_labels, logits=logits)

  return tf.reduce_mean(losses)


def Encode(source, ckpt_prefix, hparams):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  tf.reset_default_graph()

  g = tf.Graph()
  session = tf.Session(graph=g)

  with g.as_default(), session.as_default():
    A = nx.adjacency_matrix(source, weight=None)

    x = tf.one_hot(
        list(source.nodes()), source.number_of_nodes(), dtype=tf.float64)
    y = tf.convert_to_tensor(A.todense(), dtype=tf.float64)

    layer = tf.layers.dense(x, hparams.embedding_size, use_bias=False)
    for _ in range(hparams.num_dnn_layers):
      layer = tf.layers.dense(
          layer, hparams.embedding_size * 4, activation=tf.nn.tanh)
    logits = tf.layers.dense(
        layer, source.number_of_nodes(), activation=tf.nn.tanh)

    loss = AdjMatrixLoss(logits, y)

    train_op = contrib_training.create_train_op(
        loss,
        tf.train.AdamOptimizer(hparams.learning_rate),
        summarize_gradients=False)

    session.run(tf.global_variables_initializer())

    for _ in range(hparams.train_num_epochs):
      session.run(train_op)

    tf.train.Saver(tf.trainable_variables()).save(session, ckpt_prefix)


def Score(source, target, ckpt_prefix, hparams):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  tf.reset_default_graph()

  g = tf.Graph()
  session = tf.Session(graph=g)

  with g.as_default(), session.as_default():
    A = nx.adjacency_matrix(target, weight=None)

    x = tf.one_hot(
        list(target.nodes()), target.number_of_nodes(), dtype=tf.float64)
    y = tf.convert_to_tensor(A.todense(), dtype=tf.float64)

    with tf.variable_scope('attention'):
      attention = tf.layers.dense(x, source.number_of_nodes(), use_bias=False)
      source_node_prob = tf.nn.softmax(attention)

    layer = tf.layers.dense(
        source_node_prob, hparams.embedding_size, use_bias=False)
    for _ in range(hparams.num_dnn_layers):
      layer = tf.layers.dense(
          layer, hparams.embedding_size * 4, activation=tf.nn.tanh)
    logits = tf.layers.dense(
        layer, source.number_of_nodes(), activation=tf.nn.tanh)

    with tf.variable_scope('attention_reverse'):
      attention_reverse = tf.layers.dense(logits, target.number_of_nodes())
      target_neighbors_pred = tf.nn.sigmoid(attention_reverse)
      target_neighbors_prob = ProbFromCounts(target_neighbors_pred)

    loss = AdjMatrixLoss(attention_reverse, y)

    if hparams.get('node_label_loss_coefficient', None):
      label_loss = NodeLabelLoss(source, source_node_prob, target,
                                 hparams.num_node_labels)
      label_loss += NeighborNodesLabelLoss(target_neighbors_prob, target,
                                           hparams.num_node_labels)
      loss += label_loss * hparams.node_label_loss_coefficient

    if hparams.get('incident_label_loss_coefficient', None):
      edge_loss = EdgeLabelLoss(source, source_node_prob, target,
                                hparams.num_edge_labels)
      edge_loss += NeighborEdgesLabelsLoss(target_neighbors_prob, target,
                                           hparams.num_edge_labels)
      loss += edge_loss * hparams.incident_label_loss_coefficient

    vars_to_restore = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='(?!attention)')
    vars_to_train = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='attention')

    train_op = contrib_training.create_train_op(
        loss,
        tf.train.AdamOptimizer(hparams.learning_rate),
        variables_to_train=vars_to_train,
        summarize_gradients=False)

    session.run(tf.global_variables_initializer())

    tf.train.Saver(vars_to_restore).restore(session, ckpt_prefix)

    losses = []

    for _ in range(hparams.score_num_epochs):
      losses.append(session.run([train_op, loss])[1])

  return losses[-hparams.score_window:]
