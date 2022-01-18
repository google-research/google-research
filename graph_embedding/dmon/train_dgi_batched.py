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
"""TODO(tsitsulin): add headers, tests, and improve style."""
from absl import app
from absl import flags
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
import tensorflow.compat.v2 as tf

from graph_embedding.dmon.layers.gcn import GCN
from graph_embedding.dmon.models.dgi import deep_graph_infomax
from graph_embedding.dmon.synthetic_data.graph_util import construct_knn_graph
from graph_embedding.dmon.synthetic_data.overlapping_gaussians import line_gaussians
from graph_embedding.dmon.utilities.batching import make_batch
from graph_embedding.dmon.utilities.batching import random_batch
from graph_embedding.dmon.utilities.shuffling import shuffle_inbatch

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'n_nodes', 1000, 'Number of nodes for the synthetic graph.', lower_bound=0)
flags.DEFINE_integer(
    'n_clusters',
    2,
    'Number of clusters for the synthetic graph.',
    lower_bound=0)
flags.DEFINE_integer(
    'batch_size', 16, 'Batch size to use for training.', lower_bound=0)
flags.DEFINE_float(
    'train_size', 0.2, 'Training data proportion.', lower_bound=0)
flags.DEFINE_integer(
    'n_epochs', 200, 'Number of epochs to train.', lower_bound=0)
flags.DEFINE_float(
    'learning_rate', 0.01, 'Optimizer\'s learning rate.', lower_bound=0)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('Bröther may i have some self-lööps')
  n_nodes = FLAGS.n_nodes
  n_clusters = FLAGS.n_clusters
  train_size = FLAGS.train_size
  batch_size = FLAGS.batch_size
  data_clean, data_dirty, labels = line_gaussians(n_nodes, n_clusters)
  graph_clean = construct_knn_graph(data_clean)
  n_neighbors = [15, 10]  # TODO(tsitsulin): move to FLAGS.
  total_matrix_size = 1 + np.cumprod(n_neighbors).sum()

  train_mask = np.zeros(n_nodes, dtype=np.bool)
  train_mask[np.random.choice(
      np.arange(n_nodes), int(n_nodes * train_size), replace=False)] = True
  test_mask = ~train_mask
  print(
      f'Data shape: {data_clean.shape}, graph shape: {graph_clean.shape}'
  )
  print(f'Train size: {train_mask.sum()}, test size: {test_mask.sum()}')

  input_features = tf.keras.layers.Input(shape=(
      total_matrix_size,
      2,
  ))
  input_features_corrupted = tf.keras.layers.Input(
      shape=(
          total_matrix_size,
          2,
      ))
  input_graph = tf.keras.layers.Input((
      total_matrix_size,
      total_matrix_size,
  ))

  encoder = [GCN(64), GCN(32), tf.keras.layers.Lambda(lambda x: x[0][:, 0, :])]
  model = deep_graph_infomax(
      [input_features, input_features_corrupted, input_graph], encoder)

  def loss(model, x, y, training):
    _, y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)

  def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
      loss_value = loss(model, inputs, targets, training=True)
      for loss_internal in model.losses:
        loss_value += loss_internal
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

  labels_dgi = tf.concat([tf.zeros([batch_size, 1]),
                          tf.ones([batch_size, 1])], 0)
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

  for epoch in range(FLAGS.n_epochs):
    subgraph_mat, features_mat, _, nonzero_indices = random_batch(
        graph_clean, data_dirty, batch_size, n_neighbors)
    perc_shuffle = 1  # np.linspace(1, 0.25, max_epoch)[epoch]
    features_corrupted = shuffle_inbatch(features_mat, nonzero_indices,
                                         perc_shuffle)
    loss_value, grads = grad(model,
                             [features_mat, features_corrupted, subgraph_mat],
                             labels_dgi)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(
        f'epoch {epoch}, loss: {loss_value.numpy():.4f}, shuffle %: {100*perc_shuffle:.2f}'
    )
  subgraph_mat, features_mat, _ = make_batch(graph_clean, data_dirty,
                                             np.arange(n_nodes), n_neighbors)
  representations, _ = model([features_mat, features_mat, subgraph_mat],
                             training=False)
  representations = representations.numpy()
  clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
  clf.fit(representations[train_mask], labels[train_mask])
  clusters = clf.predict(representations[test_mask])
  print(
      'NMI:',
      normalized_mutual_info_score(
          labels[test_mask], clusters, average_method='arithmetic'))
  print('Accuracy:', 100 * accuracy_score(labels[test_mask], clusters))


if __name__ == '__main__':
  app.run(main)
