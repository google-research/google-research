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
"""TODO(tsitsulin): add headers, tests, and improve style."""
import collections

from absl import app
from absl import flags
from sklearn.metrics import normalized_mutual_info_score
import tensorflow.compat.v2 as tf

from graph_embedding.dmon.models.gcn_modularity import gcn_modularity
from graph_embedding.dmon.synthetic_data.overlapping_gaussians import circular_gaussians
from graph_embedding.dmon.utilities.graph import construct_knn_graph
from graph_embedding.dmon.utilities.graph import normalize_graph
from graph_embedding.dmon.utilities.graph import scipy_to_tf
from graph_embedding.dmon.utilities.metrics import conductance
from graph_embedding.dmon.utilities.metrics import modularity
from graph_embedding.dmon.utilities.metrics import precision
from graph_embedding.dmon.utilities.metrics import recall

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'n_nodes', 1000, 'Number of nodes for the synthetic graph.', lower_bound=0)
flags.DEFINE_integer(
    'n_clusters',
    10,
    'Number of clusters for the synthetic graph.',
    lower_bound=0)
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
  data_clean, data_dirty, labels = circular_gaussians(n_nodes, n_clusters)
  n_nodes = data_clean.shape[0]
  graph_clean_ = construct_knn_graph(data_clean)
  graph_clean_normalized_ = normalize_graph(graph_clean_, normalized=True)

  graph_clean = scipy_to_tf(graph_clean_)
  graph_clean_normalized = scipy_to_tf(graph_clean_normalized_)

  input_features = tf.keras.layers.Input(shape=(2,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

  model = gcn_modularity([input_features, input_graph, input_adjacency],
                         [64, 32, 16])

  def grad(model, inputs):
    with tf.GradientTape() as tape:
      _ = model(inputs, training=True)
      loss_value = sum(model.losses)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  model.compile(optimizer, None)

  for epoch in range(FLAGS.n_epochs):
    loss_value, grads = grad(model,
                             [data_dirty, graph_clean_normalized, graph_clean])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'epoch {epoch}, loss: {loss_value.numpy():.4f}')
  _, assignments = model([data_dirty, graph_clean_normalized, graph_clean],
                         training=False)
  clusters = assignments.numpy().argmax(axis=1)
  print('Conductance:', conductance(graph_clean_, clusters))
  print('Modularity:', modularity(graph_clean_, clusters))
  print(
      'NMI:',
      normalized_mutual_info_score(
          labels, clusters, average_method='arithmetic'))
  print('Precision:', precision(labels, clusters))
  print('Recall:', recall(labels, clusters))
  print('Cluster sizes for %d clusters:' % (len(set(clusters))))
  print(collections.Counter(clusters))


if __name__ == '__main__':
  app.run(main)
