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
import os

from absl import app
from absl import flags
from sklearn.metrics import normalized_mutual_info_score
import tensorflow.compat.v2 as tf

from graph_embedding.dmon.models.gcn_modularity import gcn_modularity
from graph_embedding.dmon.utilities.graph import load_kipf_data
from graph_embedding.dmon.utilities.graph import load_npz_to_sparse_graph
from graph_embedding.dmon.utilities.graph import normalize_graph
from graph_embedding.dmon.utilities.graph import scipy_to_tf
from graph_embedding.dmon.utilities.metrics import conductance
from graph_embedding.dmon.utilities.metrics import modularity
from graph_embedding.dmon.utilities.metrics import precision
from graph_embedding.dmon.utilities.metrics import recall

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string('graph_path', None, 'Input graph path')
flags.DEFINE_string('output_path', None, 'Output results path')
flags.DEFINE_string('architecture', None, 'Network architecture')
flags.DEFINE_string('load_strategy', 'schur', 'Graph format')
flags.DEFINE_string('postfix', '', 'File postfix')
flags.DEFINE_float(
    'orthogonality_regularization',
    1,
    'Orthogonality regularization parameter',
    lower_bound=0)
flags.DEFINE_float(
    'cluster_size_regularization',
    1,
    'Cluster size regularization parameter',
    lower_bound=0)
flags.DEFINE_float(
    'dropout_rate',
    0,
    'Orthogonality regularization parameter',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_integer('n_clusters', 16, 'Number of clusters', lower_bound=0)
flags.DEFINE_integer('n_epochs', 1000, 'Number of epochs', lower_bound=0)
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate', lower_bound=0)


def format_filename():
  graph_name = os.path.split(FLAGS.graph_path)[1]
  architecture_str = FLAGS.architecture.strip('[]')
  return (f'{FLAGS.output_path}/{graph_name}-'
          f'nclusters-{FLAGS.n_clusters}-'
          f'architecture-{architecture_str}-'
          f'ortho-{FLAGS.orthogonality_regularization}-'
          f'clustersize-{FLAGS.cluster_size_regularization}-'
          f'dropout-{FLAGS.dropout_rate}-'
          f'lr-{FLAGS.learning_rate}-'
          f'epochs-{FLAGS.n_epochs}'
          f'postfix-{FLAGS.postfix}'
          '.txt')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('Starting', format_filename())
  if FLAGS.load_strategy == 'schur':
    adjacency, features, labels, label_mask = load_npz_to_sparse_graph(
        FLAGS.graph_path)
  elif FLAGS.load_strategy == 'kipf':
    adjacency, features, labels, label_mask = load_kipf_data(
        *os.path.split(FLAGS.graph_path))
  else:
    raise Exception('Unknown loading strategy!')
  n_nodes = adjacency.shape[0]
  feature_size = features.shape[1]
  architecture = [int(x) for x in FLAGS.architecture.strip('[]').split('_')]
  architecture.append(FLAGS.n_clusters)
  graph_clean = scipy_to_tf(adjacency)
  graph_clean_normalized = scipy_to_tf(
      normalize_graph(adjacency.copy(), normalized=True))

  input_features = tf.keras.layers.Input(shape=(feature_size,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

  model = gcn_modularity(
      [input_features, input_graph, input_adjacency],
      architecture,
      dropout_rate=FLAGS.dropout_rate,
      orthogonality_regularization=FLAGS.orthogonality_regularization,
      cluster_size_regularization=FLAGS.cluster_size_regularization)

  def grad(model, inputs):
    with tf.GradientTape() as tape:
      _ = model(inputs, training=True)
      loss_value = sum(model.losses)
    return model.losses, tape.gradient(loss_value, model.trainable_variables)

  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  model.compile(optimizer, None)

  for epoch in range(FLAGS.n_epochs):
    loss_values, grads = grad(model,
                              [features, graph_clean_normalized, graph_clean])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'epoch {epoch}, losses: ' +
          ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))

  _, assignments = model([features, graph_clean_normalized, graph_clean],
                         training=False)
  assignments = assignments.numpy()
  clusters = assignments.argmax(axis=1)
  print('Conductance:', conductance(adjacency, clusters))
  print('Modularity:', modularity(adjacency, clusters))
  print(
      'NMI:',
      normalized_mutual_info_score(
          labels, clusters[label_mask], average_method='arithmetic'))
  print('Precision:', precision(labels, clusters[label_mask]))
  print('Recall:', recall(labels, clusters[label_mask]))
  with open(format_filename(), 'w') as out_file:
    print('Conductance:', conductance(adjacency, clusters), file=out_file)
    print('Modularity:', modularity(adjacency, clusters), file=out_file)
    print(
        'NMI:',
        normalized_mutual_info_score(
            labels, clusters[label_mask], average_method='arithmetic'),
        file=out_file)
    print('Precision:', precision(labels, clusters[label_mask]), file=out_file)
    print('Recall:', recall(labels, clusters[label_mask]), file=out_file)


if __name__ == '__main__':
  app.run(main)
