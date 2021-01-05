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

"""Main script for training GCN models."""

import time
import models
import numpy as np
import partition_utils
import tensorflow.compat.v1 as tf
import utils

tf.logging.set_verbosity(tf.logging.INFO)
# Set random seed
seed = 1
np.random.seed(seed)

# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('save_name', './mymodel.ckpt', 'Path for saving model')
flags.DEFINE_string('dataset', 'ppi', 'Dataset string.')
flags.DEFINE_string('data_prefix', 'data/', 'Datapath prefix.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('num_clusters', 50, 'Number of clusters.')
flags.DEFINE_integer('bsize', 1, 'Number of clusters for each batch.')
flags.DEFINE_integer('num_clusters_val', 5,
                     'Number of clusters for validation.')
flags.DEFINE_integer('num_clusters_test', 1, 'Number of clusters for test.')
flags.DEFINE_integer('num_layers', 5, 'Number of GCN layers.')
flags.DEFINE_float(
    'diag_lambda', 1,
    'A positive number for diagonal enhancement, -1 indicates normalization without diagonal enhancement'
)
flags.DEFINE_bool('multilabel', True, 'Multilabel or multiclass.')
flags.DEFINE_bool('layernorm', True, 'Whether to use layer normalization.')
flags.DEFINE_bool(
    'precalc', True,
    'Whether to pre-calculate the first layer (AX preprocessing).')
flags.DEFINE_bool('validation', True,
                  'Print validation accuracy after each epoch.')


def load_data(data_prefix, dataset_str, precalc):
  """Return the required data formats for GCN models."""
  (num_data, train_adj, full_adj, feats, train_feats, test_feats, labels,
   train_data, val_data,
   test_data) = utils.load_graphsage_data(data_prefix, dataset_str)
  visible_data = train_data

  y_train = np.zeros(labels.shape)
  y_val = np.zeros(labels.shape)
  y_test = np.zeros(labels.shape)
  y_train[train_data, :] = labels[train_data, :]
  y_val[val_data, :] = labels[val_data, :]
  y_test[test_data, :] = labels[test_data, :]

  train_mask = utils.sample_mask(train_data, labels.shape[0])
  val_mask = utils.sample_mask(val_data, labels.shape[0])
  test_mask = utils.sample_mask(test_data, labels.shape[0])

  if precalc:
    train_feats = train_adj.dot(feats)
    train_feats = np.hstack((train_feats, feats))
    test_feats = full_adj.dot(feats)
    test_feats = np.hstack((test_feats, feats))

  return (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
          train_mask, val_mask, test_mask, train_data, val_data, test_data,
          num_data, visible_data)


# Define model evaluation function
def evaluate(sess, model, val_features_batches, val_support_batches,
             y_val_batches, val_mask_batches, val_data, placeholders):
  """evaluate GCN model."""
  total_pred = []
  total_lab = []
  total_loss = 0
  total_acc = 0

  num_batches = len(val_features_batches)
  for i in range(num_batches):
    features_b = val_features_batches[i]
    support_b = val_support_batches[i]
    y_val_b = y_val_batches[i]
    val_mask_b = val_mask_batches[i]
    num_data_b = np.sum(val_mask_b)
    if num_data_b == 0:
      continue
    else:
      feed_dict = utils.construct_feed_dict(features_b, support_b, y_val_b,
                                            val_mask_b, placeholders)
      outs = sess.run([model.loss, model.accuracy, model.outputs],
                      feed_dict=feed_dict)

    total_pred.append(outs[2][val_mask_b])
    total_lab.append(y_val_b[val_mask_b])
    total_loss += outs[0] * num_data_b
    total_acc += outs[1] * num_data_b

  total_pred = np.vstack(total_pred)
  total_lab = np.vstack(total_lab)
  loss = total_loss / len(val_data)
  acc = total_acc / len(val_data)

  micro, macro = utils.calc_f1(total_pred, total_lab, FLAGS.multilabel)
  return loss, acc, micro, macro


def main(unused_argv):
  """Main function for running experiments."""
  # Load data
  (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
   train_mask, val_mask, test_mask, _, val_data, test_data, num_data,
   visible_data) = load_data(FLAGS.data_prefix, FLAGS.dataset, FLAGS.precalc)

  # Partition graph and do preprocessing
  if FLAGS.bsize > 1:
    _, parts = partition_utils.partition_graph(train_adj, visible_data,
                                               FLAGS.num_clusters)
    parts = [np.array(pt) for pt in parts]
  else:
    (parts, features_batches, support_batches, y_train_batches,
     train_mask_batches) = utils.preprocess(train_adj, train_feats, y_train,
                                            train_mask, visible_data,
                                            FLAGS.num_clusters,
                                            FLAGS.diag_lambda)

  (_, val_features_batches, val_support_batches, y_val_batches,
   val_mask_batches) = utils.preprocess(full_adj, test_feats, y_val, val_mask,
                                        np.arange(num_data),
                                        FLAGS.num_clusters_val,
                                        FLAGS.diag_lambda)

  (_, test_features_batches, test_support_batches, y_test_batches,
   test_mask_batches) = utils.preprocess(full_adj, test_feats, y_test,
                                         test_mask, np.arange(num_data),
                                         FLAGS.num_clusters_test,
                                         FLAGS.diag_lambda)
  idx_parts = list(range(len(parts)))

  # Some preprocessing
  model_func = models.GCN

  # Define placeholders
  placeholders = {
      'support':
          tf.sparse_placeholder(tf.float32),
      'features':
          tf.placeholder(tf.float32),
      'labels':
          tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
      'labels_mask':
          tf.placeholder(tf.int32),
      'dropout':
          tf.placeholder_with_default(0., shape=()),
      'num_features_nonzero':
          tf.placeholder(tf.int32)  # helper variable for sparse dropout
  }

  # Create model
  model = model_func(
      placeholders,
      input_dim=test_feats.shape[1],
      logging=True,
      multilabel=FLAGS.multilabel,
      norm=FLAGS.layernorm,
      precalc=FLAGS.precalc,
      num_layers=FLAGS.num_layers)

  # Initialize session
  sess = tf.Session()
  tf.set_random_seed(seed)

  # Init variables
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  cost_val = []
  total_training_time = 0.0
  # Train model
  for epoch in range(FLAGS.epochs):
    t = time.time()
    np.random.shuffle(idx_parts)
    if FLAGS.bsize > 1:
      (features_batches, support_batches, y_train_batches,
       train_mask_batches) = utils.preprocess_multicluster(
           train_adj, parts, train_feats, y_train, train_mask,
           FLAGS.num_clusters, FLAGS.bsize, FLAGS.diag_lambda)
      for pid in range(len(features_batches)):
        # Use preprocessed batch data
        features_b = features_batches[pid]
        support_b = support_batches[pid]
        y_train_b = y_train_batches[pid]
        train_mask_b = train_mask_batches[pid]
        # Construct feed dictionary
        feed_dict = utils.construct_feed_dict(features_b, support_b, y_train_b,
                                              train_mask_b, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy],
                        feed_dict=feed_dict)
    else:
      np.random.shuffle(idx_parts)
      for pid in idx_parts:
        # Use preprocessed batch data
        features_b = features_batches[pid]
        support_b = support_batches[pid]
        y_train_b = y_train_batches[pid]
        train_mask_b = train_mask_batches[pid]
        # Construct feed dictionary
        feed_dict = utils.construct_feed_dict(features_b, support_b, y_train_b,
                                              train_mask_b, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy],
                        feed_dict=feed_dict)

    total_training_time += time.time() - t
    print_str = 'Epoch: %04d ' % (epoch + 1) + 'training time: {:.5f} '.format(
        total_training_time) + 'train_acc= {:.5f} '.format(outs[2])

    # Validation
    if FLAGS.validation:
      cost, acc, micro, macro = evaluate(sess, model, val_features_batches,
                                         val_support_batches, y_val_batches,
                                         val_mask_batches, val_data,
                                         placeholders)
      cost_val.append(cost)
      print_str += 'val_acc= {:.5f} '.format(
          acc) + 'mi F1= {:.5f} ma F1= {:.5f} '.format(micro, macro)

    tf.logging.info(print_str)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
        cost_val[-(FLAGS.early_stopping + 1):-1]):
      tf.logging.info('Early stopping...')
      break

  tf.logging.info('Optimization Finished!')

  # Save model
  saver.save(sess, FLAGS.save_name)

  # Load model (using CPU for inference)
  with tf.device('/cpu:0'):
    sess_cpu = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    sess_cpu.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess_cpu, FLAGS.save_name)
    # Testing
    test_cost, test_acc, micro, macro = evaluate(
        sess_cpu, model, test_features_batches, test_support_batches,
        y_test_batches, test_mask_batches, test_data, placeholders)
    print_str = 'Test set results: ' + 'cost= {:.5f} '.format(
        test_cost) + 'accuracy= {:.5f} '.format(
            test_acc) + 'mi F1= {:.5f} ma F1= {:.5f}'.format(micro, macro)
    tf.logging.info(print_str)


if __name__ == '__main__':
  tf.app.run(main)
