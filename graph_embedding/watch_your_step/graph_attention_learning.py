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

"""Graph attention method to learn the correct context over random walks.

Reference implementation for the NIPS 2018 paper:

Watch Your Step: Learning Graph Embeddings Through Attention
Sami Abu-El-Haija, Bryan Perozzi, Rami Al-Rfou, Alex Alemi
https://arxiv.org/abs/1710.09599

Example Usage:
==============
1. First, install relevant requirements

    # From google-research/
    pip install -r graph_embedding/watch_your_step/requirements.txt

2. Second download datasets from [Abu-El-Haija et al, CIKM'17]:

    # From google-research/
    curl http://sami.haija.org/graph/datasets.tgz > datasets.tgz
    tar zxvf datasets.tgz
    export DATA_DIR=~datasets

** Second, run the code:

    # From google-research/
    python -m graph_embedding.watch_your_step.graph_attention_learning --dataset_dir ${DATA_DIR}/wiki-vote   #pylint: disable=line-too-long

To save the output, please use --output_dir. Consider other flags for options.
Output file will contain train/test metrics, embeddings, as well as learned
context distributions.
"""

import json
import os
import pickle

from absl import app
from absl import flags
from absl import logging

import numpy
from sklearn import metrics
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as contrib_slim

flags.DEFINE_integer('max_number_of_steps', 100,
                     'The maximum number of gradient steps.')
flags.DEFINE_float('learning_rate', 0.2, 'PercentDelta learning rate.')
flags.DEFINE_string(
    'dataset_dir', None,
    'Directory where all dataset files live. All data files '
    'must be located here. Including {train,test}.txt.npy and '
    '{train,test}.neg.txt.npy. ')
flags.mark_flag_as_required('dataset_dir')

flags.DEFINE_string('output_dir', None,
                    'If set, output metrics will be written.')

flags.DEFINE_integer('d', 4, 'embedding dimensions')

flags.DEFINE_integer(
    'transition_powers', 5,
    'Highest power of normalized adjacency (transition) '
    'matrix.')

flags.DEFINE_float(
    'context_regularizer', 0.1,
    'Regularization co-efficient to the context distribution '
    'parameters.')

flags.DEFINE_string('objective', 'nlgl',
                    'Choices are "rmse" or "nlgl" (neg. Log Graph Likelihood)')

flags.DEFINE_bool(
    'share_embeddings', False,
    'If set, left and right embedding dictionary will be shared.')

FLAGS = flags.FLAGS

NUM_NODES = 0
IS_DIRECTED = None


def IsDirected():
  global IS_DIRECTED
  if IS_DIRECTED is not None:
    return IS_DIRECTED
  IS_DIRECTED = os.path.exists(
      os.path.join(FLAGS.dataset_dir, 'test.directed.neg.txt.npy'))
  return IS_DIRECTED


def GetNumNodes():
  global NUM_NODES
  if NUM_NODES == 0:
    index = pickle.load(
        open(os.path.join(FLAGS.dataset_dir, 'index.pkl'), 'rb'))
    NUM_NODES = len(index['index'])
  return NUM_NODES


def Description():
  return 'ds.%s.e.%i.o.%s' % (os.path.basename(FLAGS.dataset_dir), FLAGS.d,
                              FLAGS.objective)


def GetOrMakeAdjacencyMatrix():
  """Creates Adjacency matrix and caches it on disk with name a.npy."""
  a_file = os.path.join(FLAGS.dataset_dir, 'a.npy')
  if os.path.exists(a_file):
    return numpy.load(open(a_file, 'rb'))

  num_nodes = GetNumNodes()
  a = numpy.zeros(shape=(num_nodes, num_nodes), dtype='float32')
  train_edges = numpy.load(
      open(os.path.join(FLAGS.dataset_dir, 'train.txt.npy'), 'rb'))
  a[train_edges[:, 0], train_edges[:, 1]] = 1.0
  if not IsDirected():
    a[train_edges[:, 1], train_edges[:, 0]] = 1.0

  numpy.save(open(a_file, 'wb'), a)
  return a


def GetPowerTransitionPairs(highest_power):
  return list(IterPowerTransitionPairs(highest_power))


def IterPowerTransitionPairs(highest_power):
  """Yields powers of transition matrix (T, T*T, T*T*T, ...).

  It caches them on disk as t_<i>.npy, where <i> is the power. The first power
  (i = 1) is not cached as it is trivially computed from the adjacency matrix.

  Args:
    highest_power: integer representing the highest power of the transition
      matrix. This will be the number of yields.
  """
  num_nodes = GetNumNodes()

  for i in range(highest_power):
    if i == 0:
      a = GetOrMakeAdjacencyMatrix()
      transition = a.T
      degree = transition.sum(axis=0)
      transition /= degree + 0.0000001
      power_array = transition
    else:
      power_filename = os.path.join(FLAGS.dataset_dir, 't_%i.npy' % (i + 1))
      if os.path.exists(power_filename):
        power_array = numpy.load(open(power_filename, 'rb'))
      else:
        power_array = power_array.dot(transition)
        print('Computing T^%i  ...' % (i + 1))  # pylint: disable=superfluous-parens
        numpy.save(open(power_filename, 'wb'), power_array)
        print(' ... Saved T^%i' % (i + 1))  # pylint: disable=superfluous-parens

    placeholder = tf.placeholder(tf.float32, shape=(num_nodes, num_nodes))
    yield (placeholder, power_array)


def GetParametrizedExpectation(references):
  r"""Calculates E[D; q_1, q_2, ...]: a parametrized (tensor) matrix D.

  Which is defined as:
  E[D; q] = P_0 * (Q_1*T + Q_2*T^2 + Q_3*T^3 + ...)

  where Q_1, Q_2, ... = softmax(q_1, q_2, ...)
  and vector (q_1, q_2, ...) is created as a "trainable variable".

  Args:
    references: Dict that will be populated as key-value pairs:
      'combination': \sum_j Q_j T^j (i.e. E[D] excluding P_0).
      'normed': The vector Q_1, Q_2, ... (sums to 1).
      'mults': The vector q_1, q_2, ... (Before softmax, does not sum to 1).

  Returns:
    Tuple (E[D; q], feed_dict) where the first entry contains placeholders and
    the feed_dict contains is a dictionary from the placeholders to numpy arrays
    of the transition powers.
  """
  feed_dict = {}
  n = FLAGS.transition_powers
  regularizer = FLAGS.context_regularizer
  a = GetOrMakeAdjacencyMatrix()
  transition = a.T
  degree = transition.sum(axis=0)

  # transition /= degree + 0.0000001
  # transition_pow_n = transition
  convex_combination = []

  # vector q
  mults = tf.Variable(numpy.ones(shape=(n), dtype='float32'))
  # vector Q (output of softmax)
  normed = tf.squeeze(tf.nn.softmax(tf.expand_dims(mults, 0)), 0)

  references['mults'] = mults
  references['normed'] = normed
  transition_powers = GetPowerTransitionPairs(n)
  for k, (placeholder, transition_pow) in enumerate(transition_powers):
    feed_dict[placeholder] = transition_pow
    convex_combination.append(normed[k] * placeholder)
  d_sum = tf.add_n(convex_combination)

  d_sum *= degree

  tf.losses.add_loss(tf.reduce_mean(mults**2) * regularizer)

  references['combination'] = convex_combination
  return tf.transpose(d_sum) * GetNumNodes() * 80, feed_dict


# Helper function 1/3 for PercentDelta.
def GetPD(target_num_steps):
  global_step = tf.train.get_or_create_global_step()
  global_step = tf.cast(global_step, tf.float32)
  # gs = 0,         target = 1
  # gs = num_steps, target = 0.01
  # Solve: y = mx + c
  # gives: c = 1
  #        m = dy / dx = (1 - 0.01) / (0 - num_steps) = - 0.99 / num_steps
  # Therefore, y = 1 - (0.99/num_steps) * x
  return -global_step * 0.99 / target_num_steps + 1


# Helper function 2/3 for PercentDelta.
def PlusEpsilon(x, eps=1e-5):
  """Returns x+epsilon, without changing element-wise sign of x."""
  return x + (tf.cast(x < 0, tf.float32) * -eps) + (
      tf.cast(x >= 0, tf.float32) * eps)


# Helper function 3/3 for PercentDelta.
def CreateGradMultipliers(loss):
  """Returns a gradient multiplier so that SGD becomes PercentDelta."""
  variables = tf.trainable_variables()  # tf.global_variables()
  gradients = tf.gradients(loss, variables)
  multipliers = {}
  target_pd = GetPD(FLAGS.max_number_of_steps)
  for v, g in zip(variables, gradients):
    if g is None:
      continue
    multipliers[v] = target_pd / PlusEpsilon(
        tf.reduce_mean(tf.abs(g / PlusEpsilon(v))))
  return multipliers


def CreateEmbeddingDictionary(side, size):
  num_nodes = GetNumNodes()
  embeddings = numpy.array(
      numpy.random.uniform(low=-0.1, high=0.1, size=(num_nodes, size)),
      dtype='float32')
  embeddings = tf.Variable(embeddings, name=side + 'E')
  tf.losses.add_loss(tf.reduce_mean(embeddings**2) * 1e-6)
  return embeddings


def CreateObjective(g, target_matrix):
  """Returns the objective function (can be nlgl or rmse)."""
  if FLAGS.objective == 'nlgl':  # Negative log likelihood
    # target_matrix is E[D; q], which is used in the "positive part" of the
    # likelihood objective. We use true adjacency for the "negative part", as
    # described in our paper.
    true_adjacency = tf.Variable(
        GetOrMakeAdjacencyMatrix(), name='adjacency', trainable=False)
    logistic = tf.sigmoid(g)
    return -tf.reduce_mean(
        tf.multiply(target_matrix, tf.log(PlusEpsilon(logistic))) +
        tf.multiply(1 - true_adjacency, tf.log(PlusEpsilon(1 - logistic))))
  elif FLAGS.objective == 'rmse':  # Root mean squared error
    return tf.reduce_mean((g - target_matrix)**2)
  else:
    logging.fatal('unknown objective "%s".', FLAGS.objective)


def CreateGFn(net_l, net_r):
  return tf.matmul(net_l, tf.transpose(net_r))


def LogMsg(msg):
  logging.info(msg)


def Write(eval_metrics):
  if FLAGS.output_dir:
    out_json = os.path.join(FLAGS.output_dir, Description() + '.json')
    open(out_json, 'w').write(json.dumps(eval_metrics))


BEST_EVAL = None
BEST_TF_PARAMS = None


def RunEval(sess, g, test_pos_arr, test_neg_arr, train_pos_arr, train_neg_arr,
            i, v_total_loss, v_objective_loss, eval_metrics, feed_dict):
  """Calls sess.run(g) and computes AUC metric for test and train."""
  scores = sess.run(g, feed_dict=feed_dict)

  # Compute test auc:
  test_pos_prods = scores[test_pos_arr[:, 0], test_pos_arr[:, 1]]
  test_neg_prods = scores[test_neg_arr[:, 0], test_neg_arr[:, 1]]
  test_y = [0] * len(test_neg_prods) + [1] * len(test_pos_prods)
  test_y_pred = numpy.concatenate([test_neg_prods, test_pos_prods], 0)
  test_auc = metrics.roc_auc_score(test_y, test_y_pred)

  # Compute train auc:
  train_pos_prods = scores[train_pos_arr[:, 0], train_pos_arr[:, 1]]
  train_neg_prods = scores[train_neg_arr[:, 0], train_neg_arr[:, 1]]
  train_y = [0] * len(train_neg_prods) + [1] * len(train_pos_prods)
  train_y_pred = numpy.concatenate([train_neg_prods, train_pos_prods], 0)
  train_auc = metrics.roc_auc_score(train_y, train_y_pred)

  LogMsg('@%i test/train auc=%f/%f obj.loss=%f total.loss=%f' %
         (i, test_auc, train_auc, v_objective_loss, v_total_loss))

  # Populate metrics.
  eval_metrics['train auc'].append(float(train_auc))
  eval_metrics['test auc'].append(float(test_auc))
  eval_metrics['i'].append(i)
  eval_metrics['total loss'].append(float(v_total_loss))
  eval_metrics['objective loss'].append(float(v_objective_loss))
  if train_auc > eval_metrics['best train auc']:
    eval_metrics['best train auc'] = float(train_auc)
    eval_metrics['test auc at best train'] = float(test_auc)
    eval_metrics['i at best train'] = i

  return train_auc


def main(argv=()):
  del argv  # Unused.

  if FLAGS.output_dir and not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  references = {}
  net_l = CreateEmbeddingDictionary('L', FLAGS.d)

  if FLAGS.share_embeddings:
    net_r = net_l
  else:
    net_r = CreateEmbeddingDictionary('R', FLAGS.d)

  g = CreateGFn(net_l, net_r)

  target_matrix, feed_dict = GetParametrizedExpectation(references)
  if not isinstance(target_matrix, tf.Tensor):
    target_matrix = tf.Variable(target_matrix, name='target', trainable=False)

  objective_loss = CreateObjective(g, target_matrix)
  tf.losses.add_loss(objective_loss)
  loss = tf.losses.get_total_loss()

  optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
  # Set up training.
  grad_mults = CreateGradMultipliers(loss)
  train_op = contrib_slim.learning.create_train_op(
      loss, optimizer, gradient_multipliers=grad_mults)

  if IsDirected():
    test_neg_file = os.path.join(FLAGS.dataset_dir, 'test.directed.neg.txt.npy')
    test_neg_arr = numpy.load(open(test_neg_file, 'rb'))
  else:
    test_neg_file = os.path.join(FLAGS.dataset_dir, 'test.neg.txt.npy')
    test_neg_arr = numpy.load(open(test_neg_file, 'rb'))
  test_pos_file = os.path.join(FLAGS.dataset_dir, 'test.txt.npy')
  test_pos_arr = numpy.load(open(test_pos_file, 'rb'))

  train_pos_file = os.path.join(FLAGS.dataset_dir, 'train.txt.npy')
  train_neg_file = os.path.join(FLAGS.dataset_dir, 'train.neg.txt.npy')
  train_pos_arr = numpy.load(open(train_pos_file, 'rb'))
  train_neg_arr = numpy.load(open(train_neg_file, 'rb'))

  sess = tf.Session()
  coord = tf.train.Coordinator()
  tf.train.start_queue_runners(coord=coord, sess=sess)
  sess.run(tf.global_variables_initializer())

  eval_metrics = {
      'train auc': [],
      'test auc': [],
      'i': [],
      'i at best train': 0,
      'best train auc': 0,
      'test auc at best train': 0,
      'total loss': [],
      'objective loss': [],
      'mults': [],
      'normed_mults': [],
  }
  # IPython.embed()

  all_variables = tf.trainable_variables() + (
      [tf.train.get_or_create_global_step()])
  best_train_values = None
  best_train_auc = 0

  for i in range(FLAGS.max_number_of_steps):
    # import pdb; pdb.set_trace()
    _, v_total_loss, v_objective_loss = sess.run(
        (train_op, loss, objective_loss), feed_dict=feed_dict)
    if 'update' in references:
      references['update'](sess)

    if i % 4 == 0:  # Compute eval every 4th step.
      train_auc = RunEval(sess, g, test_pos_arr, test_neg_arr, train_pos_arr,
                          train_neg_arr, i, v_total_loss, v_objective_loss,
                          eval_metrics, feed_dict)
      if 'mults' in references:
        mults, normed_mults = sess.run((references['mults'],
                                        references['normed']))
        eval_metrics['mults'].append(list(map(float, list(mults))))
        eval_metrics['normed_mults'].append(
            list(map(float, list(normed_mults))))

      if train_auc > best_train_auc:  # Found new best.
        best_train_auc = train_auc

        # Memorize variables.
        best_train_values = sess.run(all_variables)

    if i % 100 == 0:
      Write(eval_metrics)

    if i - 100 > eval_metrics['i at best train']:
      LogMsg('Reached peak a while ago. Terminating...')
      break

  Write(eval_metrics)

  if FLAGS.output_dir:
    # Write trained parameters.
    last_params = os.path.join(FLAGS.output_dir, Description() + '.last.pkl')
    best_params = os.path.join(FLAGS.output_dir, Description() + '.best.pkl')
    names = [v.name for v in all_variables]

    last_train_values = sess.run(all_variables)

    pickle.dump(list(zip(names, last_train_values)), open(last_params, 'wb'))
    pickle.dump(list(zip(names, best_train_values)), open(best_params, 'wb'))

  return 0


if __name__ == '__main__':
  app.run(main)
