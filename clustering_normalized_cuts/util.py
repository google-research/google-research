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

"""Contains various utility functions used in the models."""
from __future__ import division

from __future__ import print_function
import math

from munkres import Munkres
import numpy as np
import sklearn.metrics
from sklearn.neighbors import NearestNeighbors
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import Callback


def make_batches(size, batch_size):
  """generates a list of tuples for batching data.

  generates a list of (start_idx, end_idx) tuples for batching data
  of the given size and batch_size

  Args:
    size:       size of the data to create batches for
    batch_size: batch size

  Returns:
  list of tuples of indices for data
  """
  num_batches = (size + batch_size - 1) // batch_size  # round up
  return [(i * batch_size, min(size, (i + 1) * batch_size))
          for i in range(num_batches)]


def train_gen(pairs_train, dist_train, batch_size):
  """generator used for training the siamese net with keras.

  Args:
    pairs_train:    training pairs
    dist_train:     training labels
    batch_size:     batch size

  Yields:
    generator instance
  """
  batches = make_batches(len(pairs_train), batch_size)
  while 1:
    random_idx = np.random.permutation(len(pairs_train))
    for batch_start, batch_end in batches:
      p_ = random_idx[batch_start:batch_end]
      x1, x2 = pairs_train[p_, 0], pairs_train[p_, 1]
      y = dist_train[p_]
      yield ([x1, x2], y)


def make_layer_list(arch, network_type=None, reg=None, dropout=0):
  """generates the list of layers.

  generates the list of layers specified by arch, to be stacked
  by stack_layers

  Args:
    arch:           list of dicts, where each dict contains the arguments to the
      corresponding layer function in stack_layers
    network_type:   siamese or cnc net. used only to name layers
    reg:            L2 regularization (if any)
    dropout:        dropout (if any)

  Returns:
    appropriately formatted stack_layers dictionary
  """
  layers = []
  for i, a in enumerate(arch):
    layer = {'l2_reg': reg}
    layer.update(a)
    if network_type:
      layer['name'] = '{}_{}'.format(network_type, i)
    layers.append(layer)
    if a['type'] != 'Flatten' and dropout != 0:
      dropout_layer = {
          'type': 'Dropout',
          'rate': dropout,
      }
      if network_type:
        dropout_layer['name'] = '{}_dropout_{}'.format(network_type, i)
      layers.append(dropout_layer)
  return layers


class LearningHandler(Callback):
  """Class for managing the learning rate scheduling and early stopping criteria.

  Learning rate scheduling is implemented by multiplying the learning rate
  by 'drop' everytime the validation loss does not see any improvement
  for 'patience' training steps
  """

  def __init__(self,
               lr,
               drop,
               lr_tensor,
               patience,
               tau_tensor=None,
               tau=1,
               min_tem=1,
               gumble=False):
    """initializer.

    Args:
      lr: initial learning rate
      drop: factor by which learning rate is reduced
      lr_tensor: tensorflow (or keras) tensor for the learning rate
      patience: patience of the learning rate scheduler
      tau_tensor: tensor to kepp the changed temperature
      tau: temperature
      min_tem: minimum temperature
      gumble: True if gumble is used
    """
    super(LearningHandler, self).__init__()
    self.lr = lr
    self.drop = drop
    self.lr_tensor = lr_tensor
    self.patience = patience
    self.tau = tau
    self.tau_tensor = tau_tensor
    self.min_tem = min_tem
    self.gumble = gumble

  def on_train_begin(self, logs=None):
    """Initialize the parameters at the start of training."""
    self.assign_op = tf.no_op()
    self.scheduler_stage = 0
    self.best_loss = np.inf
    self.wait = 0

  def on_epoch_end(self, epoch, logs=None):
    """For managing learning rate, early stopping, and temperature."""
    stop_training = False
    min_tem = self.min_tem
    anneal_rate = 0.00003
    if self.gumble and epoch % 20 == 0:
      self.tau = np.maximum(self.tau * np.exp(-anneal_rate * epoch), min_tem)
      K.set_value(self.tau_tensor, self.tau)
    # check if we need to stop or increase scheduler stage
    if isinstance(logs, dict):
      loss = logs['loss']
    else:
      loss = logs
    if loss <= self.best_loss:
      self.best_loss = loss
      self.wait = 0
    else:
      self.wait += 1
      if self.wait > self.patience:
        self.scheduler_stage += 1
        self.wait = 0
    if math.isnan(loss):
      stop_training = True
    # calculate and set learning rate
    lr = self.lr * np.power(self.drop, self.scheduler_stage)
    K.set_value(self.lr_tensor, lr)

    # built in stopping if lr is way too small
    if lr <= 1e-9:
      stop_training = True

    # for keras
    if hasattr(self, 'model') and self.model is not None:
      self.model.stop_training = stop_training

    return stop_training


def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)."""
  samples = tf.random_uniform(shape, minval=0, maxval=1)
  return -tf.log(-tf.log(samples + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
  """Draw a sample from the Gumbel-Softmax distribution."""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.

  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y

  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it
    will be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y


def get_scale(x, batch_size, n_nbrs):
  """Calculates the scale.

    The scale is based on the median distance of the kth
    neighbors of each point of x*, a m-sized sample of x, where
    k = n_nbrs and m = batch_size

  Args:
    x:          data for which to compute scale.
    batch_size: m in the aforementioned calculation.
    n_nbrs:     k in the aforementeiond calculation.

  Returns:
    the scale which is the variance term of the gaussian affinity matrix used by
    ncutnet
  """
  n = len(x)

  # sample a random batch of size batch_size
  sample = x[np.random.randint(n, size=batch_size), :]
  # flatten it
  sample = sample.reshape((batch_size, np.prod(sample.shape[1:])))

  # compute distances of the nearest neighbors
  nbrs = NearestNeighbors(n_neighbors=n_nbrs).fit(sample)
  distances, _ = nbrs.kneighbors(sample)

  # return the median distance
  return np.median(distances[:, n_nbrs - 1])


def calculate_cost_matrix(cluster, n_clusters):
  cost_matrix = np.zeros((n_clusters, n_clusters))

  # cost_matrix[i,j] will be the cost of assigning cluster i to label j
  for j in range(n_clusters):
    s = np.sum(cluster[:, j])  # number of examples in cluster i
    for i in range(n_clusters):
      t = cluster[i, j]
      cost_matrix[j, i] = s - t
  return cost_matrix


def get_cluster_labels_from_indices(indices):
  n_clusters = len(indices)
  cluster_labels = np.zeros(n_clusters)
  for i in range(n_clusters):
    cluster_labels[i] = indices[i][1]
  return cluster_labels


def get_accuracy(cluster_assignments, y_true, n_clusters):
  """Computes accuracy.

  Computes the accuracy based on the cluster assignments
  and true labels, using the Munkres algorithm

  Args:
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

  Returns:
    a tuple containing the accuracy and confusion matrix, in that order
  """
  y_pred, confusion_matrix = get_y_preds(cluster_assignments, y_true,
                                         n_clusters)
  # calculate the accuracy
  return np.mean(y_pred == y_true), confusion_matrix


def print_accuracy(cluster_assignments,
                   y_true,
                   n_clusters,
                   extra_identifier=''):
  """prints the accuracy."""
  # get accuracy
  accuracy, confusion_matrix = get_accuracy(cluster_assignments, y_true,
                                            n_clusters)
  # get the confusion matrix
  print('confusion matrix{}: '.format(extra_identifier))
  print(confusion_matrix)
  print(('Cnc_net{} accuracy: '.format(extra_identifier) +
         str(np.round(accuracy, 3))))
  return str(np.round(accuracy, 3))


def get_y_preds(cluster_assignments, y_true, n_clusters):
  """Computes the predicted labels.

  Label assignments now correspond to the actual labels in
  y_true (as estimated by Munkres)

  Args:
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

  Returns:
    a tuple containing the accuracy and confusion matrix, in that order
  """
  confusion_matrix = sklearn.metrics.confusion_matrix(
      y_true, cluster_assignments, labels=None)
  # compute accuracy based on optimal 1:1 assignment of clusters to labels
  cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
  indices = Munkres().compute(cost_matrix)
  true_cluster_labels = get_cluster_labels_from_indices(indices)
  y_pred = true_cluster_labels[cluster_assignments]
  return y_pred, confusion_matrix
