# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""A classifier - fully-connected or convolutional."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf
from extrapolation.utils import dataset_utils

FLAGS = flags.FLAGS
EPS = 1e-12



class Classifier(tf.keras.Model):
  """A classifier."""

  def __init__(self, n_classes, onehot=False):
    """Initializes an n-way classifier.

    Args:
      n_classes (int): number of classes in the problem.
      onehot (bool): whether the labels take one-hot or integer format.
    """
    super(Classifier, self).__init__()
    self.n_classes = n_classes
    self.onehot = onehot

  def build_layers(self):
    """Build the internal layers of the network."""
    raise NotImplementedError

  def call(self, inputs):
    """Feed inputs forward through the network.

    Args:
      inputs (tensor): input batch we wish to encode.
    Returns:
      logits (tensor): the unnormalized output of the model on this batch.
    """
    raise NotImplementedError

  def get_loss(self, batch_x, batch_y, return_preds=False):
    if not self.onehot:
      batch_y = dataset_utils.make_onehot(batch_y, self.n_classes)
    return self.get_loss_with_onehot_labels(batch_x, batch_y, return_preds)

  def get_loss_dampened(self, batch_x, batch_y, lam=0.0):
    loss, _ = self.get_loss(batch_x, batch_y, return_preds=False)
    reg_loss = 0.5 * sum([tf.reduce_sum(tf.square(w)) for w in self.weights])
    loss = loss + lam * reg_loss
    return loss

  def get_loss_with_onehot_labels(self, batch_x, batch_y, return_preds):
    """Run classifier on input batch and return loss, error, and predictions.

    Args:
      batch_x (tensor): input batch we wish to run on.
      batch_y (tensor): onehot input labels we wish to predict.
      return_preds (bool): whether or not to return prediction tensor.
    Returns:
      loss (tensor): cross-entropy loss for each element in batch.
      err (tensor): classification error for each element in batch.
      preds (tensor): optional, model softmax outputs.
      reprs (tensor): optional, internal model representations.
    """
    logits, reprs = self.call(batch_x)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(batch_y), logits=logits)
    class_preds = tf.cast(tf.argmax(logits, axis=1), tf.int64)
    class_true = tf.cast(tf.argmax(batch_y, axis=1), tf.int64)
    err = 1 - tf.cast(
        tf.math.equal(class_preds,
                      tf.squeeze(class_true)), tf.float64)
    if return_preds:
      preds = tf.clip_by_value(tf.nn.softmax(logits), EPS, 1 - EPS)
      return loss, err, preds, reprs
    else:
      return loss, err


class MLP(Classifier):
  """A fully-connected classifier."""

  def __init__(self, layer_dims, n_classes, **kwargs):
    super(MLP, self).__init__(n_classes, **kwargs)
    self.layer_dims = layer_dims
    self.dense_layers = []
    self.build_layers()

  def build_layers(self):
    for dim in self.layer_dims:
      self.dense_layers.append(
          tf.keras.layers.Dense(dim, activation=tf.nn.relu))
    self.dense_layers.append(tf.keras.layers.Dense(self.n_classes))

  def call(self, inputs):
    reprs = []
    logits = tf.keras.layers.Flatten()(inputs)
    for l in self.dense_layers:
      logits = l(logits)
      reprs.append(logits)
    return logits, reprs


class CNN(Classifier):
  """A CNN classifier."""

  def __init__(self, conv_dims, conv_sizes, dense_sizes, n_classes, **kwargs):
    super(CNN, self).__init__(n_classes, **kwargs)
    self.conv_dims = conv_dims
    self.conv_sizes = conv_sizes
    self.dense_sizes = dense_sizes
    self.conv_layers = []
    self.maxpool_layers = []
    self.dense_layers = []
    self.build_layers()

  def build_layers(self):
    for dim, size in zip(self.conv_dims, self.conv_sizes):
      self.conv_layers.append(tf.keras.layers.Conv2D(dim, [size] * 2,
                                                     activation=tf.nn.relu))
      self.maxpool_layers.append(
          tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
    for size in self.dense_sizes:
      self.dense_layers.append(tf.keras.layers.Dense(size,
                                                     activation=tf.nn.relu))
    self.dense_layers.append(tf.keras.layers.Dense(self.n_classes))

  def call(self, inputs):
    """Call this model on some input tensor inputs.

    Args:
      inputs (tensor): inputs to this model.
    Returns:
      logits (tensor): output class logits (unnormalized).
      reprs (list): list of intermediate tensors in the model's forward pass.
    """
    reprs = []
    u = inputs
    for i in range(len(self.conv_layers)):
      u = self.conv_layers[i](u)
      u = self.maxpool_layers[i](u)
      reprs.append(u)
    flat_u = tf.keras.layers.Flatten()(u)
    for i in range(len(self.dense_layers)):
      flat_u = self.dense_layers[i](flat_u)
      reprs.append(flat_u)
    logits = flat_u
    return logits, reprs

