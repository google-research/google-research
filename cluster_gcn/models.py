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

"""Collections of different Models."""

import layers
import metrics
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS


class Model(object):
  """Model class to be inherited."""

  def __init__(self, **kwargs):
    allowed_kwargs = {
        'name', 'logging', 'multilabel', 'norm', 'precalc', 'num_layers'
    }
    for kwarg, _ in kwargs.items():
      assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
    name = kwargs.get('name')
    if not name:
      name = self.__class__.__name__.lower()
    self.name = name

    logging = kwargs.get('logging', False)
    self.logging = logging

    self.vars = {}
    self.placeholders = {}

    self.layers = []
    self.activations = []

    self.inputs = None
    self.outputs = None

    self.loss = 0
    self.accuracy = 0
    self.pred = 0
    self.optimizer = None
    self.opt_op = None
    self.multilabel = kwargs.get('multilabel', False)
    self.norm = kwargs.get('norm', False)
    self.precalc = kwargs.get('precalc', False)
    self.num_layers = kwargs.get('num_layers', 2)

  def _build(self):
    raise NotImplementedError

  def build(self):
    """Wrapper for _build()."""
    with tf.variable_scope(self.name):
      self._build()

    # Build sequential layer model
    self.activations.append(self.inputs)
    for layer in self.layers:
      hidden = layer(self.activations[-1])
      if isinstance(hidden, tuple):
        tf.logging.info('{} shape = {}'.format(layer.name,
                                               hidden[0].get_shape()))
      else:
        tf.logging.info('{} shape = {}'.format(layer.name, hidden.get_shape()))
      self.activations.append(hidden)
    self.outputs = self.activations[-1]

    # Store model variables for easy access
    variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    self.vars = variables
    for k in self.vars:
      tf.logging.info((k.name, k.get_shape()))

    # Build metrics
    self._loss()
    self._accuracy()
    self._predict()

    self.opt_op = self.optimizer.minimize(self.loss)

  def _loss(self):
    """Construct the loss function."""
    # Weight decay loss
    if FLAGS.weight_decay > 0.0:
      for var in self.layers[0].vars.values():
        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    # Cross entropy error
    if self.multilabel:
      self.loss += metrics.masked_sigmoid_cross_entropy(
          self.outputs, self.placeholders['labels'],
          self.placeholders['labels_mask'])
    else:
      self.loss += metrics.masked_softmax_cross_entropy(
          self.outputs, self.placeholders['labels'],
          self.placeholders['labels_mask'])

  def _accuracy(self):
    if self.multilabel:
      self.accuracy = metrics.masked_accuracy_multilabel(
          self.outputs, self.placeholders['labels'],
          self.placeholders['labels_mask'])
    else:
      self.accuracy = metrics.masked_accuracy(self.outputs,
                                              self.placeholders['labels'],
                                              self.placeholders['labels_mask'])

  def _predict(self):
    if self.multilabel:
      self.pred = tf.nn.sigmoid(self.outputs)
    else:
      self.pred = tf.nn.softmax(self.outputs)

  def save(self, sess=None):
    if not sess:
      raise AttributeError('TensorFlow session not provided.')
    saver = tf.train.Saver(self.vars)
    save_path = saver.save(sess, 'tmp/%s.ckpt' % self.name)
    tf.logging.info('Model saved in file:', save_path)

  def load(self, sess=None):
    if not sess:
      raise AttributeError('TensorFlow session not provided.')
    saver = tf.train.Saver(self.vars)
    save_path = 'tmp/%s.ckpt' % self.name
    saver.restore(sess, save_path)
    tf.logging.info('Model restored from file:', save_path)


class GCN(Model):
  """Implementation of GCN model."""

  def __init__(self, placeholders, input_dim, **kwargs):
    super(GCN, self).__init__(**kwargs)

    self.inputs = placeholders['features']
    self.input_dim = input_dim
    self.output_dim = placeholders['labels'].get_shape().as_list()[1]
    self.placeholders = placeholders

    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    self.build()

  def _build(self):

    self.layers.append(
        layers.GraphConvolution(
            input_dim=self.input_dim if self.precalc else self.input_dim * 2,
            output_dim=FLAGS.hidden1,
            placeholders=self.placeholders,
            act=tf.nn.relu,
            dropout=True,
            sparse_inputs=False,
            logging=self.logging,
            norm=self.norm,
            precalc=self.precalc))

    for _ in range(self.num_layers - 2):
      self.layers.append(
          layers.GraphConvolution(
              input_dim=FLAGS.hidden1 * 2,
              output_dim=FLAGS.hidden1,
              placeholders=self.placeholders,
              act=tf.nn.relu,
              dropout=True,
              sparse_inputs=False,
              logging=self.logging,
              norm=self.norm,
              precalc=False))

    self.layers.append(
        layers.GraphConvolution(
            input_dim=FLAGS.hidden1 * 2,
            output_dim=self.output_dim,
            placeholders=self.placeholders,
            act=lambda x: x,
            dropout=True,
            logging=self.logging,
            norm=False,
            precalc=False))
