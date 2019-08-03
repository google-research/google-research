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

"""Implementations of different layers."""

import inits
import tensorflow as tf

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
  """Helper function, assigns unique layer IDs."""
  if layer_name not in _LAYER_UIDS:
    _LAYER_UIDS[layer_name] = 1
    return 1
  else:
    _LAYER_UIDS[layer_name] += 1
    return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
  """Dropout for sparse tensors."""
  random_tensor = keep_prob
  random_tensor += tf.random_uniform(noise_shape)
  dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
  pre_out = tf.sparse_retain(x, dropout_mask)
  return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
  """Wrapper for tf.matmul (sparse vs dense)."""
  if sparse:
    res = tf.sparse_tensor_dense_matmul(x, y)
  else:
    res = tf.matmul(x, y)
  return res


def layernorm(x, offset, scale):
  mean, variance = tf.nn.moments(x, axes=[1], keep_dims=True)
  return tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-9)


class Layer(object):
  """Base layer class.

  Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
  """

  def __init__(self, **kwargs):
    allowed_kwargs = {'name', 'logging'}
    for kwarg, _ in kwargs.items():
      assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
    name = kwargs.get('name')
    if not name:
      layer = self.__class__.__name__.lower()
      name = layer + '_' + str(get_layer_uid(layer))
    self.name = name
    self.vars = {}
    logging = kwargs.get('logging', False)
    self.logging = logging
    self.sparse_inputs = False

  def _call(self, inputs):
    return inputs

  def __call__(self, inputs):
    with tf.name_scope(self.name):
      if self.logging and not self.sparse_inputs:
        tf.summary.histogram(self.name + '/inputs', inputs)
      outputs = self._call(inputs)
      if self.logging:
        tf.summary.histogram(self.name + '/outputs', outputs)
      return outputs

  def _log_vars(self):
    for var in self.vars:
      tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
  """Dense layer."""

  def __init__(self,
               input_dim,
               output_dim,
               placeholders,
               dropout=0.,
               sparse_inputs=False,
               act=tf.nn.relu,
               bias=False,
               featureless=False,
               norm=False,
               **kwargs):
    super(Dense, self).__init__(**kwargs)

    if dropout:
      self.dropout = placeholders['dropout']
    else:
      self.dropout = 0.

    self.act = act
    self.sparse_inputs = sparse_inputs
    self.featureless = featureless
    self.bias = bias
    self.norm = norm

    # helper variable for sparse dropout
    self.num_features_nonzero = placeholders['num_features_nonzero']

    with tf.variable_scope(self.name + '_vars'):
      self.vars['weights'] = inits.glorot([input_dim, output_dim],
                                          name='weights')
      if self.bias:
        self.vars['bias'] = inits.zeros([output_dim], name='bias')

      if self.norm:
        self.vars['offset'] = inits.zeros([1, output_dim], name='offset')
        self.vars['scale'] = inits.ones([1, output_dim], name='scale')

    if self.logging:
      self._log_vars()

  def _call(self, inputs):
    x = inputs

    # dropout
    if self.sparse_inputs:
      x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
    else:
      x = tf.nn.dropout(x, 1 - self.dropout)

    # transform
    output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

    # bias
    if self.bias:
      output += self.vars['bias']

    with tf.variable_scope(self.name + '_vars'):
      if self.norm:
        output = layernorm(output, self.vars['offset'], self.vars['scale'])

    return self.act(output)


class GraphConvolution(Layer):
  """Graph convolution layer."""

  def __init__(self,
               input_dim,
               output_dim,
               placeholders,
               dropout=0.,
               sparse_inputs=False,
               act=tf.nn.relu,
               bias=False,
               featureless=False,
               norm=False,
               precalc=False,
               **kwargs):
    super(GraphConvolution, self).__init__(**kwargs)

    if dropout:
      self.dropout = placeholders['dropout']
    else:
      self.dropout = 0.

    self.act = act
    self.support = placeholders['support']
    self.sparse_inputs = sparse_inputs
    self.featureless = featureless
    self.bias = bias
    self.norm = norm
    self.precalc = precalc

    # helper variable for sparse dropout
    self.num_features_nonzero = placeholders['num_features_nonzero']

    with tf.variable_scope(self.name + '_vars'):
      self.vars['weights'] = inits.glorot([input_dim, output_dim],
                                          name='weights')
      if self.bias:
        self.vars['bias'] = inits.zeros([output_dim], name='bias')

      if self.norm:
        self.vars['offset'] = inits.zeros([1, output_dim], name='offset')
        self.vars['scale'] = inits.ones([1, output_dim], name='scale')

    if self.logging:
      self._log_vars()

  def _call(self, inputs):
    x = inputs

    # convolve
    if self.precalc:
      support = x
    else:
      support = dot(self.support, x, sparse=True)
      support = tf.concat((support, x), axis=1)

    # dropout
    support = tf.nn.dropout(support, 1 - self.dropout)

    output = dot(support, self.vars['weights'], sparse=self.sparse_inputs)

    # bias
    if self.bias:
      output += self.vars['bias']

    with tf.variable_scope(self.name + '_vars'):
      if self.norm:
        output = layernorm(output, self.vars['offset'], self.vars['scale'])

    return self.act(output)
