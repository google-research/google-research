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

"""Implementation of various neural modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow import nest


class BatchLinear(snt.AbstractModule):
  """Performs k independent linear transformations of k vectors."""

  def __init__(self, n_units, tile_dims=(0,), use_bias=True, initializers=None):
    """Builds the module.

    Args:
      n_units: int, number of output dimensions.
      tile_dims: sequence of ints; weights along these dimensions are shared.
      use_bias: boolean.
      initializers: dict (as in sonnet).
    """
    super(BatchLinear, self).__init__()
    self._n_units = n_units
    self._tile_dims = sorted(tile_dims)
    self._use_bias = use_bias
    self.initializers = snt.python.modules.util.check_initializers(
        initializers, {'w', 'b'} if use_bias else {'w'})

  def _build(self, x):
    """Applies the module.

    Args:
      x: tensor of shape [B, k, d].

    Returns:
      Tensor of shape [B, k, n_units].
    """

    # batch_size, n_inputs, n_dims = x.shape.as_list()
    shape = x.shape.as_list()

    if 'w' not in self.initializers:
      stddev = 1 / math.sqrt(shape[-1])
      self.initializers['w'] = tf.truncated_normal_initializer(
          stddev=stddev)

    weights_shape = shape + [self._n_units]
    tiles = []
    for i in self._tile_dims:
      tiles.append(weights_shape[i])
      weights_shape[i] = 1

    weights = tf.get_variable('weights', shape=weights_shape,
                              initializer=self._init('w'))

    weights = snt.TileByDim(self._tile_dims, tiles)(weights)

    x = tf.expand_dims(x, -2)
    y = tf.matmul(x, weights)
    y = tf.squeeze(y, -2)

    if self._use_bias:
      if 'b' not in self.initializers:
        self.initializers['b'] = tf.zeros_initializer()

      init = dict(b=self._init('b'))
      bias_dims = [i for i in range(len(shape)) if i not in self._tile_dims]
      add_bias = snt.AddBias(bias_dims=bias_dims, initializers=init)
      y = add_bias(y)

    return y

  def _init(self, key):
    if self.initializers:
      return self.initializers.get(key, None)


class BatchMLP(snt.AbstractModule):
  """Applies k independent MLPs on k inputs."""

  def __init__(self, n_hiddens,
               activation=tf.nn.relu,
               activate_final=False,
               initializers=None,
               use_bias=True,
               tile_dims=(0,)):

    super(BatchMLP, self).__init__()
    self._n_hiddens = nest.flatten(n_hiddens)
    self._activation = activation
    self._activate_final = activate_final
    self._initializers = initializers
    self._use_bias = use_bias
    self._tile_dims = tile_dims

  def _build(self, x):

    h = x
    for n_hidden in self._n_hiddens[:-1]:
      layer = BatchLinear(n_hidden, initializers=self._initializers,
                          use_bias=True)
      h = self._activation(layer(h))

    layer = BatchLinear(self._n_hiddens[-1], initializers=self._initializers,
                        use_bias=self._use_bias)
    h = layer(h)
    if self._activate_final:
      h = self._activation(h)

    return h


