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

"""PathNet building blocks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import models


class ModelHeadComponent(object):
  """Component that computes the task loss."""

  def __init__(self, loss_fn, auxiliary_loss_fn=None):
    """Constructor.

    Args:
      loss_fn: function to compute task loss. Given two tf.Tensors of the same
        shape, corresponding to a batch of labels and a batch of model
        predictions respectively, `loss_fn` should return a scalar loss.
      auxiliary_loss_fn: function to compute auxiliary losses or None. If
        a function is provided, it will receive as the single argument the
        `state` passed to `__call__`. The scalar value returned from
        `auxiliary_loss_fn` will be added to `task_loss`. Note that the
        `auxiliary_loss_fn` can take into account more than just the `state`:
        if it is created during construction of the PathNet, it can access
        parts of the PathNet from the outer scope e.g. components and routers.
    """

    self.name = 'Head'
    self.loss_fn = loss_fn
    self.auxiliary_loss_fn = auxiliary_loss_fn

  def __call__(self, state):
    """Calls the component to compute the task loss.

    Args:
      state: a dictionary with keys: 'in_tensor' containing a batch of
        predictions, and 'labels' containing a batch of labels.

    Returns:
      A dictionary with the scalar task loss under the 'task_loss' key.
    """
    predictions = state['in_tensor']
    labels = state['labels']

    task_loss = self.loss_fn(labels, predictions)
    tf.contrib.summary.scalar('task_loss', tf.reduce_mean(task_loss))

    if self.auxiliary_loss_fn:
      aux_loss = self.auxiliary_loss_fn(state)
      tf.contrib.summary.scalar('aux_loss', aux_loss)
    else:
      aux_loss = tf.constant(0.0)

    # The PathNet optimizer uses the entry under `task_loss` as the target for
    # optimization, so that entry has to include all losses. The other entries
    # are there just for summaries.
    return {
        'task_loss': task_loss + aux_loss,
        'task_loss_without_aux_losses': task_loss
    }

  def zero_output(self, state):
    del state
    return {
        'task_loss': tf.zeros([1]),
        'task_loss_without_aux_losses': tf.zeros([1])
    }


class KerasComponent(object):
  """Wraps a Keras network into a component that can be used with PathNet."""

  def __init__(self, name, network, out_shape):
    """Constructor.

    Args:
      name: (string) name for this component.
      network: keras network which is going to be wrapped.
      out_shape: expected output shape for `network` (excluding batch
        dimension) as a sequence of ints.
    """

    self.name = name
    self.out_shape = out_shape
    self.network = network

  def __call__(self, state):
    x = state['in_tensor']
    training = state['training']

    x = self.network(x, training=training)

    return {'in_tensor': x}

  def zero_output(self, state):
    del state
    return {'in_tensor': tf.zeros([1] + self.out_shape)}


class IdentityComponent(KerasComponent):
  """Component that implements an identity function."""

  def __init__(self, out_shape):
    """Constructor.

    Args:
      out_shape: shape for both input and output of this component.
    """
    super(IdentityComponent, self).__init__(
        'Identity', models.Sequential(), out_shape)

  def __call__(self, state):
    assert state['in_tensor'].shape[1:] == self.out_shape
    return super(IdentityComponent, self).__call__(state)


class FCLComponent(KerasComponent):
  """Component that implements a fully connected neural network."""

  def __init__(
      self, numbers_of_units, hidden_activation=None, out_activation=None):
    """Constructor.

    Args:
      numbers_of_units: (list of int) number of hidden units for every layer
        (including the output layer)
      hidden_activation: activation function to apply after each hidden layer,
        ignored if there are no hidden layers.
      out_activation: activation function to apply at the output layer.
    """

    num_layers = len(numbers_of_units)
    assert num_layers >= 1

    activations = [hidden_activation] * (num_layers - 1) + [out_activation]

    network = models.Sequential([
        layers.Dense(units, activation=activation)
        for units, activation in zip(numbers_of_units, activations)
    ])

    super(FCLComponent, self).__init__(
        '%sFCL' % num_layers, network, [numbers_of_units[-1]])
