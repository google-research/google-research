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
"""Exponential family policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from action_gap_rl.policies import layers_lib
import tensorflow.compat.v2 as tf


Distribution = collections.namedtuple('Distribution', 'params,log_pdf,mode')


"""
Define distirbutions accessible by `config`. Maps distribution name to
`Distribution` namedtuple.

Distribution:
  log_pdf: function (action, *coembedding) --> unnormalized log-pdf
      `coembedding` is a tuple of distribution parameters which will be the
      output of the NN.
  mode: function (*coembedding) --> mode action
  params: binary list of length `len(coembedding)`. For each distribution param
      in `coembedding`, specify whether this param must be non-negative.
      1 indicates non-negative, 0 indicates entire reals.


Rationale:
    We want to specify what type of exponential distribution to use in the
    hparam config. Rather than saving python functions to disk (possible via
    pickle, but messy), we define a mapping between distribution names and
    python functions that specify its unnormalized log-pdf and argmax. That way
    in the config (saved to disk), only the string name is needed.


How to add to DISTS
--------------------
Decide on the number of distribution parameters (number of outputs from the NN).
This is the size of the observation co-embedding. Simply be consistent about
that number of parameters in your definition.

This example uses 2 paramers: p0 and p1.
```
DISTS = {
    ...,
    'my-distribution': Distribution(
        log_pdf=lambda a, p0, p1: ...,  # Function of action and your parameters
        mode=lambda p0, p1: ...,  # Function of just your parameters

        # Specify the domain of your params. In this case p0 is over the reals
        # (unrestricted), and p1 is over the non-negative reals
        # (soft-plus activation).
        params=[0, 1],
    ),
}
```
"""
DISTS = {
    'quadratic': Distribution(
        params=[1, 0, 0],
        log_pdf=lambda a, p0, p1, p2: -a**2*p0 + a*p1 + p2,
        mode=lambda p0, p1, _: p1/p0),
}


class ExponentialFamilyPolicy(tf.keras.Model):
  """A policy that takes an arbitrary function as the un-normalized log pdf."""

  def __init__(self, config, name=None):
    super(ExponentialFamilyPolicy, self).__init__(
        name=name or self.__class__.__name__)
    self._config = config
    hidden_widths = config.hidden_widths
    self._dist = DISTS[config.dist_name]
    params = self._dist.params
    self._log_pdf = self._dist.log_pdf
    self._mode = self._dist.mode
    if config.embed:
      transformation_layers = [layers_lib.soft_hot_layer(**config.embed)]
    else:
      transformation_layers = []
    self._body = tf.keras.Sequential(
        transformation_layers
        + [tf.keras.layers.Dense(w, activation='relu') for w in hidden_widths]
        + [tf.keras.layers.Dense(len(params), activation=None),
           tf.keras.layers.Lambda(
               lambda x: tf.stack(  # pylint: disable=g-long-lambda
                   [tf.math.softplus(x[Ellipsis, i]) if is_non_neg else x[Ellipsis, i]
                    for i, is_non_neg in enumerate(params)],
                   -1)
               )]
    )

  def call(self, states, actions):
    return self._log_pdf(actions, *tf.unstack(self._body(states), axis=-1))

  def regularizer(self, states):
    return 0.0

  def loss(self, states, actions, targets):
    # L2 loss.
    return tf.reduce_mean(tf.square(self(states, actions) - targets))

  def argmax(self, states):
    return tf.expand_dims(
        self._mode(*tf.unstack(self._body(states), axis=-1)), axis=-1)
