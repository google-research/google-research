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

"""Classification model.

Model that takes state and makes a class prediction.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from saccader import model_utils


class ClassificationNetwork(object):
  """Classification network.

  Network that receives a state and makes a class prediction.
  Attributes:
    regularizer: Regularizer for network.
    num_classes: (Integer) indicating number of classes.
    num_units_fc_layers: List with number of units for fully connected layers
      (None or []: no fully connect layers are used).
    activation: Activation function (None: for linear network).
    var_list: List of network variables.
    init_op: Initialization operations for model variables.

  """

  def __init__(self, config):
    """Init.

    Args:
      config: ConfigDict object with model parameters (see dram_config.py).
    """
    self.regularizer = config.regularizer
    if config.num_units_fc_layers is None:
      self.num_units_fc_layers = []
    else:
      self.num_units_fc_layers = config.num_units_fc_layers
    self.num_classes = config.num_classes
    self.activation = config.activation
    self.var_list = []
    self.init_op = None

  def collect_variables(self):
    """Collects model variables.

    Populates self.var_list with model variables and self.init_op with
    variables' initializer. This function is only called once with __call__.

    """
    self.var_list = [
        v for v in tf.global_variables() if "classification_network" in v.name
    ]
    self.init_op = tf.variables_initializer(var_list=self.var_list)

  def __call__(self, input_state):
    """Builds classification network.

    Args:
      input_state: 2-D Tensor of shape [batch, state dimensionality]

    Returns:
      logits: Network logits.
      endpoints: Dictionary with activations at different layers.
    """
    if self.var_list:
      reuse = True
    else:
      reuse = False

    tf.logging.info("Builds Classification Network")
    endpoints = {}
    net = input_state

    # Fully connected layers.
    with tf.variable_scope("classification_network", reuse=reuse):
      net, endpoints_ = model_utils.build_fc_layers(
          net, self.num_units_fc_layers, activation=self.activation,
          regularizer=self.regularizer)
    endpoints.update(endpoints_)

    # Linear output layer.
    with tf.variable_scope("classification_network/output", reuse=reuse):
      logits, _ = model_utils.build_fc_layers(
          net, [self.num_classes], activation=None,
          regularizer=self.regularizer)
    endpoints["logits"] = logits

    if not reuse:
      self.collect_variables()
    return logits, endpoints
