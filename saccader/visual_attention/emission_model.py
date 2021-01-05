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

"""Emission model.

Model that takes an input state and makes a prediction to where to look next in
an image.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from saccader import model_utils


class EmissionNetwork(object):
  """Emission network.

  Network that receives a state and makes a prediction about a location in an
  image to look at next.
  Attributes:
    regularizer: Regularizer for network.
    location_scale: Location scale (default 1., which gives -1, 1 location
      range).
    num_units_fc_layers: List with number of units for fully connected layers
      (None or []: no fully connect layers are used).
    activation: Activation function (None: for linear network).
    location_dims: (integer) Dimensionality of location.
    input_dims: (integer) Input dimensionality.
    var_list: List of network variables.
    init_op: Initialization operations for model variables.
  """

  def __init__(self, config):
    """Init.

    Args:
      config: ConfigDict object with model parameters.
    """
    self.regularizer = config.regularizer
    if config.num_units_fc_layers is None:
      self.num_units_fc_layers = []
    else:
      self.num_units_fc_layers = config.num_units_fc_layers
    self.location_dims = 2
    self.activation = config.activation
    self.var_list = []
    self.init_op = []

  def collect_variables(self):
    """Collect model variables call needs to be run at least once."""
    self.var_list = [
        v for v in tf.global_variables() if "emission_network" in v.name
    ]
    self.init_op = tf.variables_initializer(var_list=self.var_list)

  def __call__(self,
               input_state,
               location_scale,
               prev_locations=None,
               is_training=False,
               policy="learned",
               sampling_stddev=1e-5):
    """Builds emission network.

    Args:
      input_state: 2-D Tensor of shape [batch, state dimensionality]
      location_scale: <= 1. and >= 0. the normalized location range
        [-location_scale, location_scale]
      prev_locations: if not None add prev_location to current proposed location
        (ie using relative locations)
      is_training: (Boolean) to indicate training or inference modes.
      policy: (String) 'learned': uses learned policy, 'random': uses random
        policy, or 'center': uses center look policy.
      sampling_stddev: Sampling distribution standard deviation.

    Returns:
      locations: network output reflecting next location to look at
        (normalized to range [-location_scale, location_scale]).
        The image locations mapping to locs are as follows:
          (-1, -1): upper left corner.
          (-1, 1): upper right corner.
          (1, 1): lower right corner.
          (1, -1): lower left corner.
      endpoints: dictionary with activations at different layers.
    """
    if self.var_list:
      reuse = True
    else:
      reuse = False

    batch_size = input_state.shape.as_list()[0]

    tf.logging.info("BUILD Emission Network")
    endpoints = {}
    net = input_state

    # Fully connected layers.
    with tf.variable_scope("emission_network", reuse=reuse):
      net, endpoints_ = model_utils.build_fc_layers(
          net,
          self.num_units_fc_layers,
          activation=self.activation,
          regularizer=self.regularizer)
    endpoints.update(endpoints_)

    # Tanh output layer.
    with tf.variable_scope("emission_network/output", reuse=reuse):
      output, _ = model_utils.build_fc_layers(
          net, [self.location_dims],
          activation=tf.nn.tanh,
          regularizer=self.regularizer)

    # scale location ([-location_scale, location_scale] range
    mean_locations = location_scale * output
    if prev_locations is not None:
      mean_locations = prev_locations + mean_locations

    if policy == "learned":
      endpoints["mean_locations"] = mean_locations
      if is_training:
        # At training samples random location.
        locations = mean_locations + tf.random_normal(
            shape=(batch_size, self.location_dims), stddev=sampling_stddev)
        # Ensures range [-location_scale, location_scale]
        locations = tf.clip_by_value(locations, -location_scale,
                                     location_scale)
        tf.logging.info("Sampling locations.")
        tf.logging.info("====================================================")
      else:
        # At inference uses the mean value for the location.
        locations = mean_locations

      locations = tf.stop_gradient(locations)
    elif policy == "random":
      # Use random policy for location.
      locations = tf.random_uniform(
          shape=(batch_size, self.location_dims),
          minval=-location_scale,
          maxval=location_scale)
      endpoints["mean_locations"] = mean_locations
    elif policy == "center":
      # Use center look policy.
      locations = tf.zeros(
          shape=(batch_size, self.location_dims))
      endpoints["mean_locations"] = mean_locations
    else:
      raise ValueError("policy can be either 'learned', 'random', or 'center'")

    if not reuse:
      self.collect_variables()
    return locations, endpoints
