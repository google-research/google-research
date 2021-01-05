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

"""Deep recurrent attention model (DRAM).

Model based on https://arxiv.org/abs/1412.7755
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow.compat.v1 as tf

from saccader.visual_attention import classification_model
from saccader.visual_attention import emission_model
from saccader.visual_attention import glimpse_model


class DRAMCell(object):
  """Deep Recurrent Attention Model Cell.

  Recurrent neural network that performs classification on images by taking
  glimpses at different locations on an image.

  Attributes:
    config: (Configuration object) With parameters:
      num_units_rnn_layers: List of Integers indicating number of units in each
        rnn layer. Length of list = 2 as there are two rnn layers.
      rnn_activation: Activation function.
      rnn_dropout_rate: Dropout rate for both input and output.
      cell_type: RNN cell type ("lstm" or "gru").
      num_resolutions: (Integer) Number of image resolutions used.

    rnn_layers: List of RNN layers.
    emission_net: Emission network object.
    glimpse_net: Glimpse network object.
    classification_net: Classification network object.
    init_op: Initialization operations for model variables.
    zero_states: List of zero states of RNN layers.
    var_list_location: List of variables for the location network.
    var_list_classification: List of variables for the classification network.
    var_list: List of all model variables.
  """

  def __init__(self, config):
    """Init.

    Args:
      config: ConfigDict object with model parameters (see dram_config.py).
    """
    self.config = copy.deepcopy(config)
    if len(self.config.num_units_rnn_layers) != 2:
      raise ValueError("num_units_rnn_layers should be a list of length 2.")
    self.cell_type = self.config.cell_type

    glimpse_model_config = self.config.glimpse_model_config
    emission_model_config = self.config.emission_model_config
    classification_model_config = self.config.classification_model_config
    classification_model_config.num_classes = self.config.num_classes

    glimpse_model_config.glimpse_shape = self.config.glimpse_shape
    glimpse_model_config.num_resolutions = self.config.num_resolutions
    self.glimpse_net = glimpse_model.GlimpseNetwork(glimpse_model_config)

    self.emission_net = emission_model.EmissionNetwork(
        emission_model_config)

    self.classification_net = classification_model.ClassificationNetwork(
        classification_model_config)

    self.rnn_layers = []
    self.zero_states = []
    for num_units in self.config.num_units_rnn_layers:
      if self.cell_type == "lstm":
        rnn_layer = tf.nn.rnn_cell.LSTMCell(
            num_units, state_is_tuple=True,
            activation=self.config.rnn_activation)
      elif self.cell_type == "gru":
        rnn_layer = tf.nn.rnn_cell.GRUCell(
            num_units, activation=self.config.rnn_activation)

      self.rnn_layers.append(rnn_layer)
      self.zero_states.append(rnn_layer.zero_state)

    self.zero_states = self.zero_states

    self.var_list = []
    self.var_list_location = []
    self.var_list_classification = []
    self.init_op = None

  def collect_variables(self):
    """Collects model variables.

    Populates self.var_list with model variables and self.init_op with
    variables' initializer. This function is only called once with __call__.
    """
    # Add glimpse network variables.
    self.var_list_classification += self.glimpse_net.var_list

    # Add emission network variables.
    self.var_list_location += self.emission_net.var_list

    # Add classification network variables.
    self.var_list_classification += self.classification_net.var_list

    # Add rnn variables for classification layer.
    self.var_list_classification += self.rnn_layers[0].weights
    # Add rnn variables for location layer.
    self.var_list_location += self.rnn_layers[1].weights
    # All variables.
    self.var_list = self.var_list_classification + self.var_list_location
    self.init_op = tf.variables_initializer(var_list=self.var_list)

  def __call__(self,
               images,
               locations,
               state_rnn,
               use_resolution,
               prev_locations=None,
               is_training=False,
               policy="learned",
               sampling_stddev=1e-5,
               stop_gradient_between_cells=False,
               stop_gradient_after_glimpse=False):
    """Builds DRAM cell.

    Args:
      images: 4-D Tensor of shape [batch, height, width, channels].
      locations: Glimpse location.
      state_rnn: Tuple of size two for the state of RNN layers.
      use_resolution: (List of Boolean of size num_resolutions) Indicates which
        resolutions to use from high (small receptive field)
        to low (wide receptive field). None indicates use all resolutions.
      prev_locations: If not None, add prev_locations to current proposed
        locations (i.e. using relative locations).
      is_training: (Boolean) To indicate training or inference modes.
      policy: (String) 'learned': uses learned policy, 'random': uses random
        policy, or 'center': uses center look policy.
      sampling_stddev: Sampling distribution standard deviation.
      stop_gradient_between_cells: (Boolean) Whether to stop the gradient
        between the classification and location sub cells of the DRAM cell.
      stop_gradient_after_glimpse: (Boolean) Whether to stop the gradient
        after the glimpse net output.
    Returns:
      logits: Model logits.
      locations: New glimpse location.
      state_rnn: Tuple of length two for the new state of RNN layers.
    """
    if self.var_list:
      reuse = True
    else:
      reuse = False

    if is_training and self.config.rnn_dropout_rate > 0:
      keep_prob = 1.0 - self.config.rnn_dropout_rate
      rnn_layers = []
      for layer in self.rnn_layers:
        rnn_layers.append(
            tf.nn.rnn_cell.DropoutWrapper(
                layer, input_keep_prob=keep_prob, output_keep_prob=keep_prob))
    else:
      rnn_layers = self.rnn_layers

    endpoints = {}
    glimpse_size = tf.cast(self.glimpse_net.glimpse_shape[0], dtype=tf.float32)
    image_size = tf.cast(tf.shape(images)[1], dtype=tf.float32)
    # Ensure glimpses within image.
    location_scale = 1. - glimpse_size / image_size
    with tf.name_scope("glimpse_network"):
      # First rnn layer (for classification).
      g, endpoints["glimpse_network"] = self.glimpse_net(
          images, locations, is_training=is_training,
          use_resolution=use_resolution)
    with tf.variable_scope("dram_cell_0", reuse=reuse):
      if stop_gradient_after_glimpse:
        input_rnn_classification = tf.stop_gradient(g)
      else:
        input_rnn_classification = g
      output_rnn0, state_rnn0 = rnn_layers[0](input_rnn_classification,
                                              state_rnn[0])

    with tf.name_scope("classification_network"):
      logits, endpoints["classification_network"] = self.classification_net(
          output_rnn0)

    # Second rnn layer (for glimpse locations).
    with tf.variable_scope("dram_cell_1", reuse=reuse):
      if stop_gradient_between_cells:
        input_rnn_location = tf.stop_gradient(output_rnn0)
      else:
        input_rnn_location = output_rnn0
      output_rnn1, state_rnn1 = rnn_layers[1](input_rnn_location, state_rnn[1])

    with tf.name_scope("emission_network"):
      locations, endpoints["emission_network"] = self.emission_net(
          output_rnn1,
          location_scale=location_scale,
          prev_locations=prev_locations,
          is_training=is_training,
          policy=policy,
          sampling_stddev=sampling_stddev)

      mean_locations = endpoints["emission_network"]["mean_locations"]
    state_rnn = (state_rnn0, state_rnn1)
    output_rnn = (output_rnn0, output_rnn1)

    endpoints["cell_outputs"] = {
        "locations": locations,
        "state_rnn": state_rnn,
        "output_rnn": output_rnn,
        "mean_locations": mean_locations,
    }

    if not reuse:
      self.collect_variables()

    return logits, endpoints


class DRAMNetwork(object):
  """Deep Recurrent Attention Model.

  Recurrent neural network that performs classification on images by taking
  glimpses at different locations on an image.

  Attributes:
    glimpse_size: 2-D tuple of integers indicating glimpse size (height, width).
    context_net: Context network object.
    init_op: initialization operations for model variables.
    dram_cell: DRAM model cell.
    location_encoding: Use "absolute" or "relative" location.
    var_list_location: List of variables for the location network.
    var_list_classification: List of variables for the classification network.
    var_list: List of all model variables.
    glimpse_shape: Tuple of two integers with and size of glimpse.
    num_resolutions: (Integer) Number of image resolutions used.
    limit_classification_rf: (Boolean) Whether to limit classification to only
      high resolution or all resolutions.
    use_resolution_for_location: (List of Boolean of size num_resolutions)
        Indicates which resolutions to use from high (small receptive field)
        to low (wide receptive field) to set the initial state of the location
        LSTM . None indicates use all resolutions.
  Raises:
    ValueError: if config.location_encoding is not 'absolute' or 'relative'.
  """

  def __init__(self, config):
    """Init.

    Args:
      config: ConfigDict object with model parameters (see dram_config.py).
    """
    self.dram_cell = DRAMCell(config)

    if config.location_encoding in ["absolute", "relative"]:
      self.location_encoding = config.location_encoding
    else:
      raise ValueError("location_encoding config can only be either "
                       "'absolute' or 'relative'")
    self.glimpse_shape = self.dram_cell.glimpse_net.glimpse_shape
    self.var_list = []
    self.var_list_classification = []
    self.var_list_location = []
    self.init_op = None
    self.num_resolutions = config.num_resolutions
    self.limit_classification_rf = config.limit_classification_rf
    # Use all resolutions to set the initial location LSTM state.
    self.use_resolution_for_location = [
        True for _ in range(self.num_resolutions)]

  def collect_variables(self):
    """Collects model variables.

    Populates variable lists with model variables and self.init_op with
    variables' initializer. This function is only called once with __call__.
    """
    self.var_list_classification += self.dram_cell.var_list_classification
    self.var_list_location += self.dram_cell.var_list_location
    self.var_list = (
        self.var_list_classification + self.var_list_location)

    self.init_op = tf.variables_initializer(var_list=self.var_list)

  def __call__(self,
               images,
               num_times,
               is_training=False,
               policy="learned",
               sampling_stddev=1e-5,
               stop_gradient_between_cells=False,
               stop_gradient_after_glimpse=False):
    """Builds DRAM network.

    Args:
      images: 4-D Tensor of shape [batch, height, width, channels].
      num_times: Integer representing number of times for the RNNs.
      is_training: (Boolean) To indicate training or inference modes.
      policy: (String) 'learned': uses learned policy, 'random': uses random
        policy, or 'center': uses center look policy.
      sampling_stddev: Sampling distribution standard deviation.
      stop_gradient_between_cells: (Boolean) Whether to stop the gradient
        between the classification and location sub cells of the DRAM cell.
      stop_gradient_after_glimpse: (Boolean) Whether to stop the gradient
        after the glimpse net output.

    Returns:
      logits_t: Model logits at each time point.
      locs_t: Glimpse locations at each time point.
    """
    batch_size = images.shape.as_list()[0]
    # Get context information for images.
    endpoints = {}
    # Ensure glimpses within image.
    prev_locations = None

    with tf.name_scope("pre_time"):
      with tf.name_scope("initial_state"):
        # Initial state zeros for rnn0 and contexts for rnn1
        state_rnn0 = self.dram_cell.zero_states[0](batch_size, tf.float32)
        state_rnn1 = self.dram_cell.zero_states[1](batch_size, tf.float32)
        state_rnn = (state_rnn0, state_rnn1)
        locations = tf.zeros(shape=(batch_size, 2))
        with tf.name_scope("dram_cell"):
          logits, endpoints_ = self.dram_cell(
              images,
              locations,
              state_rnn,
              use_resolution=self.use_resolution_for_location,
              prev_locations=prev_locations,
              is_training=is_training,
              policy=policy,
              sampling_stddev=sampling_stddev,
              stop_gradient_between_cells=stop_gradient_between_cells,
              stop_gradient_after_glimpse=stop_gradient_after_glimpse
              )
          cell_outputs = endpoints_["cell_outputs"]
          locations, mean_locations = (cell_outputs["locations"],
                                       cell_outputs["mean_locations"])
    endpoints["pre_time"] = endpoints_
    endpoints["pre_time"]["logits"] = logits
    # Set state of the classification network to 0, but keep location state.
    state_rnn = (state_rnn0, cell_outputs["state_rnn"][1])
    # Unrolling the model in time.
    logits_t = []
    locations_t = []
    mean_locations_t = []
    if self.limit_classification_rf:
      # Use only the high resolution glimpse.
      use_resolution_for_classification = [
          True] + [False for _ in range(self.num_resolutions-1)]
    else:
      # Use glimpses from all resolutions.
      use_resolution_for_classification = [
          True for _ in range(self.num_resolutions)]

    for t in range(num_times):
      endpoints["time%d" % t] = {}
      locations_t.append(locations)
      mean_locations_t.append(mean_locations)
      if self.location_encoding == "relative":
        prev_locations = mean_locations
      elif self.location_encoding == "absolute":
        prev_locations = None

      with tf.name_scope("time%d" % t):
        with tf.name_scope("dram_cell"):
          logits, endpoints_ = self.dram_cell(
              images,
              locations,
              state_rnn,
              use_resolution=use_resolution_for_classification,
              prev_locations=prev_locations,
              is_training=is_training,
              policy=policy,
              sampling_stddev=sampling_stddev,
              stop_gradient_between_cells=stop_gradient_between_cells,
              stop_gradient_after_glimpse=stop_gradient_after_glimpse
              )
          cell_outputs = endpoints_["cell_outputs"]
          locations, state_rnn, _, mean_locations = (
              cell_outputs["locations"], cell_outputs["state_rnn"],
              cell_outputs["output_rnn"], cell_outputs["mean_locations"])
          endpoints["time%d" % t].update(endpoints_)
          logits_t.append(logits)

      if t == 0:
        self.collect_variables()

    return (logits_t, locations_t, mean_locations_t, endpoints)
