# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Saccader model.

Saccader model is an image classification model with a hard attention mechanism.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from saccader import utils
from saccader.feedforward import bagnet_model

MOMENTUM = 0.9
EPS = 1e-5


def gather_2d(images, offsets):
  """Extracts a elements from input tensor based on offsets.

  Args:
    images: A Tensor of type float32. A 4-D float tensor of shape [batch_size,
      height, width, channels].
    offsets: A Tensor of type float32. A 2-D integer tensor of shape
      [batch_size, 2] containing the x, y (range -1, 1).

  Returns:
    A Tensor of type float32.
  """
  batch_size, height, width, num_channels = images.shape.as_list()
  indices_height, indices_width = utils.normalized_locations_to_indices(
      offsets, height, width)

  # Compute linear indices into flattened images. If the indices along the
  # height or width dimension fall outside the image, we clip them to be the
  # nearest pixel inside the image.
  indices_batch = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
  indices_height = tf.reshape(indices_height, [batch_size, 1, 1])
  indices_width = tf.reshape(indices_width, [batch_size, 1, 1])

  # Gather into flattened images.
  return tf.reshape(
      tf.gather_nd(
          images, tf.stack((indices_batch, indices_height, indices_width), -1)),
      (batch_size, num_channels))


def build_attention_network(features2d,
                            attention_groups,
                            attention_layers_per_group,
                            is_training):
  """Builds attention network.

  Args:
    features2d: A Tensor of type float32. A 4-D float tensor of shape
      [batch_size, height, width, channels].
    attention_groups: (Integer) Number of network groups.
    attention_layers_per_group: (Integer) Number of layers per group.
    is_training: (Boolean) To indicate training or inference modes.

  Returns:
    features_embedded:  A Tensor of type float32. A 4-D float tensor of shape
      [batch_size, height, width, channels].
  """
  channels = features2d.shape.as_list()[-1]
  with tf.variable_scope("attention_network"):
    features_embedded = features2d
    for i in range(attention_groups):
      filters = channels // 2**(i+1)
      for j in range(attention_layers_per_group):
        features_embedded = tf.layers.conv2d(
            features_embedded,
            filters=filters,
            kernel_size=3 if j == (attention_layers_per_group-1)
            else 1,
            strides=1,
            dilation_rate=(2, 2) if j == (attention_layers_per_group-1)
            else (1, 1),
            activation=None,
            use_bias=False,
            name="features2d_embedding%d_%d" %(i, j),
            padding="same")
        features_embedded = tf.layers.batch_normalization(
            features_embedded, training=is_training,
            momentum=MOMENTUM, epsilon=EPS,
            name="features2d_embedding%d_%d" %(i, j))
        features_embedded = tf.nn.relu(features_embedded)
        tf.logging.info("Constructing layer: %s", features_embedded)

    return features_embedded


def sobel_edges(images):
  """Computes edge intensity of image using sobel operator."""
  batch_size, h, w, _ = images.shape.as_list()
  edges = tf.image.sobel_edges(tf.image.rgb_to_grayscale(images))
  edges = tf.reshape(edges, (batch_size, h, w, 2))
  edge_intensity = tf.norm(edges, ord="euclidean", axis=-1)
  return edge_intensity


def engineered_policies(images, logits2d, position_channels, glimpse_shape,
                        num_times, policy):
  """Engineered policies.

  Args:
    images: A Tensor of type float32. A 4-D float tensor of shape
      [batch_size, height, width, channels].
    logits2d: 2D logits tensor of type float32 of shape
      [batch_size, height, width, classes].
    position_channels: A Tensor of type float32 containing the output of
      `utils.position_channels` called on `images`.
    glimpse_shape: (Tuple of integer) Glimpse shape.
    num_times: (Integer) Number of glimpses.
    policy: (String) 'ordered logits', 'sobel_mean', or 'sobel_var'.


  Returns:
    locations_t: List of 2D Tensors containing policy locations.

  """
  if policy == "ordered_logits":
    pred_labels = tf.argmax(tf.reduce_mean(logits2d, axis=[1, 2]), axis=-1)
    metric2d = utils.batch_gather_nd(logits2d, pred_labels, axis=-1)

  elif "sobel" in policy:
    edges = sobel_edges(images)
    edges = edges[:, :, :, tf.newaxis]
    _, orig_h, orig_w, _ = edges.shape.as_list()
    _, h, w, _ = logits2d.shape.as_list()
    ksize = [1, glimpse_shape[0], glimpse_shape[1], 1]
    strides = [1,
               int(np.ceil((orig_h-glimpse_shape[0]+1) / h)),
               int(np.ceil((orig_w-glimpse_shape[1]+1) / w)),
               1]
    mean_per_glimpse = tf.nn.avg_pool(
        edges,
        ksize=ksize,
        strides=strides,
        padding="VALID"
        )
    if "mean" in policy:
      metric2d = mean_per_glimpse
    elif "var" in policy:
      n = np.prod(glimpse_shape)
      var_per_glimpse = (n / (n - 1.)) * (tf.nn.avg_pool(
          tf.square(edges),
          ksize=ksize,
          strides=strides,
          padding="VALID"
          ) - tf.square(mean_per_glimpse))
      metric2d = var_per_glimpse

    metric2d = tf.squeeze(metric2d, axis=3)
  _, locations_t = utils.sort2d(
      metric2d,
      position_channels,
      first_k=num_times,
      direction="DESCENDING")
  locations_t = tf.unstack(locations_t, axis=0)

  return locations_t


class SaccaderCell(object):
  """Saccader Cell.

  Network that emits a glimpse location at each time and store a memory of
  previously visited locations in cell memory state.

  Attributes:
    soft_attention: Emission network object.
    num_units: Glimpse network object.
    var_list_location: List of variables for the location network.
    var_list_classification: List of variables for the classification network.
    var_list: List of all model variables.
    init_op: Initialization operations for model variables.

  """

  def __init__(self, soft_attention):
    """Init.

    Args:
      soft_attention: (Boolean) If True use soft attention prediction.
    """
    self.soft_attention = soft_attention
    self.var_list = []
    self.init_op = None

  def collect_variables(self, vs):
    """Collects model variables.

    Args:
      vs: Tensorflow variables.

    Populates self.var_list with model variables and self.init_op with
    variables' initializer. This function is only called once with __call__.
    """
    # All variables.
    self.var_list = vs
    self.init_op = tf.variables_initializer(var_list=self.var_list)

  def __call__(self,
               mixed_features2d,
               cell_state,
               logits2d,
               is_training=False,
               policy="learned"):
    """Builds Saccader cell.

    Args:
      mixed_features2d: 4-D Tensor of shape [batch, height, width, channels].
      cell_state: 4-D Tensor of shape [batch, height, width, 1] with cell state.
      logits2d: 4-D Tensor of shape [batch, height, width, channels].
      is_training: (Boolean) To indicate training or inference modes.
      policy: (String) 'learned': uses learned policy, 'random': uses random
        policy, or 'center': uses center look policy.
    Returns:
      logits: Model logits.
      cell_state: New cell state.
      endpoints: Dictionary with cell parameters.
    """
    batch_size, height, width, channels = mixed_features2d.shape.as_list()
    reuse = True if self.var_list else False
    position_channels = utils.position_channels(mixed_features2d)

    variables_before = set(tf.global_variables())
    with tf.variable_scope("saccader_cell", reuse=reuse):
      # Compute 2D weights of features across space.
      features_space_logits = tf.layers.dense(
          mixed_features2d, units=1,
          use_bias=False, name="attention_weights") / tf.math.sqrt(
              float(channels))

      features_space_logits += (cell_state * -1.e5)  # Mask used locations.
      features_space_weights = utils.softmax2d(features_space_logits)

      # Compute 1D weights of features across channels.
      features_channels_logits = tf.reduce_sum(
          mixed_features2d * features_space_weights, axis=[1, 2])
      features_channels_weights = tf.nn.softmax(
          features_channels_logits, axis=1)

      # Compute location probability.
      locations_logits2d = tf.reduce_sum(
          (mixed_features2d *
           features_channels_weights[:, tf.newaxis, tf.newaxis, :]),
          axis=-1, keepdims=True)

      locations_logits2d += (cell_state * -1e5)  # Mask used locations.
      locations_prob2d = utils.softmax2d(locations_logits2d)

    variables_after = set(tf.global_variables())
    # Compute best locations.
    locations_logits = tf.reshape(
        locations_logits2d, (batch_size, -1))
    all_positions = tf.reshape(
        position_channels, [batch_size, height*width, 2])

    best_locations_labels = tf.argmax(locations_logits, axis=-1)
    best_locations = utils.batch_gather_nd(
        all_positions, best_locations_labels, axis=1)

    # Sample locations.
    if policy == "learned":
      if is_training:
        dist = tfp.distributions.Categorical(logits=locations_logits)
        locations_labels = dist.sample()
        # At training samples location from the learned distribution.
        locations = utils.batch_gather_nd(
            all_positions, locations_labels, axis=1)
        # Ensures range [-1., 1.]
        locations = tf.clip_by_value(locations, -1., 1)
        tf.logging.info("Sampling locations.")
        tf.logging.info("==================================================")
      else:
        # At inference uses the mean value for the location.
        locations = best_locations
        locations_labels = best_locations_labels
    elif policy == "random":
      # Use random policy for location.
      locations = tf.random_uniform(
          shape=(batch_size, 2),
          minval=-1.,
          maxval=1.)
      locations_labels = None
    elif policy == "center":
      # Use center look policy.
      locations = tf.zeros(
          shape=(batch_size, 2))
      locations_labels = None

    # Update cell_state.
    cell_state += utils.onehot2d(cell_state, locations)
    cell_state = tf.clip_by_value(cell_state, 0, 1)
    #########################################################################
    # Extract logits from the 2D logits.
    if self.soft_attention:
      logits = tf.reduce_sum(logits2d * locations_prob2d, axis=[1, 2])
    else:
      logits = gather_2d(logits2d, locations)
    ############################################################
    endpoints = {}
    endpoints["cell_outputs"] = {
        "locations": locations,
        "locations_labels": locations_labels,
        "best_locations": best_locations,
        "best_locations_labels": best_locations_labels,
        "locations_logits2d": locations_logits2d,
        "locations_prob2d": locations_prob2d,
        "cell_state": cell_state,
        "features_space_logits": features_space_logits,
        "features_space_weights": features_space_weights,
        "features_channels_logits": features_channels_logits,
        "features_channels_weights": features_channels_weights,
        "locations_logits": locations_logits,
        "all_positions": all_positions,
    }
    if not reuse:
      self.collect_variables(list(variables_after - variables_before))

    return logits, cell_state, endpoints


class Saccader(object):
  """Saccader Model.

  Network that performs classification on images by taking glimpses at
  different locations on an image.

  Attributes:
    config: (Configuration object) With attributes:
      num_classes: (Integer) Number of classification classes.
      attention_groups: (Integer) Number of groups in attention network.
      attention_layers_per_group: (Integer) Number of layers in each group in
        attention network.
      representation_config: (Configuration object) for representation network.
    variable_scope: (String) Name of model variable scope.
    saccader_cell: Saccader Cell object.
    representation_network: Representation network object.
    glimpse_shape: 2-D tuple of integers indicating glimpse shape.
    var_list_representation_network: List of variables for the representation
      network.
    var_list_attention_network: List of variables for the attention network.
    var_list_saccader_cell: List of variables for the saccader cell.
    var_list_location: List of variables for the location network.
    var_list_classification: List of variables for the classification network.
    var_list: List of all model variables.
    init_op: Initialization operations for model variables.
  """

  def __init__(self, config, variable_scope="saccader"):
    self.config = copy.deepcopy(config)
    representation_config = self.config.representation_config
    representation_config.num_classes = self.config.num_classes

    self.glimpse_shape = (-1, -1)
    self.variable_scope = variable_scope
    self.saccader_cell = SaccaderCell(soft_attention=self.config.soft_attention)
    self.representation_network = bagnet_model.BagNet(representation_config)
    self.var_list_saccader_cell = []
    self.var_list_representation_network = []
    self.var_list_attention_network = []
    self.var_list_classification = []
    self.var_list_location = []
    self.var_list = []
    self.init_op = None

  def collect_variables(self):
    """Collects model variables.

    Populates variable lists with model variables and self.init_op with
    variables' initializer. This function is only called once with __call__.
    """
    self.var_list_classification = self.var_list_representation_network
    self.var_list_location = self.var_list_attention_network
    self.var_list_location.extend(self.var_list_saccader_cell)
    self.var_list = (self.var_list_classification + self.var_list_location)
    self.init_op = tf.variables_initializer(var_list=self.var_list)

  def __call__(self,
               images,
               num_times,
               is_training=False,
               policy="learned",
               stop_gradient_after_representation=False):

    endpoints = {}
    reuse = True if self.var_list else False
    with tf.variable_scope(self.variable_scope+"/representation_network",
                           reuse=reuse):
      representation_logits, endpoints_ = self.representation_network(
          images, is_training)

    if not self.var_list_representation_network:
      self.var_list_representation_network = self.representation_network.var_list

    self.glimpse_shape = self.representation_network.receptive_field
    glimpse_size = tf.cast(self.glimpse_shape[0], dtype=tf.float32)
    image_size = tf.cast(tf.shape(images)[1], dtype=tf.float32)
    # Ensure glimpses within image.
    location_scale = 1. - glimpse_size / image_size
    endpoints["location_scale"] = location_scale
    endpoints["representation_network"] = endpoints_
    endpoints["representation_network"]["logits"] = representation_logits
    features2d = endpoints_["features2d"]  # size [batch, 28, 28, 2048]
    logits2d = endpoints_["logits2d"]  # size [batch, 28, 28, 1001]
    what_features2d = endpoints_["features2d_lowd"]
    endpoints["logits2d"] = logits2d
    endpoints["features2d"] = features2d
    endpoints["what_features2d"] = what_features2d

    # Freeze the representation network weights.
    if stop_gradient_after_representation:
      features2d = tf.stop_gradient(features2d)
      logits2d = tf.stop_gradient(logits2d)
      what_features2d = tf.stop_gradient(what_features2d)

    # Attention network.
    variables_before = set(tf.global_variables())
    with tf.variable_scope(self.variable_scope, reuse=reuse):
      where_features2d = build_attention_network(
          features2d,
          self.config.attention_groups,
          self.config.attention_layers_per_group,
          is_training)
      endpoints["where_features2d"] = where_features2d
      # Mix what and where features.
      mixed_features2d = tf.layers.conv2d(
          tf.concat([where_features2d, what_features2d], axis=-1),
          filters=512,
          kernel_size=1,
          strides=1,
          activation=None,
          use_bias=True,
          name="mixed_features2d",
          padding="same")
    endpoints["mixed_features2d"] = mixed_features2d
    variables_after = set(tf.global_variables())
    if not self.var_list_attention_network:
      self.var_list_attention_network = list(
          variables_after - variables_before)
    # Unrolling the model in time.
    classification_logits_t = []
    locations_t = []
    best_locations_t = []
    locations_logits2d_t = []
    batch_size, height, width, _ = mixed_features2d.shape.as_list()
    cell_state = tf.zeros((batch_size, height, width, 1), dtype=tf.float32)
    # Engineered policies.
    if policy in ["ordered_logits", "sobel_mean", "sobel_var"]:
      locations_t = engineered_policies(
          images,
          logits2d,
          utils.position_channels(logits2d) * location_scale,
          self.glimpse_shape,
          num_times,
          policy)

      best_locations_t = locations_t
      classification_logits_t = [
          gather_2d(logits2d, locations / location_scale)
          for locations in locations_t]
      # Run for 1 time to create variables (but output is unused).
      with tf.name_scope("time%d" % 0):
        with tf.variable_scope(self.variable_scope):
          self.saccader_cell(
              mixed_features2d,
              cell_state,
              logits2d,
              is_training=is_training,
              policy="random")

    # Other policies
    elif policy in ["learned", "random", "center"]:
      for t in range(num_times):
        endpoints["time%d" % t] = {}
        with tf.name_scope("time%d" % t):
          with tf.variable_scope(self.variable_scope):
            logits, cell_state, endpoints_ = self.saccader_cell(
                mixed_features2d,
                cell_state,
                logits2d,
                is_training=is_training,
                policy=policy)
          cell_outputs = endpoints_["cell_outputs"]
          endpoints["time%d" % t].update(endpoints_)
          classification_logits_t.append(logits)
          # Convert to center glimpse location on images space.
          locations_t.append(cell_outputs["locations"] * location_scale)
          best_locations_t.append(
              cell_outputs["best_locations"] * location_scale)
          locations_logits2d_t.append(cell_outputs["locations_logits2d"])
      endpoints["locations_logits2d_t"] = locations_logits2d_t
    else:
      raise ValueError(
          "policy can be either 'learned', 'random', or 'center'")

    if not self.var_list_saccader_cell:
      self.var_list_saccader_cell = self.saccader_cell.var_list
    self.collect_variables()
    endpoints["classification_logits_t"] = classification_logits_t
    logits = tf.reduce_mean(classification_logits_t, axis=0)
    return (logits, locations_t, best_locations_t, endpoints)
