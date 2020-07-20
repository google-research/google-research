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

"""Models for experiments that involve MNIST and Omniglot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from learning_parameter_allocation import utils

from learning_parameter_allocation.pathnet import pathnet_lib as pn
from learning_parameter_allocation.pathnet.utils import \
    create_identity_input_layer

import tensorflow.compat.v1 as tf

from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import models


def get_keras_layers_for_mnist_experiment(num_components):
  """Get Keras layers for the MNIST experiment.

  Args:
    num_components: (int) number of components to use for every layer.

  Returns:
    A list of lists of `keras.layer.Layer`s, where the outer index corresponds
    to layer id, and inner index to component id within a layer.
  """
  keras_layers = []
  filters = 4

  keras_layers.append([
      layers.Conv2D(filters=filters, kernel_size=5, activation="relu")
      for _ in range(num_components)
  ])

  keras_layers.append([
      layers.AveragePooling2D(pool_size=2)
  ])

  keras_layers.append([
      layers.Conv2D(filters=filters, kernel_size=3, activation="relu")
      for _ in range(num_components)
  ])

  keras_layers.append([
      layers.AveragePooling2D(pool_size=2)
  ])

  keras_layers.append([
      layers.Conv2D(filters=filters, kernel_size=3, activation="relu")
      for _ in range(num_components)
  ])

  keras_layers.append([
      layers.Flatten()
  ])

  keras_layers.append([
      layers.Dropout(0.5)
  ])

  return keras_layers


class GroupNorm(layers.Lambda):
  """Layer that applies group normalization.

  Group Normalization is a technique introduced by the "Group Normalization"
  paper (https://arxiv.org/abs/1803.08494). This class provides a thin wrapper
  for `tf.contrib.layers.group_norm`.
  """

  def __init__(self, num_groups, *args, **kwargs):
    def group_norm_fn(in_tensor):
      return tf.contrib.layers.group_norm(in_tensor, groups=num_groups)

    super(GroupNorm, self).__init__(group_norm_fn, *args, **kwargs)


def parse_kernel_size(kernel_size_description):
  """Parses a kernel size from a string.

  Args:
    kernel_size_description: string of the form AxB where A and B are integers.

  Returns:
    A pair of integers (A, B).
  """
  a_string, b_string = kernel_size_description.split("x")
  return int(a_string), int(b_string)


def get_components_layer_for_general_diversity_and_depth_model(
    layer_description, num_filters, group_norm_num_groups, layer_strides):
  """Create a single routed layer.

  Args:
    layer_description: (list of string) description of the layer. Each element
      of the list described one component as a pipe-separated sequence of
      operations. Each operation should be one of the following: `convAxB`,
      `maxpoolAxB`, `avgpoolAxB`, `identity` where `A` and `B` are integers.
    num_filters: (int) number of filters for each convolution.
    group_norm_num_groups: (int) number of groups to use for group
      normalization; has to divide `num_filters`.
    layer_strides: (int) strides for this routed layer. In order to stride
      a routed layer, we stride every component within a layer. Since some
      components contain several operations (e.g. two convolutions), we apply
      the stride to the last operation in each component. To these operations
      we pass the value of `layer_strides` as the `strides` argument.

  Returns:
    A list of `keras.layer.Layer`s, corresponding to components in the routed
    layer.
  """
  components = []

  for component_description in layer_description:
    element_descriptions = component_description.split("|")

    component_elements = []

    for i, element_description in enumerate(element_descriptions):
      if i == len(element_descriptions) - 1:
        strides = layer_strides
      else:
        strides = 1

      if element_description == "identity":
        if strides == 1:
          component_elements.append(layers.Lambda(lambda x: x))
        else:
          # In case of strides, apply an additional 1x1 convolution.
          component_elements.append(layers.Conv2D(
              filters=num_filters,
              kernel_size=1,
              padding="same",
              strides=strides))
      elif element_description.startswith("conv"):
        kernel_size = parse_kernel_size(element_description[len("conv"):])

        component_elements.append(layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            strides=strides))

        component_elements.append(GroupNorm(num_groups=group_norm_num_groups))
        component_elements.append(layers.ReLU())
      elif element_description.startswith("maxpool"):
        pool_size = parse_kernel_size(element_description[len("maxpool"):])

        component_elements.append(layers.MaxPooling2D(
            pool_size=pool_size,
            padding="same",
            strides=strides))
      elif element_description.startswith("avgpool"):
        pool_size = parse_kernel_size(element_description[len("avgpool"):])

        component_elements.append(layers.AveragePooling2D(
            pool_size=pool_size,
            padding="same",
            strides=strides))
      else:
        assert False

    components.append(models.Sequential(component_elements))

  return components


def get_keras_layers_for_general_diversity_and_depth_model(
    layer_description, num_filters, num_layers, num_downsamples,
    group_norm_num_groups):
  """Gets Keras layers for the Omniglot and CIFAR-100 experiments.

  This model is a generalized version of the one proposed by the authors of
  "Diversity and Depth in Per-Example Routing Models"
  (https://openreview.net/pdf?id=BkxWJnC9tX).

  Args:
    layer_description: (list of string) description of a single layer, see
      `get_components_layer_for_general_diversity_and_depth_model`.
    num_filters: (int) number of filters for each convolution.
    num_layers: (int) number of layers.
    num_downsamples: (int) number of times the input should be downsampled by a
      factor of 2 before reaching the linear task-specific heads.
    group_norm_num_groups: (int) number of groups to use for group
      normalization.

  Returns:
    A list of lists of `keras.layer.Layer`s, where the outer index corresponds
    to layer id, and inner index to component id within a layer.
  """
  keras_layers = []

  # Initial shared 1x1 convolution, which increases the number of channels
  # from 1 to `num_filters`.
  keras_layers.append([
      layers.Conv2D(filters=num_filters, kernel_size=1, padding="same")
  ])

  keras_layers.append([
      GroupNorm(num_groups=group_norm_num_groups)
  ])

  keras_layers.append([
      layers.ReLU()
  ])

  downsampling_interval = num_layers / num_downsamples

  # Subset of `range(0, num_layers)` - subset of layers for downsampling.
  downsampling_layers = [
      int(downsampling_interval * i) for i in range(num_downsamples)]

  for layer_id in range(num_layers):
    if layer_id in downsampling_layers:
      layer_strides = 2
    else:
      layer_strides = 1

    keras_layers.append(
        get_components_layer_for_general_diversity_and_depth_model(
            layer_description, num_filters, group_norm_num_groups,
            layer_strides))

    keras_layers.append([
        GroupNorm(num_groups=group_norm_num_groups)
    ])

    keras_layers.append([
        layers.ReLU()
    ])

  # At this point, the feature map is `2^num_downsamples` times smaller.
  keras_layers.append([
      layers.Flatten()
  ])

  keras_layers.append([
      layers.Dropout(0.5)
  ])

  return keras_layers


def get_keras_layers_for_task_clusters_experiment():
  """Gets Keras layers for the experiment with three task clusters.

  Returns:
    A list of lists of `keras.layer.Layer`s, where the outer index corresponds
    to layer id, and inner index to component id within a layer.
  """
  return get_keras_layers_for_general_diversity_and_depth_model(
      layer_description=[
          "conv3x3", "conv3x3", "conv3x3", "conv3x3",
          "conv3x3", "conv3x3", "conv3x3", "conv3x3",
          "conv5x5", "conv5x5", "conv5x5", "conv5x5",
          "conv5x5", "conv5x5", "conv5x5", "conv5x5"],
      num_filters=8,
      num_layers=3,
      num_downsamples=3,
      group_norm_num_groups=4)


def get_keras_layers_for_omniglot_experiment():
  """Gets Keras layers for the Omniglot experiment.

  This model matches the one proposed by the authors of "Diversity and Depth
  in Per-Example Routing Models" (https://openreview.net/pdf?id=BkxWJnC9tX).

  Returns:
    A list of lists of `keras.layer.Layer`s, where the outer index corresponds
    to layer id, and inner index to component id within a layer.
  """
  return get_keras_layers_for_general_diversity_and_depth_model(
      layer_description=[
          "conv3x3|conv3x3",
          "conv5x5|conv5x5",
          "conv7x7|conv7x7",
          "conv1x7|conv7x1",
          "maxpool3x3",
          "avgpool3x3",
          "identity"],
      num_filters=48,
      num_layers=8,
      num_downsamples=5,
      group_norm_num_groups=4)


def wrap_router_fn(router_fn):
  """Wraps `router_fn` so that it is used only for non-trivial layers.

  Trivial layers, i.e. consiting of one component, will be handled simply by
  `pathnet_lib.SinglePathRouter`.

  Args:
    router_fn: function that, given a single argument `num_components`, returns
      a router (see routers in `pathnet/pathnet_lib.py`) for a layer containing
      `num_components` components.

  Returns:
    A function with the same behavior as the `router_fn` argument, except that
    when it gets passed `num_components=1`, it does not call `router_fn`,
    but instead returns `pathnet_lib.SinglePathRouter`.
  """
  def wrapped_router_fn(num_components):
    if num_components == 1:
      return pn.SinglePathRouter()
    else:
      return router_fn(num_components)

  return wrapped_router_fn


def build_model_from_keras_layers(
    input_data_shape, num_tasks, keras_layers, router_fn):
  """Creates PathNet layers from Keras layers.

  Args:
    input_data_shape: (sequence of ints) expected input shape.
    num_tasks: (int) number of tasks.
    keras_layers: (list of lists of `keras.layer.Layer`) keras layers to be
      wrapped into routed layers. The keras layers for a specific model can be
      obtained by calling `get_keras_layers_for_mnist_experiment`.
    router_fn: function that, given a single argument `num_components`, returns
      a router (see routers in `pathnet/pathnet_lib.py`) for a layer containing
      `num_components` components.

  Returns:
    A list of `pathnet_lib/ComponentsLayer`. The returned list starts with
    an input layer returned from `pathnet_lib.create_identity_input_layer`.
    The input/output shape for the subsequent layers is automatically computed
    by `utils.compute_output_shape_and_create_routed_layer`. The last layer
    uses an `IndependentTaskBasedRouter`, since it assumes the next layer
    contains task-specific heads.
  """
  pathnet_layers = []

  pathnet_layers.append(create_identity_input_layer(
      num_tasks, input_data_shape, router_out=pn.SinglePathRouter()))

  wrapped_router_fn = wrap_router_fn(router_fn)
  data_shape = input_data_shape

  for layer_index, layer in enumerate(keras_layers):
    if layer_index < len(keras_layers) - 1:
      # Not the last layer
      router_out = pn.SinglePathRouter()
    else:
      # The last layer - the layer after that will have task-specific heads
      router_out = pn.IndependentTaskBasedRouter(num_tasks=num_tasks)

    # Create routed layer and update current data shape
    new_layers, data_shape = utils.compute_output_shape_and_create_routed_layer(
        keras_components=layer,
        in_shape=data_shape,
        router_fn=wrapped_router_fn,
        router_out=router_out)

    pathnet_layers += new_layers

  return pathnet_layers
