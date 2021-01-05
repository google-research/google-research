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

# Lint as: python2, python3
"""Utils for creating TensorFlow graphs for mobile classifier searches."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Optional, Sequence, Text, Union

from tunas import basic_specs
from tunas import mobile_search_space_v3
from tunas import search_space_utils
from tunas.rematlib import layers
from tunas.rematlib import mobile_model_v3


# List of supported search space definitions
ALL_SSDS = mobile_search_space_v3.ALL_SSDS


def get_model_spec(
    ssd,
    filters = None,
    op_indices = None,
    indices = None,
    filters_multipliers = 1.0,
    path_dropout_rate = 0.0,
    training = None):
  """Get a model spec namedtuple for the given search space definition.

  Args:
    ssd: Search space definition to use.
    filters: List of filter sizes to use. Required for V2 search spaces.
    op_indices: List of integers specifying the operations to select.
    indices: List of integers specifying the values to use for all the
      operations in the search space. If specified, this will override
      op_indices and filters.
    filters_multipliers: Single value or a list of possible values, used to
      scale up/down the number of filters in each layer of the network.
    path_dropout_rate: Rate of path dropout to use during stand-alone training.
    training: Boolean. Only needs to be specified if path_dropout_rate > 0.

  Returns:
    A basic_specs.ConvTowerSpec namedtuple.
  """
  if ssd in mobile_search_space_v3.ALL_SSDS:
    model_spec = mobile_search_space_v3.get_search_space_spec(ssd)
  else:
    raise ValueError('Unsupported SSD: {}'.format(ssd))

  if indices:
    genotype = indices
  else:
    genotype = dict()
    if op_indices:
      genotype[basic_specs.OP_TAG] = op_indices
    if filters and ssd in mobile_search_space_v3.ALL_SSDS:
      genotype[basic_specs.FILTERS_TAG] = filters

  model_spec = search_space_utils.prune_model_spec(
      model_spec, genotype, prune_filters_by_value=True,
      path_dropout_rate=path_dropout_rate, training=training)
  model_spec = search_space_utils.scale_conv_tower_spec(
      model_spec, filters_multipliers)
  return model_spec


def get_model_for_search(model_spec,
                         **kwargs):
  """Create a one-shot/weight-shared model based on `model_spec`.

  NOTE: We will always force the selected algorithm to run with stateless batch
  normalization. This makes the method suitable for architecture searches but
  not stand-alone model training.

  Args:
    model_spec: basic_specs.ConvTowerSpec namedtuple.
    **kwargs: Additional keyword arguments to pass to the model builder.

  Returns:
    An instance of rematlib/layers.Layer. The layer's apply() function will
    return a tuple `(output, endpoints)`, where `output` is a Tensor and
    `endpoints` is a list of Tensors for object detection.
  """
  return mobile_model_v3.get_model(
      model_spec=model_spec,
      force_stateless_batch_norm=True,
      **kwargs)


def get_standalone_model(model_spec,
                         **kwargs):
  """Create a standalone model based on `model_spec`.

  Args:
    model_spec: basic_specs.ConvTowerSpec namedtuple.
    **kwargs: Additional keyword arguments to pass to the model builder.

  Returns:
    An instance of rematlib/layers.Layer. The layer's apply() function will
    return a tuple `(output, endpoints)`, where `output` is a Tensor and
    `endpoints` is a list of Tensors for object detection.

  Raises:
    ValueError: If `ssd` is not recognized.
  """
  return mobile_model_v3.get_model(
      model_spec=model_spec,
      force_stateless_batch_norm=False,
      **kwargs)
