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

"""Component for model networks."""

import abc
from typing import Mapping, Optional

import tensorflow.compat.v1 as tf  # tf

# Read-only map from name to Tensor. Used in components that take or return
# multiple named Tensors (e.g. network).
TensorMap = Mapping[str, tf.Tensor]


class Network(metaclass=abc.ABCMeta):
  """Interface for the model network component.

  Note that the output Tensors are expected to be embeddings (e.g. "PreLogits"
  in InceptionV3) instead of predictions (e.g. "Logits"). Head components
  accept embeddings to produce predictions; this design allows us to support
  multi-head and reusing the same network body for different predictions
  (e.g. regression and classification).

  Implementations should provide a constructor that takes parameters needed for
  the specific model.

  Example usage:
  net = network.Inception_v3(inception_v3_params)
  prelogits = net.build(images)['out']
  """

  # ---------------------------------------------------------------------------
  # Standard keys for input Tensors to build().

  IMAGES = 'Images'

  # ---------------------------------------------------------------------------
  # Standard keys for output Tensors from build().

  PRE_LOGITS = 'PreLogits'
  LOGITS = 'Logits'
  PROBABILITIES_TENSOR = 'ProbabilitiesTensor'
  PROBABILITIES = 'Probabilities'
  ARM_OUTPUT_TENSOR = 'ArmOutputTensor'

  @abc.abstractmethod
  def build(self, inputs):
    """Builds the network.

    Args:
      inputs: a map from input string names to tensors.

    Returns:
      A map from output string names to tensors.
    """

  @staticmethod
  def _get_tensor(tmap,
                  name,
                  expected_rank = None):
    """Returns the specified Tensor from a TensorMap, with error-checking.

    Args:
      tmap: a mapping from string names to Tensors.
      name: the name of the Tensor to return.
      expected_rank: expected rank of the Tensor, for error-checking. Note that
        this checks static shape (e.g. via tensor.get_shape()). Defaults to not
        checked.

    Returns:
      The selected Tensor.

    Raises:
      ValueError: tmap does not contain the specified Tensor, or the Tensor is
        not of expected rank.
    """
    tensor = tmap.get(name, None)
    if tensor is None:
      raise ValueError('Tensor {} not found in TensorMap.'.format(name))
    rank = len(tensor.get_shape())
    if expected_rank is not None and rank != expected_rank:
      raise ValueError('Tensor {} is of rank {}, but expected {}.'.format(
          name, rank, expected_rank))
    return tensor
