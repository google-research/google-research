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

"""Implement stacked layers.

Construct multiple layers and pass the output of one as input to the next layer.
Takes care of aggregating the auxiliary output (a.k.a. stats) of each layer by
stacking the PyTrees with a new leading dimension for the layers.
"""

from typing import Tuple, Union

import flax.deprecated.nn as nn
from lib import utils
from lib.typing import AuxOutput
from lib.typing import LayerInput


class StackedLayers(nn.Module):
  """Sequential application of a given layer."""

  def apply(
      self,
      x,
      layer,
      num_layers,
      *,
      is_training = True,
      with_aux_outputs = False
  ):
    """Stack `num_layers` time intependent `layer`s and aggregate aux output.

    Args:
      x: Input to the first layer.
      layer: Layer to be applied repeatedly to the input.
      num_layers: Number of repeated applications.
      is_training: Boolean passed to the layers (True if in training).
      with_aux_outputs: Specify if layer outputs auxiliary outputs that should
        be stacked and returned.

    Returns:
      The output of the last layer, if with_aux_outputs=True also returns the
      stacked auxiliary output of all the layers.
    """
    extra_outputs = []
    for _ in range(num_layers):
      x = layer(x, is_training=is_training)

      if with_aux_outputs:
        x, extra_output = x
        extra_outputs.append(extra_output)

    output = x
    if with_aux_outputs:
      # stack the extra outputs of the layers
      extra_outputs = utils.to_tree_arrays(extra_outputs)
      output = (x, extra_outputs)

    return output
