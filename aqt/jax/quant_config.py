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

"""Types for configuring quantization."""

import dataclasses
import enum
import typing

import flax

from aqt.jax.flax import struct as flax_struct

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass


class QuantGranularity(str, enum.Enum):
  """The granularity of the scale factors used for quantization."""
  # A single scale factor is used for an entire tensor.
  per_tensor = 'per_tensor'
  # For a weight tensor, a separate scale factor is used for each dimension
  # corresponding to an output channel (ie, the columns of the weight matrix if
  # a dense layer is thought of as an "activation tensor * weight tensor" matrix
  # multiplication).
  per_channel = 'per_channel'


@dataclass
class QuantContext:
  """Dynamic parameters visible to all model layers.

  A single instance of this dataclass is intended to be threaded throughout the
  entire hierarchy of model layers, making it easy when the model is called
  (whether for training or inference) for the caller to set contextual
  variables that all parts of the model that deal with quantization need.
  """

  # The construct `field_name = flax.struct.field(pytree_node=False)` causes a
  # field to be treated as a *static* argument to a JITed function. ie, the
  # function will recompile when the value of that field changes, but normal
  # Python control flow can be used inside the model (eg, `if
  # context.update_bounds:`) with those fields. Other fields will be treated as
  # *dynamic*, so the model will not recompile when those fields change, but
  # they cannot be used in normal Python control flow (typically, you would use
  # `lax.cond` instead).

  # Whether to update activation bounds.
  update_bounds: bool = flax.struct.field(pytree_node=False)

  # This will be passed to create_weight_ops in ConvAqt
  quantize_weights: bool = flax.struct.field(default=True, pytree_node=False)

  # Whether to tag activations to record statistics.
  collect_acts_stats: bool = flax.struct.field(default=False, pytree_node=False)

  # Whether to quantize activations.
  #
  # TODO(malmaud): This only applies to softmax for now. Apply it to layernorm
  # and dot operations as well.
  quantize_acts: bool = True

  # Whether to feed lax.dot inputs with an int8 dtype and accumulate to int32
  # dtype if quantizing both inputs to the dot operation to 8bits or 4bits (thus
  # ensuring the cast to int is lossless). If False, inputs are always
  # floating-point.
  prefer_int8_to_int32_dot: bool = True
