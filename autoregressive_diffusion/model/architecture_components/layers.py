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

"""This contians a layer that handles input processing."""

from flax import linen as nn
import jax.numpy as jnp

Array = jnp.ndarray


class CausalConv(nn.Conv):
  """A (possibly) causal (1D) convolution."""
  is_causal: bool = False

  def __call__(self, inputs):
    """Apply (causal) covolution to inputs."""

    kernel_size = tuple(self.kernel_size)
    def maybe_broadcast(x):  # Copied from Flax.
      if x is None:
        # Backward compatibility with using None as sentinel for
        # broadcast 1.
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return x

    inputs_shape = inputs.shape
    if self.is_causal:
      if self.padding != 'VALID':
        raise ValueError(
            'Padding must be set to VALID for causal convolutions.')
      if len(kernel_size) != 1:
        raise ValueError('Only 1D causal convolutions are supported.')

      # Pad inputs to simulate causality.
      kernel_dilation = maybe_broadcast(self.kernel_dilation)
      padding = kernel_dilation[0] * (kernel_size[0] - 1)
      inputs = jnp.pad(inputs, [(0, 0), (padding, 0), (0, 0)])

    outputs = super().__call__(inputs)
    if self.is_causal:
      outputs = outputs[:, :inputs_shape[1], :]
    return outputs
