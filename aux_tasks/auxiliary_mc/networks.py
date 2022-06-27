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

"""Networks for auxilairy MC experiments."""

from typing import Tuple

from flax import linen as nn
import gin


@gin.configurable
class Stack(nn.Module):
  """Stack of pooling and convolutional blocks with residual connections."""
  num_ch: int
  num_blocks: int
  use_max_pooling: bool = True

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    conv_out = nn.Conv(
        features=self.num_ch,
        kernel_size=(3, 3),
        strides=1,
        kernel_init=initializer,
        padding='SAME')(
            x)
    if self.use_max_pooling:
      conv_out = nn.max_pool(
          conv_out, window_shape=(3, 3), padding='SAME', strides=(2, 2))

    for _ in range(self.num_blocks):
      block_input = conv_out
      conv_out = nn.relu(conv_out)
      conv_out = nn.Conv(features=self.num_ch, kernel_size=(3, 3),
                         strides=1, padding='SAME')(conv_out)
      conv_out = nn.relu(conv_out)
      conv_out = nn.Conv(features=self.num_ch, kernel_size=(3, 3),
                         strides=1, padding='SAME')(conv_out)
      conv_out += block_input

    return conv_out


@gin.configurable
class ImpalaEncoder(nn.Module):
  """Impala Network which also outputs penultimate representation layers."""
  nn_scale: int = 1
  stack_sizes: Tuple[int, Ellipsis] = (16, 32, 32)
  num_blocks: int = 2

  @nn.compact
  def __call__(self, x):
    conv_out = x

    for stack_size in self.stack_sizes:
      conv_out = Stack(
          num_ch=stack_size * self.nn_scale, num_blocks=self.num_blocks)(
              conv_out)

    conv_out = nn.relu(conv_out)
    return conv_out
