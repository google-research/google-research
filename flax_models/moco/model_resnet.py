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


# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResNet implementation for MoCo.
"""


from flax import nn
import jax.nn
from jax.nn import initializers
import jax.numpy as jnp


class BottleneckBlock(nn.Module):
  """Bottleneck ResNet block."""

  def apply(self, x, filters, strides=(1, 1), train=True):
    needs_projection = x.shape[-1] != filters * 4 or strides != (1, 1)
    batch_norm = nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5)
    conv = nn.Conv.partial(bias=False)
    y = conv(x, filters, (1, 1), (1, 1), name='conv1')
    y = batch_norm(y, name='bn1')
    y = jax.nn.relu(y)
    y = conv(y, filters, (3, 3), strides, name='conv2')
    y = batch_norm(y, name='bn2')
    y = jax.nn.relu(y)
    y = conv(y, filters * 4, (1, 1), (1, 1), name='conv3')
    y = batch_norm(y, name='bn3', scale_init=initializers.zeros)
    if needs_projection:
      x = conv(x, filters * 4, (1, 1), strides, name='proj_conv')
      x = batch_norm(x, name='proj_bn')
    return jax.nn.relu(x + y)


class ResNet(nn.Module):
  """ResNetV1.

  ResNet class with configurable size.

  `apply` method return a tuple of:
  - final dense / classification layer output
  - features fed into dense layer
  """

  def apply(self, x, num_outputs, num_filters=64, block_sizes=[3, 4, 6, 3],  # pylint: disable=dangerous-default-value
            train=True):
    x = nn.Conv(x, num_filters, (7, 7), (2, 2), bias=False,
                name='init_conv')
    x = nn.BatchNorm(x,
                     use_running_average=not train,
                     epsilon=1e-5,
                     name='init_bn')
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = BottleneckBlock(x, num_filters * 2 ** i, strides=strides,
                            train=train)
    x = jnp.mean(x, axis=(1, 2))
    x_clf = nn.Dense(x, num_outputs, name='clf')
    # We return both the outputs from the dense layer *and* the features
    # that go into it.
    return x_clf, x


ResNet50 = ResNet.partial(block_sizes=[3, 4, 6, 3])  # pylint: disable=invalid-name
ResNet101 = ResNet.partial(block_sizes=[3, 4, 23, 3])  # pylint: disable=invalid-name
ResNet152 = ResNet.partial(block_sizes=[3, 8, 36, 3])  # pylint: disable=invalid-name

