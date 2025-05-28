# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Flax implementation of Myrtle-CNN."""

# pytype: disable=wrong-arg-count

import functools

from flax import linen as nn
import ml_collections


class MLP(nn.Module):
  """MultiLayerPerceptron model.

  Contains fully connected layers with ReLU activations between them.
  """
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):

    norm = functools.partial(
        nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5)

    for i, feat in enumerate(self.config.widths + [1]):
      x = nn.Dense(
          feat,
          name=f'layers_{i}',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)
      if i != len(self.config.widths + [1]) - 1:
        x = norm()(x)
        x = nn.relu(x)

    _ = norm()(x)

    return x / 4
