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

# python3
"""Shape and broadcasting modification modules.
"""

# pylint: disable=g-bad-import-order
import tensorflow.compat.v1 as tf

from weak_disentangle.tensorsketch.modules.base import Module


class Flatten(Module):
  """Flattens input along all dimensions except first dimension.
  """

  def forward(self, x):
    return tf.reshape(x, (x.shape[0], -1))


class Reshape(Module):
  """Reshape the input.
  """

  def __init__(self, shape):
    super().__init__()
    self.shape = shape

  def forward(self, x):
    return tf.reshape(x, self.shape)

  def extra_repr(self):
    return "({})".format(self.shape)
