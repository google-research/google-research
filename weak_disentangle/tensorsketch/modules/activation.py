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

"""Activation modules.
"""

# pylint: disable=g-bad-import-order
import tensorflow.compat.v1 as tf

from weak_disentangle.tensorsketch.modules.base import Module


class ReLU(Module):
  """Applies rectified-linear activation to input.
  """

  def __init__(self, name="relu"):
    super().__init__(name=name)

  def forward(self, x):
    return tf.nn.relu(x)


class LeakyReLU(Module):
  """Applies leaky rectified-linear activation to input.
  """

  def __init__(self, alpha=0.2, name="leaky_relu"):
    super().__init__(name=name)
    self.alpha = alpha

  def forward(self, x):
    return tf.nn.leaky_relu(x, alpha=self.alpha)

  def extra_repr(self):
    return "({})".format(self.alpha)


class Sigmoid(Module):
  """Sigmoid activation.
  """

  def forward(self, x):
    return tf.math.sigmoid(x)
