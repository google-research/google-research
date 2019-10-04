# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
"""Weight normalization utilities."""

# pylint: disable=g-importing-member, g-bad-import-order
import tensorflow as tf

from weak_disentangle.tensorsketch.modules.base import build_with_name_scope
from weak_disentangle.tensorsketch.normalization.base import KernelNorm


class WeightNorm(KernelNorm):
  """Weight Normalization class."""

  NAME = "weight_norm"

  def __init__(self, scale=True, axis=-1, epsilon=1e-5, name=None):
    super().__init__(name=name)
    self.use_scale = scale
    self.axis = axis
    self.epsilon = epsilon
    self.g = None

  @build_with_name_scope
  def build_parameters(self, kernel):
    if self.use_scale:
      # Since self.axis defines the axes of normalization,
      # the scale value should also be broadcast along those axes,
      # meaning scale resides in the remaining axes.
      # pylint: disable=protected-access
      shape = kernel._shape_as_list()
      try:
        shape[self.axis] = 1
      except TypeError:
        for i in self.axis:
          shape[i] = 1
      self.g = tf.Variable(tf.ones(shape), trainable=True)

  def reset_parameters(self):
    if self.use_scale:
      self.g.assign(tf.ones(self.g.shape))

  def forward(self, kernel):
    return self.normalize(kernel, self.g, self.axis, self.epsilon)

  @staticmethod
  def normalize(kernel, g, axis, epsilon):
    # Weight norm and what I'm currently doing are slightly different
    # in that the normalization axis is very different...
    # The easiest thing to do is to specify a normalization axis
    # So, adding 1e-3 works
    # kernel = tf.math.l2_normalize(kernel, axis=-1)
    kernel = kernel * tf.rsqrt(
        tf.reduce_sum(tf.square(kernel), axis=axis, keepdims=True) + epsilon)
    if g is not None:
      kernel = kernel * g
    return kernel

  @staticmethod
  def add(module, scale=True, axis=-1, epsilon=1e-5):
    KernelNorm.add(module, WeightNorm(scale, axis, epsilon))

  @staticmethod
  def remove(module):
    KernelNorm.remove(module, WeightNorm.NAME)

  def extra_repr(self):
    return "({}, {}, {})".format(self.use_scale,
                                 self.axis,
                                 self.epsilon)
