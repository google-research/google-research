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

"""Spectral normalization utilities."""

# pylint: disable=g-importing-member, g-bad-import-order
import tensorflow.compat.v1 as tf

from weak_disentangle.tensorsketch.modules.base import build_with_name_scope
from weak_disentangle.tensorsketch.normalization.base import KernelNorm


class SpectralNorm(KernelNorm):
  """Spectral Normalization class."""

  NAME = "spectral_norm"

  def __init__(self, norm=1, name=None):
    super().__init__(name=name)
    self.norm = norm

  @build_with_name_scope
  def build_parameters(self, kernel):
    num_input, num_output = tf.reshape(kernel, (-1, kernel.shape[-1])).shape
    self.u = tf.Variable(
        tf.math.l2_normalize(tf.random.normal((num_output, 1))),
        trainable=False)
    self.v = tf.Variable(
        tf.math.l2_normalize(tf.random.normal((num_input, 1))),
        trainable=False)

  def reset_parameters(self):
    self.u.assign(tf.math.l2_normalize(tf.random.normal(self.u.shape)))
    self.v.assign(tf.math.l2_normalize(tf.random.normal(self.v.shape)))

  def forward(self, kernel):
    return self.normalize(kernel, self.u, self.v, self.norm, self.training)

  @staticmethod
  def normalize(kernel, u, v, norm, training):
    kernel_mat = tf.reshape(kernel, (-1, kernel.shape[-1]))
    if training:
      v_new = tf.stop_gradient(
          tf.math.l2_normalize(tf.matmul(kernel_mat, u)))
      u_new = tf.stop_gradient(
          tf.math.l2_normalize(tf.matmul(kernel_mat, v_new, transpose_a=True)))

      u.assign(u_new)
      v.assign(v_new)

    sigma = tf.reshape(tf.matmul(kernel_mat @ u, v, transpose_a=True), ())
    return kernel / sigma * norm

  @staticmethod
  def add(module, norm=1):
    KernelNorm.add(module, SpectralNorm(norm))

  @staticmethod
  def remove(module):
    KernelNorm.remove(module, SpectralNorm.NAME)

  def extra_repr(self):
    return "(norm={})".format(self.norm)
