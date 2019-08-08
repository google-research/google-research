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

# Lint as: python2, python3
"""Tests for uq_benchmark_2019.gaussian_process_kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl.testing import parameterized
import tensorflow as tf

from uq_benchmark_2019 import gaussian_process_kernels

_FEATURE_SIZE = 5
_INITIAL_AMPLITUDE = [1.4]
_INITIAL_LEN_SCALE = [0.4, 0.9, -0.1, -1.2, 3.]
_BATCH_SIZE = 3


class GaussianProcessKernelsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(GaussianProcessKernelsTest, self).setUp()
    self._x1 = tf.constant([
        [0.57063887, 0.93843622, 0.7200364, 0.34000043, 0.68912818],
        [0.5429421, 0.46853128, 0.90759079, 0.44412577, 0.13539434],
        [0.4299868, 0.05476928, 0.21251964, 0.05090492, 0.28789705],
    ])
    self._x2 = tf.constant([
        [0.17382772, 0.64353101, 0.7684412, 0.05174528, 0.38692555],
        [0.46538115, 0.53920241, 0.07274304, 0.06132928, 0.18208745],
        [0.05350466, 0.8344726, 0.84254704, 0.91651488, 0.43557407],
    ])

  @parameterized.named_parameters([
      ('RBF',
       gaussian_process_kernels.RBFKernelFn,
       [3.49392, 1.774218, 0.755473]),
      ('Matern1',
       functools.partial(gaussian_process_kernels.MaternKernelFn, degree=1),
       [2.75733519, 1.35013103, 0.76525533]),
      ('Matern3',
       functools.partial(gaussian_process_kernels.MaternKernelFn, degree=3),
       [3.15654683, 1.53865254, 0.77299559]),
      ('Matern5',
       functools.partial(gaussian_process_kernels.MaternKernelFn, degree=5),
       [3.28316498, 1.60713243, 0.76870167]),
  ])
  def test_kernel_apply(self, kernel_provider_fn, true_value):
    kernel_provider = kernel_provider_fn(
        num_classes=10,
        per_class_kernel=False,
        feature_size=_FEATURE_SIZE,
        initial_amplitude=_INITIAL_AMPLITUDE,
        initial_length_scale=_INITIAL_LEN_SCALE,
        initial_linear_bias=[0.],
        initial_linear_slope=[1.],
        add_linear=True)
    kernel_value = kernel_provider.kernel.apply(
        self._x1, self._x2, example_ndims=1)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(kernel_value), true_value)


if __name__ == '__main__':
  tf.test.main()
