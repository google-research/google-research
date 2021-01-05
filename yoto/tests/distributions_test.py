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

"""Tests for yoto.utils.distributions."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf

from yoto.optimizers import distributions


class DistributionsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      dict(shape=(2, 4, 8), return_numpy=True),
      dict(shape=(2, 3), return_numpy=True),
      dict(shape=(2, 3), return_numpy=False),)
  def test_sample(self, shape=None, return_numpy=None):
    tf.disable_eager_execution()
    distribution_spec = distributions.DistributionSpec(
        distributions.DistributionType.LOG_UNIFORM,
        {"low": 0.01, "high": 5.},
        distributions.TransformType.LOG)
    sample = distributions.get_sample(shape, distribution_spec, seed=17,
                                      return_numpy=return_numpy)
    if return_numpy:
      self.assertIsInstance(sample, np.ndarray)
      self.assertEqual(sample.shape, shape)
    else:
      self.assertIsInstance(sample, tf.Tensor)
      self.assertEqual(tuple(sample.get_shape().as_list()), shape)

    self.assertAllGreater(sample, np.log(0.0099))
    self.assertAllLess(sample, np.log(5.001))

  def test_get_transform(self):
    x = tf.constant([[-1., -2., -3.], [4., 5., 6.]])
    transform_types = [distributions.TransformType.IDENTITY,
                       distributions.TransformType.LOG]
    for transform_type in transform_types:
      transform = distributions.get_transform(transform_type)
      xt = transform(x)
      self.assertEqual(x.get_shape(), xt.get_shape())

    with self.assertRaises(ValueError):
      distributions.get_transform(1)
      distributions.get_transform({1: "a"})
      distributions.get_transform(None)


if __name__ == "__main__":
  tf.test.main()
