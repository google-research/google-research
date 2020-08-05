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

# Lint as: python2, python3
"""Tests for mobile_search_space_v3."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from tunas import basic_specs
from tunas import mobile_search_space_v3
from tunas import schema
from tunas import schema_io


class MobileSearchSpaceV3Test(parameterized.TestCase):

  def test_get_strides_activation(self):
    self.assertEqual(
        (1, 1),
        mobile_search_space_v3.get_strides(mobile_search_space_v3.RELU))

  def test_get_strides_convolution(self):
    self.assertEqual(
        (1, 1),
        mobile_search_space_v3.get_strides(
            mobile_search_space_v3.ConvSpec(
                kernel_size=3,
                strides=1)))

    self.assertEqual(
        (2, 2),
        mobile_search_space_v3.get_strides(
            mobile_search_space_v3.ConvSpec(
                kernel_size=3,
                strides=2)))

  def test_get_strides_separable_convolution(self):
    self.assertEqual(
        (1, 1),
        mobile_search_space_v3.get_strides(
            mobile_search_space_v3.SeparableConvSpec(
                kernel_size=3,
                strides=1,
                activation='relu')))

    self.assertEqual(
        (2, 2),
        mobile_search_space_v3.get_strides(
            mobile_search_space_v3.SeparableConvSpec(
                kernel_size=3,
                strides=2,
                activation='relu')))

  def test_get_strides_depthwise_bottleneck(self):
    self.assertEqual(
        (1, 1),
        mobile_search_space_v3.get_strides(
            mobile_search_space_v3.DepthwiseBottleneckSpec(
                kernel_size=3,
                expansion_filters=72,
                use_squeeze_and_excite=False,
                strides=1,
                activation='relu')))

    self.assertEqual(
        (2, 2),
        mobile_search_space_v3.get_strides(
            mobile_search_space_v3.DepthwiseBottleneckSpec(
                kernel_size=3,
                expansion_filters=72,
                use_squeeze_and_excite=False,
                strides=2,
                activation='relu')))

  def test_get_strides_zero(self):
    self.assertEqual(
        (1, 1),
        mobile_search_space_v3.get_strides(basic_specs.ZeroSpec()))

  def test_get_strides_global_avg_pool(self):
    self.assertEqual(
        (None, None),
        mobile_search_space_v3.get_strides(
            mobile_search_space_v3.GlobalAveragePoolSpec()))

  def test_get_strides_residual_connection(self):
    self.assertEqual(
        (1, 1),
        mobile_search_space_v3.get_strides(
            mobile_search_space_v3.ResidualSpec(
                basic_specs.ZeroSpec())))

    with self.assertRaisesRegex(
        ValueError, 'Residual layer must have stride 1'):
      mobile_search_space_v3.get_strides(
          mobile_search_space_v3.ResidualSpec(
              mobile_search_space_v3.GlobalAveragePoolSpec()))

  def test_get_strides_oneof(self):
    self.assertEqual(
        (1, 1),
        mobile_search_space_v3.get_strides(
            schema.OneOf([
                mobile_search_space_v3.ConvSpec(kernel_size=3, strides=1),
                mobile_search_space_v3.ConvSpec(kernel_size=5, strides=1),
            ], basic_specs.OP_TAG)))

    self.assertEqual(
        (2, 2),
        mobile_search_space_v3.get_strides(
            schema.OneOf([
                mobile_search_space_v3.ConvSpec(kernel_size=3, strides=2),
                mobile_search_space_v3.ConvSpec(kernel_size=5, strides=2),
            ], basic_specs.OP_TAG)))

    with self.assertRaisesRegex(ValueError, 'Stride mismatch'):
      mobile_search_space_v3.get_strides(
          schema.OneOf([
              mobile_search_space_v3.ConvSpec(kernel_size=3, strides=1),
              mobile_search_space_v3.ConvSpec(kernel_size=3, strides=2),
          ], basic_specs.OP_TAG))

  def test_choose_filters(self):
    filters = [16, 24]
    one_of_filters = mobile_search_space_v3.choose_filters(filters)
    assert isinstance(one_of_filters, schema.OneOf)

  def test_mobilenet_v3_large(self):
    spec = mobile_search_space_v3.mobilenet_v3_large()
    self.assertIsInstance(spec, basic_specs.ConvTowerSpec)

  @parameterized.parameters(list(mobile_search_space_v3.ALL_SSDS))
  def test_get_search_space_spec(self, ssd):
    spec = mobile_search_space_v3.get_search_space_spec(ssd)
    self.assertIsInstance(spec, basic_specs.ConvTowerSpec)

  def test_serialize_and_deserialize(self):
    spec = mobile_search_space_v3.mobilenet_v3_large()

    serialized = schema_io.serialize(spec)
    self.assertIsInstance(serialized, str)

    deserialized = schema_io.deserialize(serialized)
    self.assertIsInstance(deserialized, basic_specs.ConvTowerSpec)
    self.assertEqual(deserialized, spec)


if __name__ == '__main__':
  absltest.main()
