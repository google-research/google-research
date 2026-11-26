# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for TSMixer's reversible instance-normalization layer."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf


# Load only the RevIN module, without importing optional dependencies in
# models/__init__.py needed by other TSMixer variants.
_SPEC = importlib.util.spec_from_file_location(
    'tsmixer_revin', Path(__file__).with_name('rev_in.py'))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RevNorm = _MODULE.RevNorm


class RevNormTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.inputs = tf.reshape(
        tf.cast(tf.range(48), tf.float32) / 48, (4, 3, 4))

  def test_default_and_explicit_none_preserve_shape_and_values(self):
    for affine in (True, False):
      with self.subTest(affine=affine):
        layer = RevNorm(axis=1, affine=affine)
        normalized = layer(self.inputs, mode='norm')
        for options in ({}, {'target_slice': None}):
          reconstructed = layer(normalized, mode='denorm', **options)
          self.assertEqual(reconstructed.shape, self.inputs.shape)
          np.testing.assert_allclose(
              reconstructed.numpy(), self.inputs.numpy(), atol=1e-5)

  def test_feature_slice_denormalizes_selected_channels(self):
    layer = RevNorm(axis=1)
    normalized = layer(self.inputs, mode='norm')
    reconstructed = layer(
        normalized[:, :, :2], mode='denorm', target_slice=slice(0, 2))
    self.assertEqual(reconstructed.shape, (4, 3, 2))
    np.testing.assert_allclose(
        reconstructed.numpy(), self.inputs[:, :, :2].numpy(), atol=1e-5)

  def test_default_denormalization_with_tf_function(self):
    layer = RevNorm(axis=1)

    @tf.function
    def roundtrip(inputs):
      normalized = layer(inputs, mode='norm')
      return layer(normalized, mode='denorm')

    reconstructed = roundtrip(self.inputs)
    self.assertEqual(reconstructed.shape, self.inputs.shape)
    np.testing.assert_allclose(
        reconstructed.numpy(), self.inputs.numpy(), atol=1e-5)


if __name__ == '__main__':
  unittest.main()
