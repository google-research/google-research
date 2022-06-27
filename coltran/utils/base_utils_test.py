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

"""Tests for base_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import tensorflow as tf
from coltran.utils import base_utils


class UtilsTest(tf.test.TestCase):

  def test_quantize(self):
    x = tf.range(0, 256, dtype=tf.int32)
    actual = base_utils.convert_bits(x, n_bits_in=8, n_bits_out=5).numpy()
    expected = np.repeat(np.arange(0, 32), 8)
    self.assertTrue(np.allclose(expected, actual))

  def test_dequantize(self):
    x = tf.range(0, 32, dtype=tf.int32)
    actual = base_utils.convert_bits(x, n_bits_in=5, n_bits_out=8).numpy()
    expected = np.arange(0, 256, 8)
    self.assertTrue(np.allclose(expected, actual))

  def test_rgb_to_ycbcr(self):
    x = tf.random.uniform(shape=(2, 32, 32, 3))
    ycbcr = base_utils.rgb_to_ycbcr(x)
    self.assertEqual(ycbcr.shape, (2, 32, 32, 3))

  def test_image_hist_to_bit(self):
    x = tf.random.uniform(shape=(2, 32, 32, 3), minval=0, maxval=256,
                          dtype=tf.int32)
    hist = base_utils.image_to_hist(x, num_symbols=256)
    self.assertEqual(hist.shape, (2, 3, 256))

  def test_labels_to_bins(self):
    n_bits = 3
    bins = np.arange(2**n_bits)
    triplets = itertools.product(bins, bins, bins)

    labels = np.array(list(triplets))
    labels_t = tf.convert_to_tensor(labels, dtype=tf.float32)
    bins_t = base_utils.labels_to_bins(labels_t, num_symbols_per_channel=8)
    bins_np = bins_t.numpy()
    self.assertTrue(np.allclose(bins_np, np.arange(512)))

    inv_labels_t = base_utils.bins_to_labels(bins_t, num_symbols_per_channel=8)
    inv_labels_np = inv_labels_t.numpy()
    self.assertTrue(np.allclose(labels, inv_labels_np))

  def test_bins_to_labels_random(self):
    labels_t = tf.random.uniform(shape=(1, 64, 64, 3), minval=0, maxval=8,
                                 dtype=tf.int32)
    labels_np = labels_t.numpy()
    bins_t = base_utils.labels_to_bins(labels_t, num_symbols_per_channel=8)

    inv_labels_t = base_utils.bins_to_labels(bins_t, num_symbols_per_channel=8)
    inv_labels_np = inv_labels_t.numpy()
    self.assertTrue(np.allclose(inv_labels_np, labels_np))


if __name__ == '__main__':
  tf.test.main()
