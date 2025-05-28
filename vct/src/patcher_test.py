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

"""Tests for patcher."""

from absl.testing import parameterized
import tensorflow as tf
from vct.src import extract_patches
from vct.src import patcher


class PatcherTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.product(h=(11, 16), w=(11, 16))
  def test_patch_unpatch_is_identity(self, h, w):
    x = tf.reshape(tf.range(h * w), (1, h, w, 1))
    x = tf.concat([x for _ in range(16)], -1)

    p = patcher.Patcher(stride=8)
    x_patched, (n_h, n_w) = p(x, patch_size=8)
    self.assertEqual(x_patched.shape, (4, 64, 16))
    first_patch = tf.reshape(x_patched[0, :64, :], (8, 8, 16))
    self.assertAllEqual(
        first_patch,
        x[0, :8, :8, :16],
        msg=f"{first_patch[..., 0]}\n\n{x[0, :8, :8, 0]}")

    x_unpatched = p.unpatch(x_patched, n_h, n_w, crop=(h, w))
    self.assertAllEqual(
        x_unpatched, x, msg=f"{x[0, ..., 0]}\n\n{x_unpatched[0, ..., 0]}")

  @parameterized.product(h=(13, 16), w=(13, 16))
  def test_padded_patching(self, h, w):
    x = tf.reshape(tf.range(h * w), (1, h, w, 1))
    x = tf.concat([x for _ in range(16)], -1)

    p = patcher.Patcher(stride=8)
    x_patched, _ = p(x, patch_size=16)
    self.assertEqual(x_patched.shape, (4, 256, 16))
    first_patch_without_padding = tf.reshape(
        x_patched, (4, 16, 16, 16))[0, 4:, 4:, :]
    expected_overlap = x[0, :12, :12, :]
    self.assertAllEqual(first_patch_without_padding, expected_overlap)

  @parameterized.product(h=(13, 16), w=(13, 16))
  def test_against_reshape_transpose(self, h, w):
    """Verify that patcher returns the same thing as the old code."""
    x = tf.reshape(tf.range(h * w), (1, h, w, 1))
    x = tf.concat([x for _ in range(16)], -1)
    p = patcher.Patcher(stride=8)
    x_patched, (n_h, n_w) = p(x, patch_size=8)
    b, seq_len, c = x_patched.shape
    x_patched_rt = extract_patches.window_partition(x, window_size=8, pad=True)
    self.assertEqual(x_patched_rt.shape,
                     (b // (n_h * n_w), n_h, n_w, seq_len, c))
    x_patched_rt = tf.reshape(x_patched_rt, (-1, seq_len, c))
    self.assertAllEqual(x_patched_rt, x_patched)


if __name__ == "__main__":
  tf.test.main()
