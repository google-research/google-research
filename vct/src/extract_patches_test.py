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

"""Tests for extract_patches."""

from absl.testing import parameterized

import tensorflow as tf

from vct.src import extract_patches


def extract_patches_tf(image,
                       size,
                       stride = 1,
                       padding = "SAME"):
  """Plain tf patch extraction."""
  return tf.image.extract_patches(
      image,
      [1, size, size, 1],
      [1, stride, stride, 1],
      [1] * 4,
      padding,
  )


class ExtractPatchesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      size=(1, 2, 3, 4),
      stride=(1, 2),
      padding=("SAME", "VALID"),
      )
  def test_extract_patches_conv2d(self, size, stride, padding):
    image = tf.random.normal((2, 16, 16, 5))
    output = extract_patches.extract_patches_conv2d(
        image, size=size, stride=stride, padding=padding)
    expected = extract_patches_tf(
        image, size=size, stride=stride, padding=padding)
    self.assertAllClose(output, expected)

    if size == stride:
      with self.subTest("non_overlapping"):
        expected_non_overlapping = (
            extract_patches.extract_patches_nonoverlapping(
                image, window_size=stride, pad=False))
        self.assertAllClose(output, expected_non_overlapping)


class WindowPartitionTest(tf.test.TestCase):

  def test_non_overlapping(self):
    with self.subTest("non_padding"):
      image = tf.random.normal((2, 16, 16, 4))
      patches = extract_patches.window_partition(image, 4, pad=False)
      unpatched = extract_patches.unwindow(patches, 4, unpad=None)
      self.assertAllEqual(image, unpatched)

    with self.subTest("padding"):
      image = tf.random.normal((2, 14, 14, 4))
      patches = extract_patches.window_partition(image, 4, pad=True)
      unpatched = extract_patches.unwindow(patches, 4, unpad=(14, 14))
      self.assertAllEqual(image, unpatched)


if __name__ == "__main__":
  tf.test.main()
