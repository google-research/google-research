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

"""Tests for `datasets.py`."""

import numpy as np
import tensorflow.compat.v1 as tf

from hypertransformer.tf.core import datasets


class DatasetsTest(tf.test.TestCase):

  def test_augmentation_config_randomization(self):
    """Testing randomization in AugmentationConfig."""
    aug_config = datasets.AugmentationConfig(
        random_config=datasets.RandomizedAugmentationConfig())
    rand_op = aug_config.randomize_op()
    angle = aug_config.angle.value
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(rand_op)
      v1 = sess.run(angle)
      v2 = sess.run(angle)
      sess.run(rand_op)
      v3 = sess.run(angle)
    self.assertAlmostEqual(v1, v2)
    self.assertNotAlmostEqual(v1, v3)

  def _test_augmentations(self, aug_config):
    """Just making sure that the image processing returns a proper shape."""
    shape = (4, 8, 8, 1)
    images = tf.ones(shape=shape, dtype=tf.float32)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      value = sess.run(aug_config.process(images))
    self.assertEqual(value.shape, shape)

  def test_no_augmentations(self):
    """Testing image processing without augmentations."""
    aug_config = datasets.AugmentationConfig(
        random_config=datasets.RandomizedAugmentationConfig(
            rotation_probability=0.0,
            smooth_probability=0.0,
            contrast_probability=0.0,
            resize_probability=0.0,
            negate_probability=0.0))
    self._test_augmentations(aug_config)

  def test_full_augmentations(self):
    """Testing image processing with all augmentations."""
    aug_config = datasets.AugmentationConfig(
        random_config=datasets.RandomizedAugmentationConfig(
            rotation_probability=1.0,
            smooth_probability=1.0,
            contrast_probability=1.0,
            resize_probability=1.0,
            negate_probability=1.0))
    self._test_augmentations(aug_config)

  def _make_data(self, batch_size = 4, image_size = 4):
    """Creates a dictionary for a fake NumPy dataset."""
    assert batch_size % 4 == 0
    repetitions = batch_size // 4
    ds = tf.data.Dataset.from_tensor_slices(
        {'image': np.zeros(shape=(batch_size, image_size, image_size, 1),
                           dtype=np.float32),
         'label': list(range(4)) * repetitions})

    with self.session() as sess:
      return datasets.make_numpy_data(sess, ds, batch_size=1, num_labels=4,
                                      samples_per_label=repetitions)

  def test_make_numpy_data(self):
    """Tests `make_numpy_data` function."""
    data = self._make_data()
    self.assertSequenceEqual(list(data.keys()), range(4))
    for label in range(4):
      self.assertLen(data[label], 1)

  def test_get_batch(self):
    """Tests image and label generation in the `TaskGenerator`."""
    batch_size, image_size = 8, 4
    data = self._make_data(batch_size=batch_size, image_size=image_size)
    gen = datasets.TaskGenerator(data, num_labels=4, image_size=image_size)
    aug_config = datasets.AugmentationConfig(
        random_config=datasets.RandomizedAugmentationConfig(
            rotation_probability=0.0,
            smooth_probability=0.0,
            contrast_probability=0.0))
    images, labels, classes = gen.get_batch(batch_size=batch_size,
                                            config=aug_config)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      v_images, v_labels, v_classes = sess.run((images, labels, classes))
    self.assertEqual(v_images.shape, (batch_size, image_size, image_size, 1))
    self.assertEqual(v_labels.shape, (batch_size,))
    self.assertEqual(v_classes.shape, (batch_size,))

if __name__ == '__main__':
  tf.test.main()
