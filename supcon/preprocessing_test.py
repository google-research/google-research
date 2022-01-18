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

# Lint as: python3
"""Tests for supcon.preprocessing."""

import contextlib
import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from supcon import enums
from supcon import hparams
from supcon import preprocessing


@contextlib.contextmanager
def nullcontext():
  yield


class PreprocessingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (True, False),  # decode_image
          (tf.uint8, tf.float32),  # image_dtype
          (224, 32),  # image_size
          (True, False),  # is_training
          (enums.AugmentationType.SIMCLR, enums.AugmentationType.AUTOAUGMENT,
           enums.AugmentationType.RANDAUGMENT
          ),  # augmentation_type (untested: IDENTITY, STACKED_RANDAUGMENT)
          (0., 0.5),  # warp_prob
          (0., 0.5),  # augmentation_magnitude
          (enums.EvalCropMethod.RESIZE_THEN_CROP,
           enums.EvalCropMethod.CROP_THEN_RESIZE,
           enums.EvalCropMethod.CROP_THEN_DISTORT),  # eval_crop_method
          (None, ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]))  # image_mean_std
      ))
  def test_preprocess_image(self, decode_image, image_dtype, image_size,
                            is_training, augmentation_type, warp_prob,
                            augmentation_magnitude, eval_crop_method,
                            image_mean_std):
    # tf.random.uniform() doesn't allow generating random values that are uint8.
    image = tf.cast(
        tf.random.uniform(
            shape=(300, 400, 3), minval=0, maxval=255, dtype=tf.float32),
        dtype=tf.uint8)
    if decode_image:
      image = tf.image.encode_jpeg(image)
    else:
      image = tf.cast(image, image_dtype)

    expect_error = (not decode_image and image_dtype != tf.uint8)
    if expect_error:
      context_manager = self.assertRaises(AssertionError)
    else:
      context_manager = nullcontext()
    with context_manager:
      output = preprocessing.preprocess_image(
          image,
          is_training=is_training,
          bfloat16_supported=False,
          preprocessing_options=hparams.ImagePreprocessing(
              image_size=image_size,
              augmentation_type=augmentation_type,
              warp_probability=warp_prob,
              augmentation_magnitude=augmentation_magnitude,
              eval_crop_method=eval_crop_method,
          ),
          dataset_options=preprocessing.DatasetOptions(
              image_mean_std=image_mean_std, decode_input=decode_image))
      self.assertEqual(output.dtype, tf.float32)
      self.assertEqual([image_size, image_size, 3], output.shape.as_list())

  @parameterized.parameters((0.,), (.5,), (1.,))
  def test_batch_random_blur(self, blur_prob):
    batch_size = 100
    side_length = 30
    image_batch = np.random.uniform(
        low=-1., high=1.,
        size=(batch_size, side_length, side_length, 3)).astype(np.float32)
    output = preprocessing.batch_random_blur(
        tf.constant(image_batch), side_length, blur_prob)
    with self.cached_session() as sess:
      output_np = sess.run(output)
    num_blurred = 0
    for i in range(batch_size):
      if not np.allclose(image_batch[i, Ellipsis], output_np[i, Ellipsis]):
        num_blurred += 1
    # Note there is some chance that these will fail due to randomness, but it
    # should be rare.
    if blur_prob < 1.:
      self.assertLess(num_blurred, batch_size)
    else:
      self.assertEqual(num_blurred, batch_size)
    if blur_prob > 0.:
      self.assertGreater(num_blurred, 0)
    else:
      self.assertEqual(num_blurred, 0)

  @parameterized.parameters(*itertools.product(
      (True, False),  # is_training
      (True, False),  # decode_input
      (1, 2, 3),  # num_views
  ))
  def test_image_to_multi_viewed_image_preprocessor(self, is_training,
                                                    decode_image, num_views):
    # tf.random.uniform() doesn't allow generating random values that are uint8.
    image = tf.cast(
        tf.random.uniform(
            shape=(300, 400, 3), minval=0, maxval=255, dtype=tf.float32),
        dtype=tf.uint8)
    if decode_image:
      image = tf.image.encode_jpeg(image)

    image_size = 32
    preprocessor = preprocessing.ImageToMultiViewedImagePreprocessor(
        is_training=is_training,
        preprocessing_options=hparams.ImagePreprocessing(
            image_size=image_size, num_views=num_views),
        dataset_options=preprocessing.DatasetOptions(decode_input=decode_image))
    output = preprocessor.preprocess(image)
    self.assertEqual(output.dtype, tf.float32)
    self.assertEqual([image_size, image_size, 3 * num_views],
                     output.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
