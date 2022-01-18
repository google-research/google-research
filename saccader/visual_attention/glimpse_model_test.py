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

"""Tests for glimpse model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from saccader.visual_attention import dram_config
from saccader.visual_attention import glimpse_model


class GlimpseModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(glimpse_model)

  @parameterized.named_parameters(
      ("training_mode", True),
      ("inference_mode", False),
  )
  def test_build(self, is_training):
    config = dram_config.get_config()
    image_shape = (28, 28, 1)
    batch_size = 10
    location_dims = 2
    output_dims = 10
    config.glimpse_model_config.output_dims = output_dims
    config.glimpse_model_config.glimpse_shape = config.glimpse_shape
    config.glimpse_model_config.num_resolutions = config.num_resolutions

    images = tf.placeholder(shape=(batch_size,) + image_shape, dtype=tf.float32)
    locations = tf.placeholder(shape=(batch_size, 2), dtype=tf.float32)
    model = glimpse_model.GlimpseNetwork(config.glimpse_model_config)

    g, _ = model(images, locations, is_training,
                 use_resolution=[True] * model.num_resolutions)
    init_op = model.init_op

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertEqual(
          (batch_size, output_dims),
          sess.run(
              g,
              feed_dict={
                  images: np.random.rand(*((batch_size,) + image_shape)),
                  locations: np.random.rand(batch_size, location_dims)
              }).shape)

  @parameterized.parameters(
      itertools.product(
          [True, False], itertools.product([True, False], repeat=3))
  )
  def test_use_resolution(self, is_training, use_resolution):
    config = dram_config.get_config()
    image_shape = (28, 28, 1)
    batch_size = 5
    output_dims = 10
    config.glimpse_model_config.output_dims = output_dims
    config.glimpse_model_config.glimpse_shape = config.glimpse_shape
    config.glimpse_model_config.num_resolutions = config.num_resolutions
    config.glimpse_model_config.glimpse_shape = (8, 8)
    config.glimpse_model_config.num_resolutions = 3
    locations = tf.placeholder(shape=(batch_size, 2), dtype=tf.float32)
    model = glimpse_model.GlimpseNetwork(config.glimpse_model_config)
    images = tf.random_uniform(
        minval=-1, maxval=1,
        shape=(batch_size,) + image_shape, dtype=tf.float32)
    locations = tf.zeros(shape=(batch_size, 2), dtype=tf.float32)
    model = glimpse_model.GlimpseNetwork(config.glimpse_model_config)
    g, endpoints = model(images, locations, is_training=is_training,
                         use_resolution=use_resolution)
    gnorms = [tf.norm(grad)
              for grad in tf.gradients(g[:, 0], endpoints["model_input_list"])]
    self.evaluate(tf.global_variables_initializer())
    gnorms = self.evaluate(gnorms)

    for use, gnorm in zip(use_resolution, gnorms):
      if use:
        self.assertGreater(gnorm, 0.)
      else:
        self.assertEqual(gnorm, 0.)

  @parameterized.parameters(True, False)
  def test_apply_stop_gradient(self, apply_stop_gradient):
    config = dram_config.get_config()
    image_shape = (28, 28, 1)
    batch_size = 10
    config.glimpse_model_config.output_dims = 10
    config.glimpse_model_config.glimpse_shape = config.glimpse_shape
    config.glimpse_model_config.num_resolutions = config.num_resolutions
    config.glimpse_model_config.apply_stop_gradient = apply_stop_gradient

    images = tf.placeholder(shape=(batch_size,) + image_shape, dtype=tf.float32)
    locations = tf.placeholder(shape=(batch_size, 2), dtype=tf.float32)
    model = glimpse_model.GlimpseNetwork(config.glimpse_model_config)

    outputs, _ = model(images, locations, False,
                       use_resolution=[True] * model.num_resolutions)
    gradients = tf.gradients([outputs[0, 0]], images)

    if apply_stop_gradient:
      self.assertEqual(gradients, [None])
    else:
      self.assertEqual(gradients[0].shape, images.shape)


if __name__ == "__main__":
  tf.test.main()
