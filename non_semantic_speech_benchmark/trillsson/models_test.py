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

"""Tests for non_semantic_speech_benchmark.distillation.models."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

import tensorflow as tf

from non_semantic_speech_benchmark.trillsson import models


class ModelsTest(parameterized.TestCase):


  @parameterized.parameters(
      {'model_type': 'efficientnetv2b0'},
  )
  @flagsaver.flagsaver
  def test_models_handle_arbitrary_len(self, model_type):
    flags.FLAGS.frame_hop = 195
    flags.FLAGS.num_mel_bins = 80
    flags.FLAGS.frame_width = 195
    flags.FLAGS.n_required = 32000
    m = models.get_keras_model(model_type=model_type)

    samples = tf.zeros([2, 63000], tf.float32)
    o = m(samples)
    self.assertIsInstance(o, dict)
    self.assertLen(o, 1)
    self.assertIn('embedding', o)
    o['embedding'].shape.assert_is_compatible_with([2, 1024])

  @flagsaver.flagsaver
  def test_frontend_shapes(self):
    flags.FLAGS.frame_hop = 195
    flags.FLAGS.num_mel_bins = 80
    flags.FLAGS.frame_width = 195
    flags.FLAGS.n_required = 32000

    model_in = tf.keras.Input((None,), name='audio_samples')

    # Set up manually average model.
    feats = models.frontend_keras(model_in, frame_hop=195)
    feats.shape.assert_is_compatible_with([None, None, None, 1])
    m1 = tf.keras.Model(inputs=[model_in], outputs=feats)
    # Run inference with manually average model.
    o1 = m1(tf.zeros([2, 32000]))
    self.assertEqual(o1.shape, (2, 195, 80, 1))
    o2 = m1(tf.zeros([2, 64000]))
    self.assertEqual(o2.shape, (4, 195, 80, 1))


if __name__ == '__main__':
  assert tf.executing_eagerly()
  absltest.main()
