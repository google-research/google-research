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

"""Tests for frontend."""

from absl.testing import absltest

import tensorflow as tf

from non_semantic_speech_benchmark.distillation import frontend_lib


class FrontendTest(absltest.TestCase):

  def test_default_shape(self):
    self.assertEqual(frontend_lib.get_frontend_output_shape(), [1, 96, 64])

  def test_sample_to_features_nopadding(self):
    frontend_args = {
        'frame_hop': 5,
        'n_required': 16000,
        'num_mel_bins': 80,
        'frame_width': 5,
        'pad_mode': 'CONSTANT',
    }
    samples = tf.zeros([17000])
    feats = frontend_lib._sample_to_features(samples, frontend_args, False)
    print(f'feats: {feats}')
    self.assertEqual(feats.shape, (20, 5, 80))

  def test_sample_to_features_yespadding(self):
    frontend_args = {
        'frame_hop': 5,
        'n_required': 16000,
        'num_mel_bins': 80,
        'frame_width': 5,
        'pad_mode': 'CONSTANT',
    }
    samples = tf.zeros([5000])
    feats = frontend_lib._sample_to_features(samples, frontend_args, False)
    print(f'feats: {feats}')
    self.assertEqual(feats.shape, (19, 5, 80))

  def test_keras_layer(self):
    frontend_args = {
        'frame_hop': 5,
        'n_required': 16000,
        'num_mel_bins': 80,
        'frame_width': 5,
        'pad_mode': 'SYMMETRIC',
    }
    samples = tf.zeros([2, 5000])
    feats = frontend_lib.SamplesToFeats(False, frontend_args)(samples)
    print(f'feats: {feats}')
    self.assertEqual(feats.shape, (2, 19, 5, 80))


if __name__ == '__main__':
  absltest.main()
