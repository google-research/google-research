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

from non_semantic_speech_benchmark.trillsson import frontend_lib


class FrontendTest(absltest.TestCase):

  def test_keras_layer(self):
    layer = frontend_lib.SamplesToFeats()
    self.assertEqual(layer.frontend_args['frame_hop'], 195)
    self.assertEqual(layer.frontend_args['num_mel_bins'], 80)
    self.assertEqual(layer.frontend_args['frame_width'], 195)
    self.assertEqual(layer.frontend_args['pad_mode'], 'SYMMETRIC')
    self.assertEqual(layer.frontend_args['n_required'], 32000)

    for l, expected_l in [(5000, 1), (32000, 1), (63000, 2)]:
      samples = tf.zeros([2, l])
      feats = layer(samples)
      self.assertEqual(feats.shape, (2, expected_l, 195, 80))

  def test_keras_arg_overrides(self):
    frontend_args = {
        'n_required': 5000,
    }
    layer = frontend_lib.SamplesToFeats(frontend_args)
    self.assertEqual(layer.frontend_args['n_required'], 5000)


if __name__ == '__main__':
  absltest.main()
