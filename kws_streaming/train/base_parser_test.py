# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for base_parser default values."""

import tensorflow as tf
from kws_streaming.train import base_parser

FLAGS = None


class BaseParserTest(tf.test.TestCase):

  def test_default_values(self):
    # validate default parameters to avoid regression
    self.assertEqual(FLAGS.lr_schedule, 'linear')
    self.assertEqual(FLAGS.optimizer, 'adam')
    self.assertEqual(FLAGS.background_volume, 0.1)
    self.assertEqual(FLAGS.l2_weight_decay, 0.0)
    self.assertEqual(FLAGS.background_frequency, 0.8)
    self.assertEqual(FLAGS.split_data, 1)
    self.assertEqual(FLAGS.silence_percentage, 10.0)
    self.assertEqual(FLAGS.unknown_percentage, 10.0)
    self.assertEqual(FLAGS.time_shift_ms, 100.0)
    self.assertEqual(FLAGS.testing_percentage, 10)
    self.assertEqual(FLAGS.validation_percentage, 10)
    self.assertEqual(FLAGS.how_many_training_steps, '10000,10000,10000')
    self.assertEqual(FLAGS.eval_step_interval, 400)
    self.assertEqual(FLAGS.learning_rate, '0.0005,0.0001,0.00002')
    self.assertEqual(FLAGS.batch_size, 100)
    self.assertEqual(FLAGS.optimizer_epsilon, 1e-08)
    self.assertEqual(FLAGS.resample, 0.15)
    self.assertEqual(FLAGS.sample_rate, 16000)
    self.assertEqual(FLAGS.clip_duration_ms, 1000)
    self.assertEqual(FLAGS.window_size_ms, 40.0)
    self.assertEqual(FLAGS.window_stride_ms, 20.0)
    self.assertEqual(FLAGS.preprocess, 'raw')
    self.assertEqual(FLAGS.feature_type, 'mfcc_tf')
    self.assertEqual(FLAGS.preemph, 0.0)
    self.assertEqual(FLAGS.window_type, 'hann')
    self.assertEqual(FLAGS.mel_lower_edge_hertz, 20.0)
    self.assertEqual(FLAGS.mel_upper_edge_hertz, 7000.0)
    self.assertEqual(FLAGS.log_epsilon, 1e-12)
    self.assertEqual(FLAGS.dct_num_features, 20)
    self.assertEqual(FLAGS.use_tf_fft, 0)
    self.assertEqual(FLAGS.mel_non_zero_only, 1)
    self.assertEqual(FLAGS.fft_magnitude_squared, 0)
    self.assertEqual(FLAGS.mel_num_bins, 40)
    self.assertEqual(FLAGS.use_spec_augment, 0)
    self.assertEqual(FLAGS.time_masks_number, 2)
    self.assertEqual(FLAGS.time_mask_max_size, 10)
    self.assertEqual(FLAGS.frequency_masks_number, 2)
    self.assertEqual(FLAGS.frequency_mask_max_size, 5)


if __name__ == '__main__':
  # parser for training/testing data and speach feature flags
  parser = base_parser.base_parser()

  FLAGS, unparsed = parser.parse_known_args()
  tf.test.main()
