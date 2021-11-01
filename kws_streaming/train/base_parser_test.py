# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
from kws_streaming.models import model_params
from kws_streaming.train import base_parser

FLAGS = None


class BaseParserTest(tf.test.TestCase):

  def test_default_values(self):
    params = model_params.Params()
    # validate default parameters to avoid regression
    self.assertEqual(FLAGS.lr_schedule, params.lr_schedule)
    self.assertEqual(FLAGS.optimizer, params.optimizer)
    self.assertEqual(FLAGS.background_volume, params.background_volume)
    self.assertEqual(FLAGS.l2_weight_decay, params.l2_weight_decay)
    self.assertEqual(FLAGS.background_frequency, params.background_frequency)
    self.assertEqual(FLAGS.split_data, params.split_data)
    self.assertEqual(FLAGS.silence_percentage, params.silence_percentage)
    self.assertEqual(FLAGS.unknown_percentage, params.unknown_percentage)
    self.assertEqual(FLAGS.time_shift_ms, params.time_shift_ms)
    self.assertEqual(FLAGS.testing_percentage, params.testing_percentage)
    self.assertEqual(FLAGS.validation_percentage, params.validation_percentage)
    self.assertEqual(FLAGS.how_many_training_steps,
                     params.how_many_training_steps)
    self.assertEqual(FLAGS.eval_step_interval, params.eval_step_interval)
    self.assertEqual(FLAGS.learning_rate, params.learning_rate)
    self.assertEqual(FLAGS.batch_size, 100)
    self.assertEqual(FLAGS.optimizer_epsilon, params.optimizer_epsilon)
    self.assertEqual(FLAGS.resample, params.resample)
    self.assertEqual(FLAGS.sample_rate, params.sample_rate)
    self.assertEqual(FLAGS.volume_resample, params.volume_resample)
    self.assertEqual(FLAGS.clip_duration_ms, 1000)
    self.assertEqual(FLAGS.window_size_ms, params.window_size_ms)
    self.assertEqual(FLAGS.window_stride_ms, params.window_stride_ms)
    self.assertEqual(FLAGS.preprocess, params.preprocess)
    self.assertEqual(FLAGS.feature_type, params.feature_type)
    self.assertEqual(FLAGS.preemph, params.preemph)
    self.assertEqual(FLAGS.window_type, params.window_type)
    self.assertEqual(FLAGS.mel_lower_edge_hertz, params.mel_lower_edge_hertz)
    self.assertEqual(FLAGS.mel_upper_edge_hertz, params.mel_upper_edge_hertz)
    self.assertEqual(FLAGS.log_epsilon, params.log_epsilon)
    self.assertEqual(FLAGS.dct_num_features, params.dct_num_features)
    self.assertEqual(FLAGS.use_tf_fft, params.use_tf_fft)
    self.assertEqual(FLAGS.mel_non_zero_only, params.mel_non_zero_only)
    self.assertEqual(FLAGS.fft_magnitude_squared, params.fft_magnitude_squared)
    self.assertEqual(FLAGS.mel_num_bins, params.mel_num_bins)
    self.assertEqual(FLAGS.use_spec_augment, params.use_spec_augment)
    self.assertEqual(FLAGS.time_masks_number, params.time_masks_number)
    self.assertEqual(FLAGS.time_mask_max_size, params.time_mask_max_size)
    self.assertEqual(FLAGS.frequency_masks_number,
                     params.frequency_masks_number)
    self.assertEqual(FLAGS.frequency_mask_max_size,
                     params.frequency_mask_max_size)
    self.assertEqual(FLAGS.return_softmax,
                     params.return_softmax)
    self.assertEqual(FLAGS.use_spec_cutout, params.use_spec_cutout)
    self.assertEqual(FLAGS.spec_cutout_masks_number,
                     params.spec_cutout_masks_number)
    self.assertEqual(FLAGS.spec_cutout_time_mask_size,
                     params.spec_cutout_time_mask_size)
    self.assertEqual(FLAGS.spec_cutout_frequency_mask_size,
                     params.spec_cutout_frequency_mask_size)
    self.assertEqual(FLAGS.pick_deterministically,
                     params.pick_deterministically)


if __name__ == '__main__':
  # parser for training/testing data and speach feature flags
  parser = base_parser.base_parser()

  FLAGS, unparsed = parser.parse_known_args()
  tf.test.main()
