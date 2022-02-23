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

"""Tests for non_semantic_speech_benchmark.trillsson.get_data."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
from non_semantic_speech_benchmark.trillsson import get_data


class GetDataTest(parameterized.TestCase):

  def setUp(self):
    super(GetDataTest, self).setUp()
    # Generate fake data.
    self.samples_key = 'samples'
    self.target_key = 'target'
    self.label_key = 'label'
    self.len = 5
    self.output_dim = 1024
    precomputed_ex1 = tf.train.Example()
    precomputed_ex1.features.feature[self.samples_key].float_list.value.extend(
        [1.0] * self.len)
    precomputed_ex1.features.feature[self.target_key].float_list.value.extend(
        [0.0] * self.output_dim)
    precomputed_ex1.features.feature[self.label_key].bytes_list.value.append(
        b'0')
    precomputed_ex2 = tf.train.Example()
    precomputed_ex2.features.feature[self.samples_key].float_list.value.extend(
        [1.0] * (self.len * 2))
    precomputed_ex2.features.feature[self.target_key].float_list.value.extend(
        [0.0] * self.output_dim)
    precomputed_ex2.features.feature[self.label_key].bytes_list.value.append(
        b'1')

    # Write to precomputed location.
    self.precomputed_file_pattern = os.path.join(
        absltest.get_default_test_tmpdir(), 'precomputed_test_data.tfrecord')
    with tf.io.TFRecordWriter(self.precomputed_file_pattern) as file_writer:
      for _ in range(2):
        file_writer.write(precomputed_ex1.SerializeToString())
      file_writer.write(precomputed_ex2.SerializeToString())
    self.precomputed_file_pattern = f'{self.precomputed_file_pattern}*'

    # Write some tfexamples with integer samples.
    precomputed_ex3 = tf.train.Example()
    precomputed_ex3.features.feature[self.samples_key].int64_list.value.extend(
        [10] * self.len)
    precomputed_ex3.features.feature[self.target_key].float_list.value.extend(
        [0.0] * self.output_dim)
    self.int_samples_file_pattern = os.path.join(
        absltest.get_default_test_tmpdir(), 'int_samples_test_data.tfrecord')
    with tf.io.TFRecordWriter(self.int_samples_file_pattern) as file_writer:
      for _ in range(3):
        file_writer.write(precomputed_ex3.SerializeToString())
    self.int_samples_file_pattern = f'{self.int_samples_file_pattern}*'

  def test_get_data(self):
    bs = 3
    ds = get_data.get_data(
        file_patterns=self.precomputed_file_pattern,
        output_dimension=self.output_dim,
        reader=tf.data.TFRecordDataset,
        samples_key=self.samples_key,
        target_key=self.target_key,
        batch_size=bs,
        loop_forever=False,
        shuffle=True,
        shuffle_buffer_size=2,
        min_samples_length=1)
    self.assertLen(ds.element_spec, 2)
    # Test that one element of the input pipeline can be successfully read.
    # Also, test that the shape is truncated to the shortest in the minibatch.
    has_elements = False
    for wav_samples, targets in ds:
      self.assertEqual(wav_samples.shape, [bs, self.len])
      self.assertEqual(targets.shape, [bs, self.output_dim])
      has_elements = True
      break
    self.assertTrue(has_elements)

  def test_get_data_out_len(self):
    bs = 3
    ds = get_data.get_data(
        file_patterns=self.precomputed_file_pattern,
        output_dimension=self.output_dim,
        reader=tf.data.TFRecordDataset,
        samples_key=self.samples_key,
        target_key=self.target_key,
        batch_size=bs,
        loop_forever=False,
        shuffle=True,
        shuffle_buffer_size=2,
        label_key='label')
    self.assertLen(ds.element_spec, 3)

  def test_get_data_max_len(self):
    bs = 3
    max_samples_length = 3
    assert max_samples_length < self.len
    ds = get_data.get_data(
        file_patterns=self.precomputed_file_pattern,
        output_dimension=self.output_dim,
        reader=tf.data.TFRecordDataset,
        samples_key=self.samples_key,
        target_key=self.target_key,
        batch_size=bs,
        loop_forever=False,
        shuffle=True,
        shuffle_buffer_size=2,
        min_samples_length=1,
        max_samples_length=max_samples_length)
    self.assertLen(ds.element_spec, 2)
    # Test that one element of the input pipeline can be successfully read.
    # Also, test that the shape is truncated to the shortest in the minibatch.
    has_elements = False
    for wav_samples, targets in ds:
      self.assertEqual(wav_samples.shape, [bs, max_samples_length])
      self.assertEqual(targets.shape, [bs, self.output_dim])
      has_elements = True
      break
    self.assertTrue(has_elements)

  def test_get_data_min_len(self):
    bs = 3
    min_samples_length = 1000
    ds = get_data.get_data(
        file_patterns=self.precomputed_file_pattern,
        output_dimension=self.output_dim,
        reader=tf.data.TFRecordDataset,
        samples_key=self.samples_key,
        target_key=self.target_key,
        batch_size=bs,
        loop_forever=False,
        shuffle=True,
        shuffle_buffer_size=2,
        min_samples_length=min_samples_length)
    has_elements = False
    for _ in ds:
      has_elements = True
      break
    self.assertFalse(has_elements)

  def test_get_data_int_samples(self):
    bs = 3
    ds = get_data.get_data(
        file_patterns=self.int_samples_file_pattern,
        output_dimension=self.output_dim,
        reader=tf.data.TFRecordDataset,
        samples_key=self.samples_key,
        target_key=self.target_key,
        batch_size=bs,
        loop_forever=False,
        shuffle=True,
        shuffle_buffer_size=2,
        min_samples_length=1,
        samples_are_float=False)
    self.assertLen(ds.element_spec, 2)
    # Test that one element of the input pipeline can be successfully read.
    # Also, test that the shape is truncated to the shortest in the minibatch.
    has_elements = False
    for wav_samples, targets in ds:
      self.assertEqual(wav_samples.shape, [bs, self.len])
      self.assertEqual(targets.shape, [bs, self.output_dim])
      has_elements = True
      break
    self.assertTrue(has_elements)


if __name__ == '__main__':
  assert tf.executing_eagerly()
  absltest.main()
