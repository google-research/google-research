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

"""Tests for non_semantic_speech_benchmark.eval_embedding.finetune.get_data."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_hub as hub
from non_semantic_speech_benchmark.distillation import get_data


HUB_HANDLE_ = 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3'


class GetDataTest(parameterized.TestCase):

  def setUp(self):
    super(GetDataTest, self).setUp()
    # Generate fake data.
    self.samples_key = 'samples'
    self.target_key = 'target'
    self.label_key = 'labels'
    self.output_dim = 5
    self.min_len = 6
    # Create tf.Example for audio-only case.
    ex = tf.train.Example()
    ex.features.feature[self.samples_key].float_list.value.extend(
        [1.0] * (self.min_len * 2))
    ex.features.feature[self.label_key].bytes_list.value.append(b'0')
    # Create tf.Example for precomputed case.
    precomputed_ex = tf.train.Example()
    precomputed_ex.features.feature[self.samples_key].float_list.value.extend(
        [1.0] * self.min_len)
    precomputed_ex.features.feature[self.target_key].float_list.value.extend(
        [0.0] * self.output_dim)
    precomputed_ex.features.feature[self.label_key].bytes_list.value.append(
        b'0')

    # Write to audio-only location.
    self.file_pattern = os.path.join(
        absltest.get_default_test_tmpdir(), 'test_data.tfrecord')
    with tf.io.TFRecordWriter(self.file_pattern) as file_writer:
      for _ in range(3):
        file_writer.write(ex.SerializeToString())
    # Write to precomputed location.
    self.precomputed_file_pattern = os.path.join(
        absltest.get_default_test_tmpdir(), 'precomputed_test_data.tfrecord')
    with tf.io.TFRecordWriter(self.precomputed_file_pattern) as file_writer:
      for _ in range(3):
        file_writer.write(precomputed_ex.SerializeToString())

  @parameterized.parameters(
      {'precomputed': True, 'read_labels': True},
      {'precomputed': False, 'read_labels': True},
      {'precomputed': True, 'read_labels': False},
      {'precomputed': False, 'read_labels': False},
  )
  def test_get_data(self, precomputed, read_labels):
    if precomputed:
      file_pattern = self.precomputed_file_pattern
      teacher_fn = None
      target_key = self.target_key
    else:
      file_pattern = self.file_pattern
      # Trivial function.
      teacher_fn = lambda x: tf.ones([tf.shape(x)[0], self.output_dim], tf.
                                     float32)
      target_key = None
    bs = 2
    ds = get_data.get_data(
        file_patterns=file_pattern,
        output_dimension=self.output_dim,
        reader=tf.data.TFRecordDataset,
        samples_key=self.samples_key,
        min_length=self.min_len,
        batch_size=bs,
        loop_forever=False,
        shuffle=True,
        teacher_fn=teacher_fn,
        target_key=target_key,
        label_key=self.label_key if read_labels else None,
        shuffle_buffer_size=2,
        normalize_to_pm_one=True)
    # Test that one element of the input pipeline can be successfully read.
    if read_labels:
      for wav_samples, targets, labels in ds:
        self.assertEqual(wav_samples.shape, [bs, self.min_len])
        self.assertEqual(targets.shape, [bs, self.output_dim])
        self.assertEqual(labels.shape, [bs])
        break
    else:
      for wav_samples, targets in ds:
        self.assertEqual(wav_samples.shape, [bs, self.min_len])
        self.assertEqual(targets.shape, [bs, self.output_dim])
        break

  def test_savedmodel_to_func(self):
    get_data.savedmodel_to_func(hub.load(HUB_HANDLE_), output_key='embedding')


if __name__ == '__main__':
  assert tf.executing_eagerly()
  absltest.main()
