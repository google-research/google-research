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

"""Tests for non_semantic_speech_benchmark.eval_embedding.finetune.get_data."""

from absl.testing import absltest
import tensorflow.compat.v2 as tf
from non_semantic_speech_benchmark.eval_embedding.finetune import get_data


class GetDataTest(absltest.TestCase):

  def test_df_pipeline(self):
    samples_key = 'sample'
    label_key = 'label'
    ds = tf.data.Dataset.from_tensors(
        {samples_key: tf.sparse.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 32000]),
         label_key: tf.constant(['test'])}).repeat()
    label_list = ['test', 'train']
    min_length = 16000
    batch_size = 3
    ds = get_data.tf_data_pipeline(
        ds, samples_key, label_key, label_list, min_length, batch_size)
    for i, (wav_samples, y_onehot) in enumerate(ds):
      wav_samples.shape.assert_is_compatible_with([batch_size, min_length])
      y_onehot.shape.assert_is_compatible_with(
          [batch_size, len(label_list)])
      if i > 2: break


if __name__ == '__main__':
  absltest.main()
