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

"""Tests for non_semantic_speech_benchmark.eval_embedding.finetune.get_data."""

from absl.testing import absltest
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from non_semantic_speech_benchmark.distillation import get_data


HUB_HANDLE_ = 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3'


class GetDataTest(absltest.TestCase):

  def test_df_pipeline(self):
    samples_key = 'sample'
    model_output_key = 'embedding'
    model_output_dim = 512
    saved_model_fn_ = get_data.savedmodel_to_func(
        hub.load(HUB_HANDLE_), output_key=model_output_key)
    ds = tf.data.Dataset.from_tensors(
        {samples_key: tf.sparse.SparseTensor(
            indices=[[0, 0]], values=[1.0], dense_shape=[1, 32000])}).repeat()
    min_length = 15360  # 960 ms
    batch_size = 3
    ds = get_data.tf_data_pipeline(
        ds, saved_model_fn_, samples_key, min_length,
        batch_size, model_output_dim)
    for i, (wav_samples, embeddings) in enumerate(ds):
      wav_samples.shape.assert_is_compatible_with([batch_size, min_length])
      embeddings.shape.assert_is_compatible_with([batch_size, model_output_dim])
      if i > 2: break


if __name__ == '__main__':
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  absltest.main()
