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

# Lint as: python3
"""Tests for NOSS data prep."""

from absl.testing import absltest
from absl.testing import parameterized
import mock
import numpy as np
import tensorflow.compat.v2 as tf
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_utils


BASE_SHAPE_ = (15, 5)


def _s2e(audio_samples, sample_rate, module_location, output_key):
  """Mock waveform-to-embedding computation."""
  del audio_samples, sample_rate, module_location, output_key
  return np.zeros(BASE_SHAPE_, dtype=np.float32)


class AudioToEmbeddingsTests(parameterized.TestCase):

  @parameterized.parameters(
      {'average_over_time': True, 'sample_rate_key': 's', 'sample_rate': None},
      {'average_over_time': False, 'sample_rate_key': 's', 'sample_rate': None},
      {'average_over_time': False, 'sample_rate_key': None, 'sample_rate': 5},
  )
  @mock.patch.object(
      audio_to_embeddings_beam_utils,
      '_samples_to_embedding',
      new=_s2e)
  @mock.patch.object(audio_to_embeddings_beam_utils.hub, 'load',
                     new=lambda _: None)
  def test_compute_embedding_map_fn(self, average_over_time, sample_rate_key,
                                    sample_rate):
    # Establish required key names.
    audio_key = 'audio_key'

    # Construct the tf.train.Example test data.
    ex = tf.train.Example()
    ex.features.feature[audio_key].float_list.value.extend(
        np.zeros(2000, np.float32))
    if sample_rate_key:
      ex.features.feature[sample_rate_key].int64_list.value.append(8000)

    old_k = 'oldkey'

    do_fn = audio_to_embeddings_beam_utils.ComputeEmbeddingMapFn(
        name='module_name',
        module='@loc',
        output_key='unnecessary',
        audio_key=audio_key,
        sample_rate_key=sample_rate_key,
        sample_rate=sample_rate,
        average_over_time=average_over_time)
    do_fn.setup()
    new_k, new_v = next(do_fn.process((old_k, ex)))

    self.assertEqual(new_k, old_k)
    expected_shape = (1, BASE_SHAPE_[1]) if average_over_time else BASE_SHAPE_
    self.assertEqual(new_v.shape, expected_shape)

  @parameterized.parameters(
      {'dataset_name': 'crema_d'},
      {'dataset_name': 'speech_commands'},
      {'dataset_name': 'savee'},
      {'dataset_name': 'dementiabank'},
      {'dataset_name': 'voxceleb'},
  )
  def test_tfds_info(self, dataset_name):
    self.assertTrue(audio_to_embeddings_beam_utils._tfds_sample_rate(
        dataset_name))
    self.assertTrue(audio_to_embeddings_beam_utils._tfds_filenames(
        dataset_name, 'train'))
    self.assertTrue(audio_to_embeddings_beam_utils._tfds_filenames(
        dataset_name, 'validation'))
    self.assertTrue(audio_to_embeddings_beam_utils._tfds_filenames(
        dataset_name, 'test'))


if __name__ == '__main__':
  absltest.main()
