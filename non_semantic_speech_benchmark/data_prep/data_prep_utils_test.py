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

"""Tests for data_prep_utils."""

import copy
import os
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import tensorflow as tf
from non_semantic_speech_benchmark.data_prep import data_prep_utils


TEST_DIR = 'non_semantic_speech_benchmark/data_prep/testdata'


class MockModule(object):

  def __init__(self, output_keys):
    self.signatures = {'waveform': self._fn}
    self.output_keys = output_keys

  def _fn(self, waveform, paddings):
    del paddings
    bs = waveform.shape[0] if waveform.ndim == 2 else 1
    assert isinstance(bs, int)
    return {k: tf.zeros([bs, 5, 10]) for k in self.output_keys}


class DataPrepUtilsTest(parameterized.TestCase):

  def test_samples_to_embedding_tfhub_sanity(self):
    ret = data_prep_utils.samples_to_embedding_tfhub(
        model_input=tf.zeros([16000], tf.float32),
        sample_rate=16000,
        mod=MockModule(['okey1']),
        output_key='okey1',
        name='name')
    self.assertEqual(ret.ndim, 2)

  def test_samples_to_embedding_tfhub_w2v2_sanity(self):
    data_prep_utils.samples_to_embedding_tfhub_w2v2(
        tf.zeros([16000], tf.float32), MockModule(['okey1']))

  @parameterized.parameters(
      {'feature_inputs': True},
      {'feature_inputs': False},
  )
  def test_tflite_inference(self, feature_inputs):
    if feature_inputs:
      test_file = 'model1_woutfrontend.tflite'
    else:
      test_file = 'model1_wfrontend.tflite'
    tflite_model_path = os.path.join(absltest.get_default_test_srcdir(),
                                     TEST_DIR, test_file)
    output_key = '0'
    interpreter = data_prep_utils.build_tflite_interpreter(
        tflite_model_path=tflite_model_path)

    model_input = np.zeros([32000], dtype=np.float32)
    sample_rate = 16000
    if feature_inputs:
      model_input = data_prep_utils.default_feature_fn(
          model_input, sample_rate)
    else:
      model_input = np.expand_dims(model_input, axis=0)

    data_prep_utils.samples_to_embedding_tflite(
        model_input, sample_rate, interpreter, output_key, 'name')

  def test_add_key_to_audio_repeatable(self):
    """Make sure that repeated keys of the same samples are the same."""
    # TODO(joelshor): This step shouldn't depend on the random audio samples,
    # but set a seed if it does.
    audio_samples = np.random.random([64000]) * 2.0 - 1  # [-1, 1)
    ex = tf.train.Example()
    ex.features.feature['aud'].float_list.value.extend(audio_samples)
    # Use deepcopy to run the test with different objects that have the same
    # samples. In practice, this is more likely to be the way we expect this
    # function to behave.
    ex1 = data_prep_utils.add_key_to_audio(
        copy.deepcopy(ex), 'aud', 'k')
    ex2 = data_prep_utils.add_key_to_audio(
        copy.deepcopy(ex), 'aud', 'k')
    self.assertEqual(ex1.features.feature['k'].bytes_list.value[0],
                     ex2.features.feature['k'].bytes_list.value[0],)

  @parameterized.parameters(
      {'dataset_name': 'crema_d'},
      {'dataset_name': 'speech_commands'},
      {'dataset_name': 'savee'},
      {'dataset_name': 'dementiabank'},
      {'dataset_name': 'voxceleb'},
  )
  def test_tfds_info(self, dataset_name):
    self.assertTrue(data_prep_utils._tfds_sample_rate(
        dataset_name))
    self.assertTrue(data_prep_utils.tfds_filenames(
        dataset_name, 'train'))
    self.assertTrue(data_prep_utils.tfds_filenames(
        dataset_name, 'validation'))
    self.assertTrue(data_prep_utils.tfds_filenames(
        dataset_name, 'test'))

  def test_single_audio_emb_to_tfex(self):
    k = 'k'
    audio = np.zeros([16000], np.float32)
    embedding = np.ones([1024], np.float32)
    out_k, ex = data_prep_utils.single_audio_emb_to_tfex(
        k_v=('k', audio, embedding),
        embedding_name='ename',
        audio_key='audio_key',
        embedding_length=1024)
    self.assertEqual(out_k, k)
    self.assertIn('audio_key', ex.features.feature)
    self.assertIn('embedding/ename', ex.features.feature)

  def test_combine_multiple_embeddings_to_tfex(self):
    ex = tf.train.Example()
    audio_samples = np.zeros([64000], np.int64)
    ex.features.feature['aud'].int64_list.value.extend(audio_samples)
    ex.features.feature['lbl'].bytes_list.value.append(b'test')
    k, new_ex = data_prep_utils.combine_multiple_embeddings_to_tfex(
        ('key', ex, {'emb1': np.zeros([1, 10], np.float32)}),
        delete_audio_from_output=False,
        audio_key='aud',
        label_key='lbl',
        speaker_id_key=None)
    self.assertEqual(k, 'key')
    self.assertIn('aud', new_ex.features.feature)
    self.assertIsNotNone(new_ex.features.feature['aud'].float_list.value)

if __name__ == '__main__':
  absltest.main()
