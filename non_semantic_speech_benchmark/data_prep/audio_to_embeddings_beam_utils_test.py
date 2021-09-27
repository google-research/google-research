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

# Lint as: python3
"""Tests for NOSS data prep."""

import copy
import os
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_utils
from non_semantic_speech_benchmark.export_model import tf_frontend

BASE_SHAPE_ = (15, 5)

TEST_DIR = 'non_semantic_speech_benchmark/data_prep/testdata'


def _s2e(audio_samples, sample_rate, module_location, output_key, name):
  """Mock waveform-to-embedding computation."""
  del audio_samples, sample_rate, module_location, output_key, name
  return np.zeros(BASE_SHAPE_, dtype=np.float32)


def build_tflite_interpreter_dummy(tflite_model_path):
  del tflite_model_path
  return None


class FakeMod(object):

  def __call__(self, *args, **kwargs):
    del args, kwargs
    return {'output_key':
                np.zeros([BASE_SHAPE_[0], 1, BASE_SHAPE_[1]], np.float32)}


class AudioToEmbeddingsTests(parameterized.TestCase):

  @parameterized.parameters(
      {'average_over_time': True, 'sample_rate_key': 's', 'sample_rate': None},
      {'average_over_time': False, 'sample_rate_key': 's', 'sample_rate': None},
      {'average_over_time': False, 'sample_rate_key': None, 'sample_rate': 5},
  )
  # TODO(joelshor): Add test for signatures.
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
        output_key='output_key',
        audio_key=audio_key,
        sample_rate_key=sample_rate_key,
        sample_rate=sample_rate,
        average_over_time=average_over_time,
        setup_fn=lambda _: FakeMod())
    do_fn.setup()
    new_k, new_v = next(do_fn.process((old_k, ex)))

    self.assertEqual(new_k, old_k)
    expected_shape = (1, BASE_SHAPE_[1]) if average_over_time else BASE_SHAPE_
    self.assertEqual(new_v.shape, expected_shape)

  @parameterized.parameters(
      {'average_over_time': True, 'sample_rate_key': 's', 'sample_rate': None},
      {'average_over_time': False, 'sample_rate_key': 's', 'sample_rate': None},
  )
  def test_compute_embedding_map_fn_custom_call(self, average_over_time,
                                                sample_rate_key, sample_rate):
    # Establish required key names.
    audio_key = 'audio_key'
    custom_call_shape = (5, 25)

    # Custom call function for embedding generation.
    def test_call_fn(audio_samples, sample_rate, module_location, output_key,
                     name):
      """Mock waveform-to-embedding computation."""
      del audio_samples, sample_rate, module_location, output_key, name
      return np.zeros(custom_call_shape, dtype=np.float32)

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
        average_over_time=average_over_time,
        module_call_fn=test_call_fn,
        setup_fn=lambda _: None)
    do_fn.setup()
    new_k, new_v = next(do_fn.process((old_k, ex)))

    self.assertEqual(new_k, old_k)
    expected_shape = (
        1, custom_call_shape[1]) if average_over_time else custom_call_shape
    self.assertEqual(new_v.shape, expected_shape)

  @parameterized.parameters(
      {'average_over_time': True, 'sample_rate_key': 's', 'sample_rate': None},
      {'average_over_time': False, 'sample_rate_key': 's', 'sample_rate': None},
      {'average_over_time': False, 'sample_rate_key': None, 'sample_rate': 5},
  )
  def test_compute_embedding_map_fn_tflite(
      self, average_over_time, sample_rate_key, sample_rate):
    # Establish required key names.
    audio_key = 'audio_key'

    # Construct the tf.train.Example test data.
    ex = tf.train.Example()
    ex.features.feature[audio_key].float_list.value.extend(
        np.zeros(2000, np.float32))
    if sample_rate_key:
      ex.features.feature[sample_rate_key].int64_list.value.append(8000)

    old_k = 'oldkey'

    def _feature_fn(x, s):
      return tf.expand_dims(
          tf_frontend.compute_frontend_features(x, s, frame_hop=17),
          axis=-1).numpy().astype(np.float32)
    do_fn = audio_to_embeddings_beam_utils.ComputeEmbeddingMapFn(
        name='module_name',
        module='file.tflite',
        output_key=0,
        audio_key=audio_key,
        sample_rate_key=sample_rate_key,
        sample_rate=sample_rate,
        average_over_time=average_over_time,
        feature_fn=_feature_fn,
        module_call_fn=_s2e,
        setup_fn=build_tflite_interpreter_dummy)
    do_fn.setup()
    new_k, new_v = next(do_fn.process((old_k, ex)))

    self.assertEqual(new_k, old_k)
    expected_shape = (1, BASE_SHAPE_[1]) if average_over_time else BASE_SHAPE_
    self.assertEqual(new_v.shape, expected_shape)

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
    interpreter = audio_to_embeddings_beam_utils.build_tflite_interpreter(
        tflite_model_path=tflite_model_path)

    model_input = np.zeros([32000], dtype=np.float32)
    sample_rate = 16000
    if feature_inputs:
      model_input = audio_to_embeddings_beam_utils._default_feature_fn(
          model_input, sample_rate)
    else:
      model_input = np.expand_dims(model_input, axis=0)

    audio_to_embeddings_beam_utils.samples_to_embedding_tflite(
        model_input, sample_rate, interpreter, output_key, 'name')

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
    ex1 = audio_to_embeddings_beam_utils.add_key_to_audio(
        copy.deepcopy(ex), 'aud', 'k')
    ex2 = audio_to_embeddings_beam_utils.add_key_to_audio(
        copy.deepcopy(ex), 'aud', 'k')
    self.assertEqual(ex1.features.feature['k'].bytes_list.value[0],
                     ex2.features.feature['k'].bytes_list.value[0],)

  @parameterized.parameters(
      {'input_glob': True},
      {'input_glob': False},
  )
  def test_validate_inputs(self, input_glob):
    file_glob = os.path.join(absltest.get_default_test_srcdir(), TEST_DIR, '*')
    if input_glob:
      input_filenames_list = [[file_glob]]
    else:
      filenames = tf.io.gfile.glob(file_glob)
      input_filenames_list = [filenames]
    output_filenames = [
        os.path.join(absltest.get_default_test_tmpdir(), 'fake1')]
    embedding_modules = ['m1', 'm2']
    embedding_names = ['n1', 'n2']
    module_output_keys = ['k1', 'k2']
    # Check that inputs and flags are formatted correctly.
    audio_to_embeddings_beam_utils.validate_inputs(
        input_filenames_list=input_filenames_list,
        output_filenames=output_filenames,
        embedding_modules=embedding_modules,
        embedding_names=embedding_names,
        module_output_keys=module_output_keys)


if __name__ == '__main__':
  absltest.main()
