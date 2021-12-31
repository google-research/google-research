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

import os
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import apache_beam as beam
import tensorflow as tf
# Import `main` for the flags.
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_main  # pylint:disable=unused-import
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_utils

BASE_SHAPE_ = (15, 5)

TEST_DIR = 'non_semantic_speech_benchmark/data_prep/testdata'

FLAGS = flags.FLAGS


def make_tfexample(l):
  ex = tf.train.Example()
  ex.features.feature['audio'].float_list.value.extend([0.0] * l)
  ex.features.feature['label'].bytes_list.value.append(b'dummy_lbl')
  ex.features.feature['speaker_id'].bytes_list.value.append(b'dummy_spkr')
  return ex


class MockModule(object):

  def __init__(self, output_keys):
    self.signatures = {'waveform': self._fn}
    self.output_keys = output_keys

  def _fn(self, waveform, paddings):
    del paddings
    bs = waveform.shape[0]
    assert isinstance(bs, int)
    return {k: tf.zeros([bs, 5, 10]) for k in self.output_keys}


class AudioToEmbeddingsTests(parameterized.TestCase):

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

  def test_multiple_embeddings_from_single_model_pipeline(self):
    # Write some examples to dummy location.
    tmp_input = os.path.join(absltest.get_default_test_tmpdir(),
                             'input.tfrecord')
    tmp_output = os.path.join(absltest.get_default_test_tmpdir(),
                              'multiple_embs.tfrecord')
    with tf.io.TFRecordWriter(tmp_input) as writer:
      for _ in range(3):
        ex = make_tfexample(5)
        writer.write(ex.SerializeToString())

    with beam.Pipeline() as root:
      audio_to_embeddings_beam_utils.multiple_embeddings_from_single_model_pipeline(
          root,
          input_filenames=[tmp_input],
          output_filename=tmp_output,
          sample_rate=5,
          debug=True,
          embedding_names=['em1', 'em2'],
          embedding_modules=['dummy_mod_loc'],
          module_output_keys=['k1', 'k2'],
          audio_key='audio',
          sample_rate_key=None,
          label_key='label',
          speaker_id_key='speaker_id',
          average_over_time=True,
          delete_audio_from_output=False,
          chunk_len=0,
          embedding_length=10,
          input_format='tfrecord',
          output_format='tfrecord',
          setup_fn=lambda _: MockModule(['k1', 'k2']))

  def test_precompute_chunked_audio_pipeline(self):
    # Write some examples to dummy location.
    tmp_input = os.path.join(absltest.get_default_test_tmpdir(),
                             'input.tfrecord')
    tmp_output = os.path.join(absltest.get_default_test_tmpdir(),
                              'chunked.tfrecord')
    with tf.io.TFRecordWriter(tmp_input) as writer:
      for _ in range(3):
        ex = make_tfexample(5)
        writer.write(ex.SerializeToString())

    with beam.Pipeline() as root:
      audio_to_embeddings_beam_utils.precompute_chunked_audio_pipeline(
          root,
          input_filenames=[tmp_input],
          sample_rate=5,
          debug=False,
          embedding_names=['em1', 'em2'],
          embedding_modules=['dummy_mod_loc'],
          module_output_keys=['k1', 'k2'],
          audio_key='audio',
          sample_rate_key=None,
          output_filename=tmp_output,
          label_key='label',
          speaker_id_key='speaker_id',
          chunk_len=0,
          embedding_length=10,
          input_format='tfrecord',
          output_format='tfrecord',
          setup_fn=lambda _: MockModule(['k1', 'k2']))

  @parameterized.parameters(
      {'data_prep_behavior': 'many_models'},
      {'data_prep_behavior': 'many_embeddings_single_model'},
      {'data_prep_behavior': 'chunked_audio'},
      {'data_prep_behavior': 'batched_single_model'},
  )
  @flagsaver.flagsaver
  def test_read_flags_and_create_pipeline(self, data_prep_behavior):
    """Test that the read-from-flags and pipeline creation are synced."""
    FLAGS.input_glob = os.path.join(
        absltest.get_default_test_srcdir(), TEST_DIR, '*')
    FLAGS.output_filename = os.path.join(absltest.get_default_test_tmpdir(),
                                         f'{data_prep_behavior}.tfrecord')
    FLAGS.data_prep_behavior = data_prep_behavior

    FLAGS.embedding_modules = ['dummy_mod_loc']

    if data_prep_behavior == 'batched_single_model':
      FLAGS.embedding_names = ['em1']
      FLAGS.module_output_keys = ['k1']
    else:
      FLAGS.embedding_names = ['em1', 'em2']
      FLAGS.module_output_keys = ['k1', 'k2']
    FLAGS.sample_rate = 5
    FLAGS.audio_key = 'audio_key'
    FLAGS.label_key = 'label_key'
    FLAGS.batch_size = 2
    input_filenames_list, output_filenames, beam_params = audio_to_embeddings_beam_utils.get_beam_params_from_flags(
    )
    # Use the defaults, unless we are using TFLite models.
    self.assertNotIn('module_call_fn', beam_params)
    self.assertNotIn('setup_fn', beam_params)

    # Check that the arguments run through.
    audio_to_embeddings_beam_utils.data_prep_pipeline(
        root=beam.Pipeline(),
        input_filenames_or_glob=input_filenames_list[0],
        output_filename=output_filenames[0],
        data_prep_behavior=FLAGS.data_prep_behavior,
        beam_params=beam_params,
        suffix='s')


if __name__ == '__main__':
  absltest.main()
