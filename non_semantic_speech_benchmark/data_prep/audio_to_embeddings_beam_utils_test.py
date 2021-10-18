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
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import tensorflow as tf
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_utils

BASE_SHAPE_ = (15, 5)

TEST_DIR = 'non_semantic_speech_benchmark/data_prep/testdata'


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
    with tf.io.TFRecordWriter(tmp_input) as writer:
      for _ in range(3):
        ex = make_tfexample(5)
        writer.write(ex.SerializeToString())

    with beam.Pipeline() as root:
      audio_to_embeddings_beam_utils.multiple_embeddings_from_single_model_pipeline(
          root,
          input_filenames=[tmp_input],
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
          output_filename=os.path.join(absltest.get_default_test_tmpdir(),
                                       'output.tfrecord'),
          chunk_len=0,
          embedding_length=10,
          input_format='tfrecord',
          output_format='tfrecord',
          setup_fn=lambda _: MockModule(['k1', 'k2']))

  def test_precompute_chunked_audio_pipeline(self):
    # Write some examples to dummy location.
    tmp_input = os.path.join(absltest.get_default_test_tmpdir(),
                             'input.tfrecord')
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
          output_filename=os.path.join(absltest.get_default_test_tmpdir(),
                                       'output.tfrecord'),
          label_key='label',
          speaker_id_key='speaker_id',
          chunk_len=0,
          embedding_length=10,
          input_format='tfrecord',
          output_format='tfrecord',
          setup_fn=lambda _: MockModule(['k1', 'k2']))


if __name__ == '__main__':
  absltest.main()
