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

"""Tests for data_prep_utils."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam

import tensorflow as tf
from non_semantic_speech_benchmark.data_prep import data_prep_utils


class MockModule(object):

  def __init__(self, output_keys):
    self.signatures = {'waveform': self._fn}
    self.output_keys = output_keys

  def _fn(self, waveform, paddings):
    del paddings
    bs = waveform.shape[0]
    assert isinstance(bs, int)
    return {k: tf.zeros([bs, 5, 10]) for k in self.output_keys}


def make_tfexample(l):
  ex = tf.train.Example()
  ex.features.feature['audio'].float_list.value.extend([0.0] * l)
  ex.features.feature['label'].bytes_list.value.append(b'dummy_lbl')
  ex.features.feature['speaker_id'].bytes_list.value.append(b'dummy_spkr')
  return ex


class DataPrepUtilsTest(parameterized.TestCase):

  def test_samples_to_embedding_tfhub_sanity(self):
    data_prep_utils._samples_to_embedding_tfhub(
        tf.zeros([16000], tf.float32), MockModule(['okey1']))

  @parameterized.parameters(
      [{'chunk_len': 0, 'average_over_time': True},
       {'chunk_len': 8000, 'average_over_time': True},
       {'chunk_len': 0, 'average_over_time': False},
       {'chunk_len': 8000, 'average_over_time': False},
      ])
  def test_chunk_audio(self, chunk_len, average_over_time):
    dofn = data_prep_utils.ChunkAudioAndComputeEmbeddings(
        name='all',
        module='dummy_name',
        output_key=['okey1', 'okey2'],
        embedding_names=['em1', 'em2'],
        audio_key='audio',
        label_key='label',
        speaker_id_key='speaker_id',
        sample_rate_key=None,
        sample_rate=16000,
        average_over_time=average_over_time,
        chunk_len=chunk_len,
        setup_fn=lambda _: MockModule(['okey1', 'okey2']))
    dofn.setup()
    for l in [8000, 16000, 32000]:
      k = f'key_{l}'
      ex = make_tfexample(l)

      for i, (kn, aud, lbl, spkr, embs_d) in enumerate(dofn.process((k, ex))):
        self.assertEqual(f'{k}_{i}', kn)
        if chunk_len:
          expected_chunk_len = chunk_len if l > chunk_len else l
        else:
          expected_chunk_len = l
        self.assertLen(aud, expected_chunk_len)
        self.assertEqual(lbl, b'dummy_lbl')
        self.assertEqual(spkr, b'dummy_spkr')
        for _, emb in embs_d.items():
          self.assertEqual(emb.shape, (1 if average_over_time else 5, 10))

        # Now run the next stage of the pipeline on it.
        # TODO(joelshor): Add correctness checks on the output.
        data_prep_utils.chunked_audio_to_tfex(
            (kn, aud, lbl, spkr, embs_d),
            delete_audio_from_output=True,
            chunk_len=chunk_len, embedding_dimension=10)

  @parameterized.parameters(
      [{'chunk_len': 0, 'average_over_time': True},
       {'chunk_len': 8000, 'average_over_time': True},
       {'chunk_len': 0, 'average_over_time': False},
       {'chunk_len': 8000, 'average_over_time': False},
      ])
  def test_multiple_embeddings(self, chunk_len, average_over_time):
    dofn = data_prep_utils.ComputeMultipleEmbeddingsFromSingleModel(
        name='all',
        module='dummy_name',
        output_key=['k1', 'k2'],  # Sneak the list in.
        audio_key='audio',
        sample_rate_key=None,
        sample_rate=16000,
        average_over_time=average_over_time,
        feature_fn=None,
        embedding_names=['em1', 'em2'],
        embedding_length=10,
        chunk_len=chunk_len,
        setup_fn=lambda _: MockModule(['k1', 'k2'])
    )
    dofn.setup()
    for l in [8000, 16000, 32000]:
      k = f'key_{l}'
      ex = make_tfexample(l)
      kn, exn, emb_dict = list(dofn.process((k, ex)))[0]
      self.assertEqual(k, kn)
      self.assertLen(emb_dict, 2)
      self.assertSetEqual(set(emb_dict.keys()), set(['em1', 'em2']))

      # Now run the next stage of the pipeline on it.
      # TODO(joelshor): Add correctness checks on the output.
      data_prep_utils.add_embedding_fn((kn, exn, emb_dict),
                                       delete_audio_from_output=True,
                                       audio_key='audio',
                                       label_key='label',
                                       speaker_id_key='speaker_id')

  def test_mini_beam_pipeline(self):
    with beam.Pipeline() as root:
      _ = (root
           | beam.Create([('k1', make_tfexample(5)), ('k2', make_tfexample(5))])
           | beam.ParDo(
               data_prep_utils.ComputeMultipleEmbeddingsFromSingleModel(
                   name='all',
                   module='dummy_mod_loc',
                   output_key=['k1', 'k2'],
                   audio_key='audio',
                   sample_rate_key=None,
                   sample_rate=5,
                   average_over_time=True,
                   feature_fn=None,
                   embedding_names=['em1', 'em2'],
                   embedding_length=10,
                   chunk_len=0,
                   setup_fn=lambda _: MockModule(['k1', 'k2'])))
           | beam.Map(
               data_prep_utils.add_embedding_fn,
               delete_audio_from_output=True,
               audio_key='audio',
               label_key='label',
               speaker_id_key='speaker_id'))

  def test_multiple_embeddings_from_single_model_pipeline(self):
    # Write some examples to dummy location.
    tmp_input = os.path.join(absltest.get_default_test_tmpdir(),
                             'input.tfrecord')
    with tf.io.TFRecordWriter(tmp_input) as writer:
      for _ in range(3):
        ex = make_tfexample(5)
        writer.write(ex.SerializeToString())

    with beam.Pipeline() as root:
      data_prep_utils.multiple_embeddings_from_single_model_pipeline(
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
      data_prep_utils.precompute_chunked_audio_pipeline(
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
