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

"""Tests for beam_dofns."""

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import numpy as np
import tensorflow as tf
from non_semantic_speech_benchmark.data_prep import beam_dofns
from non_semantic_speech_benchmark.data_prep import data_prep_utils
from non_semantic_speech_benchmark.export_model import tf_frontend


BASE_SHAPE_ = (15, 5)


class FakeMod(object):

  def __call__(self, *args, **kwargs):
    del args, kwargs
    return {'output_key':
                np.zeros([BASE_SHAPE_[0], 1, BASE_SHAPE_[1]], np.float32)}


def build_tflite_interpreter_dummy(tflite_model_path):
  del tflite_model_path
  return None


def _s2e(audio_samples, sample_rate, module_location, output_key, name):
  """Mock waveform-to-embedding computation."""
  del audio_samples, sample_rate, module_location, output_key, name
  return np.zeros(BASE_SHAPE_, dtype=np.float32)


def make_tfexample(l):
  ex = tf.train.Example()
  ex.features.feature['audio'].float_list.value.extend([0.0] * l)
  ex.features.feature['label'].bytes_list.value.append(b'dummy_lbl')
  ex.features.feature['speaker_id'].bytes_list.value.append(b'dummy_spkr')
  ex.features.feature['sample_rate'].int64_list.value.append(32000)
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


class BeamDofnsTest(parameterized.TestCase):

  @parameterized.parameters(
      {'average_over_time': True, 'sample_rate_key': 's', 'sample_rate': None},
      {'average_over_time': False, 'sample_rate_key': 's', 'sample_rate': None},
      {'average_over_time': False, 'sample_rate_key': None, 'sample_rate': 5},
  )
  def test_compute_embedding_dofn(self, average_over_time, sample_rate_key,
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

    do_fn = beam_dofns.ComputeEmbeddingMapFn(
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
  def test_compute_embedding_dofn_custom_call(self, average_over_time,
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

    do_fn = beam_dofns.ComputeEmbeddingMapFn(
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
    do_fn = beam_dofns.ComputeEmbeddingMapFn(
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

  @parameterized.parameters([
      {
          'chunk_len': 0,
          'average_over_time': True,
          'emb_on_chnks': True
      },
      {
          'chunk_len': 8000,
          'average_over_time': True,
          'emb_on_chnks': True
      },
      {
          'chunk_len': 0,
          'average_over_time': False,
          'emb_on_chnks': True
      },
      {
          'chunk_len': 8000,
          'average_over_time': False,
          'emb_on_chnks': True
      },
      {
          'chunk_len': 0,
          'average_over_time': True,
          'emb_on_chnks': False
      },
      {
          'chunk_len': 8000,
          'average_over_time': True,
          'emb_on_chnks': False
      },
      {
          'chunk_len': 0,
          'average_over_time': False,
          'emb_on_chnks': False
      },
      {
          'chunk_len': 8000,
          'average_over_time': False,
          'emb_on_chnks': False
      },
  ])
  def test_chunk_audio(self, chunk_len, average_over_time, emb_on_chnks):
    dofn = beam_dofns.ChunkAudioAndComputeEmbeddings(
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
        compute_embeddings_on_chunked_audio=emb_on_chnks,
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
        data_prep_utils.chunked_audio_to_tfex((kn, aud, lbl, spkr, embs_d),
                                              delete_audio_from_output=True,
                                              chunk_len=chunk_len,
                                              embedding_length=10)

  @parameterized.parameters(
      [{'chunk_len': 0, 'average_over_time': True},
       {'chunk_len': 8000, 'average_over_time': True},
       {'chunk_len': 0, 'average_over_time': False},
       {'chunk_len': 8000, 'average_over_time': False},
      ])
  def test_multiple_embeddings(self, chunk_len, average_over_time):
    dofn = beam_dofns.ComputeMultipleEmbeddingsFromSingleModel(
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
      data_prep_utils.combine_multiple_embeddings_to_tfex(
          (kn, exn, emb_dict),
          delete_audio_from_output=True,
          audio_key='audio',
          label_key='label',
          speaker_id_key='speaker_id')

  @parameterized.parameters([
      {'process_fn': 'ComputeEmbeddingMapFn', 'chunk_len': 0},
      {'process_fn': 'ComputeMultipleEmbeddings', 'chunk_len': 0},
      {'process_fn': 'ComputeMultipleEmbeddings', 'chunk_len': 200},
      {'process_fn': 'ChunkAudioAndComputeEmbeddings', 'chunk_len': 0},
      {'process_fn': 'ChunkAudioAndComputeEmbeddings', 'chunk_len': 200},
  ])
  def test_pipeline_padding(self, process_fn, chunk_len):
    """Check that the model input is of sufficient length."""
    k, ex = 'key', make_tfexample(100)
    common_args = dict(
        name='name',
        module=None,
        output_key=['output_key'],
        audio_key='audio',
        sample_rate_key='sample_rate',
        sample_rate=None,
        average_over_time=True,
        model_input_min_length=400,
        setup_fn=lambda _: FakeMod())
    if process_fn == 'ComputeEmbeddingMapFn':
      beam_dofn = beam_dofns.ComputeEmbeddingMapFn(**common_args)
    elif process_fn == 'ComputeMultipleEmbeddings':
      beam_dofn = beam_dofns.ComputeMultipleEmbeddingsFromSingleModel(
          embedding_names=['em1'], chunk_len=chunk_len, **common_args)
    else:
      assert process_fn == 'ChunkAudioAndComputeEmbeddings'
      beam_dofn = beam_dofns.ChunkAudioAndComputeEmbeddings(
          embedding_names=['em1'], chunk_len=chunk_len, **common_args)

    # Run preprocessing step.
    beam_dofn.setup()
    if process_fn == 'ComputeEmbeddingMapFn':
      model_input, sample_rate = beam_dofn.read_and_preprocess_audio(k, ex)
      expected_output_shape = (400,)
    else:
      model_input, sample_rate = beam_dofn.tfex_to_chunked_audio(k, ex)
      expected_output_shape = (2, chunk_len) if chunk_len else (400,)

    # Original audio is too short, so it should be padded to
    # `model_input_min_length`.

    self.assertEqual(model_input.shape, expected_output_shape)

    # Having a non-standard sample rate should trigger resampling and cause the
    # output to be 16kHz.
    self.assertEqual(sample_rate, 16000)

  def test_mini_beam_pipeline(self):
    with beam.Pipeline() as root:
      _ = (
          root
          | beam.Create([('k1', make_tfexample(5)), ('k2', make_tfexample(5))])
          | beam.ParDo(
              beam_dofns.ComputeMultipleEmbeddingsFromSingleModel(
                  name='all',
                  module='dummy_mod_loc',
                  output_key=['k1', 'k2'],
                  audio_key='audio',
                  sample_rate_key='sample_rate',
                  sample_rate=None,
                  average_over_time=True,
                  feature_fn=None,
                  embedding_names=['em1', 'em2'],
                  embedding_length=10,
                  chunk_len=0,
                  setup_fn=lambda _: MockModule(['k1', 'k2'])))
          | beam.Map(
              data_prep_utils.combine_multiple_embeddings_to_tfex,
              delete_audio_from_output=True,
              audio_key='audio',
              label_key='label',
              speaker_id_key='speaker_id'))


if __name__ == '__main__':
  absltest.main()
