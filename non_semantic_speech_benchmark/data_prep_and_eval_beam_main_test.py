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

"""Tests for data_prep_and_eval_beam_main."""

import os
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import mock

from non_semantic_speech_benchmark import data_prep_and_eval_beam_main


def _validate(*args, **kwargs):
  del args, kwargs
  return None


def _read_glob(input_glob_flag, *args, **kwargs):
  del args, kwargs
  if input_glob_flag:
    return ([input_glob_flag], [f'{input_glob_flag}o'], 16000)
  else:
    return (['i1', 'i2', 'i3'], ['o1', 'o2', 'o3'], 16000)


def _none(*args, **kwargs):
  del args, kwargs
  return None


def _train_and_get_score(*args, **kwargs):
  del args, kwargs
  return (0.0, 0.0)


class DataPrepAndEvalBeamMainTest(parameterized.TestCase):

  @parameterized.parameters(
      {'tfds': True, 'data_prep_behavior': 'many_models'},
      {'tfds': False, 'data_prep_behavior': 'many_models'},
      {'tfds': True, 'data_prep_behavior': 'many_embeddings_single_model'},
      {'tfds': False, 'data_prep_behavior': 'many_embeddings_single_model'},
      {'tfds': True, 'data_prep_behavior': 'chunked_audio'},
      {'tfds': False, 'data_prep_behavior': 'chunked_audio'},
  )
  # Main validation mocks.
  @mock.patch.object(
      data_prep_and_eval_beam_main.old_prep_utils,
      'validate_inputs',
      new=_validate)
  @mock.patch.object(
      data_prep_and_eval_beam_main.sklearn_utils,
      'validate_flags',
      new=_validate)
  @mock.patch.object(
      data_prep_and_eval_beam_main.old_prep_utils,
      'read_input_glob_and_sample_rate_from_flags',
      new=_read_glob)
  # Data prep pipeline creation mocks.
  @mock.patch.object(
      data_prep_and_eval_beam_main.old_prep_utils,
      'make_beam_pipeline',
      new=_none)
  @mock.patch.object(
      data_prep_and_eval_beam_main.new_prep_utils,
      'multiple_embeddings_from_single_model_pipeline',
      new=_none)
  @mock.patch.object(
      data_prep_and_eval_beam_main.new_prep_utils,
      'precompute_chunked_audio_pipeline',
      new=_none)
  # Embedding eval pipeline creation mocks.
  @mock.patch.object(
      data_prep_and_eval_beam_main.sklearn_utils,
      'train_and_get_score',
      new=_train_and_get_score)
  @flagsaver.flagsaver
  def test_full_flow(self, tfds, data_prep_behavior):
    if tfds:
      flags.FLAGS.tfds_dataset = 'speech_commands'
    else:
      flags.FLAGS.train_input_glob = 'fn1'
      flags.FLAGS.validation_input_glob = 'fn2'
      flags.FLAGS.test_input_glob = 'fn3'
    flags.FLAGS.data_prep_behavior = data_prep_behavior
    flags.FLAGS.chunk_len = 10  # Ignored if "many_models".
    flags.FLAGS.audio_key = 'audio'
    flags.FLAGS.label_key = 'label'
    flags.FLAGS.output_filename = os.path.join(
        absltest.get_default_test_tmpdir(), 'tmp')
    flags.FLAGS.results_output_file = os.path.join(
        absltest.get_default_test_tmpdir(), 'sklearn_output')
    flags.FLAGS.embedding_names = ['emb1', 'emb2']
    if data_prep_behavior == 'many_models':
      flags.FLAGS.embedding_modules = ['mod1', 'mod2']
    else:
      flags.FLAGS.embedding_modules = ['mod1']
    flags.FLAGS.module_output_keys = ['k1', 'k2']
    flags.FLAGS.debug = True

    data_prep_and_eval_beam_main.main(None)


if __name__ == '__main__':
  absltest.main()
