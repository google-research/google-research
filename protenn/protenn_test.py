# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for protenn binary."""

from absl.testing import parameterized
import pandas as pd
import tensorflow.compat.v1 as tf
from protenn import predict
from protenn import test_util
from protenn import utils


class ProtennTest(parameterized.TestCase):

  def test_gcs_path_to_relative_unzipped_path(self):
    actual = predict._gcs_path_to_relative_unzipped_path(
        utils.OSS_PFAM_ZIPPED_MODELS_URLS[0]
    )
    expected = '31382545'
    self.assertEqual(actual, expected)

  def test_parse_input(self):
    input_file_path = self.create_tempfile(content='>SEQUENCE_NAME\nACDE')
    input_text = predict.parse_input_to_text(input_file_path.full_path)
    actual_df = predict.input_text_to_df(input_text)
    expected = pd.DataFrame({
        'sequence_name': ['SEQUENCE_NAME'],
        'sequence': ['ACDE'],
    })

    # BioPython parses sequences as Bio.Seq.Seq which can, in most cases,
    # act as sequences, but in others can lead to surprising behavior. Ensure
    # we actually have a str.
    self.assertEqual(type(actual_df.sequence.values[0]), str)
    test_util.assert_dataframes_equal(self, actual_df, expected)

  def test_parse_input_malformed_fasta(self):
    # input is missing fasta header line marker.
    input_file_path = self.create_tempfile(content='SEQUENCE_NAME\nACDE')
    with self.assertRaisesRegex(ValueError, 'Failed to parse'):
      predict.parse_input_to_text(input_file_path.full_path)

  def test_format_output_adds_description(self):
    input_df = pd.DataFrame({
        'sequence_name': ['SEQ_A'],
        'predicted_label': ['PF000042'],
        'start': [1],
        'end': [100],
    })
    label_to_description = {'PF000042': 'Oxygen carrier'}

    actual = predict.format_df_for_output(
        input_df,
        label_to_description=label_to_description,
    )
    expected = pd.DataFrame({
        'sequence_name': ['SEQ_A'],
        'predicted_label': ['PF000042'],
        'description': ['Oxygen carrier'],
        'start': [1],
        'end': [100],
    })

    test_util.assert_dataframes_equal(self, actual, expected)

  def test_load_models_raises_on_model_missing_no_ensemble(self):
    expected_err_contents = ('Unable to find cached models in FAKE_PATH. Make '
                             'sure you have installed the models by running\n'
                             '    install_models.py '
                             '--model_cache_path=FAKE_PATH\nThen try rerunning '
                             'this script.')
    with self.assertRaises(ValueError) as exc:
      predict.load_models(model_cache_path='FAKE_PATH', num_ensemble_elements=1)

    actual = exc.exception.args[0]

    self.assertIn(expected_err_contents, actual)

  def test_load_models_raises_on_model_missing_with_ensemble(self):
    expected_err_contents = ('Unable to find cached models in FAKE_PATH. Make '
                             'sure you have installed the entire ensemble of '
                             'models by running\n    install_models.py '
                             '--install_ensemble '
                             '--model_cache_path=FAKE_PATH\nThen try rerunning '
                             'this script.')
    with self.assertRaises(ValueError) as exc:
      predict.load_models(model_cache_path='FAKE_PATH', num_ensemble_elements=3)

    actual = exc.exception.args[0]
    self.assertIn(expected_err_contents, actual)

  @parameterized.named_parameters(
      dict(
          testcase_name='given same sequence name, orders by start index',
          input_df=pd.DataFrame({
              'sequence_name': ['SEQ_A', 'SEQ_A'],
              'predicted_label': ['Pfam:PF00001', 'Pfam:PF00002'],
              'start': [100, 1],
              'end': [120, 20],
              'description': ['ZZZZ', 'AAAA'],
          }),
          expected=pd.DataFrame({
              'sequence_name': ['SEQ_A', 'SEQ_A'],
              'predicted_label': ['Pfam:PF00002', 'Pfam:PF00001'],
              'start': [1, 100],
              'end': [20, 120],
              'description': ['AAAA', 'ZZZZ'],
          }),
      ),
  )
  def test_order_df_for_output(self, input_df, expected):
    actual = predict.order_df_for_output(input_df)
    test_util.assert_dataframes_equal(self, actual, expected)


if __name__ == '__main__':
  tf.test.main()
