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

from absl.testing import absltest
import pandas as pd
from cisc.src import confidence_extraction
from cisc.src.post_processing import per_trace_processing
from cisc.src.post_processing import util

_CONFIG = confidence_extraction.AggregatedConfidenceExtractionConfig(
    verbal_confidence=confidence_extraction.ConfidenceExtractionType.BINARY.value,
    confidence_likelihoods=confidence_extraction.ConfidenceExtractionType.BINARY.value,
    run_sequence_probability=False,
)


class RunPostProcessingTest(absltest.TestCase):

  def test_post_process_results_dataframe(self):
    input_df = pd.DataFrame({
        'verbal_confidence': [0.1, 0.2, 0.3],
        'response_probability': [None, None, None],
        'answer': ['ans1', 'ans2', 'ans3'],
        'exception': ['', '', ''],
        'golden_label': ['ans1', 'ans1', 'ans1'],
        'is_correct': [True, False, False],
        'confidence_likelihoods': [[1, 5], None, None],
    })
    expected_df = input_df.copy()
    output_df, _ = per_trace_processing.post_process_results_dataframe(
        input_df,
        confidence_config=_CONFIG,
        config=per_trace_processing.PostProcessingConfig(),
    )
    expected_df['logit_confidence'] = [
        util.softmax([1, 5], temp=1.0)[1],
        None,
        None,
    ]
    expected_df['binary_confidence'] = [1.0, 0.0, 0.0]
    pd.testing.assert_frame_equal(output_df, expected_df)

  def test_post_process_results_dataframe_modifications(self):
    input_df = pd.DataFrame({
        'verbal_confidence': [0.1, 0.2, 0.3],
        'response_probability': [None, None, None],
        'answer': ['ans1', 'ans2', None],
        'exception': ['', '', ''],
        'golden_label': ['ans1', 'ans1', 'ans1'],
        'is_correct': [False, False, False],
        'confidence_likelihoods': [[1, 5], None, None],
    })
    expected_df = input_df.copy()
    output_df = per_trace_processing.post_process_results_dataframe(
        input_df,
        confidence_config=_CONFIG,
        config=per_trace_processing.PostProcessingConfig(),
    )[0]
    # The confidence for the invalid answer is set to 0.
    expected_df['verbal_confidence'] = [0.1, 0.2, 0]
    # None answer are converted to an empty string. This was an arbitrary
    # decision. Probably should have been fine to keep them as Nones.
    expected_df['answer'] = ['ans1', 'ans2', '']
    # The is_correct column is re-computed.
    expected_df['is_correct'] = [True, False, False]
    expected_df['logit_confidence'] = [
        util.softmax([1, 5], temp=1.0)[1],
        None,
        None,
    ]
    expected_df['binary_confidence'] = [1.0, 0.0, 0.0]
    pd.testing.assert_frame_equal(output_df, expected_df)

  def test_post_process_results_dataframe_none_is_different_than_zero(self):
    input_df = pd.DataFrame({
        'verbal_confidence': [0.1, -5, 0.3],
        'confidence_likelihoods': [None, None, None],
        'response_probability': [None, None, None],
        'answer': ['ans1', 'ans2', None],
        'exception': ['', '', ''],
        'golden_label': ['ans1', 'ans1', 'ans1'],
        'is_correct': [False, False, False],
        'confidence_likelihoods': [[1, 5], None, None],
    })
    expected_df = input_df.copy()
    output_df = per_trace_processing.post_process_results_dataframe(
        input_df,
        confidence_config=_CONFIG,
        config=per_trace_processing.PostProcessingConfig(),
    )[0]
    # Out of range confidences are set to 0.  While None answer makes the
    # confidence None.
    expected_df['verbal_confidence'] = [0.1, 0, 0]
    expected_df['binary_confidence'] = [1.0, 0.0, 0.0]
    pd.testing.assert_series_equal(
        output_df.verbal_confidence, expected_df.verbal_confidence
    )

  def test_post_process_results_dataframe_filters_answers_when_needed(self):
    input_df = pd.DataFrame({
        'verbal_confidence': [0.1, 0.2, 0.3],
        'answer': ['ans1', 'ans2', None],
        'exception': ['', '', ''],
        'golden_label': ['ans1', 'ans1', 'ans1'],
        'is_correct': [False, False, False],
        'confidence_likelihoods': [[1, 5], None, None],
        'response_probability': [None, None, None],
    })
    expected_df = input_df[:-1]
    output_df = per_trace_processing.post_process_results_dataframe(
        input_df,
        confidence_config=_CONFIG,
        config=per_trace_processing.PostProcessingConfig(filter_answers=True),
    )[0]
    # The is_correct column is re-computed.
    expected_df['is_correct'] = [True, False]
    expected_df['logit_confidence'] = [util.softmax([1, 5], temp=1.0)[1], None]
    expected_df['binary_confidence'] = [1.0, 0.0]
    pd.testing.assert_frame_equal(output_df, expected_df)


if __name__ == '__main__':
  absltest.main()
