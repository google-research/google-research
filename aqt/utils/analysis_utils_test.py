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

"""Tests for analysis_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized

import pandas as pd

from aqt.utils import analysis_utils
from aqt.utils import report_utils


class AnalysisUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Create a test ExperimentReport instance
    compute_cost_dict = {
        'compute_cost': 4,
        'memory_cost': 5,
    }

    metrics_dict = {
        'eval': {
            'accuracy': 0.6,
        },
        'train': {
            'accuracy': 0.7,
            'loss': 1.3,
        }
    }
    unsmoothed_metrics_dict = {
        'eval': {
            'accuracy': 0.5,
        },
        'train': {
            'accuracy': 0.8,
            'loss': 1.2,
        }
    }


    report_query_args = {
        'early_stop_attr': 'loss',
        'early_stop_agg': 'MIN',
        'early_stop_ds_dir': 'eval',
        'other_ds_dirs': ['train'],
        'tags_to_include': [
            'accuracy',
            'loss',
        ],
        'smoothing_kernel': 'RECTANGULAR',
        'window_size_in_steps': 50,
        'start_step': 100
    }

    self.test_report = report_utils.ExperimentReport(
        model_dir='/your/model_dir',
        metrics=metrics_dict,
        unsmoothed_metrics=unsmoothed_metrics_dict,
        early_stop_step=1500,
        num_train_steps=2000,
        report_query_args=report_query_args,
        experiment_name='test_exp',
        user_name='test_user',
        launch_time='202102016T112344',
        eval_freq=10,
        first_nan_step=None,
        tensorboard_id='12345',
    )

  def test_convert_report_to_flat_dict_default(self):
    res = analysis_utils.convert_report_to_flat_dict_default(self.test_report)
    exp = {
        'experiment_name':
            'test_exp',
        'model_dir':
            '/your/model_dir',
        'user_name':
            'test_user',
        'launch_time':
            '202102016T112344',
        'eval_freq':
            10,
        'first_nan_step':
            None,
        'tensorboard_id': '12345',
        'early_stop_step':
            1500,
        'num_train_steps':
            2000,
        'eval/accuracy':
            0.6,
        'train/accuracy':
            0.7,
        'train/loss':
            1.3,
        'unsmoothed/eval/accuracy':
            0.5,
        'unsmoothed/train/accuracy':
            0.8,
        'unsmoothed/train/loss':
            1.2,
        'report_query_args': {
            'early_stop_attr': 'loss',
            'early_stop_agg': 'MIN',
            'early_stop_ds_dir': 'eval',
            'other_ds_dirs': ['train'],
            'tags_to_include': [
                'accuracy',
                'loss',
            ],
            'smoothing_kernel': 'RECTANGULAR',
            'window_size_in_steps': 50,
            'start_step': 100
        },
    }
    self.assertDictEqual(res, exp)

  def test_convert_reports_to_dataframe(self):
    res = analysis_utils.convert_reports_to_dataframe(
        [self.test_report, self.test_report])
    exp_df_shape = (2, 20)
    self.assertEqual(res.shape, exp_df_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='nested_two_levels',
          nested_dict={
              'a': {
                  'foo': 1,
                  'bar': 2
              },
              'b': 'zoo'
          },
          exp={
              'a/foo': 1,
              'a/bar': 2,
              'b': 'zoo'
          }),
      dict(
          testcase_name='nested_three_levels',
          nested_dict={
              'a': {
                  'foo': {
                      'gan': 1,
                  },
                  'bar': 2
              },
              'b': 'zoo'
          },
          exp={
              'a/foo/gan': 1,
              'a/bar': 2,
              'b': 'zoo'
          }),
      dict(
          testcase_name='nested_with_list',
          nested_dict={
              'a': {
                  'foo': 1,
                  'bar': 2
              },
              'b': ['yaa', 'zoo']
          },
          exp={
              'a/foo': 1,
              'a/bar': 2,
              'b/0': 'yaa',
              'b/1': 'zoo',
          }),
  )
  def test_flatten_with_joined_string_paths(self, nested_dict, exp):
    res = analysis_utils.flatten_with_joined_string_paths(nested_dict)
    self.assertDictEqual(res, exp)

  def test_clickable_link(self):
    link = 'http://test.url'
    display_str = 'test_link'
    res = analysis_utils.clickable_link(link, display_str)
    exp = '<a href="http://test.url">test_link</a>'
    self.assertEqual(res, exp)



if __name__ == '__main__':
  absltest.main()
