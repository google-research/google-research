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

"""Tests for suite code."""

from absl.testing import absltest
from absl.testing import parameterized
import mock
from distracting_control import suite

DAVIS_PATH = '/tmp/davis'

class SuiteTest(parameterized.TestCase):

  @parameterized.named_parameters(('none', None),
                                  ('easy', 'easy'),
                                  ('medium', 'medium'),
                                  ('hard', 'hard'))
  @mock.patch.object(suite, 'pixels')
  @mock.patch.object(suite, 'suite')
  def test_suite_load_with_difficulty(self, difficulty, mock_dm_suite,
                                      mock_pixels):
    domain_name = 'cartpole'
    task_name = 'balance'
    suite.load(
        domain_name,
        task_name,
        difficulty,
        background_dataset_path=DAVIS_PATH)

    mock_dm_suite.load.assert_called_with(
        domain_name,
        task_name,
        environment_kwargs=None,
        task_kwargs=None,
        visualize_reward=False)

    mock_pixels.Wrapper.assert_called_with(
        mock.ANY,
        observation_key='pixels',
        pixels_only=True,
        render_kwargs={'camera_id': 0})


if __name__ == '__main__':
  absltest.main()
