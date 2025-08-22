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

"""Tests for executors."""

from absl.testing import absltest

from imp.max.execution import executors


class ExecutorsTest(absltest.TestCase):

  def test_profiler(self):
    path = self.create_tempdir('path/to/').full_path
    profiler = executors.Profiler(path=path,
                                  wait_for_steps=-1)

    for i in range(10):
      profiler.update(i, 'x', {})

    metrics = profiler.get_metrics()
    self.assertSetEqual(set(metrics.keys()), {
        'profile/steps_per_sec',
        'profile/sec_per_step',
        'profile/uptime',
        'profile/examples_per_sec',
        'profile/tokens_per_sec',
        'profile/examples_per_sec_per_tpu',
        'profile/tokens_per_sec_per_tpu',
        'x/percent_steps_sampled',
    })

    self.assertEqual(metrics['profile/uptime'], 10)
    self.assertEqual(metrics['profile/examples_per_sec'], 0)
    self.assertEqual(metrics['profile/tokens_per_sec'], 0)
    self.assertEqual(metrics['profile/examples_per_sec_per_tpu'], 0)
    self.assertEqual(metrics['profile/tokens_per_sec_per_tpu'], 0)
    self.assertEqual(metrics['x/percent_steps_sampled'], 100)


if __name__ == '__main__':
  absltest.main()
