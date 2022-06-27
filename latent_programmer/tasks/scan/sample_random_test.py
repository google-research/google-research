# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for latent_programmer.tasks.scan.sample_random."""

from absl.testing import absltest
from absl.testing import parameterized

from latent_programmer.tasks.scan import sample_random


class SampleRandomTest(parameterized.TestCase):

  @parameterized.named_parameters(
      *[(e.name, e.name) for e in sample_random.ScanExperiment]
  )
  def test_sample_task(self, experiment):
    # Make sure rejection sampling doesn't hang.
    train_tasks = [sample_random.sample_task(experiment, is_train=True)
                   for _ in range(100)]
    test_tasks = [sample_random.sample_task(experiment, is_train=False)
                  for _ in range(100)]
    train_set = set(map(tuple, train_tasks))
    test_set = set(map(tuple, test_tasks))

    # There should be a lot of variety in tasks.
    print('{} train distinct tasks: {}'.format(experiment, len(train_set)))
    print('{} test distinct tasks: {}'.format(experiment, len(test_set)))
    if experiment == sample_random.ScanExperiment.LENGTH_1_TO_2_6.name:
      # This is the only experiment where tasks are commonly duplicated, since
      # there's only 1 part per task.
      self.assertGreater(len(train_set), 40)
    else:
      self.assertGreater(len(train_set), 80)
    self.assertGreater(len(test_set), 80)

    # There should be zero overlap between train and test, unless the experiment
    # is NONE.
    if experiment != sample_random.ScanExperiment.NONE.name:
      self.assertEmpty(train_set & test_set)


if __name__ == '__main__':
  absltest.main()
