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

"""Tests for ipagnn.datasets.control_flow_programs.control_flow_programs_version."""

from absl.testing import absltest
from ipagnn.datasets.control_flow_programs import control_flow_programs_version


class ControlFlowProgramsVersionTest(absltest.TestCase):

  def test_version_at_least(self):
    self.assertTrue(control_flow_programs_version.at_least('0.0.40'))

  def test_version_as_tuple(self):
    self.assertEqual(control_flow_programs_version.as_tuple('0.0.40'),
                     (0, 0, 40))


if __name__ == '__main__':
  absltest.main()
