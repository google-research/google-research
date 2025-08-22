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

"""Tests for ode_datasets."""

from absl.testing import absltest

from simulation_research.diffusion import ode_datasets


class OdeDatasetsTest(absltest.TestCase):

  def test_generate_data(self):
    for name in ['LorenzDataset', 'FitzHughDataset', 'NPendulum']:
      ds = getattr(ode_datasets, name)(N=20)
      num_trajectories, num_steps, _ = ds.Zs.shape
      assert num_trajectories == 20
      assert num_steps >= 60
      assert len(ds.T_long) == num_steps


if __name__ == '__main__':
  absltest.main()
