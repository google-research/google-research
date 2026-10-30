# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""CPU device coverage for the PyTorch Shampoo optimizer."""

import unittest

import torch

import shampoo


class ShampooCpuTest(unittest.TestCase):

  def test_cpu_step_initializes_state_for_each_graft_type(self):
    for graft_type in shampoo.LayerwiseGrafting:
      with self.subTest(graft_type=graft_type):
        parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        initial = parameter.detach().clone()
        parameter.grad = torch.ones_like(parameter)
        optimizer = shampoo.Shampoo(
            [parameter],
            lr=0.01,
            hyperparams=shampoo.ShampooHyperParams(graft_type=graft_type))

        optimizer.step()

        self.assertEqual(
            parameter.device, optimizer.state[parameter]['momentum'].device)
        self.assertFalse(torch.equal(parameter.detach(), initial))


if __name__ == '__main__':
  unittest.main()
