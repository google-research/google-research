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
from unittest import mock

import torch

import matrix_functions
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

  def test_cpu_step_runs_matrix_preconditioning(self):
    parameter = torch.nn.Parameter(torch.ones(32, 32))
    parameter.grad = torch.ones_like(parameter)
    optimizer = shampoo.Shampoo([parameter], lr=0.01)

    with mock.patch.object(
        matrix_functions,
        'ComputePower',
        wraps=matrix_functions.ComputePower) as compute_power:
      with mock.patch.object(
          matrix_functions,
          'PowerIter',
          wraps=matrix_functions.PowerIter) as power_iter:
        optimizer.step()

    state = optimizer.state[parameter]
    preconditioner = state[shampoo.PRECONDITIONER]
    self.assertEqual((32, 32), tuple(preconditioner._transformed_shape))
    self.assertEqual(2, compute_power.call_count)
    self.assertEqual(2, power_iter.call_count)
    self.assertEqual('cpu', parameter.device.type)
    self.assertEqual(parameter.device, state[shampoo.MOMENTUM].device)
    for tensor in preconditioner.statistics + preconditioner.preconditioners:
      self.assertEqual(parameter.device, tensor.device)
      self.assertEqual('cpu', tensor.device.type)


if __name__ == '__main__':
  unittest.main()
