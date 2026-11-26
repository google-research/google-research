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

"""Tests that Shampoo's Nesterov update preserves its momentum buffers."""

import unittest

import torch
import shampoo


class _CpuParameter(torch.nn.Parameter):
  """Allows the GPU-oriented optimizer to run on CPU for focused tests."""

  def get_device(self):
    # Legacy Shampoo passes var.get_device() to torch.zeros_like; on CPU that
    # returns -1, which current PyTorch rejects as a device selector.
    return self.device


class ShampooNesterovTest(unittest.TestCase):

  def _compare_with_torch_sgd(self, graft_type, start_step, nesterov):
    initial = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    parameter = _CpuParameter(initial.clone())
    reference_parameter = torch.nn.Parameter(initial.clone())
    momentum = 0.6
    lr = 0.05
    hps = shampoo.ShampooHyperParams(
        graft_type=graft_type,
        start_preconditioning_step=start_step,
        statistics_compute_steps=1000,
        preconditioning_compute_steps=1000,
        nesterov=nesterov)
    optimizer = shampoo.Shampoo(
        [parameter], lr=lr, momentum=momentum, hyperparams=hps)
    reference = torch.optim.SGD(
        [reference_parameter], lr=lr, momentum=momentum, nesterov=nesterov)

    expected_buffer = torch.zeros_like(initial)
    gradients = (
        torch.tensor([[1.0, -0.5], [0.3, -0.1]]),
        torch.tensor([[2.0, -0.5], [0.3, -0.1]]),
        torch.tensor([[3.0, -0.5], [0.3, -0.1]]),
    )
    for grad in gradients:
      parameter.grad = grad.clone()
      reference_parameter.grad = grad.clone()
      optimizer.step()
      reference.step()

      expected_buffer = momentum * expected_buffer + grad
      # Before preconditioning starts, the SGD graft supplies the update.
      # Afterwards, Shampoo's own buffer supplies it.
      current_buffer = (
          optimizer.state[parameter][shampoo.GRAFT].momentum
          if start_step > len(gradients) else
          optimizer.state[parameter][shampoo.MOMENTUM])
      torch.testing.assert_close(
          current_buffer, expected_buffer, atol=1e-5, rtol=1e-5)
      torch.testing.assert_close(
          parameter.detach(), reference_parameter.detach(),
          atol=1e-5, rtol=1e-5)

  def test_shampoo_nesterov_does_not_modify_running_momentum(self):
    self._compare_with_torch_sgd(
        shampoo.LayerwiseGrafting.NONE, start_step=1, nesterov=True)

  def test_sgd_graft_nesterov_does_not_modify_running_momentum(self):
    self._compare_with_torch_sgd(
        shampoo.LayerwiseGrafting.SGD, start_step=10, nesterov=True)

  def test_non_nesterov_retains_momentum_recurrence(self):
    self._compare_with_torch_sgd(
        shampoo.LayerwiseGrafting.NONE, start_step=1, nesterov=False)
    self._compare_with_torch_sgd(
        shampoo.LayerwiseGrafting.SGD, start_step=10, nesterov=False)


if __name__ == '__main__':
  unittest.main()
