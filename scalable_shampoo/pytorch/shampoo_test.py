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

"""Tests for the PyTorch Shampoo optimizer.

Run from the repository root with this directory on the path, e.g.:

  PYTHONPATH=scalable_shampoo/pytorch python3 \
      scalable_shampoo/pytorch/shampoo_test.py
"""

from absl.testing import absltest
import torch

import shampoo


class ShampooCpuDeviceTest(absltest.TestCase):
  """Regression tests for optimizer state allocation on CPU parameters.

  `torch.Tensor.get_device()` returns -1 for CPU tensors, so passing its result
  as the `device` argument raises "Device index must not be negative."  State
  allocation must use `var.device` instead.
  """

  def _make_param(self, shape):
    param = torch.nn.Parameter(torch.randn(shape))
    param.grad = torch.randn(shape)
    return param

  def test_vector_parameter_step_completes(self):
    """A 1-D CPU parameter can take a step without error."""
    param = self._make_param([4])
    optimizer = shampoo.Shampoo([param])
    optimizer.step()

  def test_matrix_parameter_step_completes(self):
    """A 2-D CPU parameter exercises the preconditioner path (PowerIter)."""
    param = self._make_param([4, 5])
    hyperparams = shampoo.ShampooHyperParams(block_size=2)
    optimizer = shampoo.Shampoo([param], hyperparams=hyperparams)
    optimizer.step()

  def test_adagrad_graft_step_completes(self):
    """The AdaGrad graft allocates its own state and must work on CPU."""
    param = self._make_param([3, 3])
    hyperparams = shampoo.ShampooHyperParams(
        graft_type=shampoo.LayerwiseGrafting.ADAGRAD, block_size=2
    )
    optimizer = shampoo.Shampoo([param], hyperparams=hyperparams)
    optimizer.step()

  def test_state_tensors_are_on_parameter_device(self):
    """Every allocated state tensor lives on the parameter's device."""
    param = self._make_param([4, 5])
    optimizer = shampoo.Shampoo([param])
    optimizer.step()

    state = optimizer.state[param]
    self.assertEqual(param.device, torch.device("cpu"))

    self.assertEqual(state[shampoo.MOMENTUM].device, param.device)
    for statistic in state[shampoo.PRECONDITIONER].statistics:
      self.assertEqual(statistic.device, param.device)
    for preconditioner in state[shampoo.PRECONDITIONER].preconditioners:
      self.assertEqual(preconditioner.device, param.device)
    self.assertEqual(state[shampoo.GRAFT].momentum.device, param.device)

  def test_optimizer_update_is_applied(self):
    """The step actually moves the parameters (guards against a no-op fix)."""
    param = self._make_param([2])
    before = param.detach().clone()
    optimizer = shampoo.Shampoo([param])
    optimizer.step()
    self.assertFalse(torch.equal(before, param.detach()))


if __name__ == "__main__":
  absltest.main()
