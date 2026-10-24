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

"""Tests that the PyTorch Shampoo update never mutates its inputs.

`Shampoo.step` builds working tensors out of the parameter's gradient and then
scales them in place. Any of those helpers that return the gradient itself
rather than a copy corrupts `param.grad`, so these tests assert that the
gradient survives a step unchanged.

Run with:

  PYTHONPATH=scalable_shampoo/pytorch python3 \
      scalable_shampoo/pytorch/graft_test.py
"""

from absl.testing import absltest
import torch

import shampoo


def _make_param(shape, seed=0):
  torch.manual_seed(seed)
  param = torch.nn.Parameter(torch.randn(shape))
  param.grad = torch.randn(shape)
  return param


class GradientIsNotMutatedTest(absltest.TestCase):
  """`step()` must leave `param.grad` exactly as it found it."""

  def _assert_grad_unchanged(self, param, optimizer):
    before = param.grad.detach().clone()
    optimizer.step()
    torch.testing.assert_close(param.grad.detach(), before)

  def test_vector_parameter_grad_unchanged(self):
    """1-D parameters take the `not self.preconditioners` passthrough path."""
    param = _make_param([6])
    self._assert_grad_unchanged(param, shampoo.Shampoo([param]))

  def test_vector_parameter_grad_unchanged_with_adagrad_graft(self):
    """1-D parameter whose graft scale is not 1.0.

    With the default SGD graft both working gradients are the same tensor, so
    the in-place rescaling multiplies by exactly 1.0 and any aliasing is
    invisible. The AdaGrad graft rescales by a different factor, which exposes
    it.
    """
    param = _make_param([6])
    hyperparams = shampoo.ShampooHyperParams(
        graft_type=shampoo.LayerwiseGrafting.ADAGRAD
    )
    self._assert_grad_unchanged(
        param, shampoo.Shampoo([param], hyperparams=hyperparams)
    )

  def test_matrix_parameter_grad_unchanged(self):
    param = _make_param([4, 5])
    hyperparams = shampoo.ShampooHyperParams(block_size=2)
    self._assert_grad_unchanged(
        param, shampoo.Shampoo([param], hyperparams=hyperparams)
    )

  def test_grad_unchanged_with_weight_decay(self):
    """Weight decay adds the parameter into both working gradients."""
    param = _make_param([4, 5])
    hyperparams = shampoo.ShampooHyperParams(block_size=2, weight_decay=0.1)
    self._assert_grad_unchanged(
        param, shampoo.Shampoo([param], hyperparams=hyperparams)
    )

  def test_grad_unchanged_before_preconditioning_starts(self):
    """Before `start_preconditioning_step` the raw gradient is used directly."""
    param = _make_param([4, 5])
    hyperparams = shampoo.ShampooHyperParams(
        block_size=2, start_preconditioning_step=5
    )
    optimizer = shampoo.Shampoo([param], hyperparams=hyperparams)
    self._assert_grad_unchanged(param, optimizer)
    self._assert_grad_unchanged(param, optimizer)

  def test_grad_unchanged_with_adagrad_graft(self):
    param = _make_param([4, 5])
    hyperparams = shampoo.ShampooHyperParams(
        block_size=2, graft_type=shampoo.LayerwiseGrafting.ADAGRAD
    )
    self._assert_grad_unchanged(
        param, shampoo.Shampoo([param], hyperparams=hyperparams)
    )


class NesterovMomentumTest(absltest.TestCase):
  """Nesterov must not apply momentum to the buffer a second time."""

  def _buffer_after_one_step(self, nesterov):
    param = _make_param([4, 5], seed=3)
    hyperparams = shampoo.ShampooHyperParams(
        block_size=2, graft_type=shampoo.LayerwiseGrafting.SGD,
        nesterov=nesterov,
    )
    optimizer = shampoo.Shampoo(
        [param], lr=0.1, momentum=0.9, hyperparams=hyperparams
    )
    optimizer.step()
    state = optimizer.state[param]
    return state[shampoo.MOMENTUM].detach().clone(), param.detach().clone()

  def test_momentum_buffer_matches_non_nesterov_path(self):
    """The buffer holds m * prev + grad in both cases.

    With Nesterov enabled the update direction is m * buffer + grad, but that
    must not be written back into the buffer. Otherwise the buffer becomes
    m * (m * prev + grad) + grad and momentum compounds every step.
    """
    buffer_plain, _ = self._buffer_after_one_step(nesterov=False)
    buffer_nesterov, _ = self._buffer_after_one_step(nesterov=True)
    torch.testing.assert_close(buffer_nesterov, buffer_plain)

  def test_nesterov_changes_the_update_not_the_buffer(self):
    """Nesterov still affects the parameters, so the test above is meaningful."""
    _, plain = self._buffer_after_one_step(nesterov=False)
    _, nesterov = self._buffer_after_one_step(nesterov=True)
    self.assertFalse(torch.allclose(plain, nesterov))


if __name__ == "__main__":
  absltest.main()
