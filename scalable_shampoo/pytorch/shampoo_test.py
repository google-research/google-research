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

"""Tests for the PyTorch Shampoo optimizer."""

import unittest

import torch

from shampoo import LayerwiseGrafting
from shampoo import Shampoo
from shampoo import ShampooHyperParams


class ShampooTest(unittest.TestCase):

  def test_sgd_graft_weight_decay_preserves_gradient(self):
    parameter = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]))
    # The optimizer passes get_device() to tensor constructors. On CPU,
    # PyTorch returns -1, so report the CPU device explicitly for this test.
    parameter.get_device = lambda: torch.device('cpu')
    parameter.grad = torch.tensor([[0.3, 0.4], [0.5, 0.6]])
    original_gradient = parameter.grad.clone()
    hps = ShampooHyperParams(
        graft_type=LayerwiseGrafting.SGD,
        weight_decay=0.1,
        preconditioning_compute_steps=100,
        best_effort_shape_interpretation=False,
    )
    optimizer = Shampoo([parameter], lr=0.1, momentum=0.0, hyperparams=hps)

    optimizer.step()

    self.assertTrue(torch.equal(parameter.grad, original_gradient))


if __name__ == '__main__':
  unittest.main()
