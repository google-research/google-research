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

"""Tests for Distributed Shampoo."""

from absl.testing import absltest
from flax import optim
import numpy as np
import scipy

from scalable_shampoo.jax import shampoo


class ShampooTest(absltest.TestCase):
  """Test cases for Distributed Shampoo."""

  def test_init_state(self):
    # Create an optimizer def and check the params are wired through.
    optimizer_def = shampoo.Shampoo(
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.9,
        diagonal_epsilon=0.0,
        matrix_epsilon=1e-1,
        exponent_override=2,
        weight_decay=1e-4,
        start_preconditioning_step=1,
        preconditioning_compute_steps=1,
        statistics_compute_steps=1,
        best_effort_shape_interpretation=True,
        block_size=8,
        no_preconditioning_for_layers_with_dim_gt=1024,
        graft_type=shampoo.LayerwiseGrafting.SGD,
        nesterov=False,
        batch_axis_name=None)
    expected_hyper_params = shampoo._ShampooHyperParams(
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.9,
        diagonal_eps=0.0,
        matrix_eps=1e-1,
        exponent_override=2,
        weight_decay=1e-4,
        start_preconditioning_step=1,
        preconditioning_compute_steps=1,
        statistics_compute_steps=1,
        best_effort_shape_interpretation=True,
        block_size=8,
        no_preconditioning_for_layers_with_dim_gt=1024,
        graft_type=shampoo.LayerwiseGrafting.SGD,
        nesterov=False,
        batch_axis_name=None)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)

    params = np.zeros((1,))
    state = optimizer_def.init_state(params)
    zeros_like_param = np.zeros((1,))
    expected_state = optim.OptimizerState(
        0,
        shampoo._ShampooDefaultParamState([], [], [], zeros_like_param,
                                          zeros_like_param))
    self.assertEqual(state, expected_state)

    params = np.zeros((8,))
    state = optimizer_def.init_state(params)
    identity = np.eye(8)
    statistic = identity * 1e-1  #  I * matrix_epsilon
    preconditioner = identity
    self.assertLen(state.param_states.statistics, 1)
    self.assertLen(state.param_states.statistics, 1)
    np.testing.assert_allclose(state.param_states.statistics[0], statistic)
    np.testing.assert_allclose(state.param_states.preconditioners[0],
                               preconditioner)

    params = np.zeros((8, 8))
    state = optimizer_def.init_state(params)
    identity = np.eye(8)
    statistic = identity * 1e-1  #  I * matrix_epsilon
    preconditioner = identity
    self.assertLen(state.param_states.statistics, 2)
    self.assertLen(state.param_states.statistics, 2)
    np.testing.assert_allclose(state.param_states.statistics[0], statistic)
    np.testing.assert_allclose(state.param_states.statistics[1], statistic)
    np.testing.assert_allclose(state.param_states.preconditioners[0],
                               preconditioner)
    np.testing.assert_allclose(state.param_states.preconditioners[1],
                               preconditioner)

    params = np.zeros((16, 16))
    state = optimizer_def.init_state(params)
    zeros_like_param = np.zeros((8,))
    identity = np.eye(8)
    statistic = identity * 1e-1  #  I * matrix_epsilon
    preconditioner = identity
    self.assertLen(state.param_states.statistics, 8)
    self.assertLen(state.param_states.statistics, 8)
    for i in range(8):
      np.testing.assert_allclose(state.param_states.statistics[i], statistic)
      np.testing.assert_allclose(state.param_states.preconditioners[i],
                                 preconditioner)

    # Test best_effort_shape_interpretation
    # (3, 2, 16) wil be reshaped to (6, 16)
    # Last dim will be split into two (6, 8) and (6, 8)
    params = np.zeros((3, 2, 16))
    state = optimizer_def.init_state(params)
    zeros_like_param = np.zeros((8,))
    identity_left = np.eye(6)
    statistic_left = identity_left * 1e-1  #  I * matrix_epsilon
    preconditioner_left = identity_left
    identity_right = np.eye(8)
    statistic_right = identity_right * 1e-1  #  I * matrix_epsilon
    preconditioner_right = identity_right
    self.assertLen(state.param_states.statistics, 4)
    self.assertLen(state.param_states.statistics, 4)
    for i in range(4):
      if i % 2 == 0:
        np.testing.assert_allclose(state.param_states.statistics[i],
                                   statistic_left)
        np.testing.assert_allclose(state.param_states.preconditioners[i],
                                   preconditioner_left)
      else:
        np.testing.assert_allclose(state.param_states.statistics[i],
                                   statistic_right)
        np.testing.assert_allclose(state.param_states.preconditioners[i],
                                   preconditioner_right)

  def test_matrix_inverse_root(self):
    """Test for matrix inverse pth root."""

    def _gen_symmetrix_matrix(dim, condition_number):
      u = scipy.stats.ortho_group.rvs(dim=dim).astype(np.float64)
      v = u.T
      diag = np.diag([condition_number ** (-i/(dim-1)) for i in range(dim)])
      return u @ diag @ v

    # Fails after it reaches a particular condition number.
    for e in range(2, 12):
      condition_number = 10 ** e
      ms = _gen_symmetrix_matrix(16, condition_number)
      self.assertLess(
          np.abs(np.linalg.cond(ms) - condition_number),
          condition_number * 0.01)
      error = shampoo.matrix_inverse_pth_root(
          ms.astype(np.float32), 4, ridge_epsilon=1e-12)[1]
      if e < 7:
        self.assertLess(error, 0.1)
      else:
        # No guarantee of success after e >= 7
        pass

if __name__ == '__main__':
  absltest.main()
