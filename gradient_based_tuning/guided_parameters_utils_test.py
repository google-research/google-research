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
"""Tests for guided_parameters_utils."""

import math

from absl.testing import absltest
import numpy as np

from gradient_based_tuning import guided_parameters_utils


class GuidedParametersUtilsTest(absltest.TestCase):

  def test_get_init_by_name_fails_unrecognized_name(self):
    with self.assertRaises(ValueError):
      _ = guided_parameters_utils.get_init_by_name('bad_name')

  def test_get_activation_fn_by_name_fails_unrecognized_name(self):
    with self.assertRaises(ValueError):
      _ = guided_parameters_utils.get_activation_fn_by_name('bad_name')

  def test_get_activation_inverter_by_name_fails_unrecognized_name(self):
    with self.assertRaises(ValueError):
      _ = guided_parameters_utils.get_activation_inverter_by_name('bad_name')

  def test_init_linear_fn(self):
    out = guided_parameters_utils.init_linear_fn(num_params=11)
    self.assertSequenceAlmostEqual(
        out.tolist(), [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

  def test_init_const_fn(self):
    out = guided_parameters_utils.init_const_fn(coordinate_shape=5, const=1.0)
    self.assertListEqual(out.tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])

  def test_activation_exp_fn_uniform(self):
    out = guided_parameters_utils.activation_exp_fn(
        input_array=np.array([1.0, 1.0, 1.0]), steepness=1)
    self.assertSequenceAlmostEqual(
        out.tolist(), [math.e, math.e, math.e], delta=1e-5)

  def test_activation_exp_fn_varying(self):
    out = guided_parameters_utils.activation_exp_fn(
        input_array=np.array([0.5, 1.0, 2.0]), steepness=1)
    self.assertSequenceAlmostEqual(
        out.tolist(), [1.648721, math.e, 7.389056], delta=1e-5)

  def test_activation_exp_fn_with_lo_floor(self):
    out = guided_parameters_utils.activation_exp_fn(
        input_array=np.array([0.5, 1.0, 2.0]), steepness=1, floor=1e-2)
    self.assertSequenceAlmostEqual(
        out.tolist(), [1.648721 + 1e-2, math.e + 1e-2, 7.389056 + 1e-2],
        delta=1e-5)

  def test_activation_exp_fn_with_hi_floor(self):
    out = guided_parameters_utils.activation_exp_fn(
        input_array=np.array([0.5, 1.0, 2.0]), steepness=1, floor=1e2)
    self.assertSequenceAlmostEqual(
        out.tolist(), [1.648721 + 1e2, math.e + 1e2, 7.389056 + 1e2],
        delta=1e-5)

  def test_activation_exp_inverter_uniform(self):
    out = guided_parameters_utils.activation_exp_inverter(
        input_array=np.array([math.e, math.e, math.e]), steepness=1)
    self.assertSequenceAlmostEqual(out.tolist(), [1.0, 1.0, 1.0], delta=1e-5)

  def test_activation_exp_inverter_varying(self):
    out = guided_parameters_utils.activation_exp_inverter(
        input_array=np.array([1.648721, math.e, 7.389056]), steepness=1)
    self.assertSequenceAlmostEqual(out.tolist(), [0.5, 1.0, 2.0], delta=1e-5)

  def test_activation_exp_inverter_with_lo_floor(self):
    out = guided_parameters_utils.activation_exp_inverter(
        input_array=np.array([1.648721 + 1e-2, math.e + 1e-2, 7.389056 + 1e-2]),
        steepness=1,
        floor=1e-2)
    self.assertSequenceAlmostEqual(out.tolist(), [0.5, 1.0, 2.0], delta=1e-5)

  def test_activation_exp_inverter_with_hi_floor(self):
    out = guided_parameters_utils.activation_exp_inverter(
        input_array=np.array([1.648721 + 1e2, math.e + 1e2, 7.389056 + 1e2]),
        steepness=1,
        floor=1e2)
    self.assertSequenceAlmostEqual(out.tolist(), [0.5, 1.0, 2.0], delta=1e-5)

  def test_activation_self_fn(self):
    out = guided_parameters_utils.activation_self_fn(
        input_array=np.array([-1, 0, 1, 2, 3]))
    self.assertListEqual(out.tolist(), [-1, 0, 1, 2, 3])

  def test_activation_relu_fn(self):
    out = guided_parameters_utils.activation_relu_fn(
        input_array=np.array([-1, 0, 1, 2, 3]))
    self.assertListEqual(out.tolist(), [0, 0, 1, 2, 3])

  def test_activation_softplus_fn(self):
    out = guided_parameters_utils.activation_softplus_fn(
        input_array=np.array([-1, 0, 1, 2, 3]))
    self.assertSequenceAlmostEqual(
        out.tolist(), [0.31326, 0.69315, 1.31326, 2.12693, 3.04859], delta=1e-5)

  def test_activation_softplus_inverter(self):
    out = guided_parameters_utils.activation_softplus_inverter(
        input_array=np.array([0.31326, 0.69315, 1.31326, 2.12693, 3.04859]))
    self.assertSequenceAlmostEqual(out.tolist(), [-1, 0, 1, 2, 3], delta=1e-5)

  def test_activation_linear_fn(self):
    out = guided_parameters_utils.activation_linear_fn(
        input_array=np.array([-1, 0, 1, 2, 3]), steepness=2)
    self.assertListEqual(out.tolist(), [-2, 0, 2, 4, 6])

  def test_activation_linear_inverter(self):
    out = guided_parameters_utils.activation_linear_inverter(
        input_array=np.array([-2, 0, 2, 4, 6]), steepness=2)
    self.assertListEqual(out.tolist(), [-1, 0, 1, 2, 3])

  def test_activation_sig_fn(self):
    out = guided_parameters_utils.activation_sig_fn(
        input_array=np.array([-1, 0, 1, 2, 3]), steepness=2)
    self.assertSequenceAlmostEqual(
        out.tolist(), [0.238405, 1.0, 1.76159, 1.96402, 1.99505], delta=1e-5)

  def test_activation_sig_inverter(self):
    out = guided_parameters_utils.activation_sig_inverter(
        input_array=np.array([0.2384054, 1.0, 1.76159, 1.964027524]),
        steepness=2)
    self.assertSequenceAlmostEqual(out.tolist(), [-1, 0, 1, 2], delta=1e-5)

  def test_activation_sig_fn_ceiling(self):
    out = guided_parameters_utils.activation_sig_fn(
        input_array=np.array([-1, 0, 1, 2, 3]), steepness=2, ceiling=5)
    self.assertSequenceAlmostEqual(
        out.tolist(), [0.5960125, 2.5, 4.403985, 4.91007, 4.987637], delta=1e-5)

  def test_activation_sig_inverter_ceiling(self):
    out = guided_parameters_utils.activation_sig_inverter(
        input_array=np.array([0.5960125, 2.5, 4.403985, 4.91007, 4.987637]),
        steepness=2,
        ceiling=5)
    self.assertSequenceAlmostEqual(out.tolist(), [-1, 0, 1, 2, 3], delta=1e-5)

  def test_activation_sig_fn_floor(self):
    out = guided_parameters_utils.activation_sig_fn(
        input_array=np.array([-1, 0, 1, 2, 3]), steepness=2, floor=1)
    self.assertSequenceAlmostEqual(
        out.tolist(), [1.1192025, 1.5, 1.880799, 1.982014, 1.9975274],
        delta=1e-5)

  def test_activation_sig_inverter_floor(self):
    out = guided_parameters_utils.activation_sig_inverter(
        input_array=np.array([1.1192025, 1.5, 1.880799, 1.982014, 1.9975274]),
        steepness=2,
        floor=1)
    self.assertSequenceAlmostEqual(out.tolist(), [-1, 0, 1, 2, 3], delta=1e-5)

  def test_activation_sig_fn_ceiling_and_floor(self):
    out = guided_parameters_utils.activation_sig_fn(
        input_array=np.array([-1, 0, 1, 2, 3]), steepness=2, ceiling=5, floor=1)
    self.assertSequenceAlmostEqual(
        out.tolist(), [1.476811, 3.0, 4.523188, 4.92805, 4.9901095], delta=1e-5)

  def test_activation_sig_inverter_ceiling_and_floor(self):
    out = guided_parameters_utils.activation_sig_inverter(
        input_array=np.array([1.476811, 3.0, 4.523188, 4.928055, 4.9901095]),
        steepness=2,
        ceiling=5,
        floor=1)
    self.assertSequenceAlmostEqual(out.tolist(), [-1, 0, 1, 2, 3], delta=1e-5)


if __name__ == '__main__':
  absltest.main()
