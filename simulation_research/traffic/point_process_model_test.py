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

# Lint as: python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import logging
from absl.testing import absltest
import matplotlib.pyplot as plt
import numpy as np

from simulation_research.traffic import point_process_model


class PointProcessModelTest(absltest.TestCase):

  def setUp(self):
    super(PointProcessModelTest, self).setUp()
    self.model = point_process_model.PointProcessModel()
    np.random.seed(0)
    self._output_dir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())

  def test_generator_homogeneous_poisson(self):
    lmbd = 1
    time_step_size = 10
    rates = np.ones(100000) * lmbd
    events = self.model.generator(rates, time_step_size)
    actual_mean = np.mean(events)
    actual_std = np.std(events)
    target_mean = lmbd * time_step_size
    target_std = np.sqrt(target_mean)
    self.assertAlmostEqual(
        np.abs(target_mean - actual_mean) / target_mean, 0, places=2)
    self.assertAlmostEqual(
        np.abs(target_std - actual_std) / target_std, 0, places=2)

  def test_model_fitting_homogeneous_poisson(self):
    lmbd = 1
    time_step_size = 10
    rates = np.ones(100000) * lmbd
    events = self.model.generator(rates, time_step_size)
    actual_lmbd = self.model.fit_homo_poisson(events, time_step_size)
    self.assertAlmostEqual(actual_lmbd, lmbd, places=2)

  def test_bspline_basis(self):
    spline_order = 3
    knots = [0, 0.333, 0.6667, 1]
    basis, time_line = point_process_model.create_bspline_basis(
        knots, spline_order)

    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot(111)
    plt.plot(time_line, basis)
    output_file = os.path.join(self._output_dir, 'B-spline_basis_order3.png')
    logging.info('Save file in: %s', self._output_dir)
    plt.savefig(output_file)
    plt.close()

    spline_order = 2
    knots = [0, 0.2, 0.4, 0.6, 0.8, 1]
    basis, time_line = point_process_model.create_bspline_basis(
        knots, spline_order)

    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot(111)
    plt.plot(time_line, basis)
    output_file = os.path.join(self._output_dir, 'B-spline_basis_order2.png')
    logging.info('Save file in: %s', self._output_dir)
    plt.savefig(output_file)
    plt.close()

  def test_fit_inhomo_poisson(self):
    # Fit the homogeneous Poisson process using inhomogeneous methods.
    lmbd = 1
    time_step_size = 10
    rates = np.ones(50) * lmbd
    events = self.model.generator(rates, time_step_size)
    rates_hat = self.model.fit_inhomo_poisson(
        events, time_step_size, num_knots=3)

    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot(111)
    plt.plot(rates, label='True rate')
    plt.plot(rates_hat, label='Estimated rate')
    output_file = os.path.join(
        self._output_dir, 'inhomo model fit homo process.png')
    plt.ylim(-1, 3)
    plt.legend()
    logging.info('Save file in: %s', self._output_dir)
    plt.savefig(output_file)
    plt.close()

    # Fit the inhomogeneous Poisson process.
    spline_order = 3
    knots = [0, 0.2, 0.4, 0.6, 0.8, 1]  # knots on the unit range.
    basis, _ = point_process_model.create_bspline_basis(
        knots, spline_order, 0.02)
    beta_target = np.array([0, 0, 0, 2, 1, -1, 0, 0, 2])
    time_step_size = 10
    rates_target = np.exp(basis @ beta_target) / time_step_size
    # Generates events according to the inhomogeneous rates.
    events = self.model.generator(rates_target, time_step_size)
    rates_hat = self.model.fit_inhomo_poisson(
        events, time_step_size, num_knots=4)

    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot(111)
    plt.plot(rates_target, label='True rate')
    plt.plot(rates_hat, label='Estimated rate')
    output_file = os.path.join(
        self._output_dir, 'inhomo model fit inhomo process.png')
    plt.ylim(-2, 5)
    plt.legend()
    logging.info('Save file in: %s', self._output_dir)
    plt.savefig(output_file)
    plt.close()

if __name__ == '__main__':
  absltest.main()
