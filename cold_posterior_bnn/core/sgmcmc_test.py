# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for google_research.google_research.cold_posterior_bnn.core.sgmcmc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from cold_posterior_bnn.core import sgmcmc
from cold_posterior_bnn.core import sgmcmc_testlib


class SgmcmcTest(parameterized.TestCase, tf.test.TestCase):

  _LEARNING_RATES = [0.25, 0.1]
  _LANGEVIN_MOMENTUMS = [0.8]
  _LANGEVIN_LEARNING_RATES = [0.05]
  _TEMPS = [0.1, 0.5, 1.0]

  def _run_optimizer_test_1d(self,
                             optimizer,
                             nsamples=50000,
                             tol_mean=0.05,
                             tol_stddev=0.1,
                             efficiency_lb=0.01,
                             temp=1.0):
    """Run SG-MCMC method on 1D model."""
    tf.set_random_seed(1)
    model = tf.keras.Sequential([sgmcmc_testlib.StandardNormal1D()])
    tf.keras.backend.set_value(optimizer.temp, temp)
    samples = sgmcmc_testlib.sample_model(model, optimizer, nsamples)
    samples = sgmcmc_testlib.flatten(samples)

    mean = np.mean(samples)
    stddev = np.std(samples)
    _, efficiency = sgmcmc_testlib.compute_ess(samples)

    name = optimizer.get_config()['name']
    lr = optimizer.get_config()['learning_rate']
    momentum_decay = optimizer.get_config().get('momentum_decay', -1.0)
    opt_temp = optimizer.get_config().get('temp', 1.0)

    logging.info(
        '%s(lr=%.4f, momentum_decay=%.4f, temp=%.4f)  mean %.4f  stddev %.4f  '
        'eff %.3f', name, lr, momentum_decay, opt_temp, mean, stddev,
        efficiency)

    self.assertAlmostEqual(
        mean,
        0.0,
        delta=tol_mean,
        msg='Empirical average %.3f disagrees with true '
        'mean of 0.0.' % mean)

    self.assertAlmostEqual(
        stddev,
        math.sqrt(temp),  # var_T = T*var, thus stddev_T = sqrt(T)*stddev
        delta=tol_stddev,
        msg='Empirical standard deviation %.3f disagrees '
        'with true target of 1.0.' % stddev)

    self.assertGreaterEqual(
        efficiency,
        efficiency_lb,
        msg='Efficiency %.3f is below limit 0.01.' % efficiency)

  def _kl2d(self, pmodel, mean, cov):
    """Compute the Kullback-Leibler divergence between N(mean,cov) and P."""
    res = tf.linalg.trace(tf.matmul(tf.linalg.inv(tf.cast(cov, tf.float32)),
                                    pmodel.cov))
    res += tf.reduce_sum(tf.reshape(
        tf.linalg.solve(cov, tf.reshape(mean, (2, 1))), (2,)) * mean)
    res -= 2.0
    res += tf.linalg.logdet(
        tf.cast(cov, tf.float32)) - tf.linalg.logdet(pmodel.cov)
    res *= 0.5

    return res

  def _run_optimizer_test_2d(self,
                             optimizer,
                             noise_sigma=0.0,
                             nsamples=25000,
                             tol_mean=0.05,
                             tol_kl=0.01,
                             efficiency_lb=0.01):
    """Run SG-MCMC method on correlated 2D Normal model with gradient noise."""
    tf.set_random_seed(1)
    pmodel = sgmcmc_testlib.Normal2D(noise_sigma=noise_sigma)
    model = tf.keras.Sequential([pmodel])
    samples = sgmcmc_testlib.sample_model(model, optimizer, nsamples)

    mean = np.mean(samples, axis=1)
    cov = np.cov(samples)
    _, efficiency_all = sgmcmc_testlib.compute_ess_multidimensional(samples)
    efficiency = np.min(efficiency_all)

    name = optimizer.get_config()['name']
    lr = optimizer.get_config()['learning_rate']
    momentum_decay = optimizer.get_config().get('momentum_decay', -1.0)

    kl = self._kl2d(pmodel, mean, cov)
    logging.info(
        '%s(lr=%.4f, momentum_decay=%.4f)  mean (%.4f,%.4f)  kl %.4f  '
        'eff %.3f', name, lr, momentum_decay, mean[0], mean[1],
        kl, efficiency)

    self.assertAlmostEqual(
        mean[0], 0.0, delta=tol_mean,
        msg='Empirical average %.3f differs from true mean of 0.0.' % mean[0])
    self.assertAlmostEqual(
        mean[1], 0.0, delta=tol_mean,
        msg='Empirical average %.3f differs from true mean of 0.0.' % mean[1])

    self.assertLess(kl, tol_kl,
                    msg='Kullback-Leibler divergence %.5f larger than '
                    'acceptable tolerance %.4f' % (kl, tol_kl))

    self.assertGreaterEqual(
        efficiency,
        efficiency_lb,
        msg='Efficiency %.3f is below limit 0.01.' % efficiency)

  def _get_normal2d_gradients(self, model):
    with tf.GradientTape() as tape:
      nll = model(tf.zeros(1, 1), tf.zeros(1, 1))
    gradients = tape.gradient(nll, model.trainable_variables)
    return gradients[0]

  @parameterized.parameters(
      itertools.product([-0.5, 0.0, 0.5, 0.99],
                        [0.0, 0.5, 5.0],
                        [True, False]))
  def test_normal2d(self, correlation, noise_sigma, uniform_noise):
    model = tf.keras.Sequential([sgmcmc_testlib.Normal2D(
        correlation=correlation,
        noise_sigma=noise_sigma,
        uniform_noise=uniform_noise)])

    gradients1 = self._get_normal2d_gradients(model)
    self.assertEqual(gradients1.shape[0], 2, msg='Wrong gradient shape')

    gradients2 = self._get_normal2d_gradients(model)
    if uniform_noise:
      if noise_sigma == 0.0:
        self.assertAllEqual(gradients1, gradients2, msg='Differing gradients')
      else:
        abs_diff = tf.abs(gradients1 - gradients2)
        self.assertAllGreater(abs_diff, 1.0e-6)

  @parameterized.parameters(itertools.product(_LEARNING_RATES, _TEMPS))
  def test_sgld_mcmc(self, learning_rate, temp):
    optimizer = sgmcmc.StochasticGradientLangevinMCMC(
        total_sample_size=1, learning_rate=learning_rate)
    self._run_optimizer_test_1d(optimizer, tol_mean=0.1, temp=temp)

  @parameterized.parameters(
      itertools.product(_LANGEVIN_LEARNING_RATES, _LANGEVIN_MOMENTUMS, _TEMPS))
  def test_spv_mcmc(self, learning_rate, momentum, temp):
    optimizer = sgmcmc.StochasticPositionVerletMCMC(
        total_sample_size=1,
        learning_rate=learning_rate, momentum_decay=momentum)
    self._run_optimizer_test_1d(optimizer, temp=temp)

  @parameterized.parameters(
      itertools.product(_LANGEVIN_LEARNING_RATES, _LANGEVIN_MOMENTUMS, _TEMPS)
  )
  def test_baoab_mcmc(self, learning_rate, momentum, temp):
    optimizer = sgmcmc.BAOABMCMC(
        total_sample_size=1,
        learning_rate=learning_rate, momentum_decay=momentum)
    self._run_optimizer_test_1d(optimizer, temp=temp)

  _LS_PARAMETER_CHOICES = ['version1', 'version2']

  def test_msgnht_mcmc_1d(self):
    optimizer = sgmcmc.MultivariateNoseHooverMCMC(
        total_sample_size=1, learning_rate=0.001, momentum_decay=0.99)
    self._run_optimizer_test_1d(optimizer, tol_mean=0.15)

  def test_msgnht_mcmc_2d(self):
    optimizer = sgmcmc.MultivariateNoseHooverMCMC(
        total_sample_size=1, learning_rate=0.001, momentum_decay=0.99)

    # Because we use a large amount of noise, we tolerate low efficiency and
    # reasonable accuracy
    self._run_optimizer_test_2d(optimizer,
                                nsamples=100000,
                                noise_sigma=1.0, tol_mean=0.1,
                                tol_kl=0.025, efficiency_lb=0.001)

  @parameterized.parameters(
      itertools.product(_LANGEVIN_LEARNING_RATES, _LANGEVIN_MOMENTUMS))
  def test_bbk_mcmc(self, learning_rate, momentum):
    optimizer = sgmcmc.BBKMCMC(
        total_sample_size=1,
        learning_rate=learning_rate, momentum_decay=momentum)
    self._run_optimizer_test_1d(optimizer)

  @parameterized.parameters(itertools.product(_LEARNING_RATES))
  def test_lm_mcmc(self, learning_rate):
    optimizer = sgmcmc.LMMCMC(total_sample_size=1, learning_rate=learning_rate)
    self._run_optimizer_test_1d(optimizer)

  def test_timestep_factor(self):
    pmodel = sgmcmc_testlib.Normal2D(
        correlation=0.99,
        noise_sigma=0.25,
        uniform_noise=True)
    model = tf.keras.Sequential([pmodel])
    optimizer = sgmcmc.NaiveSymplecticEulerMCMC(
        total_sample_size=1,
        learning_rate=0.01,
        momentum_decay=0.9,
        timestep_factor=1.0)

    nburnin = 4096
    nsamples = 262144

    # Check that accuracy improves when we half the timestep_factor and
    # efficiency as measured by ESS goes down by half.
    kl_prev = None
    efficiency_prev = None
    for timestep_factor in [1.0, 0.5, 0.25, 0.125]:
      optimizer.timestep_factor.assign(timestep_factor)
      samples = sgmcmc_testlib.sample_model(model, optimizer, nburnin)
      samples = sgmcmc_testlib.sample_model(model, optimizer, nsamples)

      mean = np.mean(samples, axis=1)
      cov = np.cov(samples)
      _, efficiency_all = sgmcmc_testlib.compute_ess_multidimensional(samples)
      efficiency = np.min(efficiency_all)
      kl = self._kl2d(pmodel, mean, cov)

      name = optimizer.get_config()['name']
      lr = optimizer.get_config()['learning_rate']
      momentum_decay = optimizer.get_config().get('momentum_decay', -1.0)
      dlr, dmomentum_decay = optimizer.dynamics_parameters(tf.float32)
      dlr = float(dlr)
      dmomentum_decay = float(dmomentum_decay)

      logging.info(
          '%s(lr=%.4f, momentum_decay=%.4f, timestep_factor=%.4f) => '
          '(dlr=%.4f, dmomentum_decay=%.4f) => '
          'mean (%.4f,%.4f)  kl %.4f  eff %.5f',
          name, lr, momentum_decay, timestep_factor,
          dlr, dmomentum_decay, mean[0], mean[1], kl, efficiency)

      # Check values
      if kl_prev is not None:
        self.assertLess(kl, kl_prev + 0.006,
                        msg='Decreasing timestep_factor to %.4f increased KL '
                        'from %.5f to %.5f' % (timestep_factor, kl_prev, kl))
        self.assertAlmostEqual(efficiency / efficiency_prev, 0.5, delta=0.05,
                               msg='Decreasing timestep_factor to %.4f '
                               'produced efficiency %.5f, compared to previous '
                               'efficiency of %.5f, but ratio %.5f not close '
                               'to 1/2.' % (timestep_factor, efficiency,
                                            efficiency_prev,
                                            efficiency / efficiency_prev))

      kl_prev = kl
      efficiency_prev = efficiency

  @parameterized.parameters(itertools.product([8, 10, 11, 16, 35, 80],
                                              ['cosine', 'glide'],
                                              [0.0, 0.01]))
  def test_cyclical_rate(self, period_length, schedule, min_value):
    rate_begin, end_of_period = sgmcmc.cyclical_rate(period_length, 1,
                                                     schedule=schedule,
                                                     min_value=min_value)
    self.assertAlmostEqual(rate_begin, 1.0, delta=1.0e-6,
                           msg='Cyclical learning rate at beginning of period '
                           'not equals to one.')
    self.assertEqual(bool(end_of_period), False, msg='Beginning marked as end.')

    for pi in range(2, period_length):
      rate_prev, end_of_period = sgmcmc.cyclical_rate(period_length, pi-1,
                                                      schedule=schedule,
                                                      min_value=min_value)
      rate_cur, _ = sgmcmc.cyclical_rate(period_length,
                                         pi,
                                         schedule=schedule,
                                         min_value=min_value)
      self.assertLess(rate_cur, rate_prev+1.0e-6,
                      msg='Cyclical rate increasing from %.5f to %.5f in '
                      'period %d to %d' % (rate_prev, rate_cur, pi-1, pi))
      self.assertFalse(bool(end_of_period),
                       msg='End of period in the middle of period at index %d '
                       'with rate %.5f' % (pi-1, rate_prev))
      self.assertGreaterEqual(rate_cur, min_value,
                              msg='Minimum value of %.5f not obeyed by '
                              'rate %.5f' % (min_value, rate_cur))

    _, end_of_period = sgmcmc.cyclical_rate(period_length,
                                            period_length,
                                            schedule=schedule,
                                            min_value=min_value)
    self.assertTrue(bool(end_of_period), msg='End of period not detected')

    with self.assertRaises(ValueError):
      sgmcmc.cyclical_rate(period_length, 0)

  def test_cyclical_rate_constant(self):
    rate_begin, end_of_period = sgmcmc.cyclical_rate(1, 1)
    self.assertAlmostEqual(rate_begin, 1.0, delta=1.0e-6,
                           msg='Rate for period_length=1 is not 1.0')
    self.assertEqual(bool(end_of_period), True,
                     msg='Period length of one but not an end-step.')

  def _check_kinetic_temperature_regions(self, model, optimizer):
    """Check that all kinetic temperatures are in their 99% hpd region."""

    for var in model.trainable_variables:
      ktemp = optimizer.kinetic_temperature([var])
      ktemp_lb, ktemp_ub = optimizer.kinetic_temperature_region(tf.size(var),
                                                                hpd_level=0.999)
      self.assertGreater(
          ktemp, ktemp_lb,
          msg='Variable "%s" kinetic temperature %.3f smaller than lower '
          'bound %.3f' % (var.name, ktemp, ktemp_lb))
      self.assertLess(
          ktemp, ktemp_ub,
          msg='Variable "%s" kinetic temperature %.3f larger than upper '
          'bound %.3f' % (var.name, ktemp, ktemp_ub))

  def test_fixed_preconditioner(self):
    pmodel = sgmcmc_testlib.Normal2D(noise_sigma=1.0)
    model = tf.keras.Sequential([pmodel])
    model.build(input_shape=(1, 1))
    var0 = model.trainable_variables[0]

    optimizer = sgmcmc.NaiveSymplecticEulerMCMC(
        total_sample_size=1,
        learning_rate=0.01,
        momentum_decay=0.7,
        preconditioner='fixed')

    # Initial preconditioner: identity
    precond_dict0 = {var0.name: 1.0}
    optimizer.set_preconditioner_dict(precond_dict0, model.trainable_variables)
    sgmcmc_testlib.sample_model(model, optimizer, 2000)
    self._check_kinetic_temperature_regions(model, optimizer)

    # Adjust preconditioner
    mom_old = tf.identity(optimizer.get_slot(var0, 'moments'))
    precond_dict1 = {var0.name: 100.0}
    optimizer.set_preconditioner_dict(precond_dict1, model.trainable_variables)
    mom_new = tf.identity(optimizer.get_slot(var0, 'moments'))

    # Ensure moments are properly scaled and kinetic temperatures are ok
    mom_new_target = tf.sqrt(100.0 / 1.0) * mom_old
    self.assertAllClose(mom_new, mom_new_target,
                        msg='Moments not adjusted on preconditioner update.')
    self._check_kinetic_temperature_regions(model, optimizer)

    # Check kinetic temperature is ok after adjustment
    sgmcmc_testlib.sample_model(model, optimizer, 2000)
    self._check_kinetic_temperature_regions(model, optimizer)

  def test_serialization(self):
    opt = sgmcmc.BAOABMCMC(name='TestName',
                           learning_rate=0.02,
                           momentum_decay=0.7,
                           timestep_factor=0.5,
                           preconditioner_update=False,
                           total_sample_size=1)
    sdict = opt.get_config()
    self.assertEqual(sdict['name'], 'TestName')
    self.assertAlmostEqual(sdict['learning_rate'], 0.02, delta=1.0e-7)
    self.assertAlmostEqual(sdict['momentum_decay'], 0.7, delta=1.0e-7)
    self.assertAlmostEqual(sdict['timestep_factor'], 0.5, delta=1.0e-7)
    self.assertAlmostEqual(sdict['total_sample_size'], 1)
    self.assertEqual(sdict['preconditioner_update'], False)

    opt_d = sgmcmc.BAOABMCMC(**sdict)
    sdict_d = opt_d.get_config()
    self.assertLen(sdict_d, len(sdict),
                   msg='Serialization roundtrip dictionary mismatch.')
    for key in sdict:
      self.assertEqual(sdict[key], sdict_d[key],
                       msg='Element mismatch in serialization.')

if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
