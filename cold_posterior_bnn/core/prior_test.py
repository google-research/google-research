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

"""Tests for google_research.google_research.cold_posterior_bnn.core.prior."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math
from absl.testing import parameterized
import scipy.integrate as integrate

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from cold_posterior_bnn.core import prior

tfd = tfp.distributions

REGULARIZER_INSTANCES_LOGPDF = [
    prior.NormalRegularizer(stddev=0.1, weight=0.5),
    prior.ShiftedNormalRegularizer(mean=0.2, stddev=0.1, weight=0.5),
    prior.StretchedNormalRegularizer(offset=0.2, scale=1.2, weight=0.1),
    prior.LaplaceRegularizer(stddev=0.1),
    prior.CauchyRegularizer(scale=0.2),
    prior.SpikeAndSlabRegularizer(scale_spike=0.01, scale_slab=0.3,
                                  mass_spike=0.6, weight=0.8),
]
REGULARIZER_INSTANCES_NOLOGPDF = [
    prior.HeNormalRegularizer(scale=0.8, weight=0.7),
    prior.GlorotNormalRegularizer(scale=0.8, weight=0.7),
    prior.EmpiricalBayesNormal(ig_shape=2.5, ig_scale=0.25, weight=0.3),
    prior.HeNormalEBRegularizer(scale=0.25, weight=0.1),
]
REGULARIZER_INSTANCES = REGULARIZER_INSTANCES_LOGPDF + \
    REGULARIZER_INSTANCES_NOLOGPDF

# Regularizers with scale parameters, scale=1
REGULARIZER_INSTANCES_SCALE1 = [
    ('stddev', prior.NormalRegularizer(stddev=1.0, weight=0.5)),
    ('stddev', prior.ShiftedNormalRegularizer(mean=0., stddev=1.0,
                                              weight=0.5)),
    ('stddev', prior.LaplaceRegularizer(stddev=1.0)),
    ('scale', prior.CauchyRegularizer(scale=1.0)),
]


class PriorTest(parameterized.TestCase, tf.test.TestCase):

  def _check_regularizer(self, reg):
    weights = tf.random.normal((12, 16))
    nll = reg(weights)
    self.assertTrue(bool(tf.is_finite(nll)), msg='Invalid prior nll returned.')
    self.assertFalse(bool(tf.is_nan(nll)), msg='Prior nll is NaN.')

  def test_tfd_prior_spike_and_slab(self):
    self._check_regularizer(prior.SpikeAndSlabRegularizer())

  def test_tfd_prior_ebnormal(self):
    self._check_regularizer(prior.EmpiricalBayesNormal())

  def test_inverse_gamma_initialization(self):
    mean = 2.0
    stddev = 0.5
    ig_shape, ig_scale = prior.inverse_gamma_shape_scale_from_mean_stddev(
        mean, stddev)

    # Note: bug in tfp: second parameter is a scale parameter, not a rate
    # parameter.
    pig = tfd.InverseGamma(ig_shape, ig_scale)
    self.assertAlmostEqual(float(pig.mean()), mean,
                           msg='InverseGamma initialization wrong in mean.')
    self.assertAlmostEqual(float(pig.stddev()), stddev,
                           msg='InverseGamma initialization wrong in stddev.')

  def test_he_regularizer(self):
    reg = prior.HeNormalRegularizer(weight=1.0)
    result = float(reg(tf.ones((3, 5))))
    self.assertAlmostEqual(result, 11.25,
                           msg='HeNormalRegularizer regularization wrong.')

  def test_glorot_regularizer(self):
    reg = prior.GlorotNormalRegularizer(weight=1.0)
    result = float(reg(tf.ones((3, 5))))
    self.assertAlmostEqual(result, 30.,
                           msg='GlorotNormalRegularizer regularization wrong.')

  def _normal_nll(self, w, stddev):
    n = tf.cast(tf.size(w), tf.float32)
    v = stddev**2.0
    logp = -0.5*n*tf.math.log(2.0*math.pi)
    logp += -0.5*n*tf.math.log(v)
    logp += -0.5*tf.reduce_sum(tf.square(w / stddev))

    return -logp

  @parameterized.parameters(
      itertools.product(
          [0.1, 1.0, 2.0],
          [0.1, 1.0, 2.0]))
  def test_ebnormal(self, gen_stddev, eb_prior_stddev):
    tf.set_random_seed(1)
    eb_prior = prior.EmpiricalBayesNormal.from_stddev(eb_prior_stddev)
    w = gen_stddev*tf.random_normal((1024, 2048))
    normal_nll = self._normal_nll(w, gen_stddev)  # -log N(w), generating
    eb_nll = eb_prior(w)  # -log N(w; 0, vhat), EB
    normal_nll /= 1024.0*2048.0
    eb_nll /= 1024.0*2048.0
    self.assertAlmostEqual(normal_nll, eb_nll, delta=0.001,
                           msg='Parameters score NLL=%.6f on generating '
                           'Normal and NLL=%.6f on EB-fitted Normal, '
                           'too much difference.' % (normal_nll, eb_nll))

  def test_eb_regularizer(self):
    eb_reg = prior.HeNormalEBRegularizer()
    eb_prior = prior.EmpiricalBayesNormal.from_stddev(math.sqrt(2.0/512.0))
    w = tf.random_normal((1024, 512))
    value_reg = eb_reg(w)
    value_prior = eb_prior(w)
    self.assertAlmostEqual(value_reg, value_prior, delta=1.0e-6,
                           msg='Regularizer value %.6f disagrees with nll of '
                           'prior %.6f' % (value_reg, value_prior))

  def test_cauchy_regularizer(self):
    # Check values against values obtained from Julia's Distributions.jl
    # package via:
    # jl>>> -logpdf.(Cauchy(0.0,0.5),[0.0,0.3,1.2,23.5]) .- log(pi) .- log(0.5)
    cauchy_reg = prior.CauchyRegularizer(scale=0.5, weight=1.0)
    for position, reg_true_value in zip(
        [0.0, 0.3, 1.2, 23.5],
        [0.0, 0.30748469974796055, 1.9110228900548725, 7.700747794511799]):
      reg_value = cauchy_reg(position)
      self.assertAlmostEqual(reg_value, reg_true_value, delta=1.0e-6,
                             msg='Cauchy regularization value of %.6f at '
                             'x=%.5f disagrees with true value of %.6f' % (
                                 reg_value, position, reg_true_value))

  @parameterized.parameters([
      prior.NormalRegularizer(stddev=0.1),
      prior.NormalRegularizer(stddev=0.5),
      prior.ShiftedNormalRegularizer(mean=0.2, stddev=0.5),
      prior.ShiftedNormalRegularizer(mean=-1.2, stddev=1.1),
      prior.StretchedNormalRegularizer(offset=0.2, scale=1.2),
      prior.StretchedNormalRegularizer(offset=0.5, scale=0.1),
      prior.LaplaceRegularizer(stddev=0.1),
      prior.LaplaceRegularizer(stddev=0.2),
      prior.CauchyRegularizer(scale=0.1),
      prior.CauchyRegularizer(scale=0.2),
  ])
  def test_regularizer_logpdf(self, reg):
    def pdf(x):
      logpdf = reg.logpdf(tf.convert_to_tensor(x, dtype=tf.float32))
      logpdf = float(logpdf)
      return math.exp(logpdf)

    area, _ = integrate.quad(pdf, -15.0, 15.0)
    self.assertAlmostEqual(area, 1.0, delta=0.01,
                           msg='Density does not integrate to one.')

  def to_tensor(self, x):
    return tf.convert_to_tensor(x, dtype=tf.float32)

  def test_regularizer_logpdf_values(self):
    # Compare against reference values obtained from Julia's Distributions.jl
    # package.
    # jl> logpdf(Normal(0,0.3), 0.7)
    self.assertAlmostEqual(
        prior.NormalRegularizer(stddev=0.3).logpdf(self.to_tensor(0.7)),
        -2.4371879511,
        delta=1.0e-6)
    # jl> logpdf(Normal(0,2.5), 0.0)
    self.assertAlmostEqual(
        prior.NormalRegularizer(stddev=2.5).logpdf(self.to_tensor(0.0)),
        -1.8352292650,
        delta=1.0e-6)

    # jl> logpdf(Normal(0.2,0.3), 0.7)
    self.assertAlmostEqual(
        prior.ShiftedNormalRegularizer(mean=0.2,
                                       stddev=0.3).logpdf(self.to_tensor(0.7)),
        -1.1038546177,
        delta=1.0e-6)
    # jl> logpdf(Normal(-0.3,2.5), 0.0)
    self.assertAlmostEqual(
        prior.ShiftedNormalRegularizer(mean=-0.3,
                                       stddev=2.5).logpdf(self.to_tensor(0.0)),
        -1.8424292650,
        delta=1.0e-6)

    # jl> logpdf(Laplace(0,1),0)
    self.assertAlmostEqual(
        prior.LaplaceRegularizer(stddev=math.sqrt(2.0)*1.0).logpdf(
            self.to_tensor(0.0)),
        -0.6931471805,
        delta=1.0e-6)
    # jl> logpdf(Laplace(0,0.3),1.7)
    self.assertAlmostEqual(
        prior.LaplaceRegularizer(stddev=math.sqrt(2.0)*0.3).logpdf(
            self.to_tensor(1.7)),
        -5.1558410429,
        delta=1.0e-6)

    # jl> logpdf(Cauchy(0,1),0)
    self.assertAlmostEqual(
        prior.CauchyRegularizer(scale=1.0).logpdf(self.to_tensor(0.0)),
        -1.1447298858,
        delta=1.0e-6)
    # jl> logpdf(Cauchy(0,0.3),1.1)
    self.assertAlmostEqual(
        prior.CauchyRegularizer(scale=0.3).logpdf(self.to_tensor(1.1)),
        -2.6110669546,
        delta=1.0e-6)

  @parameterized.parameters(itertools.product(
      REGULARIZER_INSTANCES,
      [0.1, 2.5]))
  def test_regularizer_weighting(self, reg0, weight_factor):
    config = reg0.get_config()
    config['weight'] *= weight_factor
    reg1 = reg0.__class__(**config)

    data = tf.random.normal((3, 5))
    res0 = float(reg0(data))
    res1 = float(reg1(data))
    self.assertAlmostEqual(weight_factor*res0, res1, delta=1.0e-3,
                           msg='Regularizers value disagree after '
                           'weighting (not linear in weight).')

  @parameterized.parameters(REGULARIZER_INSTANCES_LOGPDF)
  def test_regularizer_serialization_logpdf(self, reg0):
    # Test a round-trip serialization to make sure the entire state is captured
    config0 = reg0.get_config()
    reg1 = reg0.__class__(**config0)  # create from config dict
    config1 = reg1.get_config()
    self.assertEqual(config0, config1,
                     msg='Serialization did create different regularizer.')
    data = tf.random.normal((3, 5))
    logpdf0 = reg0.logpdf(data)
    logpdf1 = reg1.logpdf(data)
    self.assertAlmostEqual(logpdf0, logpdf1, delta=1.0e-6,
                           msg='Regularizers logpdf value disagree after '
                           'serialization.')

  @parameterized.parameters(itertools.product(
      REGULARIZER_INSTANCES_SCALE1,
      [0.2, 0.7, 1.2, 4.3]))
  def test_regularizer_logpdf_scale_parameters(self, sname_reg1, scale):
    # Check that the definition of a scale parameter is upheld
    # (see https://en.wikipedia.org/wiki/Scale_parameter).
    scale_name, reg1 = sname_reg1
    config = reg1.get_config()
    config[scale_name] = scale
    regs = reg1.__class__(**config)

    # Scale relationship: log f(x; s) = log f(x/s; 1) - log s
    data = tf.random.normal((3, 5))
    logpdf_scale_1 = float(reg1.logpdf(data / scale))
    logpdf_scale_s = float(regs.logpdf(data))
    self.assertAlmostEqual(
        logpdf_scale_s,
        logpdf_scale_1 - \
            float(tf.cast(tf.size(data), tf.float32) *
                  math.log(scale)*reg1.weight),
        delta=1.0e-5,
        msg='Scale relationship violated')


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
