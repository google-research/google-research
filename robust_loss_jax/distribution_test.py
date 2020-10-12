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

"""Tests for distribution.py."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import jax.random as random
import scipy.stats
from robust_loss_jax import distribution


class DistributionTest(chex.TestCase):

  def setUp(self):
    self._distribution = distribution.Distribution()
    super(DistributionTest, self).setUp()

  @chex.all_variants()
  def testSplineCurveIsC1Smooth(self):
    """Tests that partition_spline_curve() and its derivative are continuous."""
    x1 = jnp.linspace(0., 8., 10000)
    x2 = x1 + 1e-7

    fn = self.variant(distribution.partition_spline_curve)
    y1 = fn(x1)
    y2 = fn(x2)
    grad = jax.grad(lambda z: jnp.sum(fn(z)))
    dy1 = grad(x1)
    dy2 = grad(x2)

    chex.assert_tree_all_close(y1, y2, atol=1e-5, rtol=1e-5)
    chex.assert_tree_all_close(dy1, dy2, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testPartitionIsCorrectGolden(self):
    """Tests _log_base_partition_function against some golden data."""
    # Here we enumerate a set of positive rational numbers n/d alongside
    # numerically approximated values of Z(n / d) up to 10 digits of precision,
    # stored as (n, d, Z(n/d)). This was generated with an external mathematica
    # script.
    golden = (
        (1, 7, 4.080330073), (1, 6, 4.038544331), (1, 5, 3.984791180),
        (1, 4, 3.912448576), (1, 3, 3.808203509), (2, 5, 3.735479786),
        (3, 7, 3.706553276), (1, 2, 3.638993131), (3, 5, 3.553489270),
        (2, 3, 3.501024540), (3, 4, 3.439385624), (4, 5, 3.404121259),
        (1, 1, 3.272306973), (6, 5, 3.149249092), (5, 4, 3.119044506),
        (4, 3, 3.068687433), (7, 5, 3.028084866), (3, 2, 2.965924889),
        (8, 5, 2.901059987), (5, 3, 2.855391798), (7, 4, 2.794052016),
        (7, 3, 2.260434598), (5, 2, 2.218882601), (8, 3, 2.190349858),
        (3, 1, 2.153202857), (4, 1, 2.101960916), (7, 2, 2.121140098),
        (5, 1, 2.080000512), (9, 2, 2.089161164), (6, 1, 2.067751267),
        (7, 1, 2.059929623), (8, 1, 2.054500222), (10, 3, 2.129863884),
        (11, 3, 2.113763384), (13, 3, 2.092928254), (14, 3, 2.085788350),
        (16, 3, 2.075212740), (11, 2, 2.073116001), (17, 3, 2.071185791),
        (13, 2, 2.063452243), (15, 2, 2.056990258))  # pyformat: disable
    alpha, z_true = tuple(jnp.array([(n / d, z) for (n, d, z) in golden]).T)
    log_z_true = jnp.log(z_true)
    log_z = self.variant(self._distribution.log_base_partition_function)(alpha)
    chex.assert_tree_all_close(log_z, log_z_true, atol=1e-7, rtol=1e-7)

  @chex.all_variants()
  def testLogPartitionInfinityIsAccurate(self):
    """Tests that the partition function is accurate at infinity."""
    alpha = float('inf')
    log_z_true = 0.70526025442  # From mathematica.
    log_z = self.variant(self._distribution.log_base_partition_function)(alpha)
    chex.assert_tree_all_close(log_z, log_z_true, atol=1e-7, rtol=1e-7)

  @chex.all_variants()
  def testSplineCurveInverseIsCorrect(self):
    """Tests that the inverse curve is indeed the inverse of the curve."""
    x_knot = jnp.arange(0, 16, 0.01)
    alpha = self.variant(distribution.inv_partition_spline_curve)(x_knot)
    x_recon = self.variant(distribution.partition_spline_curve)(alpha)
    chex.assert_tree_all_close(x_recon, x_knot, atol=1e-5, rtol=1e-5)

  def testAlphaZeroSamplesMatchACauchyDistribution(self):
    """Tests that samples when alpha=0 match a Cauchy distribution."""
    num_samples = 16384
    scale = 1.7
    rng = random.PRNGKey(0)
    samples = self._distribution.draw_samples(rng, jnp.zeros(num_samples),
                                              scale * jnp.ones(num_samples))
    # Perform the Kolmogorov-Smirnov test against a Cauchy distribution.
    ks_statistic = scipy.stats.kstest(samples, 'cauchy',
                                      (0, scale * jnp.sqrt(2))).statistic
    self.assertLess(ks_statistic, 0.02)

  def testAlphaTwoSamplesMatchANormalDistribution(self):
    """Tests that samples when alpha=2 match a normal distribution."""
    num_samples = 16384
    scale = 1.7
    rng = random.PRNGKey(0)
    samples = self._distribution.draw_samples(rng, 2 * jnp.ones(num_samples),
                                              scale * jnp.ones(num_samples))
    # Perform the Kolmogorov-Smirnov test against a normal distribution.
    ks_statistic = scipy.stats.kstest(samples, 'norm', (0., scale)).statistic
    self.assertLess(ks_statistic, 0.01)

  @chex.all_variants()
  def testAlphaZeroNllsMatchACauchyDistribution(self):
    """Tests that NLLs when alpha=0 match a Cauchy distribution."""
    x = jnp.linspace(-10, 10, 1000)
    scale = 1.7
    nll = self.variant(self._distribution.nllfun)(x, 0, scale)
    nll_true = -scipy.stats.cauchy(0, scale * jnp.sqrt(2)).logpdf(x)
    chex.assert_tree_all_close(nll, nll_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testAlphaTwoNllsMatchANormalDistribution(self):
    """Tests that NLLs when alpha=2 match a normal distribution."""
    x = jnp.linspace(-10, 10, 1000)
    scale = 1.7
    nll = self.variant(self._distribution.nllfun)(x, 2, scale)
    nll_true = -scipy.stats.norm(0., scale).logpdf(x)
    chex.assert_tree_all_close(nll, nll_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testPdfIntegratesToOne(self):
    """Tests that the PDF integrates to 1 for different alphas."""
    alphas = jnp.exp(jnp.linspace(-4., 8., 8))
    scale = 1.
    x = jnp.arange(-128., 128., 1 / 256.) * scale
    for alpha in alphas:
      nll = self.variant(self._distribution.nllfun)(x, alpha, scale)
      pdf_sum = jnp.sum(jnp.exp(-nll)) * (x[1] - x[0])
      chex.assert_tree_all_close(pdf_sum, 1., atol=0.005, rtol=0.005)


if __name__ == '__main__':
  absltest.main()
