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

"""Unit tests for stepfun."""

from absl.testing import absltest
from absl.testing import parameterized
from internal import stepfun
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import scipy as sp
import scipy.ndimage  # pylint: disable=unused-import
import scipy.special  # pylint: disable=unused-import
import scipy.stats  # pylint: disable=unused-import


def generate_toy_histograms():
  """A helper function for generating some histograms for use by some tests."""
  n, d = 100, 8
  rng = random.PRNGKey(0)
  key, rng = random.split(rng)
  t = jnp.sort(random.uniform(key, shape=(n, d + 1)), axis=-1)

  # Set a few t-values to be the same as the previous value.
  key, rng = random.split(rng)
  mask = random.uniform(key, shape=(n, d)) < 0.1
  t = np.concatenate(
      [t[Ellipsis, :1], np.where(mask, t[Ellipsis, :-1], t[Ellipsis, 1:])], axis=-1
  )

  key, rng = random.split(rng)
  w = random.uniform(key, shape=(n, d))

  key, rng = random.split(rng)
  p = jnp.exp(5 * random.normal(key, shape=(n, d)))

  return t, w, p


class StepFunTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('front_delta_0', 'front', 0.0),  # Include the front of each span.
      ('front_delta_0.05', 'front', 0.05),
      ('front_delta_0.099', 'front', 0.099),
      ('back_delta_1e-6', 'back', 1e-6),  # Exclude the back of each span.
      ('back_delta_0.05', 'back', 0.05),
      ('back_delta_0.099', 'back', 0.099),
      ('before', 'before', 1e-6),
      ('after', 'after', 0.0),
  )
  def test_query(self, mode, delta):
    """Test that query() behaves sensibly in easy cases."""
    n, d = 10, 8
    left = -10.0
    right = 15
    max_delta = 0.1

    key0, key1 = random.split(random.PRNGKey(0))
    # Each t value is at least max_delta more than the one before.
    t = -d / 2 + jnp.cumsum(
        random.uniform(key0, minval=max_delta, shape=(n, d + 1)), axis=-1
    )
    y = random.normal(key1, shape=(n, d))

    query = lambda tq: stepfun.query(tq, t, y, left=left, right=right)

    if mode == 'front':
      # Query the a point relative to the front of each span, shifted by delta
      # (if delta < max_delta this will not take you out of the current span).
      assert delta >= 0
      assert delta < max_delta
      yq = query(t[Ellipsis, :-1] + delta)
      np.testing.assert_array_equal(yq, y)
    elif mode == 'back':
      # Query the a point relative to the back of each span, shifted by delta
      # (if delta < max_delta this will not take you out of the current span).
      assert delta >= 0
      assert delta < max_delta
      yq = query(t[Ellipsis, 1:] - delta)
      np.testing.assert_array_equal(yq, y)
    elif mode == 'before':
      # Query values before the domain of the step function (exclusive).
      min_val = jnp.min(t, axis=-1)
      assert delta >= 0
      tq = min_val[:, None] + jnp.linspace(-10, -delta, 100)[None, :]
      yq = query(tq)
      np.testing.assert_array_equal(yq, left)
    elif mode == 'after':
      # Queries values after the domain of the step function (inclusive).
      max_val = jnp.max(t, axis=-1)
      assert delta >= 0
      tq = max_val[:, None] + jnp.linspace(delta, 10, 100)[None, :]
      yq = query(tq)
      np.testing.assert_array_equal(yq, right)

  @parameterized.parameters((None, None), (None, 10), (-5, None), (-10, 15))
  def test_query_boundaries(self, left, right):
    """Test that query() has correct boundary conditions."""
    n, d, m = 10, 8, 100
    key0, key1 = random.split(random.PRNGKey(0))
    t = n * np.sort(random.uniform(key0, shape=(n, d + 1)), axis=-1)
    t -= np.mean(t, axis=-1, keepdims=True)
    y = random.normal(key1, shape=(n, d))
    query = lambda tq: stepfun.query(tq, t, y, left=left, right=right)

    left_true = left or jnp.tile(y[Ellipsis, :1], [1, m])
    right_true = right or jnp.tile(y[Ellipsis, -1:], [1, m])

    # Left boundaries are non-inclusive, right boundaries are inclusive.
    tq_left = np.nextafter(t[Ellipsis, :1], -np.inf) - np.arange(0, m)[None]
    tq_right = t[Ellipsis, -1:] + np.arange(0, m)[None]

    np.testing.assert_array_equal(query(tq_left), left_true)
    np.testing.assert_array_equal(query(tq_right), right_true)

  def test_distortion_loss_is_shift_invariant(self):
    """Test that distortion loss is shift-invariant."""
    n, d = 10, 8
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = random.uniform(key, minval=-3, maxval=3, shape=(n, d + 1))
    t = jnp.sort(t, axis=-1)
    key, rng = random.split(rng)
    w = random.uniform(key, shape=(n, d))
    key, rng = random.split(rng)
    shift = random.normal(key, shape=(n,))

    loss = stepfun.lossfun_distortion(t, w)
    loss_shifted = stepfun.lossfun_distortion(t + shift[:, None], w)

    np.testing.assert_allclose(loss, loss_shifted, atol=1e-5, rtol=1e-5)

  def test_distortion_loss_scales_as_expected(self):
    """Check that distortion(a*t, b*w) == a * b^2 * distortion(t, w)."""
    n, d = 10, 8
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = random.uniform(key, minval=-3, maxval=3, shape=(n, d + 1))
    t = jnp.sort(t, axis=-1)
    key, rng = random.split(rng)
    w = random.uniform(key, shape=(n, d))

    key, rng = random.split(rng)
    a = jnp.exp(random.normal(key, shape=(n,)))
    key, rng = random.split(rng)
    b = jnp.exp(random.normal(key, shape=(n,)))

    loss = stepfun.lossfun_distortion(t, w)
    loss_scaled = stepfun.lossfun_distortion(t * a[:, None], w * b[:, None])

    np.testing.assert_allclose(
        loss_scaled, a * b**2 * loss, atol=1e-5, rtol=1e-5
    )

  def test_distortion_loss_against_sampling(self):
    """Test that the distortion loss matches a stochastic approximation."""
    # Construct a random step function that defines a weight distribution.
    n, d = 10, 8
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = random.uniform(key, minval=-3, maxval=3, shape=(n, d + 1))
    t = jnp.sort(t, axis=-1)
    key, rng = random.split(rng)
    logits = 2 * random.normal(key, shape=(n, d))

    # Compute the distortion loss.
    w = jax.nn.softmax(logits, axis=-1)
    losses = stepfun.lossfun_distortion(t, w)

    # Approximate the distortion loss using samples from the step function.
    key, rng = random.split(rng)
    samples = stepfun.sample(key, t, logits, 10000, single_jitter=False)
    losses_stoch = []
    for i in range(n):
      losses_stoch.append(
          jnp.mean(jnp.abs(samples[i][:, None] - samples[i][None, :]))
      )
    losses_stoch = jnp.array(losses_stoch)

    np.testing.assert_allclose(losses, losses_stoch, atol=1e-4, rtol=1e-4)

  @parameterized.named_parameters(
      ('deterministic', False, None),
      ('random_multiple_jitters', True, False),
      ('random_single_jitter', True, True),
  )
  def test_sample_train_mode(self, randomized, single_jitter):
    """Test that piecewise-constant sampling reproduces its distribution."""
    rng = random.PRNGKey(0)
    batch_size = 4
    num_bins = 16
    num_samples = 1000000
    precision = 1e5

    # Generate a series of random PDFs to sample from.
    data = []
    for _ in range(batch_size):
      rng, key = random.split(rng)
      # Randomly initialize the distances between bins.
      # We're rolling our own fixed precision here to make cumsum exact.
      bins_delta = jnp.round(
          precision
          * jnp.exp(
              random.uniform(key, shape=(num_bins + 1,), minval=-3, maxval=3)
          )
      )

      # Set some of the bin distances to 0.
      rng, key = random.split(rng)
      bins_delta *= random.uniform(key, shape=bins_delta.shape) < 0.9

      # Integrate the bins.
      bins = jnp.cumsum(bins_delta) / precision
      rng, key = random.split(rng)
      bins += random.normal(key) * num_bins / 2
      rng, key = random.split(rng)

      # Randomly generate weights, allowing some to be zero.
      weights = jnp.maximum(
          0, random.uniform(key, shape=(num_bins,), minval=-0.5, maxval=1.0)
      )
      gt_hist = weights / weights.sum()
      data.append((bins, weights, gt_hist))

    bins, weights, gt_hist = [jnp.stack(x) for x in zip(*data)]

    rng = random.PRNGKey(0) if randomized else None
    # Draw samples from the batch of PDFs.
    samples = stepfun.sample(
        key,
        bins,
        jnp.log(weights) + 0.7,
        num_samples,
        single_jitter=single_jitter,
    )
    self.assertEqual(samples.shape[-1], num_samples)

    # Check that samples are sorted.
    self.assertTrue(jnp.all(samples[Ellipsis, 1:] >= samples[Ellipsis, :-1]))

    # Verify that each set of samples resembles the target distribution.
    for i_samples, i_bins, i_gt_hist in zip(samples, bins, gt_hist):
      i_hist = jnp.float32(jnp.histogram(i_samples, i_bins)[0]) / num_samples
      i_gt_hist = jnp.array(i_gt_hist)

      # Merge any of the zero-span bins until there aren't any left.
      while jnp.any(i_bins[:-1] == i_bins[1:]):
        j = int(jnp.where(i_bins[:-1] == i_bins[1:])[0][0])
        i_hist = jnp.concatenate([
            i_hist[:j],
            jnp.array([i_hist[j] + i_hist[j + 1]]),
            i_hist[j + 2 :],
        ])
        i_gt_hist = jnp.concatenate([
            i_gt_hist[:j],
            jnp.array([i_gt_hist[j] + i_gt_hist[j + 1]]),
            i_gt_hist[j + 2 :],
        ])
        i_bins = jnp.concatenate([i_bins[:j], i_bins[j + 1 :]])

      # Angle between the two histograms in degrees.
      angle = (
          180
          / jnp.pi
          * jnp.arccos(
              jnp.minimum(
                  1.0,
                  jnp.mean(
                      (i_hist * i_gt_hist)
                      / jnp.sqrt(
                          jnp.mean(i_hist**2) * jnp.mean(i_gt_hist**2)
                      )
                  ),
              )
          )
      )
      # Jensen-Shannon divergence.
      m = (i_hist + i_gt_hist) / 2
      js_div = (
          jnp.sum(
              sp.special.kl_div(i_hist, m) + sp.special.kl_div(i_gt_hist, m)
          )
          / 2
      )
      self.assertLessEqual(angle, 0.5)
      self.assertLessEqual(js_div, 1e-5)

  @parameterized.named_parameters(
      ('deterministic', False, None),
      ('random_multiple_jitters', True, False),
      ('random_single_jitter', True, True),
  )
  def test_sample_large_flat(self, randomized, single_jitter):
    """Test sampling when given a large flat distribution."""
    key = random.PRNGKey(0) if randomized else None
    num_samples = 100
    num_bins = 100000
    bins = jnp.arange(num_bins)
    weights = np.ones(len(bins) - 1)
    samples = stepfun.sample(
        key,
        bins[None],
        jnp.log(jnp.maximum(1e-15, weights[None])),
        num_samples,
        single_jitter=single_jitter,
    )[0]
    # All samples should be within the range of the bins.
    self.assertTrue(jnp.all(samples >= bins[0]))
    self.assertTrue(jnp.all(samples <= bins[-1]))

    # Samples modded by their bin index should resemble a uniform distribution.
    samples_mod = jnp.mod(samples, 1)
    self.assertLessEqual(
        sp.stats.kstest(samples_mod, 'uniform', (0, 1)).statistic, 0.2
    )

    # All samples should collectively resemble a uniform distribution.
    self.assertLessEqual(
        sp.stats.kstest(samples, 'uniform', (bins[0], bins[-1])).statistic, 0.2
    )

  @parameterized.named_parameters(
      ('deterministic', False, None),
      ('random_multiple_jitters', True, False),
      ('random_single_jitter', True, True),
  )
  def test_sample_sparse_delta(self, randomized, single_jitter):
    """Test sampling when given a large distribution with a big delta in it."""
    key = random.PRNGKey(0) if randomized else None
    num_samples = 100
    num_bins = 100000
    bins = jnp.arange(num_bins)
    weights = np.ones(len(bins) - 1)
    delta_idx = len(weights) // 2
    weights[delta_idx] = len(weights) - 1
    samples = stepfun.sample(
        key,
        bins[None],
        jnp.log(jnp.maximum(1e-15, weights[None])),
        num_samples,
        single_jitter=single_jitter,
    )[0]

    # All samples should be within the range of the bins.
    self.assertTrue(jnp.all(samples >= bins[0]))
    self.assertTrue(jnp.all(samples <= bins[-1]))

    # Samples modded by their bin index should resemble a uniform distribution.
    samples_mod = jnp.mod(samples, 1)
    self.assertLessEqual(
        sp.stats.kstest(samples_mod, 'uniform', (0, 1)).statistic, 0.2
    )

    # The delta function bin should contain ~half of the samples.
    in_delta = (samples >= bins[delta_idx]) & (samples <= bins[delta_idx + 1])
    np.testing.assert_allclose(jnp.mean(in_delta), 0.5, atol=0.05)

  @parameterized.named_parameters(
      ('deterministic', False, None),
      ('random_multiple_jitters', True, False),
      ('random_single_jitter', True, True),
  )
  def test_sample_single_bin(self, randomized, single_jitter):
    """Test sampling when given a small `one hot' distribution."""
    key = random.PRNGKey(0) if randomized else None
    num_samples = 625
    bins = jnp.array([0, 1, 3, 6, 10], jnp.float32)
    for i in range(len(bins) - 1):
      weights = np.zeros(len(bins) - 1, jnp.float32)
      weights[i] = 1.0
      samples = stepfun.sample(
          key,
          bins[None],
          jnp.log(weights[None]),
          num_samples,
          single_jitter=single_jitter,
      )[0]

      # All samples should be within [bins[i], bins[i+1]].
      self.assertTrue(jnp.all(samples >= bins[i]))
      self.assertTrue(jnp.all(samples <= bins[i + 1]))

  @parameterized.named_parameters(
      ('deterministic', False, 0.1), ('random', True, 0.1)
  )
  def test_sample_intervals_accuracy(self, randomized, tolerance):
    """Test that resampled intervals resemble their original distribution."""
    n, d = 50, 32
    d_resample = 2 * d
    domain = -3, 3

    # Generate some step functions.
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = random.uniform(
        key, minval=domain[0], maxval=domain[1], shape=(n, d + 1)
    )
    t = jnp.sort(t, axis=-1)
    key, rng = random.split(rng)
    logits = 2 * random.normal(key, shape=(n, d))

    # Resample the step functions.
    key = random.PRNGKey(999) if randomized else None
    t_sampled = stepfun.sample_intervals(
        key, t, logits, d_resample, single_jitter=True, domain=domain
    )

    # Precompute the accumulated weights of the original intervals.
    weights = jax.nn.softmax(logits, axis=-1)
    acc_weights = stepfun.integrate_weights(weights)

    errors = []
    for i in range(t_sampled.shape[0]):
      # Resample into the original accumulated weights.
      acc_resampled = jnp.interp(t_sampled[i], t[i], acc_weights[i])
      # Differentiate the accumulation to get resampled weights (that do not
      # necessarily sum to 1 because some of the ends might get missed).
      weights_resampled = jnp.diff(acc_resampled, axis=-1)
      # Check that the resampled weights resemble a uniform distribution.
      u = 1 / len(weights_resampled)
      errors.append(float(jnp.sum(jnp.abs(weights_resampled - u))))
    errors = jnp.array(errors)
    mean_error = jnp.mean(errors)
    self.assertLess(mean_error, tolerance)

  @parameterized.named_parameters(
      ('deterministic_unbounded', False, False),
      ('random_unbounded', True, False),
      ('deterministic_bounded', False, True),
      ('random_bounded', True, True),
  )
  def test_sample_intervals_unbiased(self, randomized, bound_domain):
    """Test that resampled intervals are unbiased."""
    n, d_resample = 1000, 64
    domain = (-0.5, 0.5) if bound_domain else (-jnp.inf, jnp.inf)

    # A single interval from [-0.5, 0.5].
    t = jnp.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    logits = jnp.array([0, 0, 100.0, 0, 0])

    ts = jnp.tile(t[None], [n, 1])
    logits = jnp.tile(logits[None], [n, 1])

    # Resample the step functions.
    rng = random.PRNGKey(0) if randomized else None
    t_sampled = stepfun.sample_intervals(
        rng, ts, logits, d_resample, single_jitter=True, domain=domain
    )

    # The average sample should be close to zero.
    if randomized:
      self.assertLess(
          jnp.max(jnp.abs(jnp.mean(t_sampled, axis=-1))), 0.5 / d_resample
      )
    else:
      np.testing.assert_allclose(
          jnp.mean(t_sampled, axis=-1), jnp.zeros(n), atol=1e-5, rtol=1e-5
      )

    # The extents of the samples should be near -0.5 and 0.5.
    if bound_domain and randomized:
      np.testing.assert_allclose(jnp.median(t_sampled[:, 0]), -0.5, atol=1e-4)
      np.testing.assert_allclose(jnp.median(t_sampled[:, -1]), 0.5, atol=1e-4)

    # The interval edge near the extent should be centered around +/-0.5.
    if randomized:
      np.testing.assert_allclose(
          jnp.mean(t_sampled[:, 0] > -0.5), 0.5, atol=1 / d_resample
      )
      np.testing.assert_allclose(
          jnp.mean(t_sampled[:, -1] < 0.5), 0.5, atol=1 / d_resample
      )

  def test_sample_single_interval(self):
    """Resample a single interval and check that it's a linspace."""
    t = jnp.array([1, 2, 3, 4, 5, 6])
    logits = jnp.array([0, 0, 100, 0, 0])
    key = None
    t_sampled = stepfun.sample_intervals(key, t, logits, 10, single_jitter=True)
    np.testing.assert_allclose(
        t_sampled, jnp.linspace(3, 4, 11), atol=1e-5, rtol=1e-5
    )

  def test_weighted_percentile(self):
    """Test that step function percentiles match the empirical percentile."""
    num_samples = 1000000
    rng = random.PRNGKey(0)
    for _ in range(10):
      rng, key = random.split(rng)
      d = random.randint(key, (), minval=10, maxval=20)

      rng, key = random.split(rng)
      ps = 100 * random.uniform(key, [3])

      key, rng = random.split(rng)
      t = jnp.sort(random.normal(key, [d + 1]), axis=-1)

      key, rng = random.split(rng)
      w = jax.nn.softmax(random.normal(key, [d]))

      key, rng = random.split(rng)
      samples = stepfun.sample(
          key, t, jnp.log(w), num_samples, single_jitter=False
      )
      true_percentiles = jnp.percentile(samples, ps)

      our_percentiles = stepfun.weighted_percentile(t, w, ps)
      np.testing.assert_allclose(
          our_percentiles, true_percentiles, rtol=1e-4, atol=1e-4
      )

  def test_weighted_percentile_vectorized(self):
    rng = random.PRNGKey(0)
    shape = (3, 4)
    d = 128

    rng, key = random.split(rng)
    ps = 100 * random.uniform(key, (5,))

    key, rng = random.split(rng)
    t = jnp.sort(random.normal(key, shape + (d + 1,)), axis=-1)

    key, rng = random.split(rng)
    w = jax.nn.softmax(random.normal(key, shape + (d,)))

    percentiles_vec = stepfun.weighted_percentile(t, w, ps)

    percentiles = []
    for i in range(shape[0]):
      percentiles.append([])
      for j in range(shape[1]):
        percentiles[i].append(stepfun.weighted_percentile(t[i, j], w[i, j], ps))
      percentiles[i] = jnp.stack(percentiles[i])
    percentiles = jnp.stack(percentiles)

    np.testing.assert_allclose(
        percentiles_vec, percentiles, rtol=1e-5, atol=1e-5
    )

  @parameterized.named_parameters(('', False), ('_avg', True))
  def test_resample_self_noop(self, use_avg):
    """Resampling a step function into itself should be a no-op."""
    d = 32
    rng = random.PRNGKey(0)

    key, rng = random.split(rng)
    tp = random.normal(rng, shape=(d + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=(d,))

    vp_recon = stepfun.resample(tp, tp, vp, use_avg=use_avg)
    np.testing.assert_allclose(vp, vp_recon, atol=1e-4)

  @parameterized.named_parameters(('', False), ('_avg', True))
  def test_resample_2x_downsample(self, use_avg):
    """Check resampling for a 2d downsample."""
    d = 32
    rng = random.PRNGKey(0)

    key, rng = random.split(rng)
    tp = random.normal(rng, shape=(d + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=(d,))

    t = tp[::2]

    v = stepfun.resample(t, tp, vp, use_avg=use_avg)

    vp2 = vp.reshape([-1, 2])
    dtp2 = jnp.diff(tp).reshape([-1, 2])
    if use_avg:
      v_true = jnp.sum(vp2 * dtp2, axis=-1) / jnp.sum(dtp2, axis=-1)
    else:
      v_true = jnp.sum(vp2, axis=-1)

    np.testing.assert_allclose(v, v_true, atol=1e-4)

  @parameterized.named_parameters(('', False), ('_avg', True))
  def test_resample_entire_interval(self, use_avg):
    """Check the sum (or weighted mean) of an entire interval."""
    d = 32
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    tp = random.normal(rng, shape=(d + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=(d,))

    t = jnp.array([jnp.min(tp), jnp.max(tp)])

    v = stepfun.resample(t, tp, vp, use_avg=use_avg)[0]
    if use_avg:
      v_true = jnp.sum(vp * jnp.diff(tp)) / sum(jnp.diff(tp))
    else:
      v_true = jnp.sum(vp)

    np.testing.assert_allclose(v, v_true, atol=1e-4)

  def test_resample_entire_domain(self):
    """Check the sum of the entire input domain."""
    d = 32
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    tp = random.normal(rng, shape=(d + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=(d,))

    t = jnp.array([-1e6, 1e6])

    v = stepfun.resample(t, tp, vp)[0]
    v_true = jnp.sum(vp)

    np.testing.assert_allclose(v, v_true, atol=1e-4)

  @parameterized.named_parameters(('', False), ('_avg', True))
  def test_resample_single_span(self, use_avg):
    """Check the sum (or weighted mean) of a single span."""
    d = 32
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    tp = random.normal(rng, shape=(d + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=(d,))

    pad = (tp[d // 2 + 1] - tp[d // 2]) / 4
    t = jnp.array([tp[d // 2] + pad, tp[d // 2 + 1] - pad])

    v = stepfun.resample(t, tp, vp, use_avg=use_avg)[0]
    if use_avg:
      v_true = vp[d // 2]
    else:
      v_true = vp[d // 2] * 0.5

    np.testing.assert_allclose(v, v_true, atol=1e-4)

  @parameterized.named_parameters(('', False), ('_avg', True))
  def test_resample_vectorized(self, use_avg):
    """Check that resample works with vectorized inputs."""
    shape = (3, 4)
    dp = 32
    d = 16
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    tp = random.normal(rng, shape=shape + (dp + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=shape + (dp,))

    key, rng = random.split(rng)
    t = random.normal(rng, shape=shape + (d + 1,))
    t = jnp.sort(t)

    v_batch = stepfun.resample(t, tp, vp, use_avg=use_avg)

    v_indiv = []
    for i in range(t.shape[0]):
      v_indiv.append(
          jnp.array(
              [
                  stepfun.resample(t[i, j], tp[i, j], vp[i, j], use_avg=use_avg)
                  for j in range(t.shape[1])
              ]
          )
      )
    v_indiv = jnp.array(v_indiv)

    np.testing.assert_allclose(v_batch, v_indiv, atol=1e-4)

  @parameterized.parameters(2.0 ** np.arange(-10, 1))
  def test_blur_and_resample_toy_weights(self, hw):
    """Blur a single histogram bin next to two empty bins of different sizes."""
    t = np.array([-3.5, -0.8, 1.0, 2.0, 4.7, 8.2])
    w = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    wq = stepfun.blur_and_resample_weights(t, t, w, hw)
    wq_true = np.array([0.0, hw / 4, 1 - hw / 2, hw / 4, 0.0])
    np.testing.assert_allclose(wq, wq_true)

  def test_blur_and_resample_weights_pool_two_spikes(self):
    """Resample two small intervals into a single larger interval."""
    np.testing.assert_allclose(
        stepfun.blur_and_resample_weights(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([1.2, 1.3, 1.7, 1.8]),
            np.array([2.0, 0.0, 3.0]),
            0.1,
        ),
        np.array([0.0, 5.0, 0.0]),
        atol=1e-5,
    )

  def test_blur_and_resample_weights_extend_boundaries(self):
    """Resampling from non-overlapping step functions zeros out the weight."""
    np.testing.assert_allclose(
        stepfun.blur_and_resample_weights(
            np.array([0, 1]), np.array([2, 3]), np.array([7.0]), 0.5
        ),
        np.array([0.0]),
    )

  def test_blur_and_resample_weights_no_op(self):
    """Blurring with halfwidth=0 and resampling into yourself is a no-op."""
    n, d = 100, 10
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = jnp.sort(random.uniform(key, shape=(n, d + 1)), axis=-1)
    key, rng = random.split(rng)
    w = random.uniform(key, shape=(n, d))

    w_recon = stepfun.blur_and_resample_weights(t, t, w, 0.0)
    np.testing.assert_allclose(w, w_recon, atol=1e-4)

  @parameterized.parameters([0.001, 0.01, 0.1])
  def test_blur_and_resample_weights_preserves_normalization(self, hw):
    """Resampling into a wider set of intervals preseves the sum of weights."""
    n, d = 100, 10
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = jnp.sort(random.uniform(key, shape=(n, d + 1)), axis=-1)
    key, rng = random.split(rng)
    tq = jnp.sort(random.uniform(key, shape=(n, d + 1)), axis=-1)
    key, rng = random.split(rng)
    w = random.uniform(key, shape=(n, d))

    # Spread each tq to span [-1, 2], while keeping t in (0, 1).
    tq = tq - jnp.min(tq, axis=-1, keepdims=True)
    tq = tq / jnp.max(tq, axis=-1, keepdims=True)
    tq = 3 * tq - 1

    wq = stepfun.blur_and_resample_weights(tq, t, w, hw)
    np.testing.assert_allclose(
        jnp.sum(w, axis=-1), jnp.sum(wq, axis=-1), atol=1e-2
    )

  def test_weight_pdf_conversion_is_accurate(self):
    t, w, _ = generate_toy_histograms()

    p = stepfun.weight_to_pdf(t, w)
    w_recon = stepfun.pdf_to_weight(t, p)

    valid = np.diff(t) > 0
    np.testing.assert_allclose(w[valid], w_recon[valid], atol=1e-7)
    np.testing.assert_array_equal(w_recon[~valid], 0)

  def test_weight_to_pdf_gradient_is_finite(self):
    t, w, _ = generate_toy_histograms()

    dt, dw = jax.grad(lambda *x: jnp.sum(stepfun.weight_to_pdf(*x)), [0, 1])(
        t, w
    )
    self.assertTrue(np.all(np.isfinite(dt)))
    self.assertTrue(np.all(np.isfinite(dw)))

    # Check that dw is 0 when the t-delta is below numerical epsilon.
    np.testing.assert_array_equal(dw[np.diff(t) == 0], 0)
    self.assertTrue(np.all(dw[np.diff(t) > 0] > 0))

  def test_pdf_to_weight_gradient_is_finite(self):
    t, _, p = generate_toy_histograms()

    dt, dp = jax.grad(lambda *x: jnp.sum(stepfun.pdf_to_weight(*x)), [0, 1])(
        t, p
    )
    self.assertTrue(np.all(np.isfinite(dt)))
    self.assertTrue(np.all(np.isfinite(dp)))

    # Check that dp is 0 when the t-delta is below numerical epsilon.
    np.testing.assert_array_equal(dp[np.diff(t) == 0], 0)
    self.assertTrue(np.all(dp[np.diff(t) > 0] > 0))


if __name__ == '__main__':
  absltest.main()
