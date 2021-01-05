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
"""Diagnostics helpful in characterising deep neural network behavior.

This module implements statistics that characterise neural network behavior.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import math
import tensorflow.compat.v1 as tf


__all__ = [
    'symmetric_alpha_stable_invstability_estimator',
    'variable_gradient_stability_estimate',
]


def symmetric_alpha_stable_invstability_estimator(data, axis, nelem_per_piece):
  """Estimate the stability coefficient of the S-alpha-S distribution.

  The [symmetric alpha-stable
  distribution](https://en.wikipedia.org/wiki/Stable_distribution) contains the
  class of symmetric distributions which are closed under linear combinations.
  These distributions have recently been shown to accurately characterise the
  gradient noise distribution due to minibatch sampling in deep neural networks,
  see [(Simsekli et al., 2019)](https://arxiv.org/pdf/1901.06053.pdf).
  The relevance of this characterization is that many methods assume Gaussian
  tails of the noise distribution arising from the central limit theorem; for
  alpha < 2 these resulting average-minibatch-gradient noise tails are no longer
  Gaussian.

  This method estimates the inverse of the alpha tail index of the S-alpha-S
  distribution using the method proposed by [(Mohammadi et al.,
  2015)](https://link.springer.com/article/10.1007%2Fs00184-014-0515-7).

  The tail index alpha is in the range (0,2], where 2 corresponds to Gaussian
  tails.

  Args:
    data: Tensor, with zero-mean observations.  One designated axis corresponds
      to the sample dimension; for example data may be (100,16,8) composed of
      100 independent samples of (16,8) Tensors.  Note that the samples need to
      be independent and identically distributed (iid).
    axis: int, axis that corresponds to the sampling dimension.
    nelem_per_piece: int, how many elements to group to carry out estimation.
      A recommended value is around sqrt(data.shape[axis]).

  Returns:
    invstability_estimate: Tensor, shape is the shape of data with the sampling
      axis removed.  Each element of the Tensor contains an estimate of the
      inverse of the alpha stability coefficient of the symmetric alpha stable
      distribution.
  """
  n = data.shape[axis]
  num_pieces = n // nelem_per_piece
  # How many samples to use for estimation (discarding remainder)
  nestimate = num_pieces * nelem_per_piece
  data_samples, _ = tf.split(data, [nestimate, n-nestimate], axis=axis)
  term_all = tf.reduce_mean(tf.log(tf.abs(data_samples)), axis=axis)
  data_splits = tf.split(data_samples, num_pieces, axis=axis)
  term_batch = tf.reduce_mean(tf.stack(
      [tf.log(tf.abs(tf.reduce_sum(data_i, axis=axis)))
       for data_i in data_splits]), axis=0)
  invstability_estimate = (term_batch - term_all) / math.log(nelem_per_piece)

  return invstability_estimate


def _filter_gradient_tensors(gradients):
  """Filter a list of gradients and remove all tf.IndexedSlices instances."""
  return list(filter(
      lambda tensor: not isinstance(tensor, tf.IndexedSlices),
      gradients))


def variable_gradient_stability_estimate(model, tape, losses, batchsize,
                                         nelem_per_piece=8,
                                         aggregate_variable_estimates=True):
  """Estimate the symmetric alpha-stable tail index of gradient noise.

  We construct the estimate based on a model and gradient tape and a vector of
  per-instance losses.  The set of losses is grouped into batches and we
  compute per-batch gradients.  The total gradient is used to center the
  per-batch gradients, resulting in a set of independent gradient noise
  samples.  These zero-mean gradient noise samples form the input to a tail
  index estimator.

  Args:
    model: tf.keras.Model.
    tape: tf.GradientTape(persistent=True) that has been used to compute losses.
    losses: Tensor of shape (n,), one loss element per instance.
    batchsize: int, the number of instances per batch.
    nelem_per_piece: int, number of elements to group per block in the tail
      index estimator.  Ideally this is around sqrt(n//batchsize).
    aggregate_variable_estimates: bool, if True all estimates in a tf.Variable
      are mean-reduced.  If False individual estimates for each parameter are
      computed.

  Returns:
    stability_estimate: list of tf.Tensor objects containing the estimates of
    the tail index (stability == alpha).
  """
  n = int(tf.size(losses))  # number of instances
  with tape:
    loss_total = tf.reduce_mean(losses)
    losses_batched = tf.split(losses, n // batchsize)
    loss_batches = list(map(tf.reduce_mean, losses_batched))

  gradients_total = tape.gradient(loss_total, model.trainable_variables)
  gradients_total = _filter_gradient_tensors(gradients_total)
  gradients_batches = list(map(
      lambda loss_i: tape.gradient(loss_i, model.trainable_variables),
      loss_batches))
  gradients_batches = list(map(_filter_gradient_tensors, gradients_batches))

  gradients_noise = list(map(
      lambda gradients_batch_j: list(map(  # pylint: disable=g-long-lambda
          lambda grads: grads[1] - grads[0],
          zip(gradients_total, gradients_batch_j))),
      gradients_batches))

  noises = list(map(tf.stack, zip(*gradients_noise)))
  sample_axis = 0
  invalphas_estimate = list(map(
      lambda noise: symmetric_alpha_stable_invstability_estimator(  # pylint: disable=g-long-lambda
          noise, sample_axis, nelem_per_piece),
      noises))

  if aggregate_variable_estimates:
    stability_estimate = list(map(
        lambda invalpha: 1.0 / tf.reduce_mean(invalpha),
        invalphas_estimate))
  else:
    stability_estimate = list(map(
        lambda invalpha: 1.0 / invalpha, invalphas_estimate))

  return stability_estimate


class GradientNoiseEstimator(tf.keras.optimizers.Optimizer):
  """Optimizer class that can estimate gradient noise."""

  def __init__(self,
               name='GradientNoiseEstimator',
               preconditioner_regularization=1.0e-7,
               **kwargs):
    """Create a new gradient noise estimator object.

    Args:
      name: Optimizer name.
      preconditioner_regularization: float, >= 0.0, the estimated noise variance
        used to estimate the mass matrix in the estimate_fixed_preconditioner
        method will be regularized with this additive constant.
      **kwargs: arguments passed the tf.keras.optimizers.Optimizer base class.
    """
    super(GradientNoiseEstimator, self).__init__(name, **kwargs)
    self.preconditioner_regularization = preconditioner_regularization

  def gradient_noise_variance_estimate(self, var):
    """Estimate the gradient noise variance using a sample variance estimate.

    Args:
      var: tf.Variable to estimate the noise variance.

    Returns:
      variance_estimate: tf.Tensor of the same shape as 'var', containing a
        sample variance estimate of the gradient noise.  The resulting
        estimate is unregularized.
    """
    count = self.get_slot(var, 'count')
    m2 = self.get_slot(var, 'm2')
    variance_estimate = m2 / (count-1.0)

    return variance_estimate

  def gradient_second_moment_estimate(self, var):
    """Estimate the raw second moment of the gradient.

    We have E[G^2] = (E[G])^2 + Var[G].  Here the variance is over the
    minibatch sampling.

    Args:
      var: tf.Variable to estimate the second moment of.

    Returns:
      m2_estimate: tf.Tensor of the same shape as 'var', containing a raw second
          moment estimate of the gradient.  The resulting estimate is
          unregularized.
    """
    count = self.get_slot(var, 'count')
    mean = self.get_slot(var, 'mean')
    m2 = self.get_slot(var, 'm2')
    variance_estimate = m2 / count
    m2_estimate = tf.square(mean) + variance_estimate
    return m2_estimate

  def estimate_fixed_preconditioner(self, model, scale_to_min=True,
                                    raw_second_moment=False):
    """Produce a preconditioner dictionary suitable for SGMCMCOptimizer.

    Example:
      The following example estimates the gradient noise and then instantiates a
      SG-MCMC method using the estimated preconditioner.

      >>> grad_est = bnn.diagnostics.GradientNoiseEstimator()
      >>> @tf.function
          def train_gest_step(optimizer, model, data, labels):
            with tf.GradientTape(persistent=True) as tape:
              logits = model(data, training=True)
              ce_full = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits=logits, labels=labels)

              prior = sum(model.losses)
              loss = tf.reduce_mean(ce_full)
              obj = loss + prior

            gradients = tape.gradient(obj, model.trainable_variables)
            gradients = map(tf.convert_to_tensor, gradients)  # densify
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      >>> for batch in range(100):  # use 100 minibatch gradients to estimate
            data, labels = next(train_iter)
            train_gest_step(grad_est, model, data, labels)
      >>> precond_dict = grad_est.estimate_fixed_preconditioner(model)
      >>> optimizer = sgmcmc.BAOABMCMC(total_sample_size=50000,
                                       preconditioner='fixed',
                                       preconditioner_Mdict=precond_dict)

    Args:
      model: tf.keras.Model that the gradient noise was estimated for.
      scale_to_min: bool, if True then the resulting preconditioner is scaled
        such that the least sensitive variable has unit one, and the most
        sensitive variable has a mass higher than one.  Recommended.
      raw_second_moment: bool, if True then we estimate the raw second moment,
        akin to RMSprop.  If False we only estimate the gradient noise variance.

    Returns:
      precond_dict: dict, suitable as preconditioner_Mdict argument to the
        SGMCMCOptimizer base class.
    """
    def estimate_mass(var):
      """Estimate preconditioner mass matrix element for given variable."""
      if raw_second_moment:
        # Raw second moment (RMSprop)
        moment_estimate = self.gradient_second_moment_estimate(var)
      else:
        # Central second moment
        moment_estimate = self.gradient_noise_variance_estimate(var)

      mean_variance = tf.reduce_mean(moment_estimate)
      mean_variance_reg = mean_variance + self.preconditioner_regularization
      mass_estimate = float(tf.sqrt(mean_variance_reg))
      return mass_estimate

    precond_dict = {
        var.name: estimate_mass(var) for var in model.trainable_variables
    }

    # Scale so that smallest mass becomes one
    if scale_to_min:
      minimum_mass = min(precond_dict[name] for name in precond_dict)
      for name in precond_dict:
        precond_dict[name] /= minimum_mass

    return precond_dict

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'count', initializer='zeros')
      self.add_slot(var, 'mean', initializer='zeros')
      self.add_slot(var, 'm2', initializer='zeros')

  def _resource_apply_dense(self, grad, var):
    # Welford's streaming variance estimation update
    count = self.get_slot(var, 'count')
    mean = self.get_slot(var, 'mean')
    m2 = self.get_slot(var, 'm2')

    count_updated = count + 1.0
    delta = grad - mean
    mean_updated = mean + (delta/count_updated)
    delta2 = grad - mean_updated
    m2_updated = m2 + delta*delta2
    return tf.group(*([mean.assign(mean_updated),
                       m2.assign(m2_updated),
                       count.assign(count_updated)]))

  def get_config(self):
    config = super(GradientNoiseEstimator, self).get_config()
    config.update({
        'preconditioner_regularization': self.preconditioner_regularization,
    })
    return config


class AutoCorrelationEstimator(object):
  """Coarse-graining running estimation of autocorrelation.

  This class implements a hierarchical approximation (coarse-graining) scheme
  for autocorrelation estimation suitable for estimating the effective sample
  size and other autocorrelation statistics on Markov chain Monte Carlo (MCMC)
  output.

  The procedure is based on a procedure developed in the molecular dynamics
  community, in particular the so-called _order-n_ algorithm described in
  Section 4.4.2 of [1].

  The _order-n_ algorithm is a coarse-graining approximation which retains a
  fine estimation for small lags and an iteratively coarsened approximation for
  larger lags.

  There are two parameters defining the approximation:
    1. The `nlevels` parameter, specifying the depth of the temporal hierarchy.
    2. The `nsteps_per_level` parameter, specifying the number of time steps
       explicitly maintained by each level of the hierarchy.

  The overall memory complexity is `O(nsteps_per_level * nlevels)`.
  For each call to `update` the time complexity is `O(nsteps_per_level)`.
  The approximation is accurate for smooth autocorrelation functions.

  ### References
  [1] [(Frenkel and Smit, "Understanding Molecular Simulation: From Algorithms
      to Applications",
      1996)](https://www.sciencedirect.com/book/9780122673511/understanding-molecular-simulation).
  """

  def __init__(self, shape, nlevels=3, nsteps_per_level=32, dtype=tf.float32):
    """Create a new autocorrelation estimator.

    The estimator estimates the autocorrelation at lags between zero and
    total_time(), for a Tensor of statistics.  Each Tensor element is treated
    independent and the autocorrelation is estimated separately for all of them.

    Args:
      shape: tuple or TensorShape, the shape of the statistics passed to
        `update`.  Typically these would be statistics of a validation batch,
        e.g. a log-likelihood Tensor or a prediction statistic, e.g. the logits
        of a fixed validation batch.  It can also be the shape of a parameter
        tensor.
        The autocorrelation estimates obtained from `__call__` will be of the
        same size as the given `shape`.
      nlevels: int, >=1, the number of levels in the approximation hierarchy.
      nsteps_per_level: int, >=2, the number of explicit statistics in each
        level.
      dtype: tf.dtype, the element type of the statistics passed to `update`.
    """
    self.nlevels = nlevels
    self.nsteps_per_level = nsteps_per_level
    self.shape = shape

    # Marginal
    self.count = tf.convert_to_tensor(0, dtype=tf.int64)
    self.mean = tf.zeros(shape, dtype=dtype)
    self.moment2 = tf.zeros(shape, dtype=dtype)

    # Hierarchical raw and correlation statistics:
    #   1. corr[level][step] is Tensor with given shape, storing correlations.
    #   2. corr_count[level][step] counts the number of updates.
    #   3. stat[level][step] is Tensor with given shape, storing raw statistics.
    self.corr = []
    for _ in range(nlevels):
      self.corr.append([tf.zeros(shape, dtype=dtype)
                        for _ in range(nsteps_per_level)])

    self.corr_count = []
    for _ in range(nlevels):
      self.corr_count.append(
          [tf.convert_to_tensor(0, dtype=tf.int64)
           for _ in range(nsteps_per_level)])

    self.stat = []
    for _ in range(nlevels):
      self.stat.append(
          [tf.zeros(shape, dtype=dtype) for _ in range(nsteps_per_level)])

  def lag(self, level, step):
    """Return the time lag maintained at a particular position in the hierarchy.

    Args:
      level: int, >= 0, < nlevels, the level of the hierarchy.
      step: int, >= 0, < nsteps_per_level, the step within the level.

    Returns:
      lag: int, >= 1, <= total_time(), the lag.

    Raises:
      ValueError: level or step values outside the correct range.
    """
    if level < 0 or level >= self.nlevels:
      raise ValueError('level must be >= 0 and < nlevels.')
    if step < 0 or step >= self.nsteps_per_level:
      raise ValueError('step must be >= 0 and < nsteps_per_level')

    lag = (step+1)*(self.nsteps_per_level**level)
    return lag

  def _lag_to_level_and_step(self, time_lag):
    """Convert a lag to the rounded level and step in the hierarchy.

    Args:
      time_lag: int, >= 1, <= total_time(), the autocorrelation time lag.

    Returns:
      level: int, >= 0, < nlevels, the level in the hierarchy.
      step: int, >= 0, < nsteps_per_level, the step in the level.

    Raises:
      RuntimeError: level and step cannot be determined.
      ValueError: lag value outside the correct range.
    """
    if time_lag >= self.total_time():
      raise ValueError('Lag %d is outside valid range [1,%d].' % (
          time_lag, self.total_time()))
    if time_lag < 1:
      raise ValueError('Lag %d must be >= 1.' % time_lag)

    for level in range(self.nlevels):
      step_shift = self.nsteps_per_level**level
      step = time_lag // step_shift - 1
      if step < self.nsteps_per_level:
        return level, step

    raise RuntimeError('lag_to_level reached impossible state.')

  def _level_and_step_next(self, level, step):
    """Given a level and step, return the temporally next position.

    Args:
      level: int, >= 0, < nlevels, the level in the hierarchy.
      step: int, >= 0, < nsteps_per_level, the step in the level.

    Returns:
      level: int, >= level, < nlevels, the level in the hierarchy.
      step: int, >= 0, < nsteps_per_level, the step in the level.

    Raises:
      RuntimeError: level and step advanced beyond end of hierarchy.
      ValueError: level or step values outside the correct range.
    """
    if level < 0 or level >= self.nlevels:
      raise ValueError('level must be >= 0 and < nlevels.')
    if step < 0 or step >= self.nsteps_per_level:
      raise ValueError('step must be >= 0 and < nsteps_per_level')

    if step < (self.nsteps_per_level-1):
      return level, step+1

    level += 1
    step = 2
    if level >= self.nlevels:
      raise RuntimeError('_lag_and_step_next called after end')

    return level, step

  def _autocorr_level_step(self, level, step):
    """Return the autocorrelation at an approximation point.

    Args:
      level: int, >= 0, < nlevels, the level in the hierarchy.
      step: int, >= 0, < nsteps_per_level, the step in the level.

    Returns:
      acorr: Tensor, same shape and dtype as statistics in `update`.

    Raises:
      ValueError: level or step values outside the correct range.
    """
    if level < 0 or level >= self.nlevels:
      raise ValueError('level must be >= 0 and < nlevels.')
    if step < 0 or step >= self.nsteps_per_level:
      raise ValueError('step must be >= 0 and < nsteps_per_level')

    acorr = (self.corr[level][step] - self.mean**2.0) / self.variance()
    return acorr

  def _autocorr(self, time_lag):
    """Return the autocorrelation at a time_lag.

    Args:
      time_lag: int, >= 0, <= total_time().

    Returns:
      acorr: Tensor, shape as statistic passed to `update`, the autocorrelation
        at the approximation point at or before the given `time_lag`.

    Raises:
      ValueError: time_lag value outside the correct range.
    """
    if time_lag < 0 or time_lag > self.total_time():
      raise ValueError('time_lag must be >= 0 and <= total_time().')

    if time_lag == 0:
      return tf.ones_like(self.stat[0][0])

    level, step = self._lag_to_level_and_step(time_lag)
    return self._autocorr_level_step(level, step)

  def __call__(self, time_lag):
    """Return the interpolated autocorrelation estimate at lag `time_lag`.

    Args:
      time_lag: int, >= 0, <= total_time().

    Returns:
      acorr: Tensor, shape as statistic passed to `update`, the inteprolated
        autocorrelation estimate at the `time_lag`.

    Raises:
      ValueError: time_lag value outside the correct range.
    """
    if time_lag < 0 or time_lag > self.total_time():
      raise ValueError('time_lag must be >= 0 and <= total_time().')

    if time_lag == 0:
      return tf.ones_like(self.stat[0][0])

    level1, step1 = self._lag_to_level_and_step(time_lag)
    acorr1 = self._autocorr_level_step(level1, step1)
    level2, step2 = self._level_and_step_next(level1, step1)
    acorr2 = self._autocorr_level_step(level2, step2)

    t1 = self.lag(level1, step1)
    t2 = self.lag(level2, step2)
    if t1 == t2:
      return acorr1   # the most accurate estimate

    assert time_lag >= t1 and time_lag <= t2, 'time_lag out of bounds'

    # Linearly interpolate
    weight1 = float(t2 - time_lag) / float(t2 - t1)
    acorr = weight1*acorr1 + (1.0-weight1)*acorr2

    return acorr

  def total_time(self):
    """Return the largest lag for which we can estimate autocorrelation."""
    return self.lag(self.nlevels-1, self.nsteps_per_level-1)

  def variance(self):
    """Return an estimate of the marginal variance.

    Returns:
      var: Tensor, same shape as statistics passed to `update`.
        The estimated marginal variance (unbiased sample variance).
    """
    var = self.moment2 / (tf.cast(self.count, self.moment2.dtype)-1.0)
    return var

  def _update_stat(self, stat):
    """Update the marginal statistics.

    Args:
      stat: Tensor, same shape and dtype as the `shape` and `dtype` parameters
        in the call to the constructor.

    Raises:
      ValueError: stat.shape does not match self.shape.
    """
    # Check that the statistics shape matches
    if stat.shape != self.shape:
      raise ValueError('shape of statistic must match constructor shape.')

    # Online update, using Welford's algorithm for running mean and variance,
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    self.count += 1
    delta = stat - self.mean
    self.mean += delta / tf.cast(self.count, delta.dtype)
    delta2 = stat - self.mean
    self.moment2 += delta*delta2

  def _update_corr_level(self, stat, level):
    """Update correlation statistics at a given level.

    This method updated the autocorrelation estimates at the given `level`.
    For this it uses the raw statistics stored at the current level.

    Args:
      stat: Tensor, same shape and dtype as the `shape` and `dtype` parameters
        in the call to the constructor.
      level: int, >= 0, < nlevels, the level to update the correlation
        statistics.
    """
    assert level >= 0 and level < self.nlevels, 'level out of bounds'
    for step in range(self.nsteps_per_level):
      if self.count < self.lag(level, step):
        break

      self.corr_count[level][step] += 1
      delta = stat*self.stat[level][step] - self.corr[level][step]
      self.corr[level][step] += delta / tf.cast(
          self.corr_count[level][step], delta.dtype)

  def _update_stat_level(self, stat, level):
    """Update the raw statistics at a given level.

    This method renews the raw statistics at the given `level` so that the
    oldest statistics are discarded and `stat` is stored.

    Args:
      stat: Tensor, same shape and dtype as the `shape` and `dtype` parameters
        in the call to the constructor.
      level: int, >= 0, < nlevels, the level to update the correlation
        statistics.
    """
    assert level >= 0 and level < self.nlevels, 'level out of bounds'
    for step in range(self.nsteps_per_level-1, 0, -1):
      self.stat[level][step] = self.stat[level][step-1]
    self.stat[level][0] = stat

  def _is_hstep(self, count, level):
    """Decide whether the given `level` should be updated.

    Args:
      count: int, >= 0, global time step, i.e. the number of calls to `update`.
      level: int, >= 0, < nlevels, the level to decide about.

    Returns:
      do_update: true if layer should be updated, false otherwise.
    """
    assert level >= 0 and level < self.nlevels, 'level out of bounds'
    hstep = count % (self.nsteps_per_level**level)
    do_update = tf.equal(hstep, 0)
    return do_update

  def update(self, stat):
    """Update the autocorrelation estimates using the given statistics.

    Args:
      stat: Tensor, same shape and dtype as the `shape` and `dtype` parameters
        in the call to the constructor.
    """
    for level in range(self.nlevels):
      if self._is_hstep(self.count, level):
        self._update_corr_level(stat, level)
        self._update_stat_level(stat, level)

    self._update_stat(stat)

  def time_to_one_sample(self):
    """Estimate the time-to-one-sample (TT1).

    The time-to-one-sample (TT1) is related to the effective sample size: TT1
    measures the number of MCMC steps required to obtain one approximately
    independent sample.  The effective sample size (ESS) is related as `ESS = N
    / TT1`, where `N` is the number of iterations of the MCMC chain.

    Returns:
      time_to_one_sample: float, the number of iterations to obtain one
        approximately independent sample.
    """
    autocorr_sum = 0.5
    for level in range(self.nlevels):
      for step1 in range(self.nsteps_per_level-1):
        step2 = step1 + 1
        acorr1 = tf.reduce_mean(self._autocorr_level_step(level, step1))
        acorr2 = tf.reduce_mean(self._autocorr_level_step(level, step2))

        # Truncate computation if estimation uncertainty becomes too large.
        # To estimate this uncertainty we use the current ESS estimate (1/asum).
        # 2*autocorr_sum = 1/ess
        sigma1 = math.sqrt(2.0*autocorr_sum /
                           float(self.corr_count[level][step1]))
        sigma2 = math.sqrt(2.0*autocorr_sum /
                           float(self.corr_count[level][step2]))
        truncate = False
        if acorr1 <= sigma1 or acorr2 <= sigma2:
          # Reached too small estimation accuracy to continue
          truncate = True
          acorr1 = max(acorr1, 0.0)
          acorr2 = max(acorr2, 0.0)

        # Integrate area under step1-step2 segment
        lag1 = self.lag(level, step1)
        lag2 = self.lag(level, step2)
        delta = float(lag2 - lag1)
        autocorr_sum += 0.5*(acorr1 + acorr2)*delta

        if truncate:
          time_to_one_sample = 2.0*autocorr_sum
          return time_to_one_sample

    time_to_one_sample = 2.0*autocorr_sum
    return time_to_one_sample
