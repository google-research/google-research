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

"""Implements tf.Module to keep track of running statistics."""

import abc
import gin
import tensorflow.compat.v2 as tf


class MeanStd(tf.Module, metaclass=abc.ABCMeta):
  """Abstract base class that keeps track of mean and standard deviation."""

  @abc.abstractmethod
  def init(self, size):
    """Initializes normalization variables.

    Args:
      size: Integer with the dimensionality of the tracked tensor.
    """
    raise NotImplementedError('`init` is not implemented.')

  def normalize(self, x):
    """Normalizes target values x using past target statistics.

    Args:
      x: <float32>[(...), size] tensor.

    Returns:
      <float32>[(...), size] normalized tensor.
    """
    mean, std = self.get_mean_std()
    return (x - mean) / std

  def unnormalize(self, x):
    """Unnormalizes a corrected prediction x using past target statistics.

    Args:
      x: <float32>[(...), size] tensor.

    Returns:
      <float32>[(...), size] unnormalized tensor.
    """
    mean, std = self.get_mean_std()
    return std * x + mean

  @abc.abstractmethod
  def update(self, data):
    """Updates normalization statistics.

    Args:
      data: <float32>[(...), size].
    """
    raise NotImplementedError('`update` is not implemented.')

  @abc.abstractmethod
  def get_mean_std(self):
    """Returns mean and standard deviation for current statistics."""
    raise NotImplementedError('`get_mean_std` is not implemented.')


@gin.configurable
class EMAMeanStd(MeanStd):
  """Tracks mean and standard deviation using an exponential moving average.

  This works by keeping track of the first and second non-centralized moments
  using an exponential average of the global batch means of these moments, i.e.,
      new_1st_moment = (1-beta)*old_1st_moment + beta*mean(data)
      new_2nd_moment = (1-beta)*old_2nd_moment + beta*mean(data**2).

  Initially, mean and standard deviation are set to zero and one respectively.
  """

  def __init__(self, beta=1e-2, std_min_value=1e-6, std_max_value=1e6):
    """Creates a EMAMeanVariance.

    Args:
      beta: Float that determines how fast parameters are updated via the
        formula `new_parameters = (1-beta)* old_parameters + beta*batch_mean`.
      std_min_value: Float with the minimum value for the standard deviation.
      std_max_value: Float with the maximum value for the standard deviation.
    """
    super().__init__()
    self._beta = beta
    self._std_min_value = std_min_value
    self._std_max_value = std_max_value
    self._first_moment = None
    self._second_moment = None

  def init(self, size):
    """Initializes normalization variables.

    Args:
      size: Integer with the dimensionality of the tracked tensor.
    """
    self._first_moment = tf.Variable(
        name='first_moment',
        shape=[size],
        trainable=False,
        dtype=tf.float32,
        initial_value=tf.zeros(shape=[size]),
        aggregation=tf.VariableAggregation.MEAN)
    self._second_moment = tf.Variable(
        name='second_moment',
        shape=[size],
        trainable=False,
        dtype=tf.float32,
        initial_value=tf.ones(shape=[size]),
        aggregation=tf.VariableAggregation.MEAN)

  @property
  def first_moment(self):
    if self._first_moment is None:
      raise ValueError('`init` was not called.')
    return self._first_moment

  @property
  def second_moment(self):
    if self._second_moment is None:
      raise ValueError('`init` was not called.')
    return self._second_moment

  def update(self, data):
    """Updates normalization statistics.

    Args:
      data: <float32>[(...), size].
    """
    # Reduce tensors along all the dimensions except the last ones.
    reduce_dims = list(range(data.shape.rank))[:-1]
    batch_first_moment = tf.reduce_mean(data, reduce_dims)
    batch_second_moment = tf.reduce_mean(data**2, reduce_dims)

    # Updates the tracked moments. We do this by computing the difference to the
    # the current value as that allows us to use mean aggregation to make it
    # work with replicated tensors (e.g., when using multiple TPU cores), i.e.,
    #     new_moment = old_moment + beta*mean(data - old_moment)
    # where the mean is a mean across different replica and within the
    # mini-batches of each replica.
    first_moment_diff = self._beta * (batch_first_moment - self.first_moment)
    second_moment_diff = self._beta * (batch_second_moment - self.second_moment)

    # The following two assign_adds will average their arguments across
    # different replicas as the underlying variables have
    # `aggregation=tf.VariableAggregation.MEAN` set.
    self.first_moment.assign_add(first_moment_diff)
    self.second_moment.assign_add(second_moment_diff)

  def get_mean_std(self):
    """Returns mean and standard deviation for current statistics."""
    std = tf.sqrt(self.second_moment - self.first_moment**2)
    std = tf.clip_by_value(std, self._std_min_value, self._std_max_value)
    # Multiplication with one converts the variable to a tensor with the value
    # at the time this function is called. This is important if the python
    # reference is passed around and the variables are changed in the meantime.
    return self.first_moment * 1., std


def merge_summed_variances(v1, v2, mu1, mu2, merged_mean, n1, n2):
  """Computes the (summed) variance of a combined series.

  Args:
    v1: summed variance of the first series.
    v2: summed variance of the second series.
    mu1: mean of the first series.
    mu2: mean of the second series.
    merged_mean: mean for the combined series.
    n1: Number of datapoints in the first series.
    n2: Number of datapoints in the second series.

  Returns:
    The summed variance for the combined series.
  """
  return (v1 + n1 * tf.square(mu1 - merged_mean) + v2 +
          n2 * tf.square(mu2 - merged_mean))


def merge_means(mu1, mu2, n1, n2):
  """Merges means. Requires n1 + n2 > 0."""
  total = n1 + n2
  return (n1 * mu1 + n2 * mu2) / total


@gin.configurable
class AverageMeanStd(MeanStd):
  """Tracks mean and standard deviation across all past samples.

  This works by updating the mean and the sum of past variances with Welford's
  algorithm using batches (see https://stackoverflow.com/questions/56402955/
  whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates).

  One limitation of this class is that it uses float32 to aggregate statistics,
  which leads to inaccuracies after 7M batch due to limited float precision (see
  b/160686691 for details). Use TwoLevelAverageMeanStd to work around that.

  Attributes:
    observation_count: float32 tf.Variable with observation counts.
    update_count: int32 tf.Variable representing the number of times update() or
      merge() have been called.
    mean: float32 tf.Variable with mean.
    summed_variance: float32 tf.Variable with summed variance of all samples.
  """

  def __init__(self, std_min_value=1e-6, std_max_value=1e6):
    """Creates a AverageMeanStd.

    Args:
      std_min_value: Float with the minimum value for the standard deviation.
      std_max_value: Float with the maximum value for the standard deviation.
    """
    super().__init__()
    self._std_min_value = std_min_value
    self._std_max_value = std_max_value
    self._observation_count = None
    self._update_count = None
    self._mean = None
    self._summed_variance = None

  def init(self, size):
    """Initializes normalization variables.

    Args:
      size: Integer with the dimensionality of the tracked tensor.
    """
    self._observation_count = tf.Variable(
        name='observation_count',
        shape=[size],
        trainable=False,
        dtype=tf.float32,
        initial_value=tf.zeros(shape=[size], dtype=tf.float32),
        aggregation=tf.VariableAggregation.SUM)
    self._update_count = tf.Variable(
        name='update_count',
        shape=[],
        trainable=False,
        dtype=tf.int32,
        initial_value=tf.zeros(shape=[], dtype=tf.int32),
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    self._mean = tf.Variable(
        name='mean',
        shape=[size],
        trainable=False,
        dtype=tf.float32,
        initial_value=tf.zeros(shape=[size]),
        aggregation=tf.VariableAggregation.SUM)
    self._summed_variance = tf.Variable(
        name='summed_variance',
        shape=[size],
        trainable=False,
        dtype=tf.float32,
        initial_value=tf.zeros(shape=[size]),
        aggregation=tf.VariableAggregation.SUM)

  @property
  def observation_count(self):
    if self._observation_count is None:
      raise ValueError('`init` was not called.')
    return self._observation_count

  @property
  def update_count(self):
    if self._update_count is None:
      raise ValueError('`init` was not called.')
    return self._update_count

  @property
  def mean(self):
    if self._mean is None:
      raise ValueError('`init` was not called.')
    return self._mean

  @property
  def summed_variance(self):
    if self._summed_variance is None:
      raise ValueError('`init` was not called.')
    return self._summed_variance

  def merge(self, other, alpha=1.):
    """Merges with the stats from another AverageMeanStd, in-place.

    Args:
      other: An AverageMeanStd to merge into self.
      alpha: Performs a reset when alpha=1, no-op when alpha=0. We need this to
        work-around tensorflow limitations regarding mixing synchronization
        points and conditions.
    """
    ctx = tf.distribute.get_replica_context()
    num_replicas = tf.constant(ctx.num_replicas_in_sync, tf.float32)
    new_mean = merge_means(self.mean, other.mean, self.observation_count,
                           other.observation_count)
    new_summed_variance = merge_summed_variances(
        self.summed_variance, other.summed_variance, self.mean, other.mean,
        new_mean, self.observation_count, other.observation_count)
    # We need to divide by the number of replicas since those variables are
    # aggregated with cross-replica sums.
    self.mean.assign(
        (alpha * new_mean + (1. - alpha) * self.mean) / num_replicas)
    self.observation_count.assign_add(alpha * other.observation_count /
                                      num_replicas)
    self.summed_variance.assign(
        (alpha * new_summed_variance +
         (1. - alpha) * self.summed_variance) / num_replicas)
    self.update_count.assign_add(1)

  def reset(self, alpha=1.):
    """Resets the aggregator.

    Args:
      alpha: Performs a reset when alpha=1, no-op when alpha=0. We need this to
        work-around tensorflow limitations regarding mixing synchronization
        points and conditions.
    """
    ctx = tf.distribute.get_replica_context()
    num_replicas = tf.constant(ctx.num_replicas_in_sync, tf.float32)
    # We need to divide by the number of replicas since those variables are
    # aggregated with cross-replica sums.
    self.mean.assign((1. - alpha) * self.mean / num_replicas)
    self.observation_count.assign(
        (1. - alpha) * self.observation_count / num_replicas)
    self.summed_variance.assign(
        (1. - alpha) * self.summed_variance / num_replicas)
    # Remember, ONLY_FIRST_REPLICA aggregation.
    self.update_count.assign(
        tf.cast((1. - alpha) * tf.cast(self.update_count, tf.float32),
                tf.int32))

  def update(self, data):
    """Updates normalization statistics.

    Args:
      data: <float32>[(...), size].
    """
    # Reduce tensors along all the dimensions except the last ones.
    reduce_dims = list(range(data.shape.rank))[:-1]

    # Update the observations counts.
    count = tf.ones_like(data, dtype=tf.int32)
    aggregated_count = tf.reduce_sum(count, reduce_dims)
    # SUM across replicas.
    self.observation_count.assign_add(tf.cast(aggregated_count, tf.float32))
    self.update_count.assign_add(1)
    # Update the mean.
    diff_to_old_mean = data - self.mean
    mean_update = tf.reduce_sum(diff_to_old_mean, reduce_dims)
    mean_update /= tf.cast(self.observation_count, dtype=tf.float32)
    self.mean.assign_add(mean_update)

    # Update the variance.
    diff_to_new_mean = data - self.mean
    variance_update = diff_to_old_mean * diff_to_new_mean
    variance_update = tf.reduce_sum(variance_update, reduce_dims)
    self.summed_variance.assign_add(variance_update)

  def get_mean_std(self):
    """Returns mean and standard deviation for current statistics."""
    # The following clipping guarantees an initial variance of one.
    minval = self._std_min_value * self._std_min_value
    eff_var = tf.maximum(minval, self.summed_variance)
    eff_count = tf.cast(self.observation_count, dtype=tf.float32)
    eff_count = tf.maximum(minval, eff_count)
    std = tf.sqrt(eff_var / eff_count)
    std = tf.clip_by_value(std, self._std_min_value, self._std_max_value)
    # Multiplication with one converts the variable to a tensor with the value
    # at the time this function is called. This is important if the python
    # reference is passed around and the variables are changed in the meantime.
    return self.mean * 1., std


@gin.configurable
class FixedMeanStd(MeanStd):
  """Instance where the mean and standard deviation are fixed."""

  def __init__(self, mean=0., std=1.):
    """Creates a FixedMeanStd.

    Args:
      mean: Float with the fixed mean.
      std: Float with the fixed standard deviation.
    """
    super().__init__()
    self._mean = mean
    self._std = std
    self._size = None

  def init(self, size):
    """Initializes normalization variables.

    Args:
      size: Integer with the dimensionality of the tracked tensor.
    """
    self._size = size

  def update(self, data):
    """Updates normalization statistics.

    Args:
      data: <float32>[(...), size].
    """
    pass

  def get_mean_std(self):
    """Returns mean and standard deviation for current statistics."""
    vec = tf.ones((self._size,), dtype=tf.float32)
    mean = tf.convert_to_tensor(self._mean, dtype=tf.float32)
    std = tf.convert_to_tensor(self._std, dtype=tf.float32)
    return mean*vec, std*vec


@gin.configurable
class TwoLevelAverageMeanStd(MeanStd):
  """Drop-in replacement of AverageMeanStd without precision issues.

  AverageMeanStd uses float32s. This leads to precision issues when we update
  running statistics more than ~7e6 times.Unfortunately, we cannot switch to
  float64 due to limitation of variables with aggregation.

  TwoLevelAverageMeanStd works around the precision issues in AverageMeanStd by
  using two of them in a hierarchical fashion. The lower AverageMeanStd is used
  as a "buffer" up to N updates (typically N=1e5). When the buffer has reached
  the N updates, we reset it and update the upper AverageMeanStd with the
  summarized stats from the buffer.
  With this approach, the effective number of mantissa bits we get is around 40,
  vs 24 for float32 and 52 for float64.

  When retrieving the overal mean and std, we just have to combine the stats
  from the two AverageMeanStd levels.
  """

  def __init__(self, std_min_value=1e-6, std_max_value=1e6, buffer_size=1e5):
    """Init.

    Args:
      std_min_value: Clip the returned std by this lower bound.
      std_max_value: Clip the returned std by this upper bound.
      buffer_size: Number of updates to perform on the buffer AverageMeanStd
        before it's reset. AverageMeanStd uses float32s with ~24bit precision,
        so a default of 1e5 makes sense. It leaves about 7bits of precision for
        each variance added to summed_variance.
    """
    super().__init__()
    self.average_mean_std = AverageMeanStd(0., float('inf'))
    self.average_mean_std_buffer = AverageMeanStd(0., float('inf'))
    self._std_min_value = std_min_value
    self._std_max_value = std_max_value
    self.buffer_size = int(buffer_size)

  def init(self, size):
    """Initializes normalization variables.

    Args:
      size: (integer) size of the tracked tensor.
    """
    self.average_mean_std.init(size)
    self.average_mean_std_buffer.init(size)

  def update(self, data):
    """Updates normalization statistics.

    Args:
      data: <float32>[(...), size].
    """
    self.average_mean_std_buffer.update(data)
    reset_buffer = tf.cast(
        tf.math.greater_equal(self.average_mean_std_buffer.update_count,
                              self.buffer_size), tf.float32)
    self.average_mean_std.merge(self.average_mean_std_buffer, reset_buffer)
    self.average_mean_std_buffer.reset(reset_buffer)

  @tf.function
  def get_mean_std(self):
    """Returns mean and standard deviation for current statistics."""
    mean = self.average_mean_std.mean
    count = self.average_mean_std.observation_count
    mean_buffer = self.average_mean_std_buffer.mean
    count_buffer = self.average_mean_std_buffer.observation_count
    total_count = count + count_buffer
    # All elements of 'total_count' have the same value.
    if tf.reduce_any(tf.equal(total_count, 0.)):
      merged_mean = tf.zeros_like(mean)
      merged_std = tf.ones_like(mean)
    else:
      merged_mean = merge_means(mean, mean_buffer, count, count_buffer)
      merged_summed_variance = merge_summed_variances(
          self.average_mean_std.summed_variance,
          self.average_mean_std_buffer.summed_variance, mean, mean_buffer,
          merged_mean, count, count_buffer)
      # Due to precision issues, merged_summed_variance can be slightly
      # negative.
      merged_summed_variance = tf.maximum(0., merged_summed_variance)
      merged_std = tf.sqrt(merged_summed_variance / total_count)
    clipped_std = tf.clip_by_value(merged_std, self._std_min_value,
                                   self._std_max_value)
    return merged_mean, clipped_std
