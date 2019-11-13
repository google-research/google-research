# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Base class for models."""
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def _safe_log(x, eps=1e-8):
  return tf.log(tf.clip_by_value(x, eps, 1.0))


def get_squash(squash_eps=1e-6):
  return tfp.bijectors.Chain([
      tfp.bijectors.AffineScalar(scale=256.),
      tfp.bijectors.AffineScalar(
          shift=-squash_eps / 2., scale=(1. + squash_eps)),
      tfp.bijectors.Sigmoid(),
  ])


class GSTBernoulli(tfd.Bernoulli):
  """Gumbel-softmax Bernoulli distribution."""

  def __init__(self,
               temperature,
               logits=None,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name="GSTBernoulli",
               dtype=tf.int32):
    """Construct GSTBernoulli distributions.

    Args:
      temperature: An 0-D `Tensor`, representing the temperature of a set of
        GSTBernoulli distributions. The temperature should be positive.
      logits: An N-D `Tensor` representing the log-odds of a positive event.
        Each entry in the `Tensor` parametrizes an independent GSTBernoulli
        distribution where the probability of an event is sigmoid(logits). Only
        one of `logits` or `probs` should be passed in.
      probs: An N-D `Tensor` representing the probability of a positive event.
        Each entry in the `Tensor` parameterizes an independent Bernoulli
        distribution. Only one of `logits` or `probs` should be passed in.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
      dtype: Type of the Tesnors.

    Raises:
      ValueError: If both `probs` and `logits` are passed, or if neither.
    """
    with tf.name_scope(name, values=[logits, probs, temperature]) as name:
      self._temperature = tf.convert_to_tensor(
          temperature, name="temperature", dtype=dtype)
      if validate_args:
        with tf.control_dependencies([tf.assert_positive(temperature)]):
          self._temperature = tf.identity(self._temperature)
      super(GSTBernoulli, self).__init__(
          logits=logits,
          probs=probs,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          dtype=dtype,
          name=name)

  @property
  def temperature(self):
    """Distribution parameter for the location."""
    return self._temperature

  def _sample_n(self, n, seed=None):
    new_shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    u = tf.random_uniform(new_shape, seed=seed, dtype=self.probs.dtype)
    logistic = _safe_log(u) - _safe_log(1 - u)
    hard_sample = tf.cast(tf.greater(self.logits + logistic, 0), self.dtype)
    soft_sample = tf.math.sigmoid((self.logits + logistic) / self.temperature)
    sample = soft_sample + tf.stop_gradient(hard_sample - soft_sample)
    return tf.cast(sample, self.dtype)


def mlp(inputs,
        layer_sizes,
        hidden_activation=tf.math.tanh,
        final_activation=tf.math.log_sigmoid,
        name=None):
  """Creates a simple fully connected multi-layer perceptron."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    inputs = tf.layers.flatten(inputs)
    for i, s in enumerate(layer_sizes[:-1]):
      inputs = tf.layers.dense(
          inputs,
          units=s,
          activation=hidden_activation,
          kernel_initializer=tf.initializers.glorot_uniform,
          name="layer_%d" % (i + 1))
    output = tf.layers.dense(
        inputs,
        units=layer_sizes[-1],
        activation=final_activation,
        kernel_initializer=tf.initializers.glorot_uniform,
        name="layer_%d" % len(layer_sizes))
  return output


def conditional_normal(inputs,
                       data_dim,
                       hidden_sizes,
                       hidden_activation=tf.math.tanh,
                       scale_min=1e-5,
                       truncate=False,
                       bias_init=None,
                       scale_init=1.,
                       nn_scale=True,
                       name=None):
  """Create a conditional Normal distribution."""
  flat_data_dim = np.prod(data_dim)
  if nn_scale:
    raw_params = mlp(
        inputs,
        hidden_sizes + [2 * flat_data_dim],
        hidden_activation=hidden_activation,
        final_activation=None,
        name=name)
    loc, raw_scale = tf.split(raw_params, 2, axis=-1)
  else:
    loc = mlp(
        inputs,
        hidden_sizes + [flat_data_dim],
        hidden_activation=hidden_activation,
        final_activation=None,
        name=name + "_loc")
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      raw_scale_init = np.log(np.exp(scale_init) - 1 + scale_min)
      raw_scale = tf.get_variable(
          name="raw_sigma",
          shape=[flat_data_dim],
          dtype=tf.float32,
          initializer=tf.constant_initializer(raw_scale_init),
          trainable=True)
  scale = tf.math.maximum(scale_min, tf.math.softplus(raw_scale))
  # Reshape back to the proper data_dim
  loc = tf.reshape(loc, [-1] + data_dim)
  scale = tf.reshape(scale, [-1] + data_dim)
  # with tf.name_scope(name):
  #   tf.summary.histogram("scale", scale, family="scales")
  #   tf.summary.scalar("min_scale", tf.reduce_min(scale), family="scales")
  if truncate:
    if bias_init is not None:
      loc = loc + bias_init
    loc = tf.math.sigmoid(loc)
    return tfd.Independent(
        tfd.TruncatedNormal(loc=loc, scale=scale, low=0., high=1.),
        reinterpreted_batch_ndims=len(data_dim))
  else:
    return tfd.Independent(tfd.Normal(loc=loc, scale=scale),
                           reinterpreted_batch_ndims=len(data_dim))


def conditional_bernoulli(inputs,
                          data_dim,
                          hidden_sizes,
                          hidden_activation=tf.math.tanh,
                          bias_init=None,
                          dtype=tf.int32,
                          use_gst=False,
                          temperature=None,
                          name=None):
  """Create a conditional Bernoulli distribution."""
  flat_data_dim = np.prod(data_dim)
  bern_logits = mlp(
      inputs,
      hidden_sizes + [flat_data_dim],
      hidden_activation=hidden_activation,
      final_activation=None,
      name=name)
  bern_logits = tf.reshape(bern_logits, [-1] + data_dim)

  if bias_init is not None:
    bern_logits = bern_logits - tf.log(
        1. / tf.clip_by_value(bias_init, 0.0001, 0.9999) - 1)

  if use_gst:
    assert temperature is not None
    base_dist = GSTBernoulli(temperature, logits=bern_logits, dtype=dtype)
  else:
    base_dist = tfd.Bernoulli(logits=bern_logits, dtype=dtype)
  return tfd.Independent(base_dist)


class SquashedDistribution(object):
  """Apply a squashing bijector to a distribution."""

  def __init__(self, distribution, data_mean, squash_eps=1e-6):
    self.distribution = distribution
    self.data_mean = data_mean
    self.squash = get_squash(squash_eps)
    self.unsquashed_data_mean = self.squash.inverse(self.data_mean)

  def log_prob(self, data, num_samples=1):
    unsquashed_data = (self.squash.inverse(data) - self.unsquashed_data_mean)
    log_prob = self.distribution.log_prob(unsquashed_data,
                                          num_samples=num_samples)
    log_prob = (log_prob + self.squash.inverse_log_det_jacobian(
        data, event_ndims=tf.rank(data) - 1))

    return log_prob

  def sample(self, num_samples=1):
    samples = self.distribution.sample(num_samples)
    samples += self.unsquashed_data_mean
    samples = self.squash.forward(samples)
    return samples


class ProbabilisticModel(object):
  """Abstract class for probablistic models to inherit."""

  def log_prob(self, data, num_samples=1):
    """Reshape data so that it is [batch_size] + data_dim."""
    batch_shape = tf.shape(data)[:-len(self.data_dim)]
    reshaped_data = tf.reshape(data, [tf.math.reduce_prod(batch_shape)] +
                               self.data_dim)
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_prob(self, data, num_samples=1):
    pass


def get_independent_normal(data_dim, variance=1.0):
  """Returns an independent normal with event size the size of data_dim.

  Args:
    data_dim: List of data dimensions.
    variance: A scalar that is used as the diagonal entries of the covariance matrix.

  Returns:
    Independent normal distribution.
  """
  return tfd.Independent(
      tfd.Normal(
          loc=tf.zeros(data_dim, dtype=tf.float32),
          scale=tf.ones(data_dim, dtype=tf.float32)*tf.math.sqrt(variance)),
      reinterpreted_batch_ndims=len(data_dim))
