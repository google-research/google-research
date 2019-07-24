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

"""VAE base class."""

from __future__ import absolute_import
import functools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from eim.models import base

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class VAE(object):
  """Variational autoencoder with continuous latent space."""

  def __init__(self,
               latent_dim,
               data_dim,
               decoder,
               q,
               prior=None,
               data_mean=None,
               kl_weight=1.,
               dtype=tf.float32):
    """Creates a VAE.

    Args:
      latent_dim: The size of the latent variable of the VAE.
      data_dim: The size of the input data.
      decoder: A callable that accepts a batch of latent samples and returns a
        distribution over the data space of the VAE. The distribution should
        support sample() and log_prob().
      q: A callable that accepts a batch of data samples and returns a
        distribution over the latent space of the VAE. The distribution should
        support sample() and log_prob().
      prior: A distribution over the latent space of the VAE. The object must
        support sample() and log_prob(). If not provided, defaults to Gaussian.
      data_mean: Mean of the data used to center the input.
      kl_weight: Weighting on the KL regularizer.
      dtype: Type of the tensors.
    """
    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    self.decoder = decoder
    self.q = q
    self.kl_weight = kl_weight

    self.dtype = dtype
    if prior is None:
      self.prior = tfd.MultivariateNormalDiag(
          loc=tf.zeros([latent_dim], dtype=dtype),
          scale_diag=tf.ones([latent_dim], dtype=dtype))
    else:
      self.prior = prior

  def log_prob(self, data, num_samples=1):
    batch_shape = tf.shape(data)[0:-1]
    reshaped_data = tf.reshape(
        data, [tf.math.reduce_prod(batch_shape), self.data_dim])
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_prob(self, data, num_samples=1):
    """Compute a lower bound on the log likelihood."""
    mean_centered_data = data - self.data_mean

    # Construct approximate posterior and sample z.
    q_z = self.q(mean_centered_data)
    z = q_z.sample(sample_shape=[num_samples
                                ])  # [num_samples, batch_size, data_dim]
    log_q_z = q_z.log_prob(z)  # [num_samples, batch_size]

    # compute the prior prob of z, #[num_samples, batch_size]
    #    # Try giving the proposal lower bound extra compute if it can use it.
    #    try:
    #      log_p_z = self.prior.log_prob(z, num_samples=num_samples)
    #    except TypeError:
    log_p_z = self.prior.log_prob(z)

    # Compute the model logprob of the data
    p_x_given_z = self.decoder(z)
    log_p_x_given_z = p_x_given_z.log_prob(data)  # [num_samples, batch_size]

    elbo = (
        tf.reduce_logsumexp(
            log_p_x_given_z + self.kl_weight * (log_p_z - log_q_z), axis=0) -
        tf.log(tf.to_float(num_samples)))
    return elbo

  def sample(self, sample_shape=(1)):
    z = self.prior.sample(sample_shape)
    p_x_given_z = self.decoder(z)
    return tf.cast(p_x_given_z.sample(), self.dtype)


class GaussianVAE(VAE):
  """VAE with Gaussian generative distribution."""

  def __init__(self,
               latent_dim,
               data_dim,
               decoder_hidden_sizes,
               q_hidden_sizes,
               prior=None,
               data_mean=None,
               decoder_nn_scale=True,
               scale_min=1e-5,
               dtype=tf.float32,
               kl_weight=1.,
               name="gaussian_vae"):
    # Make the decoder with a Gaussian distribution
    decoder_fn = functools.partial(
        base.conditional_normal,
        data_dim=data_dim,
        hidden_sizes=decoder_hidden_sizes,
        scale_min=scale_min,
        nn_scale=decoder_nn_scale,
        bias_init=data_mean,
        truncate=False,
        name="%s/decoder" % name)
    q = functools.partial(
        base.conditional_normal,
        data_dim=latent_dim,
        hidden_sizes=q_hidden_sizes,
        scale_min=scale_min,
        name="%s/q" % name)

    super(GaussianVAE, self).__init__(
        latent_dim=latent_dim,
        data_dim=data_dim,
        decoder=decoder_fn,
        q=q,
        data_mean=data_mean,
        prior=prior,
        kl_weight=kl_weight,
        dtype=dtype)


class ConvGaussianVAE(VAE):
  """ConvVAE with Gaussian generative distribution."""

  def __init__(self,
               latent_dim,
               data_dim,
               base_depth,
               activation=tf.nn.leaky_relu,
               prior=None,
               data_mean=None,
               scale_min=1e-5,
               dtype=tf.float32,
               kl_weight=1.,
               scale_init=1.,
               name="gaussian_vae"):
    """Creates the ConvGaussianVAE.

    Args:
      latent_dim: The size of the latent variable of the VAE.
      data_dim: The size of the input data.
      base_depth: Base depth for conv layers.
      activation: Activation function in hidden layers.
      prior: A distribution over the latent space of the VAE. The object must
        support sample() and log_prob(). If not provided, defaults to Gaussian.
      data_mean: Mean of the data used to center the input.
      scale_min: Min on the scale of the Gaussian.
      dtype: Type of the tensors.
      kl_weight: Weighting on the KL regularizer.
      scale_init: Initial value for the scale of the Gaussian.
      name: Name to use for ops.
    """
    if data_mean is None:
      data_mean = 0.
    deconv = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      raw_scale_init = np.log(np.exp(scale_init) - 1 + scale_min)
      raw_scale = tf.get_variable(
          name="raw_sigma",
          shape=data_dim,
          dtype=tf.float32,
          initializer=tf.constant_initializer(raw_scale_init),
          trainable=True)
    decoder_scale = tf.math.maximum(scale_min, tf.math.softplus(raw_scale))
    tf.summary.histogram("decoder_scale", decoder_scale)

    decoder_fn = tf.keras.Sequential([
        tfkl.Lambda(lambda x: x[:, None, None, :]),
        deconv(4 * base_depth, 5, 2),
        deconv(4 * base_depth, 5, 2),
        deconv(2 * base_depth, 5, 2),
        deconv(2 * base_depth, 5, 2),
        deconv(base_depth, 5, 2),
        deconv(base_depth, 5, 2),
        conv(data_dim[-1], 5, activation=None),
        tfpl.DistributionLambda(lambda t: tfd.Independent(  # pylint: disable=g-long-lambda
            tfd.Normal(loc=t, scale=decoder_scale[None]))),
    ])
    encoding_layer = tfpl.IndependentNormal
    encoder_fn = tf.keras.Sequential([
        conv(base_depth, 5, 2),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 2),
        conv(2 * base_depth, 5, 2),
        conv(4 * base_depth, 5, 2),
        conv(4 * latent_dim, 5, 2),
        tfkl.Flatten(),
        tfkl.Dense(encoding_layer.params_size(latent_dim), activation=None),
        encoding_layer(latent_dim),
    ])

    super(ConvGaussianVAE, self).__init__(
        latent_dim=latent_dim,
        data_dim=data_dim,
        decoder=decoder_fn,
        q=encoder_fn,
        data_mean=data_mean,
        prior=prior,
        kl_weight=kl_weight,
        dtype=dtype)

  def log_prob(self, data, num_samples=1):
    return self._log_prob(data, num_samples=num_samples)

  def _log_prob(self, data, num_samples=1):
    mean_centered_data = data - self.data_mean

    # Construct approximate posterior and sample z.
    q_z = self.q(mean_centered_data)
    z = q_z.sample()
    log_q_z = q_z.log_prob(z)  # [num_samples, batch_size]

    # compute the prior prob of z, #[num_samples, batch_size]
    # Try giving the proposal lower bound extra compute if it can use it.
    #    try:
    #      log_p_z = self.prior.log_prob(z, num_samples=num_samples)
    #    except TypeError:
    log_p_z = self.prior.log_prob(z)

    # Compute the model logprob of the data
    p_x_given_z = self.decoder(z)
    log_p_x_given_z = p_x_given_z.log_prob(data)  # [num_samples, batch_size]

    elbo = log_p_x_given_z + self.kl_weight * (log_p_z - log_q_z)
    return elbo


class BernoulliVAE(VAE):
  """VAE with Bernoulli generative distribution."""

  def __init__(self,
               latent_dim,
               data_dim,
               decoder_hidden_sizes,
               q_hidden_sizes,
               prior=None,
               data_mean=None,
               scale_min=1e-5,
               kl_weight=1.,
               reparameterize_sample=False,
               temperature=None,
               dtype=tf.float32,
               name="bernoulli_vae"):
    # Make the decoder with a Gaussian distribution
    decoder_fn = functools.partial(
        base.conditional_bernoulli,
        data_dim=data_dim,
        hidden_sizes=decoder_hidden_sizes,
        bias_init=data_mean,
        dtype=dtype,
        use_gst=reparameterize_sample,
        temperature=temperature,
        name="%s/decoder" % name)
    q = functools.partial(
        base.conditional_normal,
        data_dim=latent_dim,
        hidden_sizes=q_hidden_sizes,
        scale_min=scale_min,
        name="%s/q" % name)

    super(BernoulliVAE, self).__init__(
        latent_dim=latent_dim,
        data_dim=data_dim,
        decoder=decoder_fn,
        q=q,
        data_mean=data_mean,
        prior=prior,
        kl_weight=kl_weight,
        dtype=dtype,
        name=name)


class ConvBernoulliVAE(VAE):
  """VAE with Bernoulli generative distribution."""

  def __init__(self,  # pylint: disable=super-init-not-called
               latent_dim,
               data_dim,
               prior=None,
               data_mean=None,
               scale_min=1e-5,
               kl_weight=1.,
               dtype=tf.float32):
    """Create ConvBernoulliVAE."""
    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = 0.

    self.kl_weight = kl_weight

    self.dtype = dtype
    if prior is None:
      self.prior = tfd.MultivariateNormalDiag(
          loc=tf.zeros([latent_dim], dtype=dtype),
          scale_diag=tf.ones([latent_dim], dtype=dtype))
    else:
      self.prior = prior

    deconv = functools.partial(tf.keras.layers.Conv2DTranspose, padding="SAME")
    conv = functools.partial(tf.keras.layers.Conv2D, padding="SAME")

    def normal_layer_fn(t):
      mu, raw_scale = tf.split(t, 2, axis=-1)
      return tfd.Independent(
          tfd.Normal(
              loc=mu,
              scale=tf.math.maximum(scale_min, tf.math.softplus(raw_scale))))

    # q(z2|x)
    self.q_z2_given_x = tf.keras.Sequential([
        conv(32, 4, 2, activation="relu"),
        conv(32, 4, 2, activation="relu"),
        conv(32, 4, 3, activation="relu"),
        tfkl.Flatten(),
        tfkl.Dense(latent_dim * 2, activation=None),
        tfpl.DistributionLambda(normal_layer_fn),
    ])

    # q(z1|x,z2)
    q_z1_x_fn = tf.keras.Sequential([
        conv(32, 4, 2, activation="relu"),
        conv(32, 4, 2, activation="relu"),
        conv(32, 4, 2, activation="relu"),
        tfkl.Flatten()
    ])

    q_z1_z2_fn = tfkl.Dense(512, activation="tanh")

    q_z1_fn = tf.keras.Sequential([
        tfkl.Dense(300, activation="tanh"),
        tfkl.Dense(latent_dim * 2, activation=None),
        tfpl.DistributionLambda(normal_layer_fn),
    ])

    def q_z1_given_x_z2(x, z2):
      x_out = q_z1_x_fn(x)
      z2_out = q_z1_z2_fn(z2)
      concat = tfkl.concatenate([x_out, z2_out])
      return tf.split(q_z1_fn(concat), 2, axis=1)

    self.q_z1_given_x_z2 = q_z1_given_x_z2

    # p(z_1|z_2)
    self.p_z1_z2_fn = tf.keras.Sequential([
        tfkl.Dense(300, activation="tanh"),
        tfkl.Dense(300, activation="tanh"),
        tfkl.Dense(latent_dim * 2, activation=None),
        tfpl.DistributionLambda(normal_layer_fn),
    ])

    # p(x|z1,z2)
    p_x_z1 = tfkl.Dense(300, activation="tanh")
    p_x_z2 = tfkl.Dense(300, activation="tanh")

    # In the LARS paper they say that RELU follows all conv layers, but I left
    # it off here.
    p_x_z1_z2_fn = tf.keras.Sequential([
        tfkl.Dense(512, activation="tanh"),
        tfkl.Reshape((4, 4, 32)),
        deconv(32, 4, 2, activation="relu"),
        deconv(32, 4, 2, activation="relu"),
        deconv(1, 4, 2, activation=None),
        tfkl.Lambda(lambda t: t[:, 2:30, 2:30, :]),
        tfpl.IndependentBernoulli(self.data_dim),
    ])

    def p_x_given_z1_z2(z1, z2):
      z1_out = p_x_z1(z1)
      z2_out = p_x_z2(z2)
      concat = tfkl.concatenate([z1_out, z2_out])
      return p_x_z1_z2_fn(concat)
      # Note that the output will be [batch_size, 28, 28, 1]
      # (trailing 1 dimensions)

    self.p_x_given_z1_z2 = p_x_given_z1_z2


#    # testing
#    # testing
#    x = tf.random_uniform([1,28,28,1])
#    y_mu, y_sigma = tf.split(q_z2_given_x(x), 2, axis=1)
#    x = tf.random_uniform([1,28,28,1])
#    z2 = tf.random_uniform([1,latent_dim])
#    y_mu, y_sigma = q_z1_given_x_z2(x,z2)
#    # testing
#    z1 = tf.random_uniform([1,latent_dim])
#    z2 = tf.random_uniform([1,latent_dim])
#    y_mu, y_sigma = p_x_given_z1_z2(z1,z2)

  def log_prob(self, data, num_samples=1):
    mean_centered_data = data - self.data_mean

    # Construct approximate posterior and sample z.
    q_z2_given_x = self.q_z2_given_x(mean_centered_data)
    z2 = q_z2_given_x.sample()
    q_z1_given_x_z2 = self.q_z1_given_x_z2(mean_centered_data, z2)
    z1 = q_z1_given_x_z2.sample()

    log_q = q_z2_given_x.log_prob(z2) + q_z1_given_x_z2.log_prob(z1)
    log_p = (
        self.prior.log_prob(z2) + self.p_z1_z2_fn.log_prob(z1) +
        self.p_x_given_z1_z2.log_prob(data))

    elbo = log_p - log_q
    return elbo
