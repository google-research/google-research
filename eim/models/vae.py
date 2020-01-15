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
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from eim.models import base

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

deconv = functools.partial(tf.keras.layers.Conv2DTranspose, padding="SAME")
conv = functools.partial(tf.keras.layers.Conv2D, padding="SAME")


class VAE(base.ProbabilisticModel):
  """Variational autoencoder with continuous latent space."""

  def __init__(self,
               latent_dim,
               data_dim,
               decoder,
               q,
               proposal=None,
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
      proposal: A distribution over the latent space of the VAE. The object must
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
    if proposal is None:
      self.proposal = base.get_independent_normal([latent_dim])
    else:
      self.proposal = proposal

  def _log_prob(self, data, num_samples=1):
    """Compute a lower bound on the log likelihood."""
    mean_centered_data = data - self.data_mean
    # Tile by num_samples on the batch dimension.
    tiled_mean_centered_data = tf.tile(mean_centered_data,
                                       [num_samples] + [1] * len(self.data_dim))
    tiled_data = tf.tile(data,
                         [num_samples] + [1] * len(self.data_dim))

    # Construct approximate posterior and sample z.
    q_z = self.q(tiled_mean_centered_data)
    z = q_z.sample()  # [num_samples * batch_size, data_dim]
    log_q_z = q_z.log_prob(z)  # [num_samples * batch_size]

    # compute the proposal prob of z, #[num_samples * batch_size]
    try:
      log_p_z = self.proposal.log_prob(z, log_q_data=log_q_z)
    except TypeError:
      log_p_z = self.proposal.log_prob(z)

    # Compute the model logprob of the data
    p_x_given_z = self.decoder(z)
    # [num_samples * batch_size]
    log_p_x_given_z = p_x_given_z.log_prob(tiled_data)

    elbo = log_p_x_given_z + self.kl_weight * (log_p_z - log_q_z)
    iwae = (tf.reduce_logsumexp(tf.reshape(elbo, [num_samples, -1]), axis=0)
            - tf.log(tf.to_float(num_samples)))
    return iwae

  def sample(self, num_samples=1):
    z = self.proposal.sample(num_samples)
    p_x_given_z = self.decoder(z)
    return tf.cast(p_x_given_z.sample(), self.dtype)


class GaussianVAE(VAE):
  """VAE with Gaussian generative distribution."""

  def __init__(self,
               latent_dim,
               data_dim,
               decoder_hidden_sizes,
               q_hidden_sizes,
               proposal=None,
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
        data_dim=[latent_dim],
        hidden_sizes=q_hidden_sizes,
        scale_min=scale_min,
        name="%s/q" % name)

    super(GaussianVAE, self).__init__(
        latent_dim=latent_dim,
        data_dim=data_dim,
        decoder=decoder_fn,
        q=q,
        data_mean=data_mean,
        proposal=proposal,
        kl_weight=kl_weight,
        dtype=dtype)


class BernoulliVAE(VAE):
  """VAE with Bernoulli generative distribution."""

  def __init__(self,
               latent_dim,
               data_dim,
               decoder_hidden_sizes,
               q_hidden_sizes,
               proposal=None,
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
        data_dim=[latent_dim],
        hidden_sizes=q_hidden_sizes,
        scale_min=scale_min,
        name="%s/q" % name)

    super(BernoulliVAE, self).__init__(
        latent_dim=latent_dim,
        data_dim=data_dim,
        decoder=decoder_fn,
        q=q,
        data_mean=data_mean,
        proposal=proposal,
        kl_weight=kl_weight,
        dtype=dtype)


class HVAE(object):
  """2 stochastic layer VAE."""

  def __init__(self,
               latent_dim,
               data_dim,
               proposal=None,
               data_mean=None,
               kl_weight=1.,
               dtype=tf.float32):
    """Create HVAE."""
    self.latent_dim = latent_dim
    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = 0.
    self.kl_weight = kl_weight
    self.dtype = dtype
    if proposal is None:
      self.proposal = base.get_independent_normal([latent_dim])
    else:
      self.proposal = proposal
    self._build()

  def _build(self):
    pass

  def log_prob(self, data, num_samples=1):
    """Computes log probability lower bound."""
    tiled_data = tf.tile(data[None],
                         [num_samples, 1] + [1] * len(self.data_dim))
    tiled_data_flat = tf.reshape(tiled_data, [-1] + self.data_dim)

    # Construct approximate posterior and sample z.
    q_z2_given_x = self.q_z2_given_x(data)
    z2 = q_z2_given_x.sample(sample_shape=[num_samples])  # [S, B, ...]
    z2_flat = tf.reshape(z2, [-1, self.latent_dim])  # [S*B, ...]
    q_z1_given_x_z2 = self.q_z1_given_x_z2(tiled_data_flat, z2_flat)
    z1 = q_z1_given_x_z2.sample()

    log_q = self.kl_weight * (
        tf.reshape(q_z2_given_x.log_prob(z2), [-1]) +
        q_z1_given_x_z2.log_prob(z1))
    log_p = (
        self.kl_weight *
        (self.proposal.log_prob(z2_flat) + self.p_z1_z2(z2_flat).log_prob(z1)) +
        self.p_x_given_z1_z2(z1, z2_flat).log_prob(tiled_data_flat))

    elbo = tf.reduce_logsumexp(tf.reshape(log_p - log_q, [num_samples, -1]),
                               axis=0) - tf.log(tf.to_float(num_samples))
    return elbo

  def sample(self, num_samples=1):
    z2 = self.proposal.sample(num_samples)
    z1 = self.p_z1_z2(z2).sample()
    p_x = self.p_x_given_z1_z2(z1, z2)
    return tf.cast(p_x.sample(), self.dtype)


class ConvBernoulliVAE(HVAE):
  """VAE with Bernoulli generative distribution."""

  def __init__(self,
               latent_dim,
               data_dim,
               proposal=None,
               data_mean=None,
               scale_min=1e-5,
               kl_weight=1.,
               dtype=tf.float32):
    """Create ConvBernoulliVAE."""
    self.scale_min = scale_min
    super(ConvBernoulliVAE, self).__init__(latent_dim,
                                           data_dim,
                                           proposal,
                                           data_mean,
                                           kl_weight,
                                           dtype)

  def _get_observation_layer(self):
    bias_init = -tf.log(1. /
                        tf.clip_by_value(self.data_mean, 0.0001, 0.9999) - 1)
    return [tfkl.Lambda(lambda t: t + bias_init),
            tfkl.Flatten(),
            tfpl.IndependentBernoulli(self.data_dim)]

  def _build(self):
    """Creates the distributions for the VAE."""
    def normal_layer_fn(t):
      mu, raw_scale = tf.split(t, 2, axis=-1)
      return tfd.Independent(
          tfd.Normal(
              loc=mu,
              scale=tf.math.maximum(self.scale_min,
                                    tf.math.softplus(raw_scale))))

    # q(z2|x)
    self.q_z2_given_x = tf.keras.Sequential([
        tfkl.Lambda(lambda t: t - self.data_mean),
        conv(32, 4, 2, activation="relu"),
        conv(32, 4, 2, activation="relu"),
        conv(32, 4, 2, activation="relu"),
        tfkl.Flatten(),
        tfkl.Dense(self.latent_dim * 2, activation=None),
        tfpl.DistributionLambda(normal_layer_fn),
    ])

    # q(z1|x,z2)
    q_z1_x_fn = tf.keras.Sequential([
        tfkl.Lambda(lambda t: t - self.data_mean),
        conv(32, 4, 2, activation="relu"),
        conv(32, 4, 2, activation="relu"),
        conv(32, 4, 2, activation="relu"),
        tfkl.Flatten()
    ])

    q_z1_z2_fn = tfkl.Dense(512, activation="tanh")

    q_z1_fn = tf.keras.Sequential([
        tfkl.Dense(300, activation="tanh"),
        tfkl.Dense(self.latent_dim * 2, activation=None),
        tfpl.DistributionLambda(normal_layer_fn),
    ])

    def q_z1_given_x_z2(x, z2):
      x_out = q_z1_x_fn(x)
      z2_out = q_z1_z2_fn(z2)
      concat = tfkl.concatenate([x_out, z2_out])
      return q_z1_fn(concat)
    self.q_z1_given_x_z2 = q_z1_given_x_z2

    # p(z_1|z_2)
    self.p_z1_z2 = tf.keras.Sequential([
        tfkl.Dense(300, activation="tanh"),
        tfkl.Dense(300, activation="tanh"),
        tfkl.Dense(self.latent_dim * 2, activation=None),
        tfpl.DistributionLambda(normal_layer_fn),
    ])

    # p(x|z1,z2)
    p_x_z1 = tfkl.Dense(300, activation="tanh")
    p_x_z2 = tfkl.Dense(300, activation="tanh")

    p_x_z1_z2_fn = tf.keras.Sequential([
        tfkl.Dense(512, activation="tanh"),
        tfkl.Reshape((4, 4, 32)),
        deconv(32, 4, 2, activation="relu"),
        # Remove the extra row/column [7 x 7 x 32] after
        tfkl.Lambda(lambda t: t[:, :-1, :-1, :]),
        deconv(32, 4, 2, activation="relu"),
        # In the LARS paper they say that RELU follows all conv layers, but I
        # left it off here.
        deconv(1, 4, 2, activation=None),] +
                                       self._get_observation_layer())

    def p_x_given_z1_z2(z1, z2):
      z1_out = p_x_z1(z1)
      z2_out = p_x_z2(z2)
      concat = tfkl.concatenate([z1_out, z2_out])
      return p_x_z1_z2_fn(concat)
      # Note that the output will be [batch_size, 28, 28, 1]
      # (trailing 1 dimensions)

    self.p_x_given_z1_z2 = p_x_given_z1_z2


class ConvGaussianVAE(ConvBernoulliVAE):
  """VAE with Gaussian generative distribution."""

  def __init__(self,
               latent_dim,
               data_dim,
               proposal=None,
               data_mean=None,
               scale_min=1e-5,
               scale_init=1.,
               kl_weight=1.,
               dtype=tf.float32,
               name="ConvGaussianVAE"):
    """Create ConvGaussianVAE."""
    self.scale_init = scale_init
    self.name = name
    super(ConvGaussianVAE, self).__init__(latent_dim,
                                          data_dim,
                                          proposal,
                                          data_mean,
                                          scale_min,
                                          kl_weight,
                                          dtype)

  def _get_observation_layer(self):
    return [tfpl.DistributionLambda(lambda t: tfd.Independent(  # pylint: disable=g-long-lambda
        tfd.Normal(loc=t, scale=self.decoder_scale[None])))]

  def _build(self):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      raw_scale_init = np.log(np.exp(self.scale_init) - 1 + self.scale_min)
      raw_scale = tf.get_variable(
          name="raw_sigma",
          shape=self.data_dim,
          dtype=tf.float32,
          initializer=tf.constant_initializer(raw_scale_init),
          trainable=True)
    self.decoder_scale = tf.math.maximum(self.scale_min,
                                         tf.math.softplus(raw_scale))

    super(ConvGaussianVAE, self)._build()


