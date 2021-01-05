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

"""A variational autoencoder with (optionally) conditional inputs."""

import gin
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow_probability import distributions as tfd

from yoto.problems.base import Problem


@gin.configurable("ConditionalVaeProblem")
class ConditionalVaeProblem(Problem):
  """A variational auto-encoder with (optionally) conditional inputs.
  """

  def __init__(self, encoder, decoder, latent_dimensions, features_key="image",
               output_distribution="gaussian", epsilon=1e-6):
    """Initializes the problem.

    Args:
      encoder: A callable that returns an encoder object which should accept one
        or two inputs, depending if you use `inputs_extra` or not. The inputs
        will be the observations themselves and the output should be  of shape
        `(batch_size, 2 * latent_dimensions)`. The first half of the output will
        be used as the mean of the approximate posterior, while the softplus of
        the second half as the corresponding variances.
      decoder: A callable that returns an decoder object which should accept one
        or two inputs, depending if you use `inputs_extra` or not. The input
        will be of shape `(batch_size, latent_dimensions)` and the output should
        have the same shape as the input to the encoder.
      latent_dimensions: Integer, the number of latent dimensions.
      features_key: Which key of the features dict holds the observations.
      output_distribution: Str. Specifies the output distribution of the VAE.
        Has to be one of {"gaussian", "bernoulli"}.
      epsilon: Float. A small value added to the variance to avoid numerical
        issues.
    """
    super(ConditionalVaeProblem, self).__init__()
    self._encoder_class = encoder
    self._decoder_class = decoder
    self._latent_dimensions = latent_dimensions
    self._features_key = features_key
    self._output_distribution = output_distribution
    self._epsilon = epsilon
    # These will be later initialized by initialize_model()
    self._encoder = None
    self._decoder = None

  def initialize_model(self):
    self._encoder = self._encoder_class()
    self._decoder = self._decoder_class()
    self._prior = tfd.MultivariateNormalDiag(
        loc=tf.zeros(self._latent_dimensions),
        scale_diag=tf.ones(self._latent_dimensions))

  def approximate_posterior(self, observations, inputs_extra=None,
                            training=False):
    """Given an observation compute Q(latent | observation)."""
    with tf.variable_scope("encoder"):
      code, endpoints = self._encoder(observations, inputs_extra,
                                      training=training)
    del endpoints
    if inputs_extra is not None:
      self._extra_shape = inputs_extra.get_shape().as_list()
    else:
      self._extra_shape = None
    self._latent_shape = code.get_shape().as_list()
    # because code includes means and variances
    self._latent_shape[-1] = self._latent_shape[-1] // 2
    code_flat = tf.reshape(code, [-1, self._latent_shape[-1] * 2])
    mean = code_flat[Ellipsis, :self._latent_dimensions]
    sigma = self._epsilon + tf.nn.softplus(
        code_flat[Ellipsis, self._latent_dimensions:])
    return tfd.MultivariateNormalDiag(loc=mean, scale_diag=sigma)

  def latent_conditional(self, latent_samples_flat, inputs_extra=None,
                         training=False):
    """Given latent samples, compute P(observation | latent)."""
    latent_samples = tf.reshape(latent_samples_flat, self._latent_shape)
    with tf.variable_scope("decoder"):
      outputs, endpoints = self._decoder(latent_samples, inputs_extra,
                                         training=training)
    del endpoints
    outputs_channels = outputs.get_shape().as_list()[-1] // 2
    if self._output_distribution == "gaussian":
      means_flat = tf.keras.layers.Flatten()(outputs[Ellipsis, :outputs_channels])
      sigmas_flat = self._epsilon + tf.nn.softplus(
          tf.keras.layers.Flatten()(outputs[Ellipsis, outputs_channels:]))
      return tfd.Independent(tfd.Normal(loc=means_flat, scale=sigmas_flat),
                             reinterpreted_batch_ndims=1)
    elif self._output_distribution == "bernoulli":
      logits_flat = tf.keras.layers.Flatten()(outputs)
      return tfd.Independent(tfd.Bernoulli(logits=logits_flat),
                             reinterpreted_batch_ndims=1)
    else:
      raise ValueError("Unknown output_distribution {}"
                       .format(self._output_distribution))

  def losses_and_metrics(self, features, inputs_extra=None, mc_samples=1,
                         training=False):
    """Compute the reconstruction and KL losses per sample."""
    observations = features[self._features_key]
    self._observations_shape = observations.get_shape().as_list()
    approx_posterior = self.approximate_posterior(observations,
                                                  inputs_extra=inputs_extra,
                                                  training=training)
    samples = approx_posterior.sample(mc_samples)[0]
    latent_conditionals = self.latent_conditional(samples,
                                                  inputs_extra=inputs_extra,
                                                  training=training)
    observations_flat = tf.keras.layers.Flatten()(observations)
    reconstruction_loss = -latent_conditionals.log_prob(observations_flat)
    kl_loss = approx_posterior.kl_divergence(self._prior)
    kl_loss = tf.reshape(kl_loss, [self._latent_shape[0], -1])
    kl_loss = tf.reduce_sum(kl_loss, axis=1)

    losses = {"reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

    metrics = {}
    return losses, metrics

  @property
  def losses_keys(self):
    return ("kl_loss", "reconstruction_loss")

  @property
  def module_spec(self):
    """Create a tf-hub module spec that should be exported."""
    def module_fn(training):
      """Hub module definition."""
      image_input = tf.placeholder(
          shape=[None] + self._observations_shape[1:], dtype=tf.float32)
      if self._extra_shape:
        inputs_extra = tf.placeholder(
            shape=[None, self._extra_shape[-1]], dtype=tf.float32)
      else:
        inputs_extra = None
      noise_shape = ([None] * (len(self._latent_shape) - 1)
                     + [self._latent_shape[-1]])
      noise_input = tf.placeholder(shape=noise_shape, dtype=tf.float32)
      with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        latents, endpoints_encoder = self._encoder(image_input, inputs_extra,
                                                   training=training)
      latent_means = latents[Ellipsis, :self._latent_shape[-1]]
      latent_log_sigmas = latents[Ellipsis, self._latent_shape[-1]:]
      with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        outputs, endpoints_decoder = self._decoder(latent_means, inputs_extra,
                                                   training=training)
      outputs_channels = outputs.get_shape().as_list()[-1] // 2
      if self._output_distribution == "gaussian":
        reconstructions = outputs[Ellipsis, :outputs_channels]
        reconstruction_log_sigmas = outputs[Ellipsis, outputs_channels:]
      elif self._output_distribution == "bernoulli":
        reconstructions = outputs
        reconstruction_log_sigmas = outputs
      recon_inputs = {"image": image_input}
      if self._extra_shape:
        recon_inputs.update({"inputs_extra": inputs_extra})
      hub.add_signature(inputs=recon_inputs, outputs=reconstructions,
                        name="reconstruct")
      outputs = {"reconstructions": reconstructions,
                 "reconstruction_log_sigmas": reconstruction_log_sigmas,
                 "latent_means": latent_means,
                 "latent_log_sigmas": latent_log_sigmas}
      outputs.update(endpoints_encoder)
      outputs.update(endpoints_decoder)
      hub.add_signature(inputs=recon_inputs, outputs=outputs,
                        name="reconstruct_with_extras")

      with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        outputs, endpoints_decoder = self._decoder(noise_input, inputs_extra,
                                                   training=training)
      if self._output_distribution == "gaussian":
        image_samples = outputs[Ellipsis, :outputs_channels]
        image_samples_log_sigmas = outputs[Ellipsis, outputs_channels:]
      elif self._output_distribution == "bernoulli":
        image_samples = outputs
        image_samples_log_sigmas = outputs

      sample_inputs = {"noise": noise_input}
      if self._extra_shape:
        sample_inputs.update({"inputs_extra": inputs_extra})
      hub.add_signature(inputs=sample_inputs, outputs=image_samples,
                        name="sample")
      outputs = {"image_samples": image_samples,
                 "image_samples_log_sigmas": image_samples_log_sigmas}
      outputs.update(endpoints_decoder)
      hub.add_signature(inputs=sample_inputs, outputs=outputs,
                        name="sample_with_extras")

    tags_and_args = [({"train"}, {"training": True}),
                     (set(), {"training": False})]
    return hub.create_module_spec(
        module_fn,
        tags_and_args=tags_and_args,
        drop_collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES])
