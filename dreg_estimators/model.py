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

"""Basic IWAE setup with DReGs estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
FLAGS = tf.flags.FLAGS

DEFAULT_INITIALIZERS = {
    "w": tf.contrib.layers.xavier_initializer(),
    "b": tf.zeros_initializer()
}


class ConditionalBernoulli(object):
  """A Bernoulli distribution conditioned on Tensor inputs via a fc net."""

  def __init__(self,
               size,
               hidden_layer_sizes,
               hidden_activation_fn=tf.nn.tanh,
               initializers=None,
               bias_init=0.0,
               name="conditional_bernoulli"):
    """Creates a conditional Bernoulli distribution.

    Args:
      size: The dimension of the random variable.
      hidden_layer_sizes: The sizes of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      hidden_activation_fn: The activation function to use on the hidden layers
        of the fully connected network.
      initializers: The variable intiializers to use for the fully connected
        network. The network is implemented using snt.nets.MLP so it must be a
        dictionary mapping the keys 'w' and 'b' to the initializers for the
        weights and biases. Defaults to xavier for the weights and zeros for the
        biases when initializers is None.
      bias_init: A scalar or vector Tensor that is added to the output of the
        fully-connected network that parameterizes the mean of this
        distribution.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.bias_init = bias_init
    self.size = size
    if initializers is None:
      initializers = DEFAULT_INITIALIZERS
    self.fcnet = snt.nets.MLP(
        output_sizes=hidden_layer_sizes + [size],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet")

  def condition(self, tensor_list):
    """Computes the p parameter of the Bernoulli distribution."""
    # Remove None's from tensor_list
    tensor_list = [t for t in tensor_list if t is not None]
    concatted_inputs = tf.concat(tensor_list, axis=-1)
    input_dim = concatted_inputs.get_shape().as_list()[-1]
    raw_input_shape = tf.shape(concatted_inputs)[:-1]
    fcnet_input_shape = [tf.reduce_prod(raw_input_shape), input_dim]
    fcnet_inputs = tf.reshape(concatted_inputs, fcnet_input_shape)
    outs = self.fcnet(fcnet_inputs) + self.bias_init
    # Reshape outputs to the original shape.
    output_size = tf.concat([raw_input_shape, [self.size]], axis=0)
    return tf.reshape(outs, output_size)

  def __call__(self, *args, **kwargs):
    p = self.condition(args)
    if kwargs.get("stop_gradient", False):
      p = tf.stop_gradient(p)
    return tfd.Bernoulli(logits=p)


class ConditionalNormal(object):
  """A Normal distribution conditioned on Tensor inputs via a fc network."""

  def __init__(self,
               size,
               hidden_layer_sizes,
               mean_center=None,
               sigma_min=0.0,
               raw_sigma_bias=0.25,
               hidden_activation_fn=tf.nn.tanh,
               initializers=None,
               name="conditional_normal"):
    """Creates a conditional Normal distribution.

    Args:
      size: The dimension of the random variable.
      hidden_layer_sizes: The sizes of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      mean_center: Optionally, mean center the data using this Tensor as the
        mean.
      sigma_min: The minimum standard deviation allowed, a scalar.
      raw_sigma_bias: A scalar that is added to the raw standard deviation
        output from the fully connected network. Set to 0.25 by default to
        prevent standard deviations close to 0.
      hidden_activation_fn: The activation function to use on the hidden layers
        of the fully connected network.
      initializers: The variable intitializers to use for the fully connected
        network. The network is implemented using snt.nets.MLP so it must be a
        dictionary mapping the keys 'w' and 'b' to the initializers for the
        weights and biases. Defaults to xavier for the weights and zeros for the
        biases when initializers is None.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.sigma_min = sigma_min
    self.raw_sigma_bias = raw_sigma_bias
    self.name = name
    self.mean_center = mean_center
    if initializers is None:
      initializers = DEFAULT_INITIALIZERS
    self.fcnet = snt.nets.MLP(
        output_sizes=hidden_layer_sizes + [2 * size],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet")

  def condition(self, tensor_list):
    """Computes the parameters of a normal distribution based on the inputs."""
    # Remove None's from tensor_list
    tensor_list = [t for t in tensor_list if t is not None]
    inputs = tf.concat(tensor_list, axis=1)
    if self.mean_center is not None:
      inputs -= self.mean_center
    outs = self.fcnet(inputs)
    mu, sigma = tf.split(outs, 2, axis=1)
    sigma = tf.maximum(
        tf.nn.softplus(sigma + self.raw_sigma_bias), self.sigma_min)
    return mu, sigma

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""
    mu, sigma = self.condition(args)

    # Optional stop_gradient argument stops the parameters of the distribution.
    # TODO(gjt): This only works for 1 latent layer networks.
    if kwargs.get("stop_gradient", False):
      mu = tf.stop_gradient(mu)
      sigma = tf.stop_gradient(sigma)
    return tfd.Normal(loc=mu, scale=sigma)


def iwae(p_z,
         p_x_given_z,
         q_z,
         observations,
         num_samples,
         cvs,
         contexts=None,
         antithetic=False):
  """Computes a gradient of the IWAE estimator.

  Args:
    p_z: The prior. Should be a callable that optionally accepts a conditioning
      context and returns a tfp.distributions.Distribution which has the
      log_prob and sample methods implemented. The distribution should be over a
      [batch_size, latent_dim] space.
    p_x_given_z: The likelihood. Should be a callable that accepts as input a
      tensor of shape [num_samples, batch_size, latent_size + context_size] and
      returns a tfd.Distribution over a [num_samples, batch_size, data_dim]
      space.
    q_z: The proposal, should be a callable which accepts a batch of
      observations of shape [batch_size, data_dim] and returns a distribution
      over [batch_size, latent_dim].
    observations: A float Tensor of shape [batch_size, data_dim] containing the
      observations.
    num_samples: The number of samples for the IWAE estimator.
    cvs: Control variate variables.
    contexts: A float Tensor of shape [batch_size, context_dim] containing the
      contexts. (Optionally, none)
    antithetic: Whether to use antithetic sampling.

  Returns:
    estimators: Dictionary of tuples (objective, neg_model_loss,
      neg_inference_network_loss).
  """
  alpha, beta, gamma, delta = cvs
  batch_size = tf.shape(observations)[0]
  proposal = q_z(observations, contexts, stop_gradient=False)
  # [num_samples, batch_size, latent_size]

  # If antithetic sampling, draw half of the samples and use the antithetics
  # for the other half.
  if antithetic:
    z_pos = proposal.sample(sample_shape=[num_samples // 2])
    z_neg = 2 * proposal.loc - z_pos
    z = tf.concat((z_pos, z_neg), axis=0)
  else:
    z = proposal.sample(sample_shape=[num_samples])

  tiled_contexts = None
  if contexts is not None:
    tiled_contexts = tf.tile(tf.expand_dims(contexts, 0), [num_samples, 1, 1])
  likelihood = p_x_given_z(z, tiled_contexts)
  # Before reduce_sum is [num_samples, batch_size, latent_dim].
  # Sum over the latent dim.
  log_q_z = tf.reduce_sum(proposal.log_prob(z), axis=-1)
  # Before reduce_sum is  [num_samples, batch_size, latent_dim].
  # Sum over latent dim.
  prior = p_z(contexts)
  log_p_z = tf.reduce_sum(prior.log_prob(z), axis=-1)
  # Before reduce_sum is [num_samples, batch_size, data_dim]
  log_p_x_given_z = tf.reduce_sum(likelihood.log_prob(observations), axis=-1)

  log_weights = log_p_z + log_p_x_given_z - log_q_z
  log_sum_weight = tf.reduce_logsumexp(log_weights, axis=0)
  log_avg_weight = log_sum_weight - tf.log(tf.to_float(num_samples))
  normalized_weights = tf.stop_gradient(tf.nn.softmax(log_weights, axis=0))

  if FLAGS.image_summary:
    best_index = tf.to_int32(tf.argmax(normalized_weights, axis=0))
    indices = tf.stack((best_index, tf.range(0, batch_size)), axis=-1)
    best_images = tf.gather_nd(likelihood.probs_parameter(), indices)

    if FLAGS.dataset == "struct_mnist":
      tf.summary.image("bottom_half",
                       tf.reshape(best_images, [batch_size, -1, 28, 1]))
    else:
      tf.summary.image("output", tf.reshape(best_images,
                                            [batch_size, -1, 28, 1]))
    tf.summary.image("input", tf.reshape(observations, [batch_size, -1, 28, 1]))

  # Compute gradient estimators
  model_loss = log_avg_weight
  estimators = {}

  estimators["iwae"] = (log_avg_weight, log_avg_weight, log_avg_weight)

  stopped_z_log_q_z = tf.reduce_sum(
      proposal.log_prob(tf.stop_gradient(z)), axis=-1)
  estimators["rws"] = (log_avg_weight, model_loss,
                       tf.reduce_sum(
                           normalized_weights * stopped_z_log_q_z, axis=0))

  # Doubly reparameterized
  stopped_proposal = q_z(observations, contexts, stop_gradient=True)
  stopped_log_q_z = tf.reduce_sum(stopped_proposal.log_prob(z), axis=-1)
  stopped_log_weights = log_p_z + log_p_x_given_z - stopped_log_q_z
  sq_normalized_weights = tf.square(normalized_weights)

  estimators["stl"] = (log_avg_weight, model_loss,
                       tf.reduce_sum(
                           normalized_weights * stopped_log_weights, axis=0))
  estimators["dreg"] = (log_avg_weight, model_loss,
                        tf.reduce_sum(
                            sq_normalized_weights * stopped_log_weights,
                            axis=0))
  estimators["rws-dreg"] = (
      log_avg_weight, model_loss,
      tf.reduce_sum(
          (normalized_weights - sq_normalized_weights) * stopped_log_weights,
          axis=0))

  # Add normed versions
  normalized_sq_normalized_weights = (
      sq_normalized_weights / tf.reduce_sum(
          sq_normalized_weights, axis=0, keepdims=True))
  estimators["dreg-norm"] = (
      log_avg_weight, model_loss,
      tf.reduce_sum(
          normalized_sq_normalized_weights * stopped_log_weights, axis=0))

  rws_dregs_weights = normalized_weights - sq_normalized_weights
  normalized_rws_dregs_weights = rws_dregs_weights / tf.reduce_sum(
      rws_dregs_weights, axis=0, keepdims=True)
  estimators["rws-dreg-norm"] = (
      log_avg_weight, model_loss,
      tf.reduce_sum(normalized_rws_dregs_weights * stopped_log_weights, axis=0))

  estimators["dreg-alpha"] = (log_avg_weight, model_loss,
                              (1 - FLAGS.alpha) * estimators["dreg"][-1] +
                              FLAGS.alpha * estimators["rws-dreg"][-1])

  # Jackknife
  loo_log_weights = tf.tile(
      tf.expand_dims(tf.transpose(log_weights), -1), [1, 1, num_samples])
  loo_log_weights = tf.matrix_set_diag(
      loo_log_weights, -np.inf * tf.ones([batch_size, num_samples]))
  loo_log_avg_weight = tf.reduce_mean(
      tf.reduce_logsumexp(loo_log_weights, axis=1) - tf.log(
          tf.to_float(num_samples - 1)),
      axis=-1)
  jk_model_loss = num_samples * log_avg_weight - (
      num_samples - 1) * loo_log_avg_weight

  estimators["jk"] = (jk_model_loss, jk_model_loss, jk_model_loss)

  # Compute JK w/ DReG for the inference network
  loo_normalized_weights = tf.reduce_mean(
      tf.square(tf.stop_gradient(tf.nn.softmax(loo_log_weights, axis=1))),
      axis=-1)
  estimators["jk-dreg"] = (
      jk_model_loss, jk_model_loss, num_samples * tf.reduce_sum(
          sq_normalized_weights * stopped_log_weights, axis=0) -
      (num_samples - 1) * tf.reduce_sum(
          tf.transpose(loo_normalized_weights) * stopped_log_weights, axis=0))

  # Compute control variates
  loo_baseline = tf.expand_dims(tf.transpose(log_weights), -1)
  loo_baseline = tf.tile(loo_baseline, [1, 1, num_samples])
  loo_baseline = tf.matrix_set_diag(
      loo_baseline, -np.inf * tf.ones_like(tf.transpose(log_weights)))
  loo_baseline = tf.reduce_logsumexp(loo_baseline, axis=1)
  loo_baseline = tf.transpose(loo_baseline)

  learning_signal = tf.stop_gradient(tf.expand_dims(
      log_avg_weight, 0)) - (1 - gamma) * tf.stop_gradient(loo_baseline)
  vimco = tf.reduce_sum(learning_signal * stopped_z_log_q_z, axis=0)

  first_part = alpha * vimco + (1 - alpha) * tf.reduce_sum(
      normalized_weights * stopped_log_weights, axis=0)
  second_part = ((1 - beta) * (tf.reduce_sum(
      ((1 - delta) / tf.to_float(num_samples) - normalized_weights) *
      stopped_z_log_q_z,
      axis=0)) + beta * tf.reduce_sum(
          (sq_normalized_weights - normalized_weights) * stopped_log_weights,
          axis=0))
  estimators["dreg-cv"] = (log_avg_weight, model_loss, first_part + second_part)

  return estimators
