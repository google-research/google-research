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

"""Trainable models for ARM++ experiments."""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.stats.leave_one_out import log_soomean_exp

keras = tf.keras
tfd = tfp.distributions

EPS = 1e-6


def safe_log_prob(p):
  return tf.math.log(tf.clip_by_value(p, EPS, 1.0))


def logit_func(prob_tensor):
  """Calculate logits."""
  return safe_log_prob(prob_tensor) - safe_log_prob(1. - prob_tensor)


def get_local_disarm_learning_signal(
    b1,
    b2,
    elbo_b1,
    elbo_b2,
    encoder_logits):
  """Get local learning signal for VIMCO-ARM++ v3."""
  # b1, b2 are of the shape [num_samples, batch_size, hidden_dims]
  # elbo_b1, elbo_b2 are of the shape [num_samples, batch_size]

  num_samples = tf.shape(elbo_b1)[0]
  num_samples_f = tf.cast(num_samples, tf.float32)

  # [num_samples, batch_size, hidden_dims] shape
  disarm_factor = ((1. - b1)*b2 + b1*(1. - b2)) * (-1.)**b2
  disarm_factor *= tf.math.sigmoid(tf.math.abs(encoder_logits))

  # [1, batch_size] shape
  l_b1 = (tf.reduce_logsumexp(elbo_b1, axis=0, keepdims=True)
          - tf.math.log(num_samples_f))

  # the following manipulation is due to tf.linalg.set_diag is applying
  # on [a, b, c, d, ..., m, m] last two dimension.
  # [batch_size, num_samples]
  transposed_elbo_b1 = tf.transpose(elbo_b1, [1, 0])
  transposed_elbo_b2 = tf.transpose(elbo_b2, [1, 0])
  # [batch_size, num_samples, num_samples]
  tiled_elbo_b1 = tf.tile(
      tf.expand_dims(transposed_elbo_b1, -1),
      [1, 1, num_samples])
  # [batch_size, num_samples, num_samples]
  elbo_b_tilde = tf.linalg.set_diag(tiled_elbo_b1,
                                    transposed_elbo_b2)
  # [batch_size, num_samples] shape
  # TODO(zhedong): the `axis=1` is frigile, fix this with better understading
  l_b2_t = (tf.reduce_logsumexp(elbo_b_tilde, axis=1, keepdims=False)
            - tf.math.log(num_samples_f))
  # [num_samples, batch_size]
  l_b2 = tf.transpose(l_b2_t, [1, 0])

  # [num_samples, batch_size, 1] shape
  local_learning_signal = tf.expand_dims(l_b1 - l_b2, -1)

  # [batch_size, hidden_dim]
  infnet_grad_factor = tf.reduce_sum(
      0.5 * local_learning_signal * disarm_factor,
      axis=0,
      keepdims=False)
  multisample_objective = tf.squeeze(l_b1, axis=0)
  return infnet_grad_factor, multisample_objective


def get_vimco_local_learning_signal(elbo_tensor):
  """Get vimco local learning signal from batched ELBO.

  Args:
    elbo_tensor: a `float` Tensor of the shape [num_samples, batch_size].

  Returns:
    local_learning_signal: a `float` Tensor of the same shape as `input_tensor`,
      contains the multiplicative factor as described in Algorithm 1 of VIMCO,
      L_hat - L_hat^[-i].
  """
  assert_op = tf.debugging.assert_rank_at_least(
      elbo_tensor, rank=2,
      message='ELBO needs at least 2D, [sample, batch].')
  with tf.control_dependencies([assert_op]):
    # Calculate the log swap-one-out mean and log mean
    # log_soomean_f is of the same shape as f: [num_samples, batch]
    # log_mean_f is of the reduced shape: [1, batch]
    log_soomean_f, log_mean_f = log_soomean_exp(elbo_tensor,
                                                axis=0,
                                                keepdims=True)
    local_learning_signal = log_mean_f - log_soomean_f
    return local_learning_signal


def get_vimco_disarm_learning_signal(
    b1, b2, elbo_b1, elbo_b2, encoder_logits,
    verbose_return=False,
    half_p_trick=False):
  """Get local learning signal for VIMCO-ARM++."""
  elbo_b1b2 = tf.concat([elbo_b1, elbo_b2], axis=0)
  l_b1b2 = (
      tf.reduce_logsumexp(elbo_b1b2, axis=0, keepdims=True)
      - tf.math.log(tf.cast(tf.shape(elbo_b1b2)[0], tf.float32)))
  local_learning_signal = l_b1b2 - 0.5 * (elbo_b1 + elbo_b2)
  local_learning_signal = tf.expand_dims(local_learning_signal, axis=-1)
  local_learning_signal = tf.stop_gradient(local_learning_signal)

  if half_p_trick:
    logprob_term1 = (tf.stop_gradient((1. - b1) * b2 + b1 * (1. - b2))
                     * (tf.math.log_sigmoid(encoder_logits) - tf.math.log(2.)))
    logprob_term2 = - (tf.stop_gradient((1. - b1) * (1. - b2))
                       * tf.math.softplus(encoder_logits))
  else:
    abs_phi = tf.math.abs(encoder_logits)

    logprob_term1 = (tf.stop_gradient((1. - b1) * b2 + b1 * (1. - b2))
                     * (-1.) * tf.math.softplus(abs_phi))
    logprob_term2 = (tf.stop_gradient((1. - b1) * (1. - b2) + b1 * b2)
                     * (tfp.math.log1mexp(-abs_phi)
                        - tf.math.softplus(-abs_phi)))
  logprob_b1b2 = logprob_term1 + logprob_term2

  if half_p_trick:
    sigma_abs_phi = tf.math.sigmoid(tf.math.abs(
        tf.math.log_sigmoid(encoder_logits-tf.math.log(2.))))
  else:
    sigma_abs_phi = tf.math.sigmoid(tf.math.abs(encoder_logits))

  vimco_signal = local_learning_signal * logprob_b1b2

  # arm pp factor, shape [sample, batch, event_dims]
  disarm_factor = ((1. - b1) * b2 + b1 * (1. - b2)) * (-1.)**b2
  disarm_factor *= sigma_abs_phi
  disarm_signal = (
      0.5 * tf.expand_dims(elbo_b1 - elbo_b2, axis=-1) * disarm_factor)

  if verbose_return:
    return dict(l_b1b2=l_b1b2,
                local_learning_signal=local_learning_signal,
                logprob_b1b2=logprob_b1b2,
                disarm_signal=disarm_signal,
                vimco_signal=vimco_signal,
                logprob_list=(logprob_term1, logprob_term2))
  else:
    return local_learning_signal, logprob_b1b2, disarm_signal, l_b1b2


class BinaryNetwork(tf.keras.Model):
  """Network generating binary samples."""

  def __init__(self,
               hidden_sizes,
               activations,
               mean_xs=None,
               demean_input=False,
               final_layer_bias_initializer='zeros',
               name='binarynet'):

    super(BinaryNetwork, self).__init__(name=name)
    assert len(activations) == len(hidden_sizes)

    num_layers = len(hidden_sizes)
    self.hidden_sizes = hidden_sizes
    self.activations = activations
    self.networks = keras.Sequential()

    if demean_input:
      if mean_xs is not None:
        self.networks.add(
            tf.keras.layers.Lambda(lambda x: x - mean_xs))
      else:
        self.networks.add(
            tf.keras.layers.Lambda(lambda x: 2.*tf.cast(x, tf.float32) - 1.))
    for i in range(num_layers-1):
      self.networks.add(
          keras.layers.Dense(
              units=hidden_sizes[i],
              activation=activations[i]))

    self.networks.add(
        keras.layers.Dense(
            units=hidden_sizes[-1],
            activation=activations[-1],
            bias_initializer=final_layer_bias_initializer))

  def __call__(self,
               input_tensor,
               samples=None,
               num_samples=(),
               half_p_trick=False):
    logits = self.get_logits(input_tensor, half_p_trick)
    dist = tfd.Bernoulli(logits=logits)
    if samples is None:
      samples = dist.sample(num_samples)
    samples = tf.cast(samples, tf.float32)
    likelihood = dist.log_prob(samples)
    return samples, likelihood, logits

  def get_logits(self, input_tensor, half_p_trick=False):
    logits = self.networks(input_tensor)
    if half_p_trick:
      logits = tf.math.log_sigmoid(logits-tf.math.log(2.))
    return logits

  def sample_uniform_variables(self, sample_shape, nfold=1):
    if nfold > 1:
      sample_shape = tf.concat(
          [sample_shape[0:1] * nfold, sample_shape[1:]],
          axis=0)
    return tf.random.uniform(shape=sample_shape, maxval=1.0)


class SingleLayerDiscreteVAE(tf.keras.Model):
  """Discrete VAE as described in ARM, (Yin and Zhou (2019))."""

  def __init__(self,
               encoder,
               decoder,
               prior_logits,
               grad_type='arm',
               half_p_trick=False,
               epsilon=0.,
               control_nn=None,
               name='dvae'):
    super(SingleLayerDiscreteVAE, self).__init__(name)
    self.encoder = encoder
    self.decoder = decoder

    self.half_p_trick = half_p_trick
    if self.half_p_trick:
      # This is to enforce the p of Bernoulli distribution is [0, 0.5].
      self.prior_logits = tf.math.log_sigmoid(prior_logits-tf.math.log(2.))
    else:
      self.prior_logits = prior_logits

    self.prior_dist = tfd.Bernoulli(logits=self.prior_logits)

    self.grad_type = grad_type.lower()

    # used for variance of gradients estiamations.
    self.ema = tf.train.ExponentialMovingAverage(0.999)

    # epsilon for the tolerrence used in VIMCO-ARM++
    self.epsilon = epsilon

    if self.grad_type == 'relax':
      self.log_temperature_variable = tf.Variable(
          initial_value=tf.math.log(0.1),  # Reasonable init
          dtype=tf.float32)

      # the scaling_factor is a trainable ~1.
      self.scaling_variable = tf.Variable(
          initial_value=1.,
          dtype=tf.float32)

      # neural network for control variates lambda * r(z)
      self.control_nn = control_nn

  def call(self, input_tensor, hidden_samples=None, num_samples=()):
    """Returns ELBO.

    Args:
      input_tensor: a `float` Tensor for input observations.
        The tensor is of the shape [batch_size, observation_dims].
      hidden_samples: a discrete Tensor for hidden states.
        The tensor is of the shape [batch_size, hidden_dims].
        Default to None, in which case the hidden samples will be generated
        based on num_samples.
      num_samples: 0-D or 1-D `int` Tensor. Shape of the generated samples.

    Returns:
      elbo: the ELBO with shape [batch_size].
    """
    hidden_sample, encoder_llk, encoder_logits = self.encoder(
        input_tensor,
        samples=hidden_samples,
        num_samples=num_samples,
        half_p_trick=self.half_p_trick)

    encoder_llk = tf.reduce_sum(encoder_llk, axis=-1)
    log_pb = tf.reduce_sum(
        self.prior_dist.log_prob(hidden_sample),
        axis=-1)

    decoder_llk = tf.reduce_sum(
        self.decoder(hidden_sample, input_tensor)[1],
        axis=-1)

    elbo = decoder_llk + log_pb - encoder_llk
    encoder_logits = self.threshold_around_zero(encoder_logits)

    return elbo, hidden_sample, encoder_logits, encoder_llk

  def get_elbo(self, input_tensor, hidden_tensor):
    """Returns ELBO.

    Args:
      input_tensor: a `float` Tensor for input observations.
        The tensor is of the shape [batch_size, observation_dims].
      hidden_tensor: a discrete Tensor for hidden states.
        The tensor is of the shape [batch_size, hidden_dims].

    Returns:
      elbo: the ELBO with shape [batch_size].
    """
    elbo = self.call(input_tensor, hidden_samples=hidden_tensor)[0]
    return elbo

  def get_layer_grad_estimation(
      self, input_tensor, grad_type=None):
    if grad_type is None:
      grad_type = self.grad_type

    encoder_logits = self.encoder.get_logits(input_tensor)
    sigma_phi = tf.math.sigmoid(encoder_logits)

    if grad_type == 'ar':
      u_noise = self.encoder.sample_uniform_variables(
          sample_shape=tf.shape(encoder_logits),
          nfold=1)
      b = tf.cast(u_noise < sigma_phi, tf.float32)
      f = self.get_elbo(input_tensor, b)[:, tf.newaxis]
      layer_grad = f * (1. - 2.*u_noise)

    elif grad_type == 'ar-2sample':
      u_noise = self.encoder.sample_uniform_variables(
          sample_shape=tf.shape(encoder_logits),
          nfold=2)
      u1, u2 = tf.split(u_noise, num_or_size_splits=2, axis=0)
      b1 = tf.cast(u1 < sigma_phi, tf.float32)
      b2 = tf.cast(u2 < sigma_phi, tf.float32)
      f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
      f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
      layer_grad = 0.5 * (f1 * (1. - 2.*u1) + f2 * (1. - 2.*u2))

    elif grad_type == 'arm':
      u_noise = self.encoder.sample_uniform_variables(
          sample_shape=tf.shape(encoder_logits),
          nfold=1)
      b1 = tf.cast(u_noise > 1. - sigma_phi, tf.float32)
      b2 = tf.cast(u_noise < sigma_phi, tf.float32)
      f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
      f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
      layer_grad = (f1 - f2) * (u_noise - 0.5)

    elif grad_type == 'armp':
      # ARM with conditioning, but no dropping of b_i = tilde b_i terms.
      u_noise = self.encoder.sample_uniform_variables(
          sample_shape=tf.shape(encoder_logits),
          nfold=1)
      b1 = tf.cast(u_noise > 1. - sigma_phi, tf.float32)
      b2 = tf.cast(u_noise < sigma_phi, tf.float32)
      f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
      f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
      # Only consider the case where b1_i \neq b2_i
      armp_factor = ((1. - b1) * (b2) + b1 * (1. - b2))  # * (-1.)**b2
      layer_grad = (f1 - f2) * (u_noise - 0.5) * armp_factor

    elif grad_type == 'disarm':
      u_noise = self.encoder.sample_uniform_variables(
          sample_shape=tf.shape(encoder_logits),
          nfold=1)
      sigma_abs_phi = tf.math.sigmoid(tf.math.abs(encoder_logits))
      b1 = tf.cast(u_noise > 1. - sigma_phi, tf.float32)
      b2 = tf.cast(u_noise < sigma_phi, tf.float32)
      f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
      f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
      # the factor is I(b1+b2=1) * (-1)**b2 * sigma(|phi|)
      disarm_factor = ((1. - b1) * (b2) + b1 * (1. - b2)) * (-1.)**b2
      disarm_factor *= sigma_abs_phi
      layer_grad = 0.5 * (f1 - f2) * disarm_factor

    elif grad_type == 'reinforce':
      u_noise = self.encoder.sample_uniform_variables(
          sample_shape=tf.shape(encoder_logits),
          nfold=1)
      b = tf.cast(u_noise < sigma_phi, tf.float32)
      f = self.get_elbo(input_tensor, b)[:, tf.newaxis]
      layer_grad = f * (b - sigma_phi)

    elif grad_type == 'reinforce-2sample':
      u_noise = self.encoder.sample_uniform_variables(
          sample_shape=tf.shape(encoder_logits),
          nfold=2)
      u1, u2 = tf.split(u_noise, num_or_size_splits=2, axis=0)
      b1 = tf.cast(u1 < sigma_phi, tf.float32)
      b2 = tf.cast(u2 < sigma_phi, tf.float32)
      f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
      f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
      layer_grad = 0.5 * (f1 * (b1 - sigma_phi) + f2 * (b2 - sigma_phi))

    elif grad_type == 'reinforce_loo_baseline':
      u_noise = self.encoder.sample_uniform_variables(
          sample_shape=tf.shape(encoder_logits),
          nfold=2)
      u1, u2 = tf.split(u_noise, num_or_size_splits=2, axis=0)
      b1 = tf.cast(u1 < sigma_phi, tf.float32)
      b2 = tf.cast(u2 < sigma_phi, tf.float32)
      f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
      f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
      layer_grad = 0.5 * ((f1 - f2) * (b1 - sigma_phi)
                          + (f2 - f1) * (b2 - sigma_phi))

    else:
      raise NotImplementedError('Gradient type %s is not supported.'%grad_type)

    return layer_grad

  def sample_binaries_with_loss(
      self,
      input_tensor,
      antithetic_sample=True):
    encoder_logits = self.encoder.get_logits(input_tensor)
    sigma_phi = tf.math.sigmoid(encoder_logits)
    scaling_factor = 0.5 if self.half_p_trick else 1.
    bernoulli_prob = scaling_factor * sigma_phi
    # returned u_noise would be of the shape [batch x num_samples, event_dim].
    u_noise = self.encoder.sample_uniform_variables(
        sample_shape=tf.shape(encoder_logits))

    theresholded_encoder_logits = self.threshold_around_zero(encoder_logits)

    if antithetic_sample:
      b1 = tf.cast(u_noise > 1. - bernoulli_prob, tf.float32)
      b2 = tf.cast(u_noise < bernoulli_prob, tf.float32)
      elbo_b1 = self.get_elbo(input_tensor, b1)
      elbo_b2 = self.get_elbo(input_tensor, b2)

      return b1, b2, elbo_b1, elbo_b2, theresholded_encoder_logits

    else:
      b = tf.cast(u_noise < bernoulli_prob, tf.float32)
      elbo = self.get_elbo(input_tensor, b)
      return b, elbo, theresholded_encoder_logits

  def get_relax_parameters(
      self,
      input_tensor,
      temperature=None,
      scaling_factor=None):
    if temperature is None:
      temperature = tf.math.exp(self.log_temperature_variable)
    if scaling_factor is None:
      scaling_factor = self.scaling_variable
    # [batch, hidden_units]
    encoder_logits = self.encoder.get_logits(input_tensor)

    # returned uniform_noise would be of the shape
    # [batch x 2, event_dim].
    uniform_noise = self.encoder.sample_uniform_variables(
        sample_shape=tf.shape(encoder_logits),
        nfold=2)
    # u_noise and v_noise are both of [batch, event_dim].
    u_noise, v_noise = tf.split(uniform_noise, num_or_size_splits=2, axis=0)

    theta = tf.math.sigmoid(encoder_logits)
    z = encoder_logits + logit_func(u_noise)
    b_sample = tf.cast(z > 0, tf.float32)

    v_prime = (b_sample * (v_noise * theta + 1 - theta)
               + (1 - b_sample) * v_noise * (1 - theta))
    # z_tilde ~ p(z | b)
    z_tilde = encoder_logits + logit_func(v_prime)

    elbo = self.get_elbo(input_tensor, b_sample)
    control_variate = self.get_relax_control_variate(
        input_tensor, z,
        temperature=temperature, scaling_factor=scaling_factor)
    conditional_control = self.get_relax_control_variate(
        input_tensor, z_tilde,
        temperature=temperature, scaling_factor=scaling_factor)

    log_q = tfd.Bernoulli(logits=encoder_logits).log_prob(b_sample)
    return elbo, control_variate, conditional_control, log_q

  def get_relax_control_variate(self, input_tensor, z_sample,
                                temperature, scaling_factor):
    control_value = (
        scaling_factor *
        self.get_elbo(input_tensor, tf.math.sigmoid(z_sample/temperature)))
    if self.control_nn is not None:
      control_nn_input = tf.concat((input_tensor, z_sample), axis=-1)
      control_value += (scaling_factor
                        * tf.squeeze(self.control_nn(control_nn_input),
                                     axis=-1))
    return control_value

  def _get_grad_variance(self, grad_variable, grad_sq_variable, grad_tensor):
    grad_variable.assign(grad_tensor)
    grad_sq_variable.assign(tf.square(grad_tensor))
    self.ema.apply([grad_variable, grad_sq_variable])

    # mean per component variance
    grad_var = (
        self.ema.average(grad_sq_variable)
        - tf.square(self.ema.average(grad_variable)))
    return grad_var

  def compute_grad_variance(
      self,
      grad_variables,
      grad_sq_variables,
      grad_tensors):
    # In order to use `tf.train.ExponentialMovingAverage`, one has to
    # use `tf.Variable`.
    grad_var = [
        tf.reshape(self._get_grad_variance(*g), [-1])
        for g in zip(grad_variables, grad_sq_variables, grad_tensors)]
    return tf.reduce_mean(tf.concat(grad_var, axis=0))

  def threshold_around_zero(self, input_tensor):
    if self.epsilon > 0.:
      return (tf.where(tf.math.greater(input_tensor, 0.),
                       tf.math.maximum(input_tensor, self.epsilon),
                       tf.math.minimum(input_tensor, -self.epsilon)))
    return input_tensor

  @property
  def encoder_vars(self):
    return self.encoder.trainable_variables

  @property
  def decoder_vars(self):
    return self.decoder.trainable_variables

  @property
  def prior_vars(self):
    return self.prior_dist.trainable_variables

  def get_vimco_losses(self, input_batch, num_samples):
    batch_size = tf.shape(input_batch)[0]
    tiled_inputs = tf.tile(input_batch, [num_samples, 1])
    elbo, _, _, encoder_llk = self.call(tiled_inputs)

    elbo = tf.reshape(elbo, [num_samples, batch_size])
    encoder_llk = tf.reshape(encoder_llk, [num_samples, batch_size])

    # [sample_size, batch_size]
    local_learning_signal = get_vimco_local_learning_signal(elbo)
    local_learning_signal = tf.stop_gradient(local_learning_signal)

    # [batch_size]
    multisample_objective = (
        tf.reduce_logsumexp(elbo, axis=0, keepdims=False) -
        tf.math.log(tf.cast(tf.shape(elbo)[0], tf.float32)))

    infnet_loss = -1. * (multisample_objective +
                         tf.reduce_sum(local_learning_signal * encoder_llk,
                                       axis=0))
    genmo_loss = -1. * multisample_objective
    return genmo_loss, infnet_loss

  def get_local_disarm_losses(self, input_batch, num_samples,
                              symmetrized=False):
    batch_size = tf.shape(input_batch)[0]
    tiled_inputs = tf.tile(input_batch, [num_samples, 1])

    # [batch_size, hidden_size]
    batch_encoder_logits = self.encoder.get_logits(input_batch)

    # b1, b2 are of the shape [num_samples * batch_size, hidden_dim]
    b1, b2, elbo_b1, elbo_b2, _ = (
        self.sample_binaries_with_loss(
            tiled_inputs,
            antithetic_sample=True))
    b1 = tf.reshape(b1, [num_samples, batch_size, -1])
    b2 = tf.reshape(b2, [num_samples, batch_size, -1])

    elbo_b1 = tf.reshape(elbo_b1, [num_samples, batch_size])
    elbo_b2 = tf.reshape(elbo_b2, [num_samples, batch_size])

    # infnet_grad_multiplier: [batch_size, hidden_dim]
    # batch_multisample_objective: [batch_size]
    if symmetrized:
      infnet_grad_multiplier_1, multisample_objective_1 = (
          get_local_disarm_learning_signal(
              b1, b2, elbo_b1, elbo_b2, batch_encoder_logits))
      infnet_grad_multiplier_2, multisample_objective_2 = (
          get_local_disarm_learning_signal(
              b2, b1, elbo_b2, elbo_b1, batch_encoder_logits))
      infnet_grad_multiplier = 0.5 * tf.stop_gradient(
          infnet_grad_multiplier_1 + infnet_grad_multiplier_2)
      multisample_objective = 0.5 * (
          multisample_objective_1 + multisample_objective_2)
    else:
      infnet_grad_multiplier, multisample_objective = (
          get_local_disarm_learning_signal(
              b1, b2, elbo_b1, elbo_b2, batch_encoder_logits))
      infnet_grad_multiplier = tf.stop_gradient(infnet_grad_multiplier)

    genmo_loss = -1. * multisample_objective

    infnet_loss = -1. * (infnet_grad_multiplier * batch_encoder_logits)
    return genmo_loss, infnet_loss

  def get_multisample_baseline_loss(self, input_batch, num_samples):
    """Computes gradients for num_samples IWAE bound.

    This estimator uses 2 * num_samples, half to compute the bound
    and the other half to compute a baseline.

    Args:
      input_batch: Input tensor [batch_size, dim].
      num_samples: Number of samples for the IWAE bound.

    Returns:
      genmo_loss: Loss function for the model params.
      infnet_loss: Loss function for the inference network params.
    """
    batch_size = tf.shape(input_batch)[0]
    tiled_inputs = tf.tile(input_batch, [2 * num_samples, 1])
    elbo, _, _, encoder_llk = self.call(tiled_inputs)

    elbo = tf.reshape(elbo, [2 * num_samples, batch_size])
    elbo_signal, elbo_baseline = tf.split(elbo, num_or_size_splits=2, axis=0)

    encoder_llk = tf.reshape(encoder_llk, [2 * num_samples, batch_size])
    encoder_llk = tf.split(encoder_llk, num_or_size_splits=2, axis=0)[0]

    # [batch_size]
    encoder_llk = tf.reduce_sum(encoder_llk, axis=0)
    control_variate = (
        tf.reduce_logsumexp(elbo_baseline, axis=0, keepdims=False) -
        tf.math.log(tf.cast(tf.shape(elbo_baseline)[0], tf.float32)))
    multisample_objective = (
        tf.reduce_logsumexp(elbo_signal, axis=0, keepdims=False) -
        tf.math.log(tf.cast(tf.shape(elbo_signal)[0], tf.float32)))

    # [num_samples, batch_size]
    learning_signal = multisample_objective - control_variate
    learning_signal = tf.stop_gradient(learning_signal)

    infnet_loss = -1. * (multisample_objective +
                         learning_signal * encoder_llk)
    genmo_loss = -1. * multisample_objective
    return genmo_loss, infnet_loss

  def get_relax_loss(self, input_batch, temperature=None, scaling_factor=None):
    # elbo, control_variate, conditional_control should be of [batch_size]
    # log_q is of [batch_size, event_dim]
    elbo, control_variate, conditional_control, log_q = (
        self.get_relax_parameters(
            input_batch,
            temperature=temperature,
            scaling_factor=scaling_factor))

    # Define losses
    genmo_loss = -1. * elbo

    reparam_loss = -1. * (control_variate - conditional_control)

    # [batch_size]
    learning_signal = -1. * (elbo - conditional_control)
    self.mean_learning_signal = tf.reduce_mean(learning_signal)

    # [batch_size, hidden_size]
    learning_signal = tf.tile(
        tf.expand_dims(learning_signal, axis=-1),
        [1, tf.shape(log_q)[-1]])

    return genmo_loss, reparam_loss, learning_signal, log_q
