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

"""Switching non-linear dynamical systems (SwitchingNLDS)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from snlds import forward_backward_algo
from snlds import utils


class SwitchingNLDS(tf.keras.Model):
  """Switching NonLinear Dynamical Systems base model.

  The SwitchingNLDS provides the implementation of the core algorithm for
  collapsed amortized variational inference in SNLDS, as described in Section 3
  of Dong et al. (2019)[1]. Subnetworks can be configured outside the
  SwitchingNLDS, which is a `tf.keras.Model` instance. The configurable
  subnetworks include the continuous dynamic transition network
  p(z[t] | z[t-1], ...), discrete state transition network
  p(s[t] | s[t-1], ...), emission network p(x[t] | z[t]), and inference
  network p(z[1:T] | x[1:T]) etc. For more details, please check the `__init__`
  function.

  References:
    [1] Dong, Zhe and Seybold, Bryan A. and Murphy, Kevin P., and Bui,
        Hung H.. Collapsed Amortized Variational Inference for Switching
        Nonlinear Dynamical Systems. 2019. https://arxiv.org/abs/1910.09588.
  """

  def __init__(self,
               continuous_transition_network,
               discrete_transition_network,
               emission_network,
               inference_network,
               initial_distribution,
               continuous_state_dim=None,
               num_categories=None,
               discrete_state_prior=None):
    """Constructor of Switching Non-Linear Dynamical System.

    The model framework, as described in Dong et al. (2019)[1].

    Args:
      continuous_transition_network: a `callable` with its `call` function
        taking batched sequences of continuous hidden states, `z[t-1]`, with
        shape [batch_size, num_steps, hidden_states], and returning a
        distribution with its `log_prob` function implemented. The `log_prob`
        function takes continuous hidden states, `z[t]`, and returns their
        likelihood, `p(z[t] | z[t-1], s[t])`.
      discrete_transition_network: a `callable` with its `call` function
        taking batch conditional inputs, `x[t-1]`, and returning the discrete
        state transition matrices, `log p(s[t] |s[t-1], x[t-1])`.
      emission_network: a `callable` with its `call` function taking
        continuous hidden states, `z[t]`, and returning a distribution,
        `p(x[t] | z[t])`. The distribution should have `mean` and `sample`
        function, similar as the classes in `tfp.distributions`.
      inference_network: inference network should be a class that has
        `sample` function, which takes input observations, `x[1:T]`,
        and outputs the sampled hidden states sequence of `q(z[1:T] | x[1:T])`
        and the entropy of the distribution.
      initial_distribution: a initial state distribution for continuous
        variables, `p(z[0])`.
      continuous_state_dim: number of continuous hidden units, `z[t]`.
      num_categories: number of discrete hidden states, `s[t]`.
      discrete_state_prior: a `float` Tensor, indicating the prior
        of discrete state distribution, `p[k] = p(s[t]=k)`. This is used by
        cross entropy regularizer, which tries to minize the difference between
        discrete_state_prior and the smoothed likelihood of the discrete states,
        `p(s[t] | x[1:T], z[1:T])`.

    Reference:
      [1] Dong, Zhe and Seybold, Bryan A. and Murphy, Kevin P., and Bui,
          Hung H.. Collapsed Amortized Variational Inference for Switching
          Nonlinear Dynamical Systems. 2019. https://arxiv.org/abs/1910.09588.
    """
    super(SwitchingNLDS, self).__init__()

    self.z_tran = continuous_transition_network
    self.s_tran = discrete_transition_network
    self.x_emit = emission_network
    self.inference_network = inference_network
    self.z0_dist = initial_distribution

    if num_categories is None:
      self.num_categ = self.s_tran.output_event_dims
    else:
      self.num_categ = num_categories

    if continuous_state_dim is None:
      self.z_dim = self.z_tran.output_event_dims
    else:
      self.z_dim = continuous_state_dim

    if discrete_state_prior is None:
      self.discrete_prior = tf.ones(
          shape=[self.num_categ], dtype=tf.float32) / self.num_categ
    else:
      self.discrete_prior = discrete_state_prior

    self.log_init = tf.Variable(
        utils.normalize_logprob(
            tf.ones(shape=[self.num_categ], dtype=tf.float32),
            axis=-1)[0],
        name="snlds_logprob_s0")

  def call(self, inputs, temperature=1.0, num_samples=1, dtype=tf.float32):

    """Inference call of SNLDS.

    Args:
      inputs: a `float` Tensor of shape `[batch_size, num_steps, event_size]`,
        containing the observation time series of the model.
      temperature: a `float` Scalar for temperature used to estimate discrete
        state transition `p(s[t] | s[t-1], x[t-1])` as described in Dong et al.
        (2019). Increasing temperature increase the uncertainty about each
        discrete states.
        Default to 1. For ''temperature annealing'', the temperature is set
        to large value initially, and decay to a smaller one. A temperature
        should be positive, but could be smaller than `1.`.
      num_samples: an `int` scalar for number of samples per time-step, for
        posterior inference networks, `z[i] ~ q(z[1:T] | x[1:T])`.
      dtype: data type for calculation. Default to `tf.float32`.

    Returns:
      return_dict: a python `dict` contains all the `Tensor`s for inference
        results. Including the following keys:
        elbo: Evidence Lower Bound, returned by `get_objective_values` function.
        iwae: IWAE Bound, returned by `get_objective_values` function.
        initial_likelihood: the likelihood of `p(s[0], z[0], x[0])`, returned
          by `get_objective_values` function.
        sequence_likelihood: the likelihood of `p(s[1:T], z[1:T], x[0:T])`,
          returned by `get_objective_values` function.
        zt_entropy: the entropy of posterior distribution `H(q(z[t] | x[1:T])`,
          returned by `get_objective_values` function.
        reconstruction: the reconstructed inputs, returned by
          `get_reconstruction` function.
        posterior_llk: the posterior likelihood, `p(s[t] | x[1:T], z[1:T])`,
          returned by `forward_backward_algo.forward_backward` function.
        sampled_z: the sampled z[1:T] from the approximate posterior.
        cross_entropy: batched cross entropy between discrete state posterior
          likelihood and its prior distribution.
    """
    inputs = tf.convert_to_tensor(inputs, dtype_hint=dtype,
                                  name="SNLDS_Input_Tensor")
    # Sample continuous hidden variable from `q(z[1:T] | x[1:T])'
    z_sampled, z_entropy, log_prob_q = self.inference_network(
        inputs, num_samples=num_samples)

    _, batch_size, num_steps, z_dim = tf.unstack(
        tf.shape(z_sampled))

    # Merge batch_size and num_samples dimensions.
    z_sampled = tf.reshape(z_sampled,
                           [num_samples * batch_size, num_steps, z_dim])
    z_entropy = tf.reshape(z_entropy, [num_samples * batch_size, num_steps])
    log_prob_q = tf.reshape(log_prob_q, [num_samples * batch_size, num_steps])

    inputs = tf.tile(inputs, [num_samples, 1, 1])

    # Base on observation inputs `x', sampled continuous dynamical states
    # `z_sampled', get `log_a(j, k) = p(s[t]=j | s[t-1]=k, x[t-1])', and
    # `log_b(k) = p(x[t] | z[t])p(z[t] | z[t-1], s[t]=k)'.
    log_b, log_a = self.calculate_likelihoods(
        inputs, z_sampled, temperature=temperature)

    # Forward-backward algorithm will return the posterior marginal of
    # discrete states `log_gamma2 = p(s[t]=k, s[t-1]=j | x[1:T], z[1:T])'
    # and `log_gamma1 = p(s[t]=k | x[1:T], z[1:T])'.
    _, _, log_gamma1, log_gamma2 = forward_backward_algo.forward_backward(
        log_a, log_b, self.log_init)

    recon_inputs = self.get_reconstruction(
        z_sampled,
        observation_shape=tf.shape(inputs),
        sample_for_reconstruction=False)

    # Calculate Evidence Lower Bound with components.
    # The return_dict currently support the following items:
    #   elbo: Evidence Lower Bound.
    #   iwae: IWAE Lower Bound.
    #   initial_likelihood: likelihood of p(s[0], z[0], x[0]).
    #   sequence_likelihood: likelihood of p(s[1:T], z[1:T], x[0:T]).
    #   zt_entropy: the entropy of posterior distribution.
    return_dict = self.get_objective_values(log_a, log_b, self.log_init,
                                            log_gamma1, log_gamma2, log_prob_q,
                                            z_entropy, num_samples)

    # Estimate the cross entropy between state prior and posterior likelihoods.
    state_crossentropy = utils.get_posterior_crossentropy(
        log_gamma1,
        prior_probs=self.discrete_prior)
    state_crossentropy = tf.reduce_mean(state_crossentropy, axis=0)

    recon_inputs = tf.reshape(recon_inputs,
                              [num_samples, batch_size, num_steps, -1])
    log_gamma1 = tf.reshape(log_gamma1,
                            [num_samples, batch_size, num_steps, -1])
    z_sampled = tf.reshape(z_sampled,
                           [num_samples, batch_size, num_steps, z_dim])

    return_dict["inputs"] = inputs
    return_dict["reconstructions"] = recon_inputs[0]
    return_dict["posterior_llk"] = log_gamma1[0]
    return_dict["sampled_z"] = z_sampled[0]
    return_dict["cross_entropy"] = state_crossentropy

    return return_dict

  def _get_log_likelihood(self, log_a, log_b, log_init, log_gamma1, log_gamma2):
    """Computes the log-likelihood based on pre-computed log-probabilities.

    Computes E_s[log p(x[1:T], z[1:T], s[1:T])] decomposed into two terms.

    Args:
      log_a: Transition tensor:
        log_a[t, i, j] = log p(s[t] = i|x[t-1], s[t-1]=j),
        size [batch_size, num_steps, num_cat, num_cat]
      log_b: Emission tensor:
        log_b[t, i] = log p(x[t], z[t] | s[t]=i, z[t-1]),
        size [batch_size, num_steps, num_cat]
      log_init: Initial tensor,
        log_init[i] = log p(s[0]=i)
        size [batch_size, num_cat]
      log_gamma1: computed by forward-backward algorithm.
        log_gamma1[t, i] = log p(s[t] = i | v[1:T]),
        size [batch_size, num_steps, num_cat]
      log_gamma2: computed by forward-backward algorithm.
        log_gamma2[t, i, j] = log p(s[t]= i, s[t-1]= j| v[1:T]),
        size [batch_size, num_steps, num_cat, num_cat]

    Returns:
      tuple (t1, t2)
        t1: sequence likelihood, E_s[log p(s[1:T], v[1:T]| s[0], v[0])], size
        [batch_size]
        t2: initial likelihood, E_s[log p(s[0], v[0])], size
        [batch_size]
    """
    gamma1 = tf.exp(log_gamma1)
    gamma2 = tf.exp(log_gamma2)
    t1 = tf.reduce_sum(gamma2[:, 1:, :, :]
                       * (log_b[:, 1:, tf.newaxis, :]
                          + log_a[:, 1:, :, :]),
                       axis=[1, 2, 3])

    gamma1_1, log_b1 = gamma1[:, 0, :], log_b[:, 0, :]
    t2 = tf.reduce_sum(gamma1_1 * (log_b1 +  log_init[tf.newaxis, :]),
                       axis=-1)
    return t1, t2

  def get_objective_values(self,
                           log_a, log_b, log_init,
                           log_gamma1, log_gamma2,
                           log_prob_q,
                           posterior_entropies,
                           num_samples):
    """Given all precalculated probabilities, return ELBO."""
    # All the sequences should be of the shape
    # [batch_size, num_steps, (data_dim)]

    sequence_likelihood, initial_likelihood = self._get_log_likelihood(
        log_a, log_b, log_init, log_gamma1, log_gamma2)

    t3 = tf.reduce_sum(posterior_entropies, axis=1)

    t1_mean = tf.reduce_mean(sequence_likelihood, axis=0)
    t2_mean = tf.reduce_mean(initial_likelihood, axis=0)
    t3_mean = tf.reduce_mean(t3, axis=0)
    elbo = t1_mean + t2_mean + t3_mean
    iwae = self._get_iwae(sequence_likelihood, initial_likelihood, log_prob_q,
                          num_samples)
    return dict(
        elbo=elbo,
        iwae=iwae,
        initial_likelihood=t2_mean,
        sequence_likelihood=t1_mean,
        zt_entropy=t3_mean)

  def _get_iwae(self, sequence_likelihood, initial_likelihood, log_prob_q,
                num_samples):
    r"""Computes the IWAE bound given the pre-computed log-probabilities.

    The IWAE Bound is given by:
      E_{z^i~q(z^i|x)}[ log 1/k \sum_i \frac{p(x, z^i)}{q(z^i | x)} ]
    where z^i and x are complete trajectories z_{1:T}^i and x_{1:T}. The {1:T}
    is omitted for simplicity of notation.

    log p(x, z) is given by E_s[log p(s, x, z)]

    Args:
      sequence_likelihood: E_s[log p(s[1:T], v[1:T] | s[0], v[0])],
        size [num_samples * batch_size]
      initial_likelihood: E_s[log p(s[0], v[0])],
        size [num_samples * batch_size]
      log_prob_q: log q(z[t]| x[1:T], z[1:t-1]),
        size [num_samples * batch_size, T]
      num_samples: number of samples per trajectory.

    Returns:
      tf.Tensor, the estimated IWAE bound.
    """
    log_likelihood = sequence_likelihood + initial_likelihood
    log_surrogate_posterior = tf.reduce_sum(log_prob_q, axis=-1)

    # Reshape likelihoods to [num_samples, batch_size]
    log_likelihood = tf.reshape(log_likelihood, [num_samples, -1])
    log_surrogate_posterior = tf.reshape(log_surrogate_posterior,
                                         [num_samples, -1])

    iwae_bound = tf.reduce_logsumexp(
        log_likelihood - log_surrogate_posterior,
        axis=0) - tf.math.log(tf.cast(num_samples, tf.float32))
    iwae_bound_mean = tf.reduce_mean(iwae_bound)
    return iwae_bound_mean

  def calculate_likelihoods(self,
                            inputs,
                            sampled_z,
                            switching_conditional_inputs=None,
                            temperature=1.0):
    """Calculate the probability by p network, `p_theta(x,z,s)`.

    Args:
      inputs: a float 3-D `Tensor` of shape [batch_size, num_steps, obs_dim],
        containing the observation time series of the model.
      sampled_z: a float 3-D `Tensor` of shape [batch_size, num_steps,
        latent_dim] for continuous hidden states, which are sampled from
        inference networks, `q(z[1:T] | x[1:T])`.
      switching_conditional_inputs: a float 3-D `Tensor` of shape [batch_size,
        num_steps, encoded_dim], which is the conditional input for discrete
        state transition probability, `p(s[t] | s[t-1], x[t-1])`.
        Default to `None`, when `inputs` will be used.
      temperature: a float scalar `Tensor`, indicates the temperature for
        transition probability, `p(s[t] | s[t-1], x[t-1])`.

    Returns:
      log_xt_zt: a float `Tensor` of size [batch_size, num_steps, num_categ]
        indicates the distribution, `log(p(x_t | z_t) p(z_t | z_t-1, s_t))`.
      prob_st_stm1: a float `Tensor` of size [batch_size, num_steps, num_categ,
        num_categ] indicates the transition probablity, `p(s_t | s_t-1, x_t-1)`.
      reconstruced_inputs: a float `Tensor` of size [batch_size, num_steps,
        obs_dim] for reconstructed inputs.
    """
    batch_size, num_steps = tf.unstack(tf.shape(inputs)[:2])
    num_steps = inputs.get_shape().with_rank_at_least(3).dims[1].value

    ########################################
    ## getting log p(z[t] | z[t-1], s[t])
    ########################################

    # Broadcasting rules of TFP dictate that: if the samples_z0 of dimension
    # [batch_size, 1, event_size], z0_dist is of [num_categ, event_size].
    # z0_dist.log_prob(samples_z0[:, None, :]) is of [batch_size, num_categ].
    sampled_z0 = sampled_z[:, 0, :]
    log_prob_z0 = self.z0_dist.log_prob(sampled_z0[:, tf.newaxis, :])
    log_prob_z0 = log_prob_z0[:, tf.newaxis, :]

    # `log_prob_zt` should be of the shape [batch_size, num_steps, self.z_dim]
    log_prob_zt = self.get_z_prior(sampled_z, log_prob_z0)

    ########################################
    ## getting log p(x[t] | z[t])
    ########################################

    emission_dist = self.x_emit(sampled_z)

    # `emission_dist' should have the same event shape as `inputs',
    # by broadcasting rule, the `log_prob_xt' should be of the shape
    # [batch_size, num_steps],
    log_prob_xt = emission_dist.log_prob(
        tf.reshape(inputs, [batch_size, num_steps, -1]))

    ########################################
    ## getting log p(s[t] |s[t-1], x[t-1])
    ########################################

    if switching_conditional_inputs is None:
      switching_conditional_inputs = inputs
    log_prob_st_stm1 = tf.reshape(
        self.s_tran(switching_conditional_inputs[:, :-1, :]),
        [batch_size, num_steps-1, self.num_categ, self.num_categ])
    # by normalizing the 3rd axis (axis=-2), we restrict A[:,:,i,j] to be
    # transiting from s[t-1]=j -> s[t]=i
    log_prob_st_stm1 = utils.normalize_logprob(
        log_prob_st_stm1, axis=-2, temperature=temperature)[0]

    log_prob_st_stm1 = tf.concat(
        [tf.eye(self.num_categ, self.num_categ, batch_shape=[batch_size, 1],
                dtype=tf.float32, name="concat_likelihoods"),
         log_prob_st_stm1], axis=1)

    # log ( p(x_t | z_t) p(z_t | z_t-1, s_t) )
    log_xt_zt = log_prob_xt[:, :, tf.newaxis] + log_prob_zt
    return log_xt_zt, log_prob_st_stm1

  def get_reconstruction(self,
                         hidden_state_sequence,
                         observation_shape=None,
                         sample_for_reconstruction=False):
    """Generate reconstructed inputs from emission distribution, `p(x[t]|z[t])`.

    Args:
      hidden_state_sequence: a `float` `Tensor` of the shape [batch_size,
        num_steps, hidden_dims], containing batched continuous hidden variable
        `z[t]`.
      observation_shape: a `TensorShape` object or `int` list, containing the
        shape of sampled `x[t]` to reshape reconstructed inputs.
        Default to `None`, in which case the output of `mean` or `sample`
        function for emission distribution will be returned directly, without
        reshape.
      sample_for_reconstruction: a `bool` scalar. When `True`, it will will use
        `emission_distribution.sample()` to generate reconstructions.
        Default to `False`, in which case the mean of distribution will be used
        as reconstructed observations.

    Returns:
      reconstructed_obs: a `float` `Tensor` of the shape [batch_size, num_steps,
        observation_dims], containing reconstructed observations.
    """
    # get the distribution for p(x[t] | z[t])
    emission_dist = self.x_emit(hidden_state_sequence)

    if sample_for_reconstruction:
      reconstructed_obs = emission_dist.sample()
    else:
      reconstructed_obs = emission_dist.mean()

    if observation_shape is not None:
      reconstructed_obs = tf.reshape(reconstructed_obs, observation_shape)

    return reconstructed_obs

  def get_z_prior(self, sampled_z, log_prob_z0):
    """p(z[t] | z[t-1], s[t]) transition."""
    prior_distributions = self.z_tran(sampled_z[:, :-1, :])

    # If `prior_distribution` of shape `[batch_size, num_steps-1, num_categ,
    # hidden_dim]`, and `obs` of shape `[batch_size, num_steps-1, 1,
    # hidden_dim]`, the `dist.log_prob(obs)` is of  `[batch_size, num_steps-1,
    # num_categ]`.
    future_tensor = sampled_z[:, 1:, :]
    log_prob_zt = prior_distributions.log_prob(
        future_tensor[:, :, tf.newaxis, :])

    log_prob_zt = tf.concat([log_prob_z0, log_prob_zt], axis=1,
                            name="concat_z_prior")
    return log_prob_zt
