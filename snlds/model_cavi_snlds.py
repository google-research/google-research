# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Collapsed Amortized Variational Inference for SNLDS.

This is a reasonable baseline model for switching non-linear dynamical system
with the following architecture:
1. an inference network, with Bidirectional-RNN for input embedding, and a
   forward RNN to get the posterior distribution of `q(z[1:T] | x[1:T])`.
2. a continuous state transition network, `p(z[t] | z[t-1], s[t])`.
3. a discrete state transition network that conditioned on the input,
   `p(s[t] | s[t-1], x[t-1])`.
4. an emission network conditioned on the continuous hidden dynamics,
   `p(x[t] | z[t])`.

It also contains a function, `create_model()`, to help to create the SNLDS
model discribed in  ``Collapsed Amortized Variational Inference for Switching
Nonlinear Dynamical Systems``. 2019. https://arxiv.org/abs/1910.09588.
All the networks are configurable through function arguments `network_*`.
"""

import collections
import tensorflow as tf
import tensorflow_probability as tfp
from snlds import model_base
from snlds import utils

namedtuple = collections.namedtuple

layers = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers

RANDOM_SEED = 131


def construct_initial_state_distribution(
    latent_dim,
    num_categ,
    use_trainable_cov=False,
    use_triangular_cov=False,
    raw_sigma_bias=0.0,
    sigma_min=1e-5,
    sigma_scale=0.05,
    dtype=tf.float32,
    name="z0"):
  """Construct the initial state distribution, `p(z[0])`.

  Args:
    latent_dim: an `int` scalar for dimension of continuous hidden states, `z`.
    num_categ: an `int` scalar for number of discrete states, `s`.
    use_trainable_cov: a `bool` scalar indicating whether the scale of `p(z[0])`
      is trainable. Default to False.
    use_triangular_cov: a `bool` scalar indicating whether to use triangular
      covariance matrices and `tfp.distributions.MultivariateNormalTriL` for
      distribution. Otherwise, a diagonal covariance matrices and
      `tfp.distributions.MultivariateNormalDiag` will be used.
    raw_sigma_bias: a `float` scalar to be added to the raw sigma, which is
      standard deviation of the distribution. Default to `0.`.
    sigma_min: a `float` scalar for minimal level of sigma to prevent
      underflow. Default to `1e-5`.
    sigma_scale: a `float` scalar for scaling the sigma. Default to `0.05`.
      The above three arguments are used as
      `sigma_scale * max(softmax(raw_sigma + raw_sigma_bias), sigma_min))`.
    dtype: data type for variables within the scope. Default to `tf.float32`.
    name: a `str` to construct names of variables.

  Returns:
    return_dist: a `tfp.distributions` instance for the initial state
      distribution, `p(z[0])`.
  """

  glorot_initializer = tf.keras.initializers.GlorotUniform()
  z0_mean = tf.Variable(
      initial_value=glorot_initializer(shape=[num_categ, latent_dim],
                                       dtype=dtype),
      name="{}_mean".format(name))

  if use_triangular_cov:
    z0_scale = tfp.math.fill_triangular(
        tf.Variable(
            initial_value=glorot_initializer(
                shape=[int(latent_dim * (latent_dim + 1) / 2)],
                dtype=dtype),
            name="{}_scale".format(name),
            trainable=use_trainable_cov))
    z0_scale = (tf.maximum(tf.nn.softmax(z0_scale + raw_sigma_bias),
                           sigma_min)
                * sigma_scale)
    return_dist = tfd.Independent(
        distribution=tfd.MultivariateNormalTriL(
            loc=z0_mean, scale_tril=z0_scale),
        reinterpreted_batch_ndims=0)

  else:
    z0_scale = tf.Variable(
        initial_value=glorot_initializer(
            shape=[latent_dim],
            dtype=dtype),
        name="{}_scale".format(name),
        trainable=use_trainable_cov)
    z0_scale = (tf.maximum(tf.nn.softmax(z0_scale + raw_sigma_bias),
                           sigma_min)
                * sigma_scale)
    return_dist = tfd.Independent(
        distribution=tfd.MultivariateNormalDiag(
            loc=z0_mean, scale_diag=z0_scale),
        reinterpreted_batch_ndims=0)

  return tfp.experimental.as_composite(return_dist)


class ContinuousStateTransition(tf.keras.Model):
  """Transition for `p(z[t] | z[t-1], s[t])`."""

  def __init__(self,
               transition_mean_networks,
               distribution_dim,
               num_categories=1,
               cov_mat=None,
               use_triangular_cov=False,
               use_trainable_cov=True,
               raw_sigma_bias=0.0,
               sigma_min=1e-5,
               sigma_scale=0.05,
               dtype=tf.float32,
               name="ContinuousStateTransition"):
    """Construct a `ContinuousStateTransition` instance.

    Args:
      transition_mean_networks: a list of `callable` networks, with the length
        of list same as `num_categories`. Each one of the networks will take
        previous step hidden state, `z[t-1]`, and returns the mean of
        transition distribution, `p(z[t] | z[t-1], s[t]=i)` for each
        discrete state `i`.
      distribution_dim: an `int` scalar for dimension of continuous hidden
        states, `z`.
      num_categories: an `int` scalar for number of discrete states, `s`.
      cov_mat: an optional `float` Tensor for predefined covariance matrix.
        Default to `None`, in which case, a `cov` variable will be created.
      use_triangular_cov: a `bool` scalar indicating whether to use triangular
        covariance matrices and `tfp.distributions.MultivariateNormalTriL` for
        distribution. Otherwise, a diagonal covariance matrices and
        `tfp.distributions.MultivariateNormalDiag` will be used.
      use_trainable_cov: a `bool` scalar indicating whether the scale of
        the distribution is trainable. Default to False.
      raw_sigma_bias: a `float` scalar to be added to the raw sigma, which is
        standard deviation of the distribution. Default to `0.`.
      sigma_min: a `float` scalar for minimal level of sigma to prevent
        underflow. Default to `1e-5`.
      sigma_scale: a `float` scalar for scaling the sigma. Default to `0.05`.
        The above three arguments are used as
        `sigma_scale * max(softmax(raw_sigma + raw_sigma_bias), sigma_min))`.
      dtype: data type for variables within the scope. Default to `tf.float32`.
      name: a `str` to construct names of variables.
    """
    super(ContinuousStateTransition, self).__init__()

    assertion_str = (
        "There has to be one transition mean networks for each discrete state")
    assert len(transition_mean_networks) == num_categories, assertion_str
    self.z_trans_networks = transition_mean_networks
    self.num_categ = num_categories
    self.use_triangular_cov = use_triangular_cov
    self.distribution_dim = distribution_dim

    if cov_mat:
      self.cov_mat = cov_mat
    elif self.use_triangular_cov:
      self.cov_mat = tfp.math.fill_triangular(
          tf.Variable(
              tf.random.uniform(
                  shape=[
                      int(self.distribution_dim
                          * (self.distribution_dim + 1) / 2)],
                  minval=0., maxval=1.,
                  dtype=dtype),
              name="{}_cov".format(name),
              dtype=dtype,
              trainable=use_trainable_cov))
      self.cov_mat = tf.maximum(tf.nn.softmax(self.cov_mat + raw_sigma_bias),
                                sigma_min) * sigma_scale
    else:
      self.cov_mat = tf.Variable(
          tf.random.uniform(shape=[self.distribution_dim],
                            minval=0.0, maxval=1.,
                            dtype=dtype),
          name="{}_cov".format(name),
          dtype=dtype,
          trainable=use_trainable_cov)
      self.cov_mat = tf.maximum(tf.nn.softmax(self.cov_mat + raw_sigma_bias),
                                sigma_min) * sigma_scale

  def call(self, input_tensor, dtype=tf.float32):

    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    batch_size, num_steps, distribution_dim = tf.unstack(tf.shape(input_tensor))

    # The shape of the mean_tensor after tf.stack is [num_categ, batch_size,
    # num_steps, distribution_dim].,
    mean_tensor = tf.transpose(
        tf.stack([
            z_net(input_tensor) for z_net in self.z_trans_networks]),
        [1, 2, 0, 3])
    mean_tensor = tf.reshape(mean_tensor,
                             [batch_size, num_steps,
                              self.num_categ, distribution_dim])

    if self.use_triangular_cov:
      output_dist = tfd.MultivariateNormalTriL(
          loc=mean_tensor,
          scale_tril=self.cov_mat)
    else:
      output_dist = tfd.MultivariateNormalDiag(
          loc=mean_tensor,
          scale_diag=self.cov_mat)

    return tfp.experimental.as_composite(output_dist)

  @property
  def output_event_dims(self):
    return self.distribution_dim


class DiscreteStateTransition(tf.keras.Model):
  """Discrete state transition p(s[t] | s[t-1], x[t-1])."""

  def __init__(self,
               transition_network,
               num_categories):
    """Construct a `DiscreteStateTransition` instance.

    Args:
      transition_network: a `callable` network taking batch conditional inputs,
        `x[t-1]`, and returning the discrete state transition matrices,
        `log p(s[t] |s[t-1], x[t-1])`.
      num_categories: an `int` scalar for number of discrete states, `s`.
    """
    super(DiscreteStateTransition, self).__init__()
    self.dense_net = transition_network
    self.num_categ = num_categories

  def call(self, input_tensor, dtype=tf.float32):

    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    batch_size, num_steps = tf.unstack(tf.shape(input_tensor)[:2])
    transition_tensor = self.dense_net(input_tensor)
    transition_tensor = tf.reshape(
        transition_tensor,
        [batch_size, num_steps, self.num_categ, self.num_categ])
    return transition_tensor

  @property
  def output_event_dims(self):
    return self.num_categ


class GaussianDistributionFromMean(tf.keras.Model):
  """Emission model p(x[t] | z[t])."""

  def __init__(self,
               emission_mean_network,
               observation_dim,
               cov_mat=None,
               use_triangular_cov=False,
               use_trainable_cov=True,
               raw_sigma_bias=0.0,
               sigma_min=1e-5,
               sigma_scale=0.05,
               dtype=tf.float32,
               name="GaussianDistributionFromMean"):
    """Construct a `GaussianDistributionFromMean` instance.

    Args:
      emission_mean_network: a `callable` network taking continuous hidden
        states, `z[t]`, and returning the mean of emission distribution,
        `p(x[t] | z[t])`.
      observation_dim: an `int` scalar for dimension of observations, `x`.
      cov_mat: an optional `float` Tensor for predefined covariance matrix.
        Default to `None`, in which case, a `cov` variable will be created.
      use_triangular_cov: a `bool` scalar indicating whether to use triangular
        covariance matrices and `tfp.distributions.MultivariateNormalTriL` for
        distribution. Otherwise, a diagonal covariance matrices and
        `tfp.distributions.MultivariateNormalDiag` will be used.
      use_trainable_cov: a `bool` scalar indicating whether the scale of
        the distribution is trainable. Default to False.
      raw_sigma_bias: a `float` scalar to be added to the raw sigma, which is
        standard deviation of the distribution. Default to `0.`.
      sigma_min: a `float` scalar for minimal level of sigma to prevent
        underflow. Default to `1e-5`.
      sigma_scale: a `float` scalar for scaling the sigma. Default to `0.05`.
        The above three arguments are used as
        `sigma_scale * max(softmax(raw_sigma + raw_sigma_bias), sigma_min))`.
      dtype: data type for variables within the scope. Default to `tf.float32`.
      name: a `str` to construct names of variables.
    """
    super(GaussianDistributionFromMean, self).__init__()
    self.observation_dim = observation_dim
    self.x_emission_net = emission_mean_network
    self.use_triangular_cov = use_triangular_cov

    if cov_mat:
      self.cov_mat = cov_mat
    elif self.use_triangular_cov:
      local_variable = tf.Variable(
          tf.random.uniform(
              shape=[int(self.observation_dim*(self.observation_dim+1)/2)],
              minval=0., maxval=1.,
              dtype=dtype),
          name="{}_cov".format(name),
          dtype=dtype,
          trainable=use_trainable_cov)
      self.cov_mat = tfp.math.fill_triangular(
          local_variable)
      self.cov_mat = tf.maximum(tf.nn.softmax(self.cov_mat + raw_sigma_bias),
                                sigma_min) * sigma_scale
    else:
      self.cov_mat = tf.Variable(
          initial_value=tf.random.uniform(shape=[self.observation_dim],
                                          minval=0.0, maxval=1.,
                                          dtype=dtype),
          name="{}_cov".format(name),
          dtype=dtype,
          trainable=use_trainable_cov)
      self.cov_mat = tf.maximum(tf.nn.softmax(self.cov_mat + raw_sigma_bias),
                                sigma_min) * sigma_scale

  def call(self, input_tensor, dtype=tf.float32):

    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)

    mean_tensor = self.x_emission_net(input_tensor)

    if self.use_triangular_cov:
      output_dist = tfd.MultivariateNormalTriL(
          loc=mean_tensor,
          scale_tril=self.cov_mat)
    else:
      output_dist = tfd.MultivariateNormalDiag(
          loc=mean_tensor,
          scale_diag=self.cov_mat)

    return tfp.experimental.as_composite(output_dist)

  @property
  def output_event_dims(self):
    return self.observation_dim


class RnnInferenceNetwork(tf.keras.Model):
  """Inference network for posterior q(z[1:T] | x[1:T])."""

  def __init__(self,
               posterior_rnn,
               posterior_dist,
               latent_dim,
               embedding_network=None):
    """Construct a `RnnInferenceNetwork` instance.

    Args:
      posterior_rnn: a RNN cell, `h[t]=f_RNN(h[t-1], z[t-1], input[t])`,
        which recursively takes previous step RNN states `h`, previous step
        sampled dynamical state `z[t-1]`, and conditioned input `input[t]`.
      posterior_dist: a distribution instance for `p(z[t] | h[t])`,
        where h[t] is the output of `posterior_rnn`.
      latent_dim: an `int` scalar for dimension of continuous hidden
        states, `z`.
      embedding_network: an optional network to embed the observations, `x[t]`.
        Default to `None`, in which case, no embedding is applied.
    """
    super(RnnInferenceNetwork, self).__init__()
    self.latent_dim = latent_dim
    self.posterior_rnn = posterior_rnn
    self.posterior_dist = posterior_dist

    if embedding_network is None:
      self.embedding_network = lambda x: x
    self.embedding_network = embedding_network

  def call(self,
           inputs,
           num_samples=1,
           dtype=tf.float32,
           random_seed=RANDOM_SEED,
           parallel_iterations=10):
    """Recursively sample z[t] ~ q(z[t]|h[t]=f_RNN(h[t-1], z[t-1], h[t]^b)).

    Args:
      inputs: a float `Tensor` of size [batch_size, num_steps, obs_dim], where
        each observation should be flattened.
      num_samples: an `int` scalar for number of samples per time-step, for
        posterior inference networks, `z[i] ~ q(z[1:T] | x[1:T])`.
      dtype: The data type of input data.
      random_seed: an `Int` as the seed for random number generator.
      parallel_iterations: a positive `Int` indicates the number of iterations
        allowed to run in parallel in `tf.while_loop`, where `tf.while_loop`
        defaults it to be 10.

    Returns:
      sampled_z: a float 3-D `Tensor` of size [num_samples, batch_size,
      num_steps, latent_dim], which stores the z_t sampled from posterior.
      entropies: a float 2-D `Tensor` of size [num_samples, batch_size,
      num_steps], which stores the entropies of posterior distributions.
      log_probs: a float 2-D `Tensor` of size [num_samples. batch_size,
      num_steps], which stores the log posterior probabilities.
    """
    inputs = tf.convert_to_tensor(inputs, dtype_hint=dtype)
    batch_size, num_steps = tf.unstack(tf.shape(inputs)[:2])
    latent_dim = self.latent_dim

    ## passing through embedding_network, e.g. bidirectional RNN
    inputs = self.embedding_network(inputs)

    ## passing through forward RNN
    ta_names = ["rnn_states", "latent_states", "entropies", "log_probs"]
    tas = [tf.TensorArray(tf.float32, num_steps, name=n) for n in ta_names]

    t0 = tf.constant(0, tf.int32)
    loopstate = namedtuple("LoopState", "rnn_state latent_encoded")

    initial_rnn_state = self.posterior_rnn.get_initial_state(
        batch_size=batch_size * num_samples,
        dtype=dtype)
    if (isinstance(self.posterior_rnn, layers.GRUCell)
        or isinstance(self.posterior_rnn, layers.SimpleRNNCell)):
      initial_rnn_state = [initial_rnn_state]

    init_state = (t0,
                  loopstate(
                      rnn_state=initial_rnn_state,
                      latent_encoded=tf.zeros(
                          [batch_size * num_samples, latent_dim],
                          dtype=tf.float32)), tas)

    def _cond(t, *unused_args):
      return t < num_steps

    def _step(t, loop_state, tas):
      """One step in tf.while_loop."""
      prev_latent_state = loop_state.latent_encoded
      prev_rnn_state = loop_state.rnn_state
      current_input = inputs[:, t, :]

      # Duplicate current observation to sample multiple trajectories.
      current_input = tf.tile(current_input, [num_samples, 1])

      rnn_input = tf.concat([current_input, prev_latent_state],
                            axis=-1)  # num_samples * BS, latent_dim+input_dim
      rnn_out, rnn_state = self.posterior_rnn(
          inputs=rnn_input,
          states=prev_rnn_state)
      dist = self.posterior_dist(rnn_out)
      latent_state = dist.sample(seed=random_seed)

      ## rnn_state is a list of [batch_size, rnn_hidden_dim],
      ## after TA.stack(), the dimension will be
      ## [num_steps, 1 for GRU/2 for LSTM, batch, rnn_dim]
      tas_updates = [rnn_state,
                     latent_state,
                     dist.entropy(),
                     dist.log_prob(latent_state)]
      tas = utils.write_updates_to_tas(tas, t, tas_updates)

      return (t+1,
              loopstate(rnn_state=rnn_state,
                        latent_encoded=latent_state),
              tas)
    ## end of _step function

    _, _, tas_final = tf.while_loop(
        _cond, _step, init_state, parallel_iterations=parallel_iterations)

    sampled_z, entropies, log_probs = [
        utils.tensor_for_ta(ta, swap_batch_time=True) for ta in tas_final[1:]
    ]

    sampled_z = tf.reshape(sampled_z,
                           [num_samples, batch_size, num_steps, latent_dim])
    entropies = tf.reshape(entropies, [num_samples, batch_size, num_steps])
    log_probs = tf.reshape(log_probs, [num_samples, batch_size, num_steps])
    return sampled_z, entropies, log_probs


def create_model(num_categ,
                 hidden_dim,
                 observation_dim,
                 config_emission,
                 config_inference,
                 config_z_initial,
                 config_z_transition,
                 network_emission,
                 network_input_embedding,
                 network_posterior_rnn,
                 network_s_transition,
                 networks_z_transition,
                 network_posterior_mlp=lambda x: x,
                 name="snlds"):
  """Construct SNLDS model.

  Args:
    num_categ: an `int` scalar for number of discrete states, `s`.
    hidden_dim: an `int` scalar for dimension of continuous hidden states, `z`.
    observation_dim: an `int` scalar for dimension of observations, `x`.
    config_emission: a `dict` for configuring emission distribution,
      `p(x[t] | z[t])`.
    config_inference: a `dict` for configuring the posterior distribution,
      `q(z[t]|h[t]=f_RNN(h[t-1], z[t-1], h[t]^b))`.
    config_z_initial: a `dict` for configuring the initial distribution of
      continuous hidden state, `p(z[0])`.
    config_z_transition: a `dict` for configuring the transition distribution
      `p(z[t] | z[t-1], s[t])`.
    network_emission: a `callable` network taking continuous hidden
      states, `z[t]`, and returning the mean of emission distribution,
      `p(x[t] | z[t])`.
    network_input_embedding: a `callable` network to embed the observations,
      `x[t]`. E.g. a bidirectional RNN to embedding `x[1:T]`.
    network_posterior_rnn: a RNN cell, `h[t]=f_RNN(h[t-1], z[t-1], input[t])`,
      which recursively takes previous step RNN states `h`, previous step
      sampled dynamical state `z[t-1]`, and conditioned input `input[t]`.
    network_s_transition: a `callable` network taking batch conditional inputs,
      `x[t-1]`, and returning the discrete state transition matrices,
      `log p(s[t] |s[t-1], x[t-1])`.
    networks_z_transition: a list of `callable` networks, with the length
      of list same as `num_categories`. Each one of the networks will take
      previous step hidden state, `z[t-1]`, and returns the mean of
      transition distribution, `p(z[t] | z[t-1], s[t]=i)` for each
      discrete state `i`.
    network_posterior_mlp: an optional network to embedding the output of
      inference RNN networks, before passing into the distribution as mean,
      `q(z[t] | mlp( h[t] ))`. Default to identity mapping.
    name: a `str` to construct names of variables.

  Returns:
    An instance of instantiated `model_base.SwitchingNLDS` model.
  """

  z_transition = ContinuousStateTransition(
      transition_mean_networks=networks_z_transition,
      distribution_dim=hidden_dim,
      num_categories=num_categ,
      cov_mat=config_z_transition.cov_mat,
      use_triangular_cov=config_z_transition.use_triangular_cov,
      use_trainable_cov=config_z_transition.use_trainable_cov,
      raw_sigma_bias=config_z_transition.raw_sigma_bias,
      sigma_min=config_z_transition.sigma_min,
      sigma_scale=config_z_transition.sigma_scale,
      name=name+"_z_trans")

  s_transition = DiscreteStateTransition(
      transition_network=network_s_transition,
      num_categories=num_categ)

  emission_network = GaussianDistributionFromMean(
      emission_mean_network=network_emission,
      observation_dim=observation_dim,
      cov_mat=config_emission.cov_mat,
      use_triangular_cov=config_emission.use_triangular_cov,
      use_trainable_cov=config_emission.use_trainable_cov,
      raw_sigma_bias=config_emission.raw_sigma_bias,
      sigma_min=config_emission.sigma_min,
      sigma_scale=config_emission.sigma_scale,
      name=name+"_x_emit")

  posterior_distribution = GaussianDistributionFromMean(
      emission_mean_network=network_posterior_mlp,
      observation_dim=hidden_dim,
      cov_mat=config_inference.cov_mat,
      use_triangular_cov=config_inference.use_triangular_cov,
      use_trainable_cov=config_inference.use_trainable_cov,
      raw_sigma_bias=config_inference.raw_sigma_bias,
      sigma_min=config_inference.sigma_min,
      sigma_scale=config_inference.sigma_scale,
      name=name+"_posterior")

  posterior_network = RnnInferenceNetwork(
      posterior_rnn=network_posterior_rnn,
      posterior_dist=posterior_distribution,
      latent_dim=hidden_dim,
      embedding_network=network_input_embedding)

  z_initial_distribution = construct_initial_state_distribution(
      latent_dim=hidden_dim,
      num_categ=num_categ,
      use_trainable_cov=config_z_initial.use_trainable_cov,
      use_triangular_cov=config_z_initial.use_triangular_cov,
      raw_sigma_bias=config_z_initial.raw_sigma_bias,
      sigma_min=config_z_initial.sigma_min,
      sigma_scale=config_z_initial.sigma_scale,
      name="init_dist")

  snlds_model = model_base.SwitchingNLDS(
      continuous_transition_network=z_transition,
      discrete_transition_network=s_transition,
      emission_network=emission_network,
      inference_network=posterior_network,
      initial_distribution=z_initial_distribution,
      continuous_state_dim=None,
      num_categories=None,
      discrete_state_prior=None)

  return snlds_model
