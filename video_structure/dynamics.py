# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Dynamics-related components of the structured video representation model.

These components model the dynamics of the keypoints extracted by the vision
components.
"""


import tensorflow.compat.v1 as tf

layers = tf.keras.layers


def build_vrnn(cfg):
  """Builds the VRNN dynamics model.

  The model takes observed keypoints from the keypoint detector as input and
  returns keypoints decoded from the model's latent belief. The model only uses
  the observed keypoints for the first cfg.observed_steps steps. For remaining
  cfg.predicted_steps steps, it predicts keypoints using only the dynamics
  model.

  Args:
    cfg: Hyperparameter ConfigDict.

  Returns:
    A tf.keras.Model and the KL loss tensor.
  """

  # Build model components. All of the weights are shared across observed and
  # predicted timesteps:
  rnn_cell = layers.GRUCell(cfg.num_rnn_units)
  prior_net = build_prior_net(cfg)
  posterior_net = build_posterior_net(cfg)
  decoder = build_decoder(cfg)
  scheduled_sampler_obs = ScheduledSampling(
      p_true_start=cfg.scheduled_sampling_p_true_start_obs,
      p_true_end=cfg.scheduled_sampling_p_true_end_obs,
      ramp_steps=cfg.scheduled_sampling_ramp_steps)
  scheduled_sampler_pred = ScheduledSampling(
      p_true_start=cfg.scheduled_sampling_p_true_start_pred,
      p_true_end=cfg.scheduled_sampling_p_true_end_pred,
      ramp_steps=cfg.scheduled_sampling_ramp_steps)

  # Format inputs:
  num_timesteps = cfg.observed_steps + cfg.predicted_steps
  input_keypoints_stack = tf.keras.Input(
      (num_timesteps, cfg.num_keypoints, 3), name='vrnn_input')

  input_keypoints_list = layers.Lambda(lambda x: tf.unstack(x, axis=1))(
      input_keypoints_stack)

  # Initialize loop variables:
  rnn_state = rnn_cell.get_initial_state(
      batch_size=tf.shape(input_keypoints_stack)[0], dtype=tf.float32)
  output_keypoints_list = [None] * num_timesteps
  kl_div_list = [None] * cfg.observed_steps

  # Process observed steps:
  for t in range(cfg.observed_steps):
    output_keypoints_list[t], rnn_state, kl_div_list[t] = _vrnn_iteration(
        cfg, input_keypoints_list[t], rnn_state, rnn_cell, prior_net, decoder,
        scheduled_sampler_obs, posterior_net)

  # Process predicted steps:
  for t in range(cfg.observed_steps, num_timesteps):
    output_keypoints_list[t], rnn_state, _ = _vrnn_iteration(
        cfg, input_keypoints_list[t], rnn_state, rnn_cell, prior_net, decoder,
        scheduled_sampler_pred)

  output_keypoints_stack = layers.Lambda(lambda x: tf.stack(x, axis=1))(
      output_keypoints_list)
  kl_div_stack = layers.Lambda(lambda x: tf.stack(x, axis=1))(kl_div_list)

  return tf.keras.Model(
      inputs=input_keypoints_stack,
      outputs=[output_keypoints_stack, kl_div_stack],
      name='vrnn')


def _vrnn_iteration(cfg,
                    input_keypoints,
                    rnn_state,
                    rnn_cell,
                    prior_net,
                    decoder,
                    scheduled_sampler,
                    posterior_net=None):
  """Performs one timestep of the VRNN.

  Args:
    cfg: ConfigDict with model hyperparameters.
    input_keypoints: [batch_size, num_keypoints, 3] tensor (one timestep of
      the sequence returned by the keypoint detector).
    rnn_state: Previous recurrent state.
    rnn_cell: A Keras RNN cell object (e.g. tf.layers.GRUCell) that holds the
      dynamics model.
    prior_net: A tf.keras.Model that computes the prior latent belief from the
      previous RNN state.
    decoder: A tf.keras.Model that decodes the latent belief into keypoints.
    scheduled_sampler: Keras layer instance that performs scheduled sampling.
    posterior_net: (Optional) A tf.keras.Model that computes the posterior
      latent belief, given observed keypoints and the previous RNN state. If no
      posterior_net is supplied, prior latent belief is used for predictions.

  Returns:
    Three tensors: The output keypoints, the new RNN state, and the KL
    divergence between the prior and posterior (None if no posterior_net is
    provided).
  """

  shape = input_keypoints.shape.as_list()[1:]
  observed_keypoints_flat = layers.Reshape([shape[0] * shape[1]])(
      input_keypoints)

  # Obtain parameters mean, std for the latent belief distibution:
  mean_prior, std_prior = prior_net(rnn_state)
  if posterior_net:
    mean, std = posterior_net([rnn_state, observed_keypoints_flat])
    kl_divergence = KLDivergence(cfg.kl_annealing_steps)(
        [mean_prior, std_prior, mean, std])
  else:
    # Having no posterior_net means that this cell is used to make predictions
    # based on the prior only, without having access to observations. In this
    # case, no posterior is available to compute the KL term of the
    # variational objective. We therefore cannot train the prior net during
    # predicted steps. Since a reconstruction error is still generated, we
    # need to stop the gradients explicitly to ensure the prior net is not
    # updated based on these errors:
    mean = layers.Lambda(tf.stop_gradient)(mean_prior)
    std = layers.Lambda(tf.stop_gradient)(std_prior)
    kl_divergence = None

  # Sample a belief from the distribution and decode it into keypoints:
  sampler = SampleBestBelief(
      cfg.num_samples_for_bom,
      decoder,
      use_mean_instead_of_sample=cfg.use_deterministic_belief)
  latent_belief, output_keypoints_flat = sampler(
      [mean, std, rnn_state, observed_keypoints_flat])
  output_keypoints = layers.Reshape(shape)(output_keypoints_flat)

  # TODO(mjlm): Think through where we need stop_gradients.

  # Step the RNN forward:
  keypoints_for_rnn = scheduled_sampler([
      observed_keypoints_flat, output_keypoints_flat])

  rnn_input = layers.Concatenate(axis=-1)([keypoints_for_rnn, latent_belief])
  _, rnn_state = rnn_cell(rnn_input, [rnn_state])
  rnn_state = rnn_state[0]  # rnn_cell needs state to be wrapped in list.

  return output_keypoints, rnn_state, kl_divergence


def build_prior_net(cfg):
  """Computes the prior belief over current keypoints, given past information.

  rnn_state[t-1] --> prior_mean[t], prior_std[t]

  Args:
    cfg: Hyperparameter ConfigDict.

  Returns:
    Keras Model object.
  """
  rnn_state = tf.keras.Input(shape=[cfg.num_rnn_units], name='rnn_state')
  hidden = layers.Dense(cfg.prior_net_dim, **cfg.dense_layer_kwargs)(rnn_state)
  means = layers.Dense(cfg.latent_code_size, name='means')(hidden)
  stds_raw = layers.Dense(cfg.latent_code_size)(hidden)
  stds = layers.Lambda(
      lambda x: tf.nn.softplus(x) + 1e-4, name='stds')(stds_raw)
  return tf.keras.Model(inputs=rnn_state, outputs=[means, stds], name='prior')


def build_decoder(cfg):
  """Decodes keypoints from the latent belief.

  rnn_state[t-1], latent_code[t] --> keypoints[t]

  Args:
    cfg: Hyperparameter ConfigDict.

  Returns:
    Keras Model object.
  """
  rnn_state = tf.keras.Input(shape=[cfg.num_rnn_units], name='rnn_state')
  latent_code = tf.keras.Input(shape=[cfg.latent_code_size], name='latent_code')
  hidden = layers.Concatenate()([rnn_state, latent_code])
  hidden = layers.Dense(128, **cfg.dense_layer_kwargs)(hidden)
  keypoints = layers.Dense(cfg.num_keypoints * 3, activation=tf.nn.tanh)(
      hidden)
  return tf.keras.Model(
      inputs=[rnn_state, latent_code], outputs=keypoints, name='decoder')


def build_posterior_net(cfg):
  """Incorporates observed information into the latent belief.

  rnn_state[t-1], observed_keypoints[t] --> posterior_mean[t], posterior_std[t]

  Args:
    cfg: Hyperparameter ConfigDict.

  Returns:
    Keras Model object.
  """
  rnn_state = tf.keras.Input(shape=[cfg.num_rnn_units], name='rnn_state')
  keypoints = tf.keras.Input(shape=[cfg.num_keypoints * 3], name='keypoints')
  hidden = layers.Concatenate()([rnn_state, keypoints])
  hidden = layers.Dense(cfg.posterior_net_dim, **cfg.dense_layer_kwargs)(hidden)
  means = layers.Dense(cfg.latent_code_size, name='means')(hidden)
  stds_raw = layers.Dense(cfg.latent_code_size)(hidden)
  stds = layers.Lambda(
      lambda x: tf.nn.softplus(x) + 1e-4, name='stds')(stds_raw)
  return tf.keras.Model(
      inputs=[rnn_state, keypoints], outputs=[means, stds], name='posterior')


class TrainingStepCounter(layers.Layer):
  """Provides a class attribute that contains the training step count."""

  def __init__(self, **kwargs):
    self.uses_learning_phase = True
    super().__init__(**kwargs)

  def build(self, input_shape):
    self.train_step = self.add_weight(
        name='train_step', shape=[], initializer='zeros', trainable=False)
    increment = tf.cast(tf.keras.backend.learning_phase(), tf.float32)
    increment_op = tf.assign_add(self.train_step, increment)
    self.add_update(increment_op)
    super().build(input_shape)

  def reset_states(self):
    self.train_step.set_value(0)


class KLDivergence(TrainingStepCounter):
  """Returns the KL divergence between the prior and posterior distributions.

  Attributes:
    kl_annealing_steps: The returned KL divergence value will be linearly
      annealed from 0 to the final value over this many training steps.
  """

  def __init__(self, kl_annealing_steps=0, **kwargs):
    self.kl_annealing_steps = kl_annealing_steps
    super().__init__(**kwargs)

  def call(self, inputs):
    mean_prior, std_prior, mean, std = inputs
    prior = tf.distributions.Normal(mean_prior, std_prior)
    posterior = tf.distributions.Normal(mean, std)
    kl_div = tf.distributions.kl_divergence(posterior, prior)
    kl_div = tf.reduce_sum(kl_div, axis=-1)  # Sum over distribution dimensions.
    if self.kl_annealing_steps:
      kl_div *= tf.minimum(self.train_step / self.kl_annealing_steps, 1.0)
    return kl_div


class ScheduledSampling(TrainingStepCounter):
  """Keras layer that implements scheduled sampling for teacher forcing.

  See https://arxiv.org/abs/1506.03099.

  For training an RNN, teacher forcing (i.e. providing the ground-truth inputs,
  rather than the previous RNN outputs) can stabilize training. However, this
  means that the RNN input distribution will be different during training and
  inference. Scheduled sampling randomly mixes ground truth and RNN predictions
  during training, slowly ramping down the ground-truth probablility as training
  progresses. Thereby, training is stabilized initially and then becomes
  gradually more realistic.

  This layer implements a linear schedule.

  To disable scheduled sampling, set p_true_start and p_true_end to 0.

  Scheduled sampling is only applied during the learning phase (i.e. when
  tf.keras.backend.learning_phase() is True). During testing, the layer always
  returns the "pred" (i.e. second) input tensor.

  Attributes:
    p_true_start: Initial probability of sampling the "true" input.
    p_true_end: Final probability of sampling the "true" input.
    ramp_steps: Number of training steps over which the output will ramp from
        p_true_start to p_true_end.

  Returns:
    Tensor containing either the true or the predicted input.
  """

  def __init__(
      self, p_true_start=1.0, p_true_end=0.2, ramp_steps=10000, **kwargs):
    self.ramp_steps = ramp_steps
    self.p_true_start = p_true_start
    self.p_true_end = p_true_end
    super().__init__(**kwargs)

  def call(self, inputs):
    """Inputs should be [true, pred], each with size [batch, ...]."""

    true, pred = inputs

    # Compute current probability of choosing the ground truth:
    ramp = self.train_step / self.ramp_steps
    ramp = tf.minimum(ramp, 1.0)
    p_true = self.p_true_start - (self.p_true_start - self.p_true_end) * ramp

    # Flip a coin based on p_true:
    return_true = tf.less(tf.random.uniform([]), p_true)

    # During testing, use `pred` tensor (i.e. no teacher forcing):
    return_true = tf.keras.backend.in_train_phase(return_true, False)

    return tf.keras.backend.switch(return_true, true, pred)


class SampleBestBelief(layers.Layer):
  """Chooses the best keypoints from a number of latent belief samples.

  This layer implements the "best of many" sample objective proposed in
  https://arxiv.org/abs/1806.07772.

  "Best" is defined to mean closest in Euclidean distance to the keypoints
  observed by the vision model.

  Attributes:
    num_samples: Number of samples to choose the best from.
    coordinate_decoder: tf.keras.Model object that decodes the latent belief
      into keypoints.
    use_mean_instead_of_sample: If true, do not sample, but just use the mean of
      the latent belief distribution.
  """

  def __init__(self,
               num_samples,
               coordinate_decoder,
               use_mean_instead_of_sample=False,
               **kwargs):
    self.num_samples = num_samples
    self.coordinate_decoder = coordinate_decoder
    self.use_mean_instead_of_sample = use_mean_instead_of_sample
    self.uses_learning_phase = True
    super().__init__(**kwargs)

  def call(self, inputs):
    latent_mean, latent_std, rnn_state, observed_keypoints_flat = inputs

    # Draw latent samples:
    if self.use_mean_instead_of_sample:
      sampled_latent = tf.stack([latent_mean] * self.num_samples)
    else:
      distribution = tf.distributions.Normal(loc=latent_mean, scale=latent_std)
      sampled_latent = distribution.sample(sample_shape=(self.num_samples,))
    sampled_latent_list = tf.unstack(sampled_latent)

    # Decode samples into coordinates:
    # sampled_keypoints has shape [num_samples, batch_size, 3 * num_keypoints].
    sampled_keypoints = tf.stack([
        self.coordinate_decoder([rnn_state, latent])
        for latent in sampled_latent_list
    ])

    # If we have only 1 sample, we can just return that:
    if self.num_samples == 1:
      return [sampled_latent_list[0], sampled_keypoints[0]]

    # Compute L2 prediction loss for all samples (note that this includes both
    # the x,y-coordinates and the keypoint scale):
    sample_losses = tf.reduce_mean(
        (sampled_keypoints - observed_keypoints_flat[tf.newaxis, Ellipsis])**2.0,
        axis=-1)  # Mean across keypoints.

    # Choose the sample based on the loss:
    return _choose_sample(sampled_latent, sampled_keypoints, sample_losses)

  def compute_output_shape(self, input_shape):
    return [input_shape[-1], input_shape[0]]


def _choose_sample(sampled_latent, sampled_keypoints, sample_losses):
  """Returns the first or lowest-loss sample, depending on learning phase.

  During training, the sample with the lowest loss is returned.
  During inference, the first sample is returned without regard to the loss.

  Args:
    sampled_latent: [num_samples, batch_size, latent_code_size] tensor.
    sampled_keypoints: [num_samples, batch_size, 3 * num_keypoints] tensor.
    sample_losses: [num_samples, batch_size] tensor.

  Returns:
    Two tensors: latent and keypoint representation of the best sample.
  """

  # Find the indices of the samples with the lowest loss:
  best_sample_ind = tf.argmin(sample_losses, axis=0)  # Shape is [batch_size].
  best_sample_ind = tf.cast(best_sample_ind, tf.int32)
  batch_ind = tf.range(tf.shape(sampled_latent)[1], dtype=tf.int32)
  indices = tf.stack([best_sample_ind, batch_ind], axis=-1)

  # Only keep the best keypoints and latent sample:
  best_latent = tf.gather_nd(sampled_latent, indices)
  best_keypoints = tf.gather_nd(sampled_keypoints, indices)

  # During training, return the best sample. During inference, return the
  # first sample:
  return [
      tf.keras.backend.in_train_phase(best_latent, sampled_latent[0]),
      tf.keras.backend.in_train_phase(best_keypoints, sampled_keypoints[0]),
  ]
