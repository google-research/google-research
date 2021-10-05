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

"""TODO(bmazoure): Helper methods for Tf code.

This file contains helper functions relying on Tensorflow, e.g. Tensorboard log
parsing into Pandas

Includes DrQ-style data augmentations
"""

from functools import partial  # pylint: disable=g-importing-member
import pickle
import typing
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import scipy.stats as stats
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents import trajectories
from tf_agents.environments import py_environment
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from tf_agents.utils import example_encoding
from tf_agents.utils import example_encoding_dataset

K = tf.keras.backend
Layer = tf.keras.layers.Layer


def create_dataset_for_given_tfrecord(
    record_path,
    load_buffer_size=None,
    shuffle_buffer_size=1,
    num_parallel_reads=100,
    decoder=None,
):
  """Creates TFDataset based on a given TFRecord file.

  TFRecords contain agent experience (as saved by the tf_agents endpoints).

  Args:
    record_path: path to TFRecord
    load_buffer_size: Buffer size
    shuffle_buffer_size: Size of shuffle
    num_parallel_reads: How many workers to use for loading
    decoder: decoder
  Returns:
    dataset: dataset
  """
  dataset = example_encoding_dataset.load_tfrecord_dataset(
      record_path,
      buffer_size=load_buffer_size,
      as_trajectories=True,
      as_experience=True,
      add_batch_dim=False,
      compress_image=True,
      num_parallel_reads=num_parallel_reads,
      decoder=decoder,
  )
  # Add dummy info field for experience.
  if shuffle_buffer_size > 1:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def convert_obs_to_float(batch):
  return trajectories.trajectory.Trajectory(
      step_type=batch.step_type,
      observation=tf.cast(batch.observation, dtype=tf.float32) / 255.0,
      action=batch.action,
      policy_info=batch.policy_info,
      next_step_type=batch.next_step_type,
      reward=batch.reward,
      discount=batch.discount,
  )


def create_data_iterator(
    record_paths_pattern,
    batch_size,
    shuffle_buffer_size_per_tfrecord=1,
    shuffle_buffer_size=10_000,
    num_shards=50,
    cycle_length=tf.data.experimental.AUTOTUNE,
    block_length=10,  # in rl unplugged they used 5
    obs_to_float=False,
):
  """Creates data iterator.

  Args:
    record_paths_pattern: path to TFRecord
    batch_size: batch size
    shuffle_buffer_size_per_tfrecord: Buffer size
    shuffle_buffer_size: Size of shuffle
    num_shards: Number of shards
    cycle_length: Cycle length
    block_length: Block length
    obs_to_float: Convert obs to float
  Returns:
    dataset: dataset iterator
  """
  del cycle_length
  record_paths = tf.io.gfile.glob(record_paths_pattern)
  record_paths = [x for x in record_paths if not x.endswith('.spec')]

  initial_len = len(record_paths)
  remainder = initial_len % num_shards
  for _ in range(num_shards - remainder):
    record_paths.append(record_paths[np.random.randint(low=0,
                                                       high=initial_len)])
  record_paths = np.array(record_paths)
  np.random.shuffle(record_paths)
  record_paths = np.array_split(record_paths, num_shards)

  record_file_ds = tf.data.Dataset.from_tensor_slices(record_paths)
  record_file_ds = record_file_ds.repeat().shuffle(len(record_paths))

  spec_path = record_paths[0][0] + example_encoding_dataset._SPEC_FILE_EXTENSION  # pylint: disable=protected-access
  record_spec = example_encoding_dataset.parse_encoded_spec_from_file(spec_path)
  decoder = example_encoding.get_example_decoder(
      record_spec,
      compress_image=True)

  example_ds = record_file_ds.interleave(
      partial(
          create_dataset_for_given_tfrecord,
          load_buffer_size=100000000,
          shuffle_buffer_size=shuffle_buffer_size_per_tfrecord,
          num_parallel_reads=1,
          decoder=decoder,
      ),
      cycle_length=10,
      block_length=block_length,
      num_parallel_calls=10,
  )
  example_ds = example_ds.shuffle(shuffle_buffer_size)
  example_ds = example_ds.batch(batch_size)
  if obs_to_float:
    example_ds = example_ds.map(
        convert_obs_to_float,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    example_ds = example_ds.prefetch(tf.data.experimental.AUTOTUNE)
  example_ds = example_ds.prefetch(10)
  return iter(example_ds)


class TfAgentsPolicy():
  """Policy wrapper.
  """

  def __init__(self, policy):
    self.policy = policy

  def log_probs(self, states, actions):
    ts_ = trajectories.TimeStep(
        tf.stack([trajectories.StepType.MID] * 256, 0),
        tf.constant([1.] * 256, dtype=tf.float32),
        tf.constant([1.] * 256, dtype=tf.float32), tf.cast(states, tf.float32))
    dist = self.policy.distribution(ts_)
    return dist.action.log_prob(actions)


class PerLevelCriticActor():
  """Wrapper to allow acting with per-level critic.
  """

  def __init__(self, critic, encoder, num_augmentations, action_dim):
    self.critic = critic
    self.encoder = encoder
    self.num_augmentations = num_augmentations
    self.action_dim = action_dim
    self.parallel = not isinstance(critic, list)

  def act(self, states, level_ids=None, data_aug=False):
    """Act from a batch of states.

    Args:
      states: batch of states
      level_ids: optional batch of level ids (Procgen only)
      data_aug: optional flag
    Returns:
      actions
    """
    del data_aug
    if self.num_augmentations > 0:
      # use pad of 2 to bump 64 to 68 with 2 + 64 + 2 on each side
      img_pad = 2
      paddings = tf.constant(
          [[0, 0], [img_pad, img_pad], [img_pad, img_pad], [0, 0]],
          dtype=tf.int32)
      states = tf.cast(
          tf.pad(tf.cast(states * 255., tf.int32), paddings, 'SYMMETRIC'),
          tf.float32) / 255.
    features = self.encoder(states)
    if self.parallel:
      # n_batch x 200 x 15
      q1, q2 = self.critic(features, actions=None)
    else:
      @tf.function
      def compute_q():
        q1_all, q2_all = [], []
        for i in range(200):
          q1, q2 = self.critic[i](features, actions=None)
          q1_all.append(q1)
          q2_all.append(q2)
        q1_all = tf.concat(q1_all, 1)
        q2_all = tf.concat(q2_all, 1)
        return q1_all, q2_all

      q1, q2 = compute_q()
    # n_batch x 200 x 15
    q = tf.minimum(q1, q2)
    # n_batch x 15
    if level_ids is None:
      q_avg = tf.reduce_mean(tf.reshape(q, (-1, 200, self.action_dim)), 1)
    else:
      q_avg = tf.reshape(
          q[:, level_ids * self.action_dim:(level_ids + 1) * self.action_dim],
          (1, -1))
    # Take action wrt average Q-value per levels
    actions = tf.argmax(q_avg, -1)
    return actions

# Layer / network definitions


class EmbedNet(tf.keras.Model):
  """An embed network."""

  def __init__(self,
               state_dim,
               embedding_dim = 256,
               num_distributions=None,
               hidden_dims=(256, 256)):
    """Creates a neural net.

    Args:
      state_dim: State size.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions for discrete
        embedding.
      hidden_dims: List of hidden dimensions.
    """
    super().__init__()

    inputs = tf.keras.Input(shape=(state_dim,))
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    assert not num_distributions or embedding_dim % num_distributions == 0
    self.embedder = create_mlp(
        inputs.shape[-1],
        self.embedding_dim,
        hidden_dims=hidden_dims,
        activation=tf.nn.swish,
        near_zero_last_layer=bool(num_distributions))

  @tf.function
  def call(self, states, stop_gradient = True):
    """Returns embeddings of states.

    Args:
      states: A batch of states.
      stop_gradient: Whether to put a stop_gradient on embedding.

    Returns:
      Embeddings of states.
    """
    if not self.num_distributions:
      out = self.embedder(states)
    else:
      all_logits = self.embedder(states)
      all_logits = tf.split(
          all_logits, num_or_size_splits=self.num_distributions, axis=-1)
      all_probs = [tf.nn.softmax(logits, -1) for logits in all_logits]
      joined_probs = tf.concat(all_probs, -1)
      all_samples = [
          tfp.distributions.Categorical(logits=logits).sample()
          for logits in all_logits
      ]
      all_onehot_samples = [
          tf.one_hot(samples, self.embedding_dim // self.num_distributions)
          for samples in all_samples
      ]
      joined_onehot_samples = tf.concat(all_onehot_samples, -1)

      # Straight-through gradients.
      out = joined_onehot_samples + joined_probs - tf.stop_gradient(
          joined_probs)

    if stop_gradient:
      return tf.stop_gradient(out)
    return out


class Bilinear(Layer):
  """Bilinear layer f(x,y)=xW^Ty .
  """

  def __init__(self, output_dim, input_dim=None, **kwargs):
    self.output_dim = output_dim
    self.input_dim = input_dim
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(Bilinear, self).__init__(**kwargs)

  def build(self, input_shape):
    mean = 0.0
    std = 1.0
    # W : k*d*d
    k = self.output_dim
    d = self.input_dim
    initial_w_values = stats.truncnorm.rvs(
        -2 * std, 2 * std, loc=mean, scale=std, size=(k, d, d))
    initial_v_values = stats.truncnorm.rvs(
        -2 * std, 2 * std, loc=mean, scale=std, size=(2 * d, k))
    self.w = tf.Variable(initial_w_values, trainable=True, dtype=tf.float32)
    self.v = tf.Variable(initial_v_values, trainable=True, dtype=tf.float32)
    self.b = K.zeros((self.input_dim,))
    # self.trainable_weights = [self.W, self.V, self.b]

  def call(self, e1, e2, mask=None):
    # e1 = inputs[0]
    # e2 = inputs[1]
    batch_size = K.shape(e1)[0]
    k = self.output_dim
    # print([e1,e2])
    feed_forward_product = K.dot(K.concatenate([e1, e2]), self.v)
    # print(feed_forward_product)
    bilinear_tensor_products = [
        K.sum((e2 * K.dot(e1, self.w[0])) + self.b, axis=1)
    ]
    # print(bilinear_tensor_products)
    for i in range(k)[1:]:
      btp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
      bilinear_tensor_products.append(btp)
    result = K.reshape(
        K.concatenate(bilinear_tensor_products, axis=0),
        (batch_size, k)) + feed_forward_product  # K.tanh( )
    # print(result)
    return result

  def compute_output_shape(self, input_shape):
    # print (input_shape)
    batch_size = input_shape[0][0]
    return (batch_size, self.output_dim)


class RNNEmbedNet(tf.keras.Model):
  """An RNN embed network."""

  def __init__(self,
               input_dim,
               embedding_dim,
               num_distributions=None,
               return_sequences=False):
    """Creates a neural net.

    Args:
      input_dim: Size of inputs
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions for discrete
        embedding.
      return_sequences: Whether to return the entire sequence embedding.
    """
    super().__init__()

    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    assert not num_distributions or embedding_dim % num_distributions == 0

    inputs = tf.keras.Input(shape=input_dim)
    outputs = tf.keras.layers.LSTM(
        embedding_dim, return_sequences=return_sequences)(
            inputs)
    self.embedder = tf.keras.Model(inputs=inputs, outputs=outputs)
    self.embedder.call = tf.function(self.embedder.call)

  @tf.function
  def call(self, states, stop_gradient = True):
    """Returns embeddings of states.

    Args:
      states: A batch of sequence of states].
      stop_gradient: Whether to put a stop_gradient on embedding.

    Returns:
      Auto-regressively computed Embeddings of the last states.
    """
    assert len(states.shape) == 3
    if not self.num_distributions:
      out = self.embedder(states)
    else:
      all_logits = self.embedder(states)
      all_logits = tf.split(
          all_logits, num_or_size_splits=self.num_distributions, axis=-1)
      all_probs = [tf.nn.softmax(logits, -1) for logits in all_logits]
      joined_probs = tf.concat(all_probs, -1)
      all_samples = [
          tfp.distributions.Categorical(logits=logits).sample()
          for logits in all_logits
      ]
      all_onehot_samples = [
          tf.one_hot(samples, self.embedding_dim // self.num_distributions)
          for samples in all_samples
      ]
      joined_onehot_samples = tf.concat(all_onehot_samples, -1)

      # Straight-through gradients.
      out = joined_onehot_samples + joined_probs - tf.stop_gradient(
          joined_probs)

    if stop_gradient:
      return tf.stop_gradient(out)
    return out


class StochasticEmbedNet(tf.keras.Model):
  """A stochastic embed network."""

  def __init__(self,
               state_dim,
               embedding_dim = 256,
               hidden_dims=(256, 256),
               num_distributions=None,
               logvar_min = -4.0,
               logvar_max = 15.0):
    """Creates a neural net.

    Args:
      state_dim: State size.
      embedding_dim: Embedding size.
      hidden_dims: List of hidden dimensions.
      num_distributions: Number of categorical distributions for discrete
        embedding.
      logvar_min: Minimum allowed logvar.
      logvar_max: Maximum allowed logvar.
    """
    super().__init__()

    inputs = tf.keras.Input(shape=(state_dim,))
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    assert not num_distributions or embedding_dim % num_distributions == 0

    distribution_dim = (2 if not num_distributions else 1) * self.embedding_dim
    self.embedder = create_mlp(
        inputs.shape[-1],
        distribution_dim,
        hidden_dims=hidden_dims,
        activation=tf.nn.swish,
        near_zero_last_layer=False)
    self.logvar_min = logvar_min
    self.logvar_max = logvar_max

  @tf.function
  def call(self,
           states,
           stop_gradient = True,
           sample = True,
           sample_and_raw_output = False):
    """Returns embeddings of states.

    Args:
      states: A batch of states.
      stop_gradient: Whether to put a stop_gradient on embedding.
      sample: Whether to sample an embedding.
      sample_and_raw_output: Whether to return the original probability in
        addition to sampled embeddings.

    Returns:
      Embeddings of states.
    """
    if not self.num_distributions:
      mean_and_logvar = self.embedder(states)
      mean, logvar = tf.split(mean_and_logvar, 2, axis=-1)
      logvar = tf.clip_by_value(logvar, self.logvar_min, self.logvar_max)
      sample_out = mean + tf.random.normal(tf.shape(mean)) * tf.exp(
          0.5 * logvar)
      raw_out = tf.concat([mean, logvar], -1)
    else:
      all_logits = self.embedder(states)
      all_logits = tf.split(
          all_logits, num_or_size_splits=self.num_distributions, axis=-1)
      all_probs = [tf.nn.softmax(logits, -1) for logits in all_logits]
      joined_probs = tf.concat(all_probs, -1)
      all_samples = [
          tfp.distributions.Categorical(logits=logits).sample()
          for logits in all_logits
      ]
      all_onehot_samples = [
          tf.one_hot(samples, self.embedding_dim // self.num_distributions)
          for samples in all_samples
      ]
      joined_onehot_samples = tf.concat(all_onehot_samples, -1)

      # Straight-through gradients.
      sample_out = joined_onehot_samples + joined_probs - tf.stop_gradient(
          joined_probs)
      raw_out = joined_probs

    if sample_and_raw_output:
      out = (sample_out, raw_out)
    elif sample:
      out = sample_out
    else:
      out = raw_out

    if stop_gradient:
      if hasattr(out, '__len__'):
        return tuple(map(tf.stop_gradient, out))
      return tf.stop_gradient(out)
    return out


# Helper method definition (data aug, moving average, numpy dataset, etc)


def create_mlp(
    input_dim,
    output_dim,
    hidden_dims = (256, 256),
    activation = tf.nn.relu,
    near_zero_last_layer = True,
    normalize_last_layer = False,
):
  """Creates an MLP.

  Args:
    input_dim: input dimensionaloty.
    output_dim: output dimensionality.
    hidden_dims: hidden layers dimensionality.
    activation: activations after hidden units.
    near_zero_last_layer: init for last layer
    normalize_last_layer: normalize last layer?

  Returns:
    An MLP model.
  """
  initialization = tf.keras.initializers.VarianceScaling(
      scale=0.333, mode='fan_in', distribution='uniform')
  near_zero_initialization = tf.keras.initializers.VarianceScaling(
      scale=1e-2, mode='fan_in', distribution='uniform')
  last_layer_initialization = (
      near_zero_initialization if near_zero_last_layer else initialization)

  layers = []
  for hidden_dim in hidden_dims:
    layers.append(
        tf.keras.layers.Dense(
            hidden_dim,
            activation=activation,
            kernel_initializer=initialization))
  layers += [
      tf.keras.layers.Dense(
          output_dim, kernel_initializer=last_layer_initialization)
  ]
  if normalize_last_layer:
    layers += [tf.keras.layers.LayerNormalization(epsilon=1e-6)]

  inputs = tf.keras.Input(shape=input_dim)
  outputs = tf.keras.Sequential(layers)(inputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.call = tf.function(model.call)
  return model


class GroupedRunningPercentile:
  """Running percentile estimator in Tf by groups (e.g. levels).
  """

  def __init__(self, percentiles, n_groups, step=0.1):
    self.percentile_estimators = [
        RunningPercentile(percentiles, step) for _ in range(n_groups)
    ]

  def push(self, observation, group):
    return self.percentile_estimators[group].push(observation)


class RunningPercentile:
  """Running percentile estimator in Tf.
  """

  def __init__(self, percentiles, step=0.1):
    self.n_q = len(percentiles)
    self.steps = step * tf.ones(shape=(self.n_q))
    self.step_up = 1.0 - tf.constant(percentiles)
    self.step_down = tf.constant(percentiles)
    self.x = None

  def push(self, observation):
    """Push observation into estimator.

    Args:
      observation: tf.tensor
    Returns:
      None
    """
    if self.x is None:
      self.x = observation * tf.ones(shape=(self.n_q))
      return
    mask = tf.cast(self.x > observation, tf.float32)
    self.x = (
        self.x - mask * (self.steps * self.step_up) +
        (1 - mask) * self.steps * self.step_down)
    halfstep_mask = tf.cast(
        tf.abs(self.x - observation) < self.steps, tf.float32)
    self.steps = self.steps / tf.maximum(halfstep_mask * 2.0, 1.0)


def image_aug(obs,  # pylint: disable=dangerous-default-value
              next_obs,
              img_pad=4,
              num_augmentations=2,
              obs_dim=84,
              channels=9,
              cropped_shape=[256, 64, 64, 3]):
  """Padding and cropping."""
  paddings = tf.constant(
      [[0, 0], [img_pad, img_pad], [img_pad, img_pad], [0, 0]], dtype=tf.int32)
  obs.set_shape([cropped_shape[0], obs_dim, obs_dim, channels])
  next_obs.set_shape([cropped_shape[0], obs_dim, obs_dim, channels])

  # cropped_shape = obs.shape
  # cropped_shape[0] = 256
  # The reference uses ReplicationPad2d in pytorch, but it is not available
  # in tf. Use 'SYMMETRIC' instead.
  if obs.dtype == tf.float32:
    obs = obs * 255.
    next_obs = next_obs * 255.

  obs = tf.pad(tf.cast(obs, tf.int32), paddings, 'SYMMETRIC')
  next_obs = tf.pad(tf.cast(next_obs, tf.int32), paddings, 'SYMMETRIC')

  def get_random_crop(padded_obs):
    return tf.image.random_crop(padded_obs, cropped_shape)

  augmented_obs = []
  augmented_next_obs = []

  for _ in range(num_augmentations):
    augmented_obs.append(tf.cast(get_random_crop(obs), tf.float32) / 255.)
    augmented_next_obs.append(
        tf.cast(get_random_crop(next_obs), tf.float32) / 255.)

  return augmented_obs, augmented_next_obs


class NumpyObserver():
  """Observer class for NumPy.
  """

  def __init__(self, shard_fn, env):
    self.env = env
    self.obs_shape = env.observation_spec().shape
    self.action_shape = env.action_spec().shape[0]
    self.shard_fn = shard_fn
    self.allocated = False

  def allocate_arrays(self, capacity):
    """Allocate buffer memory.

    Args:
      capacity: Length of array
    """
    self.capacity = capacity
    # the proprioceptive obs is stored as float32, pixels obs as uint8
    obs_dtype = np.float32 if len(self.obs_shape) == 1 else np.uint8

    self.obses = np.empty((capacity, *self.obs_shape), dtype=obs_dtype)
    self.next_obses = np.empty((capacity, *self.obs_shape), dtype=obs_dtype)
    self.actions = np.empty((capacity, self.action_shape), dtype=np.float32)
    self.rewards = np.empty((capacity, 1), dtype=np.float32)
    self.discounts = np.empty((capacity, 1), dtype=np.float32)

    self.idx = 0
    self.full = False

    self.allocated = True

  def __call__(self, trajectory):  # obs, action, reward, next_obs, discount):
    assert self.allocated, 'You need to call .allocate_arrays(SIZE) first'
    np.copyto(self.obses[self.idx], trajectory.observation[0])
    np.copyto(self.actions[self.idx], trajectory.action)
    np.copyto(self.rewards[self.idx], trajectory.reward)
    np.copyto(self.next_obses[self.idx], trajectory.observation[1])
    np.copyto(self.discounts[self.idx], trajectory.discount)

    self.idx = (self.idx + 1) % self.capacity
    self.full = self.full or self.idx == 0

  def save(self, n_shards):
    n_shard = self.capacity // n_shards
    for shard in range(n_shards):
      with tf.io.gfile.GFile(self.shard_fn(shard), 'wb') as fh:
        pickle.dump((self.obses[shard * n_shard:(shard + 1) * n_shard],
                     self.actions[shard * n_shard:(shard + 1) * n_shard],
                     self.rewards[shard * n_shard:(shard + 1) * n_shard],
                     self.next_obses[shard * n_shard:(shard + 1) * n_shard],
                     self.discounts[shard * n_shard:(shard + 1) * n_shard]), fh)

  def load(self, n_shards):
    with tf.io.gfile.GFile(self.shard_fn(0), 'rb') as fh:
      dataset = tf.data.Dataset.from_tensor_slices(pickle.load(fh))
    for i in range(1, n_shards):
      with tf.io.gfile.GFile(self.shard_fn(i), 'rb') as fh:
        tmp_dataset = tf.data.Dataset.from_tensor_slices(pickle.load(fh))
        dataset = dataset.concatenate(tmp_dataset)

    return dataset


class JointImageObservationsWrapper(wrappers.PyEnvironmentBaseWrapper):
  """Env wrapper to Flatten all image observations.

  Essentially the same as FlattenObservationsWrapper but stacks along channel
  dim and returns rank 3 tensor.

  Images are optionally resized to a specified output resolution, else images
  must all be the same size.
  """

  def __init__(self,
               env,
               out_width_height = None):
    super(JointImageObservationsWrapper, self).__init__(env)

    self.wh = out_width_height

    obs_spec: array_spec.ArraySpec = self._env.observation_spec()

    o_shape = None
    o_dtype = None
    o_name = []
    for key, obs in obs_spec.items():
      if not isinstance(obs, array_spec.ArraySpec):
        raise ValueError('Unsupported observation_spec %s' % str(obs))
      if len(obs.shape) != 3:
        if len(obs.shape) <= 0:
          obs = np.zeros(shape=(84, 84, 1))
        else:
          obs = np.zeros(shape=(84, 84, obs.shape[0]))
      # obs = obs.astype(np.float32)
      if self.wh:
        # The image size will be normalized.
        cur_shape = self.wh + (obs.shape[2],)
      else:
        cur_shape = obs.shape

      # if o_shape is None:
      #   o_shape = list(obs.shape)
      #   o_dtype = obs.dtype
      # else:
      #   o_shape[2] += obs.shape[2]
      if 'state' in key:
        o_shape = list(obs.shape)
        o_dtype = obs.dtype
      # o_name.append(obs.name)

    self._observation_spec = array_spec.ArraySpec(
        shape=o_shape,
        dtype=o_dtype,
        name='flattened')

  def _reset(self):
    return self._get_timestep(self._env.reset())

  def _step(self, action):
    return self._get_timestep(self._env.step(action))

  def _get_timestep(self, time_step):
    time_step = time_step._asdict()

    obs = []
    obs_spec: array_spec.ArraySpec = self._env.observation_spec()
    for key, _ in obs_spec.items():  # recall: obs_spec is an ordered_dict
      img = time_step['observation'][key]
      if self.wh:
        img = np.asarray(Image.fromarray(img).resize(self.wh))
        assert img.shape[0:2] == self.wh
      if len(img.shape) <= 1:
        img = np.tile(img, (84, 84, 1))
      img = img.astype(np.float32)
      obs.append(img)

    time_step['observation'] = np.concatenate(obs, axis=-1)
    # assert self.observation_spec().shape == time_step['observation'].shape
    return ts.TimeStep(**time_step)

  def observation_spec(self):
    return self._observation_spec


def load_weights(source_vars, target_vars):
  for v_t in target_vars:
    for v_s in source_vars:
      if (v_s.name == v_t.name or
          'ValueNetwork' in v_s.name) and v_s.shape == v_t.shape:
        v_t.assign(v_s)
