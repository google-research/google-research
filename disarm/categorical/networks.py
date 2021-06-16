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

# Lint as: python3
"""Trainable models for coupled estimator experiments."""
import functools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

keras = tf.keras
tfd = tfp.distributions

EPS = 1e-6


def safe_log_prob(p):
  return tf.math.log(tf.clip_by_value(p, EPS, 1.0))


def np_safe_log_prob(p):
  return np.log(np.clip(p, EPS, 1.0))


def logit_func(prob_tensor):
  """Calculate logits."""
  return safe_log_prob(prob_tensor) - safe_log_prob(1. - prob_tensor)


def get_importance_weight_vector(input_logits, threshold=1e-5):
  """Get the importance weight given the binary logits.

  Given the input_logits for binaries to construct the categorical samples.
  Suppose the number of categories is C. We need C-1 binary variables,
  b1, ..., bi, ..., b[C-1]. The categorical variable, z = n, is constructed as
  b1, ..., b[n-1] = 0, bn = 1. Therefore the importance weight is
    q(z=n)q(z_tilde=m) / p(z=n, z_tilde=m) =
      prod( sigma(|logits[i]|), i = 1, ..., min(n, m) -1)
      * (1 - p)^2 / (1 - 2*p)
  where p = sigmoid(logits[min(n, m)]).
  The sigma(|logits[i]|) is the contribution from b[i] = 0 and b_tilde[i] = 0.
  The (1 - p)^2 / (1 - 2*p) is the contribution from the antithetic pair of
  binaries that deviates.

  Args:
    input_logits: a float Tensor of the shape [batch_size, num_variables,
      num_categories-1].
    threshold: a float scalar for threholding the logits around zeros, so that
      to make the algorithm numerical stable.

  Returns:
    importance_weights: a float Tensor of the same shape as input_logits.
  """
  # q(0)q(1)/p(0,1)
  sigma_abs_logits = tf.math.sigmoid(tf.math.abs(input_logits))
  sigma_logits = tf.math.sigmoid(input_logits)
  # fi = q(b[i]=0)q(b_t[i]=0)/p(b[i]=0,bt[i]=0)
  both_zeros_multiplier = tf.where(
      input_logits >= -threshold,
      tf.zeros_like(input_logits),
      (1. - sigma_logits)**2 / (1-2*sigma_logits))
  # prod fi
  both_zeros_multiplier = tf.math.cumprod(
      both_zeros_multiplier,
      axis=-1)
  importance_weights = tf.concat(
      [sigma_abs_logits[:, :, 0:1],
       both_zeros_multiplier[:, :, :-1] * sigma_abs_logits[:, :, 1:],
       both_zeros_multiplier[:, :, -1:]],
      axis=-1)
  return importance_weights


def logit_ranking(input_logits, order=None):
  """Rank the input with sepecific order."""
  if order is None:
    return input_logits, None

  if order == 'abs':
    permutation = tf.argsort(tf.math.abs(input_logits), direction='ASCENDING')
  else:
    permutation = tf.argsort(input_logits, direction=order.upper())
  num_batch_dim = len(input_logits.shape) - 1
  sorted_logits = tf.gather(input_logits, permutation, batch_dims=num_batch_dim)
  return sorted_logits, permutation


def permute_categorical_sample(input_samples, permutations):
  """Permute the integer categorical samples."""
  # permutations doesn't have sample dims, with shape [B, V, C]
  # input_samples has sample dims, with shape [S, B, V]
  ndims = len(input_samples.shape)
  # with shape [S, B, V, C]
  permutations = tf.tile(tf.expand_dims(permutations, axis=0),
                         [input_samples.shape[0]] + [1] * ndims)
  permuted_samples = tf.gather(permutations, input_samples, batch_dims=ndims)
  return permuted_samples


def get_tree_logits_conversion_mask(i, num_cls):
  """Converting masks for converting softmax logits to tree logits."""
  order_idx = int(np.log2(i+1))
  level_slot_idx = i - 2**order_idx + 1
  denominator_mask = np.zeros([num_cls], dtype=bool)
  numerator_mask = np.zeros([num_cls], dtype=bool)
  denominator_begin_idx = int(level_slot_idx * num_cls / 2**order_idx)
  denominator_end_idx = int((level_slot_idx + 1) * num_cls / 2**order_idx)
  denominator_mask[denominator_begin_idx:denominator_end_idx] = True
  numerator_begin_idx = int((2*level_slot_idx + 1) * num_cls / 2**(order_idx+1))
  numerator_end_idx = int((level_slot_idx + 1) * num_cls / 2**order_idx)
  numerator_mask[numerator_begin_idx:numerator_end_idx] = True
  return denominator_mask, numerator_mask


def generate_softmax_to_tree_mask(num_categories):
  deno_mask_list = []
  nume_mask_list = []
  for i in range(num_categories-1):
    deno_mask, nume_mask = get_tree_logits_conversion_mask(i, num_categories)
    deno_mask_list.append(deno_mask)
    nume_mask_list.append(nume_mask)
  return (np.stack(deno_mask_list, axis=0),
          np.stack(nume_mask_list, axis=0))


def convert_softmax_logits_to_tree(
    input_logits,
    denomenator_mask,
    numerator_mask):
  """Converting softmax logits to logits for tree construction."""
  batch_size, num_variable, num_categ = input_logits.shape
  deno_mask = tf.tile(denomenator_mask[None, None, :, :],
                      [batch_size, num_variable, 1, 1])
  nume_mask = tf.tile(numerator_mask[None, None, :, :],
                      [batch_size, num_variable, 1, 1])
  tiled_input = tf.tile(input_logits[:, :, None, :],
                        [1, 1, num_categ-1, 1])
  deno = tf.where(tf.math.logical_xor(deno_mask, nume_mask),
                  tiled_input,
                  -np.inf * tf.ones_like(tiled_input))
  nume = tf.where(nume_mask,
                  tiled_input,
                  -np.inf * tf.ones_like(tiled_input))
  res = (tf.math.reduce_logsumexp(nume, axis=-1)
         - tf.math.reduce_logsumexp(deno, axis=-1))
  return res


def convert_softmax_logits_to_stickbreaking(input_logits):
  """Convert the 2-D SoftMax Logits to Stick-breaking logits."""
  ndims = len(input_logits.shape)
  tiled_logits = tf.tile(
      tf.expand_dims(input_logits, axis=-2),
      [1] * (ndims-1) +  [input_logits.shape[-1], 1])
  denominator_logits = tf.where(
      tf.linalg.band_part(tf.ones_like(tiled_logits, dtype=bool), -1, 0),
      - np.inf * tf.ones_like(tiled_logits),
      tiled_logits)
  stick_breaking_logits = (
      input_logits - tf.math.reduce_logsumexp(denominator_logits, axis=-1))
  return stick_breaking_logits[Ellipsis, :-1]


def sample_uniform_random_variable(sample_shape, num_samples=1):
  if num_samples > 1:
    sample_shape = tf.concat(
        [[sample_shape[0] * num_samples], sample_shape[1:]],
        axis=0)
  return tf.random.uniform(shape=sample_shape, maxval=1.0)


def get_first_appearance_in_2d_tensor(input_tensor, target=1, with_mask=False):
  """Find the first occurrence of the value in each row of input_tensor."""
  match_indices = tf.where(tf.equal(input_tensor, target))
  result = tf.math.segment_min(match_indices[:, 1], match_indices[:, 0])
  if with_mask:
    # one hot representation locates the first appearance
    # e.g. [[0, 0, 1, 0, 0]]
    depth = tf.shape(input_tensor)[-1]
    mask = tf.one_hot(result, depth)
    return result, mask
  else:
    return result


def get_stick_breaking_samples(encoder_logits, num_samples=1,
                               u_noise=None, tree_structure=False):
  """Generate categorical samples with stick breaking construction."""
  # in the case of stick breaking, the encoder output K-1 category logits
  batch_size, num_variables, num_categories = encoder_logits.shape

  sigma_phi = tf.math.sigmoid(encoder_logits)
  if u_noise is None:
    # [num_samples * batch_size, num_variables, num_categories]
    u_noise = sample_uniform_random_variable(
        [batch_size, num_variables, num_categories],
        num_samples=num_samples)
  sigma_phi_tiled = tf.tile(sigma_phi, [num_samples, 1, 1])
  b = tf.cast(u_noise < sigma_phi_tiled, tf.float32)

  b = tf.reshape(b, [-1, num_categories])
  # set the label of the K-th category to be 1.
  b_extended = tf.concat(
      [b, tf.ones_like(b[:, 0], dtype=tf.float32)[:, tf.newaxis]],
      axis=-1)

  if tree_structure:
    # logits_mask masks the nodes along the path of the tree
    categorical_samples, logits_mask = convert_balanced_tree_to_category(b)
  else:
    # logits_mask the first appearance of 1
    categorical_samples, logits_mask = get_first_appearance_in_2d_tensor(
        b_extended, target=1, with_mask=True)
  categorical_samples = tf.reshape(
      categorical_samples, [num_samples, batch_size, num_variables])
  logits_mask = tf.reshape(
      logits_mask,
      [num_samples, batch_size, num_variables, num_categories+1])

  binary_samples = tf.reshape(
      b_extended, [num_samples, batch_size, num_variables, -1])

  return categorical_samples, logits_mask, u_noise, binary_samples


def convert_binary_array_to_integer(binary_tensor):
  """Convert batched binary tensor to integers."""
  # The input binary tensor must be left-padded to the same length.
  # e.g. [[0, 0, 0, 0, 1, 1, 1],
  #       [0, 1, 0, 0, 1, 0, 0],
  #       [1, 1, 0, 0, 1, 0, 0]]
  # returns [7, 36, 100]
  binary_tensor = tf.convert_to_tensor(binary_tensor)
  return tf.reduce_sum(
      tf.cast(tf.reverse(tensor=binary_tensor, axis=[-1]), dtype=tf.int32)
      * 2 ** tf.range(tf.cast(tf.shape(binary_tensor)[-1], dtype=tf.int32)),
      axis=-1)


def convert_index_to_mask(index_tensor, depth):
  batch_size, event_size = index_tensor.shape
  zero_mask = tf.zeros([batch_size, depth], tf.int32)

  update_indices = tf.stack(
      [tf.repeat(tf.range(batch_size), event_size),
       tf.reshape(index_tensor, [-1])],
      axis=-1)

  return tf.tensor_scatter_nd_update(
      zero_mask, update_indices, tf.ones(batch_size * event_size, tf.int32))


def convert_balanced_tree_to_category(binary_tensor):
  """Convert a batched balanced binary tree to batched categorical samples.

  For example,
  binary_tensor = [[0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 0, 0]]
  categorical_sample = [0, 3]
  index_result = [[0, 1, 3], [0, 2, 5]]
  binary_result = [[0, 0, 0], [1, 0, 0]]

  Args:
    binary_tensor: a 1/0 integer tensor, which is of the shape [batch_size,
      num_nodes]. Each tensor of the batch corresponding to a balanced
      binary tree, where 0 indicates that the path is folked to the left child,
      and 1 to the right.

  Returns:
    categorical_sample: a integer tensor, which is of the shape [batch_size],
      contains the categorical results deduced from the tree embedded in the
      binary tensor. Depth = log2(num_nodes)
    mask: a 1/0 integer tensor, which is of the shape [batch_size, num_nodes],
      masks the elements labelled by index_result.
  """
  binary_tensor = tf.convert_to_tensor(binary_tensor)
  batch_size, num_nodes = binary_tensor.shape
  depth = int(np.log2(num_nodes+1))
  index_result = tf.zeros([batch_size, 1], dtype=tf.int32)
  binary_result = binary_tensor[:, 0][:, tf.newaxis]
  for i in range(1, depth):
    new_index = tf.cast(
        (2**i - 1) + convert_binary_array_to_integer(binary_result),
        dtype=tf.int32)
    selected_binary = tf.gather(binary_tensor, new_index, axis=1, batch_dims=1)
    index_result = tf.concat(
        [index_result, new_index[:, tf.newaxis]],
        axis=-1)
    binary_result = tf.concat(
        [binary_result, selected_binary[:, tf.newaxis]],
        axis=-1)
  categorical_sample = convert_binary_array_to_integer(binary_result)
  mask = tf.cast(convert_index_to_mask(index_result, num_nodes+1), tf.float32)
  return categorical_sample, mask


class CnnEncoderNetwork(tf.keras.Model):
  """Network generating binary samples designed for CelebA."""

  def __init__(self,
               hidden_size,
               num_categories,
               train_mean,
               base_depth=32,
               stick_breaking=False,
               name='cnnencoder'):
    self.hidden_size = hidden_size
    if stick_breaking:
      self.num_categories = num_categories-1
    else:
      self.num_categories = num_categories

    super(CnnEncoderNetwork, self).__init__(name=name)

    conv = functools.partial(
        keras.layers.Conv2D, padding='SAME', activation=tf.nn.leaky_relu)

    self.networks = keras.Sequential([
        keras.layers.Lambda(lambda x: (x - train_mean) / 256.),
        conv(base_depth, 5, 2),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 2),
        conv(2 * base_depth, 5, 2),
        conv(4 * base_depth, 5, 2),
        conv(2 * base_depth, 5, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(hidden_size * self.num_categories,
                           activation=None),
    ])

  def __call__(self,
               input_tensor,
               samples=None,
               num_samples=(),
               stick_breaking=False):
    if stick_breaking:
      raise NotImplementedError(
          'When using stick breaking, please use the get_logits, '
          'get_stick_breaking_samples function to generate the samples, '
          'and calculate the elbo.')

    else:
      logits = self.get_logits(input_tensor)
      dist = tfd.Categorical(logits=logits)
      if samples is None:
        samples = dist.sample(num_samples)
      samples = tf.cast(samples, tf.float32)
      likelihood = dist.log_prob(samples)
      return samples, likelihood, logits

  def get_logits(self, input_tensor):
    logits = tf.reshape(
        self.networks(input_tensor),
        [-1, self.hidden_size, self.num_categories])
    return logits


class CnnDecoderNetwork(tf.keras.Model):
  """Network generating categorical samples."""

  def __init__(self,
               train_mean,
               base_depth=32,
               name='cnndecoder'):

    super(CnnDecoderNetwork, self).__init__(name=name)

    deconv = functools.partial(
        keras.layers.Conv2DTranspose, padding='SAME',
        activation=tf.nn.leaky_relu)
    conv = functools.partial(
        keras.layers.Conv2D, padding='SAME',
        activation=tf.nn.leaky_relu)

    squash_eps = 1e-6
    squash_bijector = tfp.bijectors.Chain([
        tfp.bijectors.AffineScalar(scale=256.),
        tfp.bijectors.AffineScalar(
            shift=-squash_eps / 2.,
            scale=(1. + squash_eps)),
        tfp.bijectors.Sigmoid()
    ])
    unsquashed_data_mean = squash_bijector.inverse(train_mean)

    raw_scale = tf.Variable(
        name='raw_sigma',
        initial_value=tf.ones([64, 64, 3]),
        trainable=True)
    last_decoder_layer = tfp.layers.DistributionLambda(
        lambda t: tfd.Independent(  # pylint: disable=g-long-lambda
            tfd.TransformedDistribution(
                distribution=tfd.Normal(
                    loc=t + unsquashed_data_mean,
                    scale=tf.math.softplus(raw_scale)),
                bijector=squash_bijector)))

    self.networks = tf.keras.Sequential([
        keras.layers.Lambda(lambda x: x[:, None, None, :]),
        deconv(4 * base_depth, 5, 2),
        deconv(4 * base_depth, 5, 2),
        deconv(2 * base_depth, 5, 2),
        deconv(2 * base_depth, 5, 2),
        deconv(base_depth, 5, 2),
        deconv(base_depth, 5, 2),
        conv(3, 5, activation=None),
        last_decoder_layer,
    ])

  def __call__(self,
               input_tensor,
               samples=None,
               num_samples=()):
    dist = self.networks(input_tensor)
    if samples is None:
      samples = dist.sample(num_samples)
    samples = tf.cast(samples, tf.float32)
    likelihood = dist.log_prob(samples)
    return samples, likelihood


class BinaryNetwork(tf.keras.Model):
  """Network generating binary samples."""

  def __init__(self,
               hidden_sizes,
               activations,
               mean_xs=None,
               demean_input=False,
               final_layer_bias_initializer='zeros',
               name='binarynet',
               kernel_initializer='glorot_uniform'):

    super(BinaryNetwork, self).__init__(name=name)
    assert len(activations) == len(hidden_sizes)

    num_layers = len(hidden_sizes)
    self.hidden_sizes = hidden_sizes
    self.activations = activations
    self.networks = keras.Sequential()

    self.networks.add(keras.layers.Flatten())

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
              activation=activations[i],
              kernel_initializer=kernel_initializer))

    self.networks.add(
        keras.layers.Dense(
            units=hidden_sizes[-1],
            activation=activations[-1],
            kernel_initializer=kernel_initializer,
            bias_initializer=final_layer_bias_initializer))

  def __call__(self,
               input_tensor,
               samples=None,
               num_samples=()):
    logits = self.get_logits(input_tensor)
    dist = tfd.Bernoulli(logits=logits)
    if samples is None:
      samples = dist.sample(num_samples)
    samples = tf.cast(samples, tf.float32)
    likelihood = dist.log_prob(samples)
    return samples, likelihood, logits

  def get_logits(self, input_tensor):
    logits = self.networks(input_tensor)
    return logits


class CategoricalNetwork(tf.keras.Model):
  """Network generating categorical samples."""

  def __init__(self,
               hidden_sizes,
               activations,
               num_categories,
               mean_xs=None,
               demean_input=False,
               name='categnet',
               kernel_initializer='glorot_uniform'):

    super(CategoricalNetwork, self).__init__(name=name)
    assert len(activations) == len(hidden_sizes)

    num_layers = len(hidden_sizes)
    self.hidden_sizes = hidden_sizes
    self.activations = activations
    self.networks = keras.Sequential()
    self.num_categories = num_categories

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
              activation=activations[i],
              kernel_initializer=kernel_initializer))

    self.networks.add(
        keras.layers.Dense(
            units=hidden_sizes[-1] * self.num_categories,
            activation=activations[-1],
            kernel_initializer=kernel_initializer))

  def __call__(self,
               input_tensor,
               samples=None,
               num_samples=(),
               stick_breaking=False):
    if stick_breaking:
      # if stick_breaking we will convert the logits from K categories to K-1
      # and append a non-trainable zero logits to K-th categories
      raise NotImplementedError(
          'When using stick breaking, please use the get_logits, '
          'get_stick_breaking_samples function to generate the samples, '
          'and calculate the elbo.')

    else:
      logits = self.get_logits(input_tensor)
      dist = tfd.Categorical(logits=logits)
      if samples is None:
        samples = dist.sample(num_samples)
      samples = tf.cast(samples, tf.float32)
      likelihood = dist.log_prob(samples)
      return samples, likelihood, logits

  def get_logits(self, input_tensor):
    logits = tf.reshape(
        self.networks(input_tensor),
        [-1, self.hidden_sizes[-1], self.num_categories])
    return logits


class CategoricalVAE(tf.keras.Model):
  """Discrete VAE."""

  def __init__(self,
               encoder,
               decoder,
               prior_logits,
               num_categories,
               one_hot_sample=False,
               grad_type='reinforce_loo',
               name='categorical_vae'):
    super(CategoricalVAE, self).__init__(name)
    self.encoder = encoder
    self.decoder = decoder

    self.prior_logits = prior_logits
    self.prior_dist = tfd.Categorical(logits=self.prior_logits)

    self.num_categories = num_categories

    self.grad_type = grad_type.lower()

    # used for variance of gradients estiamations.
    self.ema = tf.train.ExponentialMovingAverage(0.999)

    # if True, the using one-hot categorical sample
    self.one_hot_sample = one_hot_sample

    self.denom_mask, self.numer_mask = generate_softmax_to_tree_mask(
        self.num_categories)

  def call(self, input_tensor, hidden_samples=None, num_samples=1,
           logits_mask=None,
           stick_breaking=False, tree_structure=False,
           logits_sorting_order=None):
    """Returns ELBO for single layer discrete VAE.

    Args:
      input_tensor: a `float` Tensor for input observations.
        The tensor is of the shape [batch_size, observation_dims].
      hidden_samples: a discrete Tensor for hidden states.
        The tensor is of the shape [batch_size, hidden_dims].
        Default to None, in which case the hidden samples will be generated
        based on num_samples.
      num_samples: 0-D or 1-D `int` Tensor. Shape of the generated samples.
      logits_mask: labels the binary corresponding to the
        the categorical label, e.g. [0, 0, 0, 1, 0] corresponds to 3rd category.
      stick_breaking: `boolean` scalar. Indicating whether to use stick breaking
        reparametrization of categorical variables.
      tree_structure: `boolean` scalar. Indicating whether to use tree structure
        stick breaking for categorical variables.
      logits_sorting_order: an optional string for how to order the logits in
        stick breaking, choices are [None, 'abs', 'ascending', descending'].

    Returns:
      elbo: the ELBO with shape [S, B].
      categorical_samples: with shape [S, B, V].
      encoder_logits: with shape [S, B, V, C].
      encoder_llk: with shape [S, B]
    """
    if stick_breaking:
      return self.stick_breaking_call(
          input_tensor, hidden_samples=hidden_samples, logits_mask=logits_mask,
          num_samples=num_samples, tree_structure=tree_structure,
          logits_sorting_order=logits_sorting_order)

    else:
      return self.categorical_call(
          input_tensor, hidden_samples=hidden_samples, num_samples=num_samples)

  def stick_breaking_call(
      self, input_tensor, encoder_logits=None,
      hidden_samples=None, logits_mask=None, num_samples=1,
      binary_samples=None,
      tree_structure=False,
      logits_sorting_order=None):
    batch_size = input_tensor.shape[0]

    permutation = None
    if encoder_logits is None:
      encoder_logits = self.encoder.get_logits(input_tensor)
      if tree_structure:
        encoder_logits = convert_softmax_logits_to_tree(
            encoder_logits, self.denom_mask, self.numer_mask)
      else:
        encoder_logits, permutation = logit_ranking(
            encoder_logits, order=logits_sorting_order)
        encoder_logits = convert_softmax_logits_to_stickbreaking(encoder_logits)

    if (hidden_samples is not None) and (logits_mask is not None):
      categorical_samples = hidden_samples

    else:
      # categorical_samples of the shape [samples, batch, variables]
      # logits_mask of the shape [samples, batch, variables]
      categorical_samples, logits_mask, _, binary_samples = (
          get_stick_breaking_samples(
              encoder_logits, num_samples, tree_structure=tree_structure))

    if permutation is not None:
      categorical_samples = permute_categorical_sample(
          categorical_samples, permutation)

    if tree_structure:
      # encoder_logits_4d is of the shape [S, B, V, K-1]
      # logits_mask is of the shape [S, B, V, K]
      # the last dimension is redundant, as b[K] is fixed to 1.
      encoder_logits_4d = encoder_logits[tf.newaxis, :, :, :]
      logits_mask = logits_mask[:, :, :, :-1]
      binary_samples = binary_samples[:, :, :, :-1]
      encoder_llk = -1.*(
          tf.math.softplus(encoder_logits_4d) * (1. - binary_samples)
          + tf.math.softplus(-encoder_logits_4d) * binary_samples
          ) * logits_mask
      # encoder_llk with [S, B, V] shape
      encoder_llk_sbv = tf.reduce_sum(encoder_llk, axis=[-1])
      encoder_llk = tf.reduce_sum(encoder_llk_sbv, axis=[-1])

    else:
      # logits_mask labels the binary corresponding to the
      # the categorical label, e.g. [0, 0, 0, 1, 0] corresponds to 3rd category
      # calculation_mask mask the elements before the first binary
      # e.g. [1, 1, 1, 1, 0] in the case of aforementioned example.
      calculation_mask = tf.cumsum(
          logits_mask[:, :, :, ::-1],
          axis=-1)[:, :, :, ::-1]
      # log(1 - sigma(x)) = - softplus(x)
      # log(sigma(x)) = -softplus(-x)
      encoder_logits_4d = encoder_logits[tf.newaxis, :, :, :]

      # encoder_logits is of the shape [S, B, V, K-1]
      # logits_mask and calculation_mask are of the shape [S, B, V, K]
      # the last dimension is redundant, as b[K] is fixed to 1.
      logits_mask = logits_mask[:, :, :, :-1]
      calculation_mask = calculation_mask[:, :, :, :-1]
      encoder_llk = -1.*(
          tf.math.softplus(encoder_logits_4d) * (1. - logits_mask)
          + tf.math.softplus(-encoder_logits_4d) * logits_mask
          ) * calculation_mask
      # encoder_llk with [S, B, V] shape
      encoder_llk_sbv = tf.reduce_sum(encoder_llk, axis=[-1])
      encoder_llk = tf.reduce_sum(encoder_llk_sbv, axis=[-1])

    log_pb = self.prior_dist.log_prob(categorical_samples)
    log_pb = tf.reduce_sum(log_pb, axis=-1)

    input_tensor_tiled = tf.tile(
        input_tensor,
        [num_samples] + [1] * (len(input_tensor.shape) - 1))
    if self.one_hot_sample:
      one_hot_sample = tf.one_hot(tf.cast(categorical_samples, tf.int32),
                                  depth=self.num_categories,
                                  dtype=tf.float32)
      one_hot_sample = tf.reshape(one_hot_sample,
                                  [num_samples * batch_size, -1])
      decoder_llk = self.decoder(one_hot_sample, input_tensor_tiled)[1]
    else:
      decoder_llk = self.decoder(
          tf.reshape(categorical_samples, [num_samples * batch_size, -1]),
          input_tensor_tiled)[1]

    decoder_llk = tf.reshape(decoder_llk, [num_samples * batch_size, -1])
    decoder_llk = tf.reshape(tf.reduce_sum(decoder_llk, axis=-1),
                             [num_samples, batch_size])

    f_elbo = decoder_llk + log_pb - encoder_llk

    return f_elbo, categorical_samples, encoder_logits, encoder_llk_sbv

  def categorical_call(self, input_tensor, hidden_samples=None, num_samples=1):
    batch_size = input_tensor.shape[0]

    hidden_sample, encoder_llk, encoder_logits = self.encoder(
        input_tensor,
        samples=hidden_samples,
        num_samples=num_samples)

    log_pb = self.prior_dist.log_prob(hidden_sample)

    if self.one_hot_sample:
      one_hot_sample = tf.one_hot(tf.cast(hidden_sample, tf.int32),
                                  depth=self.num_categories,
                                  dtype=tf.float32)
      one_hot_sample = tf.reshape(one_hot_sample,
                                  [num_samples * batch_size, -1])

      decoder_llk = self.decoder(one_hot_sample, input_tensor)[1]
    else:
      hidden_sample = tf.reshape(hidden_sample,
                                 [num_samples * batch_size, -1])
      decoder_llk = self.decoder(hidden_sample, input_tensor)[1]

    decoder_llk = tf.reshape(decoder_llk, [num_samples * batch_size, -1])
    elbo = (
        tf.reduce_sum(decoder_llk, axis=-1)
        + tf.reduce_sum(log_pb - encoder_llk, axis=-1))

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
      self, input_tensor, grad_type=None, num_samples=None,
      stick_breaking=False,
      tree_structure=False,
      logits_sorting_order=None,
      importance_weighting=False):
    if grad_type is None:
      grad_type = self.grad_type
    if num_samples is None:
      num_samples = self.num_categories

    # encoder_logits is of the shape [batch_size, num_variables, num_categ]
    encoder_logits = self.encoder.get_logits(input_tensor)
    categorical_dist = tfd.Categorical(logits=encoder_logits)

    batch_size, num_variables, num_categories = encoder_logits.shape

    if grad_type == 'reinforce_loo':
      if stick_breaking:
        # f_elbo of shape [num_samples, batch_size]
        # encoder_llk of shape [num_samples, batch_size]
        f_elbo, _, _, encoder_llk = self.call(
            input_tensor, num_samples=num_samples,
            stick_breaking=True, tree_structure=tree_structure)

        # f[i] - 1/(K-1) sum[f[j], j!=i, j=1:K]
        learning_signal = 1/(num_samples - 1.) * (
            f_elbo * num_samples - tf.reduce_sum(f_elbo, axis=0, keepdims=True))

        return learning_signal, encoder_llk

      else:  # not using stick breaking
        sigma_phi = tf.math.softmax(encoder_logits)

        # [num_samples, batch_size, num_variables]
        categorical_samples = categorical_dist.sample(num_samples)
        # [num_samples * batch_size, num_variables]
        categorical_samples = tf.reshape(
            categorical_samples,
            [num_samples * batch_size, num_variables])

        # [num_samples * batch_size, -1]
        input_tensor_tiled = tf.tile(
            input_tensor,
            [num_samples] + [1] * (len(input_tensor.shape) - 1))

        f_elbo = self.get_elbo(input_tensor_tiled, categorical_samples)
        f_elbo = tf.reshape(f_elbo, [num_samples, batch_size])

        # f[i] - 1/(K-1) sum[f[j], j!=i, j=1:K]
        f_elbo = 1/(num_samples - 1.) * (
            f_elbo * num_samples - tf.reduce_sum(f_elbo, axis=0, keepdims=True))

        # [ num_samples * batch_size, num_variables, num_categories]
        categorical_samples_onehot = tf.one_hot(
            categorical_samples, depth=num_categories, dtype=tf.float32)
        prefactor = tf.reshape(
            categorical_samples_onehot
            - tf.tile(sigma_phi, [num_samples, 1, 1]),
            [num_samples, batch_size, num_variables, num_categories])

        # [batch_size, num_variables, num_categories]
        layer_grad = tf.reduce_mean(
            f_elbo[:, :, tf.newaxis, tf.newaxis] * prefactor,
            axis=0)

    elif grad_type == 'arsm':
      dirichlet_dist = tfd.Dirichlet([1.0] * num_categories)
      sample_shape = [batch_size, num_variables]
      # [batch_size, num_variables, num_categories]
      dirichlet_samples = dirichlet_dist.sample(sample_shape=sample_shape)
      layer_grad = self.compute_arsm_grads(
          input_tensor, encoder_logits, dirichlet_samples)

    elif grad_type == 'arsmp':
      dirichlet_dist = tfd.Dirichlet([1.0] * num_categories)
      sample_shape = [batch_size, num_variables]
      # [batch_size, num_variables, num_categories]
      dirichlet_samples = dirichlet_dist.sample(sample_shape=sample_shape)
      layer_grad = self.compute_arsm_grads(
          input_tensor, encoder_logits, dirichlet_samples,
          partial_integration=True)

    elif grad_type == 'ars':
      dirichlet_dist = tfd.Dirichlet([1.0] * num_categories)
      sample_shape = [batch_size, num_variables]
      # [batch_size, num_variables, num_categories]
      dirichlet_samples = dirichlet_dist.sample(sample_shape=sample_shape)
      layer_grad = self.compute_ars_grads(
          input_tensor, encoder_logits, dirichlet_samples)

    elif grad_type == 'arsp':
      # 1. sample from dirichlet distribution
      # 2. get log s = log(dirichlet) - phi
      # 3. get categorical samples: z = argmin(s)
      # 4. sample reference index: j ~ uniform([1, C])
      # 5. get swapped log s(m <=> j) = log(dirichlet)[m] - phi[j]
      # 6. get mth element removed log s_exclude_m(m <=> j)
      # 7. loop m in [1: C]
      #    a. if m ! = j:
      #       upper_bound[m] = exp(
      #                  min(log s_exclude_m(m <=> j)[i], i in num_cls dim)
      #                  + phi[m])
      #    b. if m == j:
      #       lower_bound[m] = exp(
      #                  min(log s(m <=> j)[i], i in [1, C])
      #                  + phi[m])
      # 8. E(pai_j) = 0.5 * (min(upper_bound) + max(lower_bound)
      # 9. f_j = Elbo(input, argmin(log s(m <=> j), m))
      # 10, grad = (f_j - mean(f_j, j)) * (1 - C * E(pai_j)) * grad(phi)

      dirichlet_dist = tfd.Dirichlet([1.0] * num_categories)
      sample_shape = [batch_size, num_variables]
      # [batch_size, num_variables, num_categories]
      dirichlet_samples = dirichlet_dist.sample(sample_shape=sample_shape)
      layer_grad = self.compute_ars_grads(
          input_tensor, encoder_logits, dirichlet_samples,
          partial_integration=True)

    elif grad_type == 'disarm':
      if not stick_breaking:
        raise NotImplementedError(
            'DisARM only support stick breaking categorical VAEs.')

      encoder_logits = self.encoder.get_logits(input_tensor)

      permutation = None
      if tree_structure:
        encoder_logits = convert_softmax_logits_to_tree(
            encoder_logits, self.denom_mask, self.numer_mask)
      else:
        encoder_logits, permutation = logit_ranking(
            encoder_logits, order=logits_sorting_order)
        encoder_logits = convert_softmax_logits_to_stickbreaking(encoder_logits)

      z1, logits_mask1, unif_samples, b1 = get_stick_breaking_samples(
          encoder_logits, num_samples,
          tree_structure=tree_structure)
      z2, logits_mask2, _, b2 = get_stick_breaking_samples(
          encoder_logits, num_samples, u_noise=1.-unif_samples,
          tree_structure=tree_structure)

      min_z1_z2 = tf.math.minimum(z1, z2)
      if permutation is not None:
        z1 = permute_categorical_sample(z1, permutation)
        z2 = permute_categorical_sample(z2, permutation)
      # f_elbo of shape [num_samples, batch_size]
      # encoder_llk of shape [num_samples, batch_size]

      # get f(z) and f(z'), z and z', b and b'
      # z and z' are in softmax format instead of one-hot
      # encoder_llk1/2 are of the shape [S, B, V]
      elbo1, _, _, encoder_llk1 = self.stick_breaking_call(
          input_tensor, encoder_logits=encoder_logits,
          hidden_samples=z1, logits_mask=logits_mask1,
          num_samples=num_samples,
          binary_samples=b1,
          tree_structure=tree_structure)
      elbo2, _, _, encoder_llk2 = self.stick_breaking_call(
          input_tensor, encoder_logits=encoder_logits,
          hidden_samples=z2, logits_mask=logits_mask2,
          num_samples=num_samples,
          binary_samples=b2,
          tree_structure=tree_structure)

      # the mask for I( i < max(z, z_tilde))

      # logits_mask labels the binary corresponding to the
      # the categorical label, e.g. [0, 0, 0, 1, 0] corresponds to 3rd category
      # calculation_mask mask the elements before the first binary
      # e.g. [1, 1, 1, 1, 0] in the case of aforementioned example.

      if importance_weighting:
        # shape is [S, B, V]
        importance_weights = get_importance_weight_vector(
            input_logits=encoder_logits,
            threshold=1e-5)
        # expand and tile the sample dimension.
        # importance_weights has shape [B, V, C] -> [S, B, V, C]
        importance_weights = tf.tile(
            tf.expand_dims(importance_weights, axis=0),
            [min_z1_z2.shape[0]] + [1] * len(importance_weights.shape))
        # learning_signal has the shape [S, B, V]
        learning_signal = tf.gather(
            importance_weights, min_z1_z2,
            batch_dims=len(importance_weights.shape)-1)
        # if z1 == z2, elbo1 = elbo2, the learning signal vanishes.
        # but it seems explicitly check it and set the learning signal to 0
        # reduces the variance.
        learning_signal = tf.where(
            tf.math.equal(z1, z2),
            tf.zeros_like(learning_signal),
            0.5 * learning_signal * (elbo1 - elbo2)[:, :, None])
        # learning_signal = 0.5 * learning_signal * (elbo1 - elbo2)[:, :, None]
        return learning_signal, encoder_llk1 - encoder_llk2

      elif tree_structure:
        logits_mask1_bool = tf.cast(logits_mask1, tf.bool)
        logits_mask2_bool = tf.cast(logits_mask2, tf.bool)

        # mask nodes on both paths
        mask_shared_path = tf.logical_and(logits_mask1_bool, logits_mask2_bool)
        mask_shared_path = tf.cast(mask_shared_path, tf.float32)

        # mask nodes only on path 1
        mask_only_path1 = tf.logical_and(
            logits_mask1_bool,
            tf.logical_not(logits_mask2_bool))
        mask_only_path1 = tf.cast(mask_only_path1, tf.float32)

        # mask nodes only on path 2
        mask_only_path2 = tf.logical_and(
            tf.logical_not(logits_mask1_bool),
            logits_mask2_bool)
        mask_only_path2 = tf.cast(mask_only_path2, tf.float32)

        probs = tf.math.sigmoid(encoder_logits)

        sigma_abs_phi = tf.math.sigmoid(tf.math.abs(encoder_logits))
        # the factor is I(b0!=b1) * (-1)**b1 * sigma(|phi|)

        # b0, b1, and mask(i <= zmax) has the shape [S, B, V, C]
        disarm_factor = ((1. - b1) * (b2) + b1 * (1. - b2)) * (-1.)**b2
        disarm_factor *= mask_shared_path
        # encoder_logits has the shape [S, B, V, C-1] due to stick breaking
        disarm_factor = sigma_abs_phi * disarm_factor[:, :, :, :-1]

        learning_signal_1 = (
            mask_only_path1[Ellipsis, :-1] *
            0.5 * (elbo1 - elbo2)[:, :, None, None] *
            (b1[Ellipsis, :-1] - probs[None]))

        learning_signal_2 = (
            mask_only_path2[Ellipsis, :-1] *
            0.5 * (elbo2 - elbo1)[:, :, None, None] *
            (b2[Ellipsis, :-1] - probs[None]))

        learning_signal = (
            0.5 * (elbo1 - elbo2)[:, :, tf.newaxis, tf.newaxis] * disarm_factor
            + learning_signal_1 + learning_signal_2)

        return learning_signal, encoder_logits

      else:
        mask1 = tf.cast(tf.cumsum(logits_mask1, reverse=True, axis=-1), tf.bool)
        mask2 = tf.cast(tf.cumsum(logits_mask2, reverse=True, axis=-1), tf.bool)
        mask_i_le_zmax = tf.logical_and(mask1, mask2)
        mask_i_le_zmax = tf.cast(mask_i_le_zmax, tf.float32)

        probs = tf.math.sigmoid(encoder_logits)

        # Compute part from 1
        overlap_1 = tf.logical_and(tf.logical_not(mask1), mask2)
        overlap_1 = tf.cast(overlap_1, tf.float32)
        learning_signal_1 = (
            overlap_1[Ellipsis, :-1] *
            0.5 * (elbo2 - elbo1)[:, :, None, None] *
            (b2[Ellipsis, :-1] - probs[None]))

        # Compute part from 2
        overlap_2 = tf.logical_and(tf.logical_not(mask2), mask1)
        overlap_2 = tf.cast(overlap_2, tf.float32)
        learning_signal_2 = (
            overlap_2[Ellipsis, :-1] *
            0.5 * (elbo1 - elbo2)[:, :, None, None] *
            (b1[Ellipsis, :-1] - probs[None]))

        sigma_abs_phi = tf.math.sigmoid(tf.math.abs(encoder_logits))
        # the factor is I(b0!=b1) * (-1)**b1 * sigma(|phi|)

        # b0, b1, and mask(i <= zmax) has the shape [S, B, V, C]
        disarm_factor = ((1. - b1) * (b2) + b1 * (1. - b2)) * (-1.)**b2
        disarm_factor *= mask_i_le_zmax
        # encoder_logits has the shape [S, B, V, C-1] due to stick breaking
        disarm_factor = sigma_abs_phi * disarm_factor[:, :, :, :-1]

        learning_signal = (
            0.5 * (elbo1 - elbo2)[:, :, tf.newaxis, tf.newaxis] * disarm_factor
            + learning_signal_1 + learning_signal_2)

        return learning_signal, encoder_logits

    else:
      raise NotImplementedError('Gradient type %s is not supported.'%grad_type)

    return layer_grad

  def get_categorical_samples(self, dirichlet_samples, logits,
                              return_one_hot=False):
    category_label = tf.argmin(
        safe_log_prob(dirichlet_samples) - logits,
        axis=-1)
    if return_one_hot:
      return tf.one_hot(category_label, self.num_categories)
    else:
      return category_label

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

  def get_arsm_swapped_sample(self, encoder_logits, dirichlet_samples):
    def numpy_compute_loss(d_samples, e_logits):
      """Compute f(z[i<=>j])."""
      batch_size, num_variables, num_classes = e_logits.shape
      swap_sample_matrix = np.zeros(
          [batch_size, num_classes, num_classes, num_variables],
          dtype=np.float32)
      for ii in range(num_classes):
        for jj in range(ii, num_classes):
          dirich_ij = np.copy(d_samples)
          dirich_ij[:, :, [ii, jj]] = dirich_ij[:, :, [jj, ii]]
          s_ij = np.argmin(safe_log_prob(dirich_ij) - e_logits, axis=-1)
          swap_sample_matrix[:, ii, jj] = s_ij
          swap_sample_matrix[:, jj, ii] = s_ij
      return swap_sample_matrix

    # [batch_size, num_classes, num_classes, num_variables]
    swapped_matrix = tf.numpy_function(
        numpy_compute_loss,
        [dirichlet_samples, encoder_logits],
        tf.float32)
    return swapped_matrix

  def compute_arsm_grads(
      self, input_batch, encoder_logits, dirichlet_samples,
      partial_integration=False):
    batch_size, num_variables, num_classes = encoder_logits.shape
    # [batch_size, num_classes, num_classes, num_variables]
    swapped_matrix = self.get_arsm_swapped_sample(
        encoder_logits, dirichlet_samples)
    swapped_matrix = tf.reshape(
        swapped_matrix,
        [batch_size * num_classes * num_classes, num_variables])
    # input_batch is of the size [batch_size, observation_dims]
    tiled_inputs = tf.tile(input_batch[:, None, None, :],
                           [1, num_classes, num_classes, 1])
    tiled_inputs = tf.reshape(
        tiled_inputs,
        [batch_size * num_classes * num_classes, -1])
    loss_matrix = self.get_elbo(tiled_inputs, swapped_matrix)
    loss_matrix = tf.reshape(loss_matrix,
                             [batch_size, num_classes, num_classes])

    tilde_loss = (loss_matrix
                  - tf.math.reduce_mean(loss_matrix, axis=-1, keepdims=True))
    tilde_dirichlet = 1./num_classes - dirichlet_samples
    if partial_integration:
      swapped_matrix = tf.reshape(
          swapped_matrix,
          [batch_size, num_classes, num_classes, num_variables])
      tilde_dirichlet *= tf.transpose(
          tf.cast(tf.logical_not(tf.reduce_all(
              swapped_matrix == swapped_matrix[:, :, 0, :][:, :, None, :],
              axis=2)), dtype=tf.float32),
          [0, 2, 1])

    layer_grad = tf.matmul(tilde_dirichlet, tilde_loss)
    return layer_grad

  def get_ars_swapped_vector(self, encoder_logits, dirichlet_samples):
    def numpy_ars(e_logits, d_samples):
      batch_size, num_variables, num_classes = e_logits.shape
      # [batch_size, num_variables, 1]
      true_actions = np.argmin(
          safe_log_prob(d_samples) - e_logits, axis=-1)[:, :, None]
      pseudo_actions = np.full([batch_size, num_variables, num_classes],
                               true_actions, dtype=np.float32)
      jj = np.random.randint(num_classes)
      for ii in range(num_classes):
        if ii != jj:
          dirich_swap = np.copy(d_samples)
          dirich_swap[:, :, [ii, jj]] = dirich_swap[:, :, [jj, ii]]
          pseudo_actions[:, :, ii] = np.argmin(
              safe_log_prob(dirich_swap) - e_logits, axis=-1)
      reference_samples = d_samples[:, :, jj]
      return pseudo_actions, reference_samples

    # swapped_vector of the shape [batch_size, num_variables, num_classes]
    swapped_vector, reference_samples = tf.numpy_function(
        numpy_ars,
        [encoder_logits, dirichlet_samples],
        (tf.float32, tf.float32))
    # [batch_size, num_classes, num_variables]
    swapped_vector = tf.transpose(swapped_vector, [0, 2, 1])
    return swapped_vector, reference_samples

  def get_arsplus_swapped_vector(self, encoder_logits, dirichlet_samples):
    def numpy_ars(e_logits, d_samples):
      batch_size, num_variables, num_classes = e_logits.shape
      # [batch_size, num_variables, 1]
      s_vector = np_safe_log_prob(d_samples) - e_logits
      true_actions = np.argmin(s_vector, axis=-1)
      pseudo_actions = np.full([batch_size, num_variables, num_classes],
                               true_actions[:, :, None], dtype=np.float32)
      jj, l = np.random.choice(num_classes, 2, replace=False)
      lower_bound = np.zeros_like(d_samples, dtype=np.float32)
      upper_bound = np.ones_like(d_samples, dtype=np.float32)
      log_remaining_prob = np_safe_log_prob(d_samples[:, :, jj]
                                            + d_samples[:, :, l])
      for m in range(num_classes):
        if m != jj:
          dirich_swap = np.copy(d_samples)
          dirich_swap[:, :, [m, jj]] = dirich_swap[:, :, [jj, m]]
          s_swap = np_safe_log_prob(dirich_swap) - e_logits
          action_swap = np.argmin(s_swap, axis=-1)
          pseudo_actions[:, :, m] = action_swap

        else:
          s_swap = s_vector
          action_swap = true_actions

        e_logits_m = e_logits[:, :, m]

        # Need to rewrite the l entry of s_swap
        if m == l:
          # Because of swap of (m=l, j) we need to rewrite the j entry
          s_swap[:, :, jj] = (
              log_remaining_prob - e_logits[:, :, jj] - e_logits[:, :, l]
              - np_safe_log_prob(
                  np.exp(-e_logits[:, :, jj]) + np.exp(-e_logits[:, :, l])))
        else:
          s_swap[:, :, l] = (
              log_remaining_prob - e_logits[:, :, m] - e_logits[:, :, l]
              - np_safe_log_prob(
                  np.exp(-e_logits[:, :, m]) + np.exp(-e_logits[:, :, l])))

        s_swap_exclude_m = np.delete(s_swap, m, axis=-1)
        upper_bound[:, :, m] = np.where(
            action_swap == m,
            np.exp(np.min(s_swap_exclude_m, axis=-1) + e_logits_m),
            1.)
        lower_bound[:, :, m] = np.where(
            action_swap != m,
            np.exp(np.min(s_swap, axis=-1) + e_logits_m),
            0.)

      # We know that \pi_j + \pi_l = remaining_prob => \pi_j <= remaining_prob
      upper_bound = np.concatenate(
          (upper_bound, np.exp(log_remaining_prob)[Ellipsis, None]), axis=-1)
      reference_samples = 0.5 * (
          np.min(upper_bound, axis=-1)
          + np.max(lower_bound, axis=-1))
      # pseudo_action [batch_size, num_variables, num_categories]
      # reference_samples [batch_size, num_variables]
      return pseudo_actions, reference_samples

    # swapped_vector of the shape [batch_size, num_variables, num_classes]
    swapped_vector, reference_samples = tf.numpy_function(
        numpy_ars,
        [encoder_logits, dirichlet_samples],
        (tf.float32, tf.float32))
    # [batch_size, num_classes, num_variables]
    swapped_vector = tf.transpose(swapped_vector, [0, 2, 1])
    return swapped_vector, reference_samples

  def compute_ars_grads(
      self, input_batch, encoder_logits, dirichlet_samples,
      partial_integration=False):
    batch_size, num_variables, num_classes = encoder_logits.shape
    if partial_integration:
      swapped_vector, reference_samples = self.get_arsplus_swapped_vector(
          encoder_logits, dirichlet_samples)
    else:
      swapped_vector, reference_samples = self.get_ars_swapped_vector(
          encoder_logits, dirichlet_samples)
    swapped_vector = tf.reshape(
        swapped_vector,
        [batch_size * num_classes, num_variables])
    # input_batch is of the size [batch_size, observation_dims]
    tiled_inputs = tf.tile(input_batch[:, None, :],
                           [1, num_classes, 1])
    tiled_inputs = tf.reshape(
        tiled_inputs,
        [batch_size * num_classes, -1])
    loss_vector = self.get_elbo(tiled_inputs, swapped_vector)
    loss_vector = tf.reshape(loss_vector,
                             [batch_size, num_classes])
    swapped_vector = tf.reshape(swapped_vector,
                                [batch_size, num_classes, num_variables])

    tilde_loss = (loss_vector
                  - tf.math.reduce_mean(loss_vector, axis=-1, keepdims=True))
    tilde_dirichlet = 1. - num_classes * reference_samples
    if partial_integration:
      tilde_dirichlet *= tf.cast(tf.logical_not(tf.reduce_all(
          swapped_vector == swapped_vector[:, 0, :][:, None, :],
          axis=1)), dtype=tf.float32)
    layer_grad = tilde_loss[:, None, :] * tilde_dirichlet[:, :, None]
    return layer_grad

  @property
  def encoder_vars(self):
    return self.encoder.trainable_variables

  @property
  def decoder_vars(self):
    return self.decoder.trainable_variables

  @property
  def prior_vars(self):
    return self.prior_dist.trainable_variables

