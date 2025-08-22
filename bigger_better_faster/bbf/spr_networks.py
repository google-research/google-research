# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Various networks for Jax Dopamine SPR agents."""

import collections
import enum
import functools
import time
from typing import Any, Sequence, Tuple

from absl import logging
from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp
import numpy as onp


SPROutputType = collections.namedtuple(
    'RL_network',
    ['q_values', 'logits', 'probabilities', 'latent', 'representation'],
)
PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any


class EncoderType(str, enum.Enum):
  DQN = 'dqn'
  IMPALA = 'impala'
  RESNET = 'resnet'


class InitializerType(str, enum.Enum):
  XAVIER_UNIFORM = 'xavier_uniform'
  KAIMING_UNIFORM = 'kaiming_uniform'
  XAVIER_NORMAL = 'xavier_normal'
  KAIMING_NORMAL = 'kaiming_normal'
  ORTHOGONAL = 'orthogonal'


class NormType(str, enum.Enum):
  LN = 'ln'
  BN = 'bn'
  GN = 'gn'
  NONE = 'none'


def _absolute_dims(rank, dims):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


# --------------------------- < Data Augmentation >-----------------------------


def _random_crop(key, img, cropped_shape):
  """Random crop an image."""
  _, width, height = cropped_shape[:-1]
  key_x, key_y = random.split(key, 2)
  x = random.randint(key_x, shape=(), minval=0, maxval=img.shape[1] - width)
  y = random.randint(key_y, shape=(), minval=0, maxval=img.shape[2] - height)
  return img[:, x:x + width, y:y + height]


# @functools.partial(jax.jit, static_argnums=(3,))
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def _crop_with_indices(img, x, y, cropped_shape):
  cropped_image = jax.lax.dynamic_slice(img, [x, y, 0], cropped_shape[1:])
  return cropped_image


def _per_image_random_crop(key, img, cropped_shape):
  """Random crop an image."""
  batch_size, width, height = cropped_shape[:-1]
  key_x, key_y = random.split(key, 2)
  x = random.randint(
      key_x, shape=(batch_size,), minval=0, maxval=img.shape[1] - width)
  y = random.randint(
      key_y, shape=(batch_size,), minval=0, maxval=img.shape[2] - height)
  return _crop_with_indices(img, x, y, cropped_shape)


def _intensity_aug(key, x, scale=0.05):
  """Follows the code in Schwarzer et al. (2020) for intensity augmentation."""
  r = random.normal(key, shape=(x.shape[0], 1, 1, 1))
  noise = 1.0 + (scale * jnp.clip(r, -2.0, 2.0))
  return x * noise


@functools.partial(jax.jit)
def drq_image_aug(key, obs, img_pad=4):
  """Padding and cropping for DrQ."""
  flat_obs = obs.reshape(-1, *obs.shape[-3:])
  paddings = [(0, 0), (img_pad, img_pad), (img_pad, img_pad), (0, 0)]
  cropped_shape = flat_obs.shape
  # The reference uses ReplicationPad2d in pytorch, but it is not available
  # in Jax. Use 'edge' instead.
  flat_obs = jnp.pad(flat_obs, paddings, 'edge')
  key1, key2 = random.split(key, num=2)
  cropped_obs = _per_image_random_crop(key2, flat_obs, cropped_shape)
  # cropped_obs = _random_crop(key2, flat_obs, cropped_shape)
  aug_obs = _intensity_aug(key1, cropped_obs)
  return aug_obs.reshape(*obs.shape)


# --------------------------- < NoisyNetwork >---------------------------------


@gin.configurable
class NoisyNetwork(nn.Module):
  """Noisy Network from Fortunato et al. (2018)."""

  features: int = 512
  dtype: Dtype = jnp.float32
  initializer: Any = nn.initializers.xavier_uniform()
  factorized_sigma: bool = False

  @staticmethod
  def sample_noise(key, shape):
    return random.normal(key, shape)

  @staticmethod
  def f(x):
    # See (10) and (11) in Fortunato et al. (2018).
    return jnp.multiply(jnp.sign(x), jnp.power(jnp.abs(x), 0.5))

  @nn.compact
  def __call__(self, x, rng_key, bias=True, kernel_init=None, eval_mode=False):
    x = x.astype(self.dtype)

    def mu_init(key, shape):
      # Initialization of mean noise parameters (Section 3.2)
      low = -1 / jnp.power(x.shape[-1], 0.5)
      high = 1 / jnp.power(x.shape[-1], 0.5)
      return random.uniform(key, minval=low, maxval=high, shape=shape)

    def sigma_init(key, shape, dtype=jnp.float32):  # pylint: disable=unused-argument
      """Initializes sigma noise parameters.

      See the noisy nets paper, Section 3.2, for details.

      Args:
        key: Jax PRNG Key; unused, kept only to match type specs.
        shape: Weight shape (tuple).
        dtype: Dtype (float32, float16, bfloat16).

      Returns:
        Initialized weights.
      """
      return jnp.ones(shape, dtype) * (0.5 / onp.sqrt(x.shape[-1]))

    # Factored gaussian noise in (10) and (11) in Fortunato et al. (2018).
    p = NoisyNetwork.sample_noise(rng_key, [x.shape[-1], 1])
    q = NoisyNetwork.sample_noise(rng_key, [1, self.features])
    f_p = NoisyNetwork.f(p)
    f_q = NoisyNetwork.f(q)

    if self.factorized_sigma:
      w_sigma_p = self.param('kernell', sigma_init, p.shape)
      w_sigma_q = self.param('kernell', sigma_init, q.shape)
      w_epsilon = jnp.multiply(w_sigma_p, f_p) * jnp.multiply(w_sigma_q, f_q)
    else:
      w_epsilon = f_p * f_q
    b_epsilon = jnp.squeeze(f_q)

    # See (8) and (9) in Fortunato et al. (2018) for output computation.
    if self.initializer is None:
      initializer = mu_init
    else:
      initializer = self.initializer
    w_mu = self.param('kernel', initializer, (x.shape[-1], self.features))
    w_sigma = self.param('kernell', sigma_init, (x.shape[-1], self.features))
    w_epsilon = jnp.where(
        eval_mode,
        onp.zeros(shape=(x.shape[-1], self.features), dtype=onp.float32),
        w_epsilon,
    )
    w = w_mu + jnp.multiply(w_sigma, w_epsilon)
    ret = jnp.matmul(x, w)

    b_epsilon = jnp.where(
        eval_mode,
        onp.zeros(shape=(self.features,), dtype=onp.float32),
        b_epsilon,
    )
    b_mu = self.param('bias', mu_init, (self.features,))
    b_sigma = self.param('biass', sigma_init, (self.features,))
    b = b_mu + jnp.multiply(b_sigma, b_epsilon)
    return jnp.where(bias, ret + b, ret)


# --------------------------- < RainbowNetwork >--------------------------------
class FeatureLayer(nn.Module):
  """Layer encapsulating either a noisy linear or a standard linear layer.

  Attributes:
    net: The layer (nn.Module).
    noisy: Whether noisy nets are used.
    features: Output size.
    dtype: Dtype (float32 | float16 | bfloat16)
    initializer: Jax initializer.
  """
  noisy: bool
  features: int
  dtype: Dtype = jnp.float32
  initializer: Any = nn.initializers.xavier_uniform()

  def setup(self):
    if self.noisy:
      self.net = NoisyNetwork(
          features=self.features, dtype=self.dtype, initializer=self.initializer
      )
    else:
      self.net = nn.Dense(
          self.features,
          kernel_init=self.initializer,
          dtype=self.dtype,
      )

  def __call__(self, x, key, eval_mode):
    if self.noisy:
      return self.net(x, key, True, None, eval_mode)
    else:
      return self.net(x)


class LinearHead(nn.Module):
  """A linear DQN head supporting dueling networks and noisy networks.

  Attributes:
    advantage: Advantage layer.
    value: Value layer (if dueling).
    noisy: Whether to use noisy nets.
    dueling: Bool, whether to use dueling networks.
    num_actions: int, size of action space.
    num_atoms: int, number of value prediction atoms per action.
    dtype: Jax dtype.
    initializer: Jax initializer.
  """
  noisy: bool
  dueling: bool
  num_actions: int
  num_atoms: int
  dtype: Dtype = jnp.float32
  initializer: Any = nn.initializers.xavier_uniform()

  def setup(self):
    if self.dueling:
      self.advantage = FeatureLayer(
          self.noisy,
          self.num_actions * self.num_atoms,
          dtype=self.dtype,
          initializer=self.initializer,
      )
      self.value = FeatureLayer(
          self.noisy,
          self.num_atoms,
          dtype=self.dtype,
          initializer=self.initializer,
      )
    else:
      self.advantage = FeatureLayer(
          self.noisy,
          self.num_actions * self.num_atoms,
          dtype=self.dtype,
          initializer=self.initializer,
      )

  def __call__(self, x, key, eval_mode):
    if self.dueling:
      adv = self.advantage(x, key, eval_mode)
      value = self.value(x, key, eval_mode)
      adv = adv.reshape((self.num_actions, self.num_atoms))
      value = value.reshape((1, self.num_atoms))
      logits = value + (adv - (jnp.mean(adv, -2, keepdims=True)))
    else:
      x = self.advantage(x, key, eval_mode)
      logits = x.reshape((self.num_actions, self.num_atoms))
    return logits


def process_inputs(x, data_augmentation=False, rng=None, dtype=jnp.float32):
  """Input normalization and if specified, data augmentation."""

  if dtype == 'float32':
    dtype = jnp.float32
  elif dtype == 'float16':
    dtype = jnp.float16
  elif dtype == 'bfloat16':
    dtype = jnp.bfloat16

  out = x.astype(dtype) / 255.0
  if data_augmentation:
    if rng is None:
      raise ValueError('Pass rng when using data augmentation')
    out = drq_image_aug(rng, out)
  return out


def renormalize(tensor, has_batch=False):
  shape = tensor.shape
  if not has_batch:
    tensor = jnp.expand_dims(tensor, 0)
  tensor = tensor.reshape(tensor.shape[0], -1)
  max_value = jnp.max(tensor, axis=-1, keepdims=True)
  min_value = jnp.min(tensor, axis=-1, keepdims=True)
  return ((tensor - min_value) / (max_value - min_value + 1e-5)).reshape(*shape)


class ConvTMCell(nn.Module):
  """MuZero-style transition model for SPR."""

  num_actions: int
  latent_dim: int
  renormalize: bool
  dtype: Dtype = jnp.float32
  initializer: Any = nn.initializers.xavier_uniform()

  @nn.compact
  def __call__(self, x, action, eval_mode=False, key=None):
    sizes = [self.latent_dim, self.latent_dim]
    kernel_sizes = [3, 3]
    stride_sizes = [1, 1]

    action_onehot = jax.nn.one_hot(action, self.num_actions)
    action_onehot = jax.lax.broadcast(action_onehot, (x.shape[-3], x.shape[-2]))
    x = jnp.concatenate([x, action_onehot], -1)
    for layer in range(1):
      x = nn.Conv(
          features=sizes[layer],
          kernel_size=(kernel_sizes[layer], kernel_sizes[layer]),
          strides=(stride_sizes[layer], stride_sizes[layer]),
          kernel_init=self.initializer,
          dtype=self.dtype,
      )(x)
      x = nn.relu(x)
    x = nn.Conv(
        features=sizes[-1],
        kernel_size=(kernel_sizes[-1], kernel_sizes[-1]),
        strides=(stride_sizes[-1], stride_sizes[-1]),
        kernel_init=self.initializer,
        dtype=self.dtype,
    )(x)
    x = nn.relu(x)

    if self.renormalize:
      x = renormalize(x)

    return x, x


@gin.configurable
class RainbowCNN(nn.Module):
  """DQN or Rainbow-style 3-layer CNN encoder.

  Attributes:
    padding: Padding style.
    width_scale: Float, width scale.
    dtype: Jax dtype.
    dropout: Float, dropout probability. 0 to disable.
    initializer: Jax initializer.
  """
  padding: Any = 'SAME'
  dims = (32, 64, 64)
  width_scale: int = 1
  dtype: Dtype = jnp.float32
  dropout: float = 0.0
  initializer: Any = nn.initializers.xavier_uniform()

  @nn.compact
  def __call__(self, x, deterministic=None):
    # x = x[None, Ellipsis]
    kernel_sizes = [8, 4, 3]
    stride_sizes = [4, 2, 1]
    for layer in range(3):
      x = nn.Conv(
          features=int(self.dims[layer] * self.width_scale),
          kernel_size=(kernel_sizes[layer], kernel_sizes[layer]),
          strides=(stride_sizes[layer], stride_sizes[layer]),
          kernel_init=self.initializer,
          padding=self.padding,
          dtype=self.dtype,
      )(x)
      x = nn.Dropout(self.dropout, broadcast_dims=(-3, -2))(x, deterministic)
      x = nn.relu(x)
    return x


@gin.configurable
class SpatialLearnedEmbeddings(nn.Module):
  """A learned spatial embedding class that can replace flattens or pooling.

  Attributes:
    num_features: Number of features to extract per channel.
    param_dtype: Jax dtype.
    use_bias: Whether the embeddings should have a bias term.
    initializer: Jax initializer.
  """
  num_features: int = 8
  param_dtype: Any = jnp.float32
  use_bias: bool = True
  initializer: Any = nn.initializers.xavier_uniform()

  @nn.compact
  def __call__(self, spatial_latent):
    """features is B x H x W X C."""
    height = spatial_latent.shape[-3]
    width = spatial_latent.shape[-2]
    channels = spatial_latent.shape[-1]
    kernel = self.param(
        'kernel',
        self.initializer,
        (height, width, channels, self.num_features),
        self.param_dtype,
    )
    if self.use_bias:
      bias = self.param(
          'bias',
          nn.initializers.zeros,
          (channels * self.num_features,),
          self.param_dtype,
      )
    else:
      bias = 0

    spatial_latent = jnp.expand_dims(spatial_latent, -1)
    features = jnp.sum(spatial_latent * kernel, axis=(-4, -3))
    features = features.reshape(*features.shape[:-2], -1) + bias
    return features


@gin.configurable
class ImpalaCNN(nn.Module):
  """ResNet encoder based on Impala.

  Attributes:
    width_scale: Float, width scale relative to the default.
    dims: Dimensions for each stage.
    num_blocks: Number of resblocks per stage.
    norm_type: normalization to use. `none` to disable, otherwise options are
      'ln', 'bn' and 'gn'.
    dtype: Jax Dtype.
    fixup_init: Whether to do a fixup-style init (final layer of each resblock
      has weights set to 0).
    dropout: Dropout probability in [0, 1]. 0 to disable dropout.
    initializer: Jax initializer.
  """
  width_scale: int = 1
  dims: Tuple[int, Ellipsis] = (16, 32, 32)
  num_blocks: int = 2
  norm_type: str = 'none'
  dtype: Dtype = jnp.float32
  fixup_init: bool = False
  dropout: float = 0.0
  initializer: Any = nn.initializers.xavier_uniform()

  @nn.compact
  def __call__(self, x, deterministic=None):
    for width in self.dims:
      x = ResidualStage(
          dims=int(width * self.width_scale),
          num_blocks=self.num_blocks,
          dtype=self.dtype,
          norm_type=self.norm_type,
          dropout=self.dropout,
          fixup_init=self.fixup_init,
          initializer=self.initializer,
      )(x, deterministic)
    x = nn.relu(x)
    return x


class ResidualStage(nn.Module):
  """A single residual stage for an Impala-style ResNet.

  Attributes:
    dims: Number of channels.
    num_blocks: Number of blocks in the stage.
    use_max_pooling: Whether to pool (downsample) before the blocks.
    norm_type: Normalization type, in {'none', 'bn', 'ln', 'gn'}.
    dtype: Jax dtype.
    fixup_init: Whether to initialize the last weights in each block to 0.
    dropout: Dropout prob in [0, 1]. 0 disables.
    initializer: Jax initializer.
  """

  dims: int
  num_blocks: int
  use_max_pooling: bool = True
  norm_type: str = 'none'
  dtype: Dtype = jnp.float32
  fixup_init: bool = False
  dropout: float = 0.0
  initializer: Any = nn.initializers.xavier_uniform()

  @nn.compact
  def __call__(self, x, deterministic=None):
    if self.fixup_init:
      final_initializer = nn.initializers.zeros
    else:
      final_initializer = self.initializer
    if self.norm_type == NormType.LN:
      norm = functools.partial(nn.LayerNorm,
                               epsilon=1e-5,
                               dtype=self.dtype,
                               )
    elif self.norm_type == NormType.GN:
      norm = functools.partial(
          nn.GroupNorm,
          epsilon=1e-5,
          num_groups=None,
          group_size=8,
          dtype=self.dtype,
      )
    elif self.norm_type == NormType.NONE:

      def norm(*args, **kwargs):  # pylint: disable=unused-argument
        return lambda x: x

    conv_out = nn.Conv(
        features=self.dims,
        kernel_size=(3, 3),
        strides=1,
        kernel_init=self.initializer,
        padding='SAME',
        dtype=self.dtype,
    )(x)
    if self.use_max_pooling:
      conv_out = nn.max_pool(
          conv_out, window_shape=(3, 3), padding='SAME', strides=(2, 2))

    for _ in range(self.num_blocks):
      block_input = conv_out
      conv_out = nn.relu(conv_out)
      conv_out = norm()(conv_out)
      conv_out = nn.Dropout(
          self.dropout, broadcast_dims=(-3, -2))(conv_out, deterministic)
      conv_out = nn.Conv(
          features=self.dims,
          kernel_size=(3, 3),
          strides=1,
          kernel_init=self.initializer,
          padding='SAME',
          dtype=self.dtype,
      )(conv_out)
      conv_out = nn.relu(conv_out)
      conv_out = norm()(conv_out)
      conv_out = nn.Conv(
          features=self.dims,
          kernel_size=(3, 3),
          strides=1,
          kernel_init=final_initializer,
          padding='SAME',
          dtype=self.dtype,
      )(conv_out)
      conv_out += block_input
    return conv_out


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: Any
  norm: Any
  act: Any
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm()(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters, (1, 1), self.strides, name='conv_proj')(
              residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


@gin.configurable
class ResNetEncoder(nn.Module):
  """ResNet encoder defaulting to ResNet 18."""
  stage_sizes: Sequence[int] = (2, 2, 2, 2)
  num_filters: int = 64
  norm: str = 'group'
  width_scale: float = 1
  act: Any = nn.relu
  conv: Any = nn.Conv
  strides = (2, 2, 2, 1, 1)
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, deterministic=None):
    del deterministic
    logging.info(
        'Creating ResNetEncoder with %s stage_sizes, %s num_filters, %s'
        'norm, %s width_scale, %s strides', self.stage_sizes, self.num_filters,
        self.norm, self.width_scale, self.strides)

    if self.norm == 'batch':
      norm = functools.partial(
          nn.BatchNorm, use_running_average=False, momentum=0.9,
          epsilon=1e-5, dtype=self.dtype)
    elif self.norm == 'group':
      norm = functools.partial(
          nn.GroupNorm,
          epsilon=1e-5,
          num_groups=None,
          group_size=8,
          dtype=self.dtype,
      )
    elif self.norm == 'layer':
      norm = functools.partial(nn.LayerNorm, epsilon=1e-5, dtype=self.dtype)
    else:
      raise ValueError('norm not found')

    x = nn.Conv(
        int(self.num_filters*self.width_scale),
        (7, 7), (self.strides[0], self.strides[0]),
        padding=[(3, 3), (3, 3)],
        name='conv_init')(x)

    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(
        x, (3, 3), strides=(self.strides[1], self.strides[1]), padding='SAME')

    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        stride = (self.strides[i + 1],
                  self.strides[i + 1]) if i > 0 and j == 0 else (1, 1)
        x = ResNetBlock(
            int(self.num_filters * 2**i * self.width_scale),
            strides=stride,
            conv=self.conv,
            norm=norm,
            act=self.act)(x)
    return x


class TransitionModel(nn.Module):
  """An SPR-style transition model.

  Attributes:
    num_actions: Size of action conditioning input.
    latent_dim: Number of channels.
    renormalize: Whether to renormalize outputs to [0, 1] as in MuZero.
    dtype: Jax dtype.
    initializer: Jax initializer.
  """
  num_actions: int
  latent_dim: int
  renormalize: bool
  dtype: Dtype = jnp.float32
  initializer: Any = nn.initializers.xavier_uniform()

  @nn.compact
  def __call__(self, x, action):
    scan = nn.scan(
        ConvTMCell,
        in_axes=0,
        out_axes=0,
        variable_broadcast=['params'],
        split_rngs={'params': False},
    )(
        latent_dim=self.latent_dim,
        num_actions=self.num_actions,
        renormalize=self.renormalize,
        dtype=self.dtype,
        initializer=self.initializer,
    )
    return scan(x, action)


@gin.configurable
class RainbowDQNNetwork(nn.Module):
  """Jax Rainbow network for Full Rainbow.

  Attributes:
      num_actions: int, number of actions the agent can take at any state.
      num_atoms: int, the number of buckets of the value function distribution.
      noisy: bool, Whether to use noisy networks.
      dueling: bool, Whether to use dueling network architecture.
      distributional: bool, whether to use distributional RL.
  """

  num_actions: int
  num_atoms: int
  noisy: bool
  dueling: bool
  distributional: bool
  renormalize: bool = False
  padding: Any = 'SAME'
  encoder_type: str = 'dqn'
  hidden_dim: int = 512
  width_scale: float = 1.0
  dtype: Dtype = jnp.float32
  use_spatial_learned_embeddings: bool = False
  initializer_type: str = 'xavier_uniform'

  def setup(self):
    if self.initializer_type == InitializerType.XAVIER_UNIFORM:
      initializer = nn.initializers.xavier_uniform()
    elif self.initializer_type == InitializerType.XAVIER_NORMAL:
      initializer = nn.initializers.xavier_normal()
    elif self.initializer_type == InitializerType.KAIMING_UNIFORM:
      initializer = nn.initializers.kaiming_uniform()
    elif self.initializer_type == InitializerType.KAIMING_NORMAL:
      initializer = nn.initializers.kaiming_normal()
    elif self.initializer_type == InitializerType.ORTHOGONAL:
      initializer = nn.initializers.orthogonal()
    else:
      raise NotImplementedError(
          'Unsupported initializer: {}'.format(self.initializer_type)
      )

    if self.encoder_type == EncoderType.DQN:
      self.encoder = RainbowCNN(
          padding=self.padding,
          width_scale=self.width_scale,
          dtype=self.dtype,
          initializer=initializer,
      )
      latent_dim = self.encoder.dims[-1] * self.width_scale
    elif self.encoder_type == EncoderType.IMPALA:
      self.encoder = ImpalaCNN(
          width_scale=self.width_scale,
          dtype=self.dtype,
          initializer=initializer,
      )
      latent_dim = self.encoder.dims[-1] * self.width_scale
    elif self.encoder_type == EncoderType.RESNET:
      self.encoder = ResNetEncoder(
          width_scale=self.width_scale,
          dtype=self.dtype,
      )
      latent_dim = (self.encoder.num_filters
                    * self.width_scale
                    * 2**(len(self.encoder.stage_sizes) - 1))
    else:
      raise NotImplementedError()

    self.transition_model = TransitionModel(
        num_actions=self.num_actions,
        latent_dim=int(latent_dim),
        renormalize=self.renormalize,
        dtype=self.dtype,
        initializer=initializer,
    )

    if self.use_spatial_learned_embeddings:
      self.embedder = SpatialLearnedEmbeddings(initializer=initializer)

    self.projection = FeatureLayer(
        self.noisy,
        int(self.hidden_dim),
        dtype=jnp.float32,
        initializer=initializer,
    )
    self.predictor = nn.Dense(
        int(self.hidden_dim), dtype=jnp.float32, kernel_init=initializer
    )
    self.head = LinearHead(
        num_actions=self.num_actions,
        num_atoms=self.num_atoms,
        noisy=self.noisy,
        dueling=self.dueling,
        dtype=jnp.float32,
        initializer=initializer,
    )

  def encode(self, x, eval_mode=False):
    latent = self.encoder(x, deterministic=not eval_mode)
    if self.renormalize:
      latent = renormalize(latent)
    return latent

  def encode_project(self, x, key, eval_mode):
    latent = self.encode(x, eval_mode)
    representation = self.flatten_spatial_latent(latent)
    return self.project(representation, key, eval_mode)

  def project(self, x, key, eval_mode):
    projected = self.projection(x, key=key, eval_mode=eval_mode)
    return projected

  @functools.partial(jax.vmap, in_axes=(None, 0, None, None))
  def spr_predict(self, x, key, eval_mode):
    projected = self.project(x, key, eval_mode)
    return self.predictor(projected)

  def spr_rollout(self, latent, actions, key):
    _, pred_latents = self.transition_model(latent, actions)

    representations = self.flatten_spatial_latent(pred_latents, True)
    predictions = self.spr_predict(representations, key, True)
    return predictions

  def flatten_spatial_latent(self, spatial_latent, has_batch=False):
    logging.info('Spatial latent shape: %s', str(spatial_latent.shape))
    if self.use_spatial_learned_embeddings:
      representation = self.embedder(spatial_latent)
    elif has_batch:
      representation = spatial_latent.reshape(spatial_latent.shape[0], -1)
    else:
      representation = spatial_latent.reshape(-1)
    logging.info(
        'Flattened representation shape: %s', str(representation.shape)
    )
    return representation

  @nn.compact
  def __call__(
      self,
      x,
      support,
      actions=None,
      do_rollout=False,
      eval_mode=False,
      key=None,
  ):
    # Generate a random number generation key if not provided
    if key is None:
      key = random.PRNGKey(int(time.time() * 1e6))

    spatial_latent = self.encode(x, eval_mode)
    representation = self.flatten_spatial_latent(spatial_latent)
    # Single hidden layer
    x = self.project(representation, key, eval_mode)
    x = nn.relu(x)

    logits = self.head(x, key, eval_mode)

    if do_rollout:
      spatial_latent = self.spr_rollout(spatial_latent, actions, key)

    if self.distributional:
      probabilities = jnp.squeeze(nn.softmax(logits))
      q_values = jnp.squeeze(jnp.sum(support * probabilities, axis=-1))
      return SPROutputType(
          q_values, logits, probabilities, spatial_latent, representation
      )

    q_values = jnp.squeeze(logits)
    return SPROutputType(q_values, None, None, spatial_latent, representation)
