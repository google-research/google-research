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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer library imported from flax examples."""
import functools
from typing import Any, Callable, Optional, Tuple

from absl import flags
from flax import linen as nn
from flax import struct
from incontext import utils
from incontext.flax import self_attention_patch as fpatch
import jax
import jax.numpy as jnp
import numpy as np

Array = utils.Array
Dtype = utils.Dtype

flags.DEFINE_integer("n_layers", default=12, help="n_layers.")
flags.DEFINE_integer("n_heads", default=8, help="n_heads.")
flags.DEFINE_integer("hidden_size", default=512, help="hidden_size.")
flags.DEFINE_bool("norm_first", default=True, help="Layer norms comes first.")
flags.DEFINE_bool(
    "final_layer_norm", default=False, help="Apply last layer norm")
flags.DEFINE_bool(
    "disable_layer_norms", default=False, help="disable layer norms.")
flags.DEFINE_bool("inner_dim", default=None, help="MLP inner dim")

flags.DEFINE_string(
    "kernel_init", default="uniform_scaling", help="Initializer for kernels")
flags.DEFINE_string(
    "bias_init", default="uniform_scaling", help="Initializer for bias")
flags.DEFINE_string(
    "linear_w_init",
    default="uniform_scaling",
    help="Initializer for linear layers")
flags.DEFINE_string(
    "linear_bias_init",
    default="uniform_scaling",
    help="Initializer for bias in linear layer")
flags.DEFINE_string(
    "posemb_init",
    default="uniform_scaling",
    help="Initializer for positional embedding init")

flags.DEFINE_string(
    "activation_fn", default="gelu", help="Activation function.")


def uniform_scaling(dtype = jnp.float_):
  """Uniform scaling initializer.

  Initializes by sampling from a uniform distribution, but with the variance
  scaled by the inverse square root of the number of input units, multiplied
  by
  the scale.
  Args:
    dtype (Dtype): data type of the numpy array.

  Returns:
    Callable: initializer.
  """

  def init(key, shape, dtype=dtype):
    if len(shape) == 1:
      input_size = shape[0]
    else:
      input_size = np.prod(shape[:-1])
    max_val = np.sqrt(1 / input_size)
    return jax.random.uniform(key, shape, dtype, -max_val, max_val)

  return init


def sinusoidal_init(max_len = 128):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype = np.float32):
    """Sinusoidal init."""
    del key
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=dtype)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


def nn_init_parser(init_str):
  """nn initializer from string names."""
  if init_str == "uniform_scaling":
    init_fn = uniform_scaling()
  elif init_str == "ones":
    init_fn = nn.initializers.ones
  elif init_str == "zeros":
    init_fn = nn.initializers.zeros
  elif init_str.startswith("normal"):
    std = init_str.replace("normal(", "").replace(")", "")
    init_fn = nn.initializers.normal(float(std))
  else:
    raise ValueError("Unknown nn initializer!")
  return init_fn


def nn_activation_parser(activation_str):
  """nn activation functions from string names."""
  if activation_str == "gelu":
    activation_fn = functools.partial(nn.gelu, approximate=True)
  elif activation_str == "relu":
    activation_fn = nn.relu
  elif activation_str == "tanh":
    activation_fn = nn.tanh
  elif activation_str == "softplus":
    activation_fn = nn.softplus
  elif activation_str == "sofu":

    def activation_fn(x):
      return nn.softmax(x, axis=-1) * x
  else:
    raise ValueError("Unknown activation function!")
  return activation_fn


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  dtype: Any = jnp.float32
  num_heads: int = 8
  num_layers: int = 16
  hidden_size: int = 2048
  max_len: int = 128
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  kernel_init: Callable[[Any, Any, Dtype], Array] = uniform_scaling()
  bias_init: Callable[[Any, Any, Dtype], Array] = uniform_scaling()
  linear_w_init: Callable[[Any, Any, Dtype], Array] = uniform_scaling()
  linear_bias_init: Callable[[Any, Any, Dtype], Array] = uniform_scaling()
  layer_norm_init: Callable[[Any, Any, Dtype], Array] = nn.initializers.ones
  posemb_init: Optional[Callable[[Any, Any, Dtype],
                                 Array]] = nn.initializers.normal(1.0)
  inner_dim: Optional[int] = None
  loss_on_x_steps: bool = False
  norm_first: bool = False
  disable_layer_norms: bool = False
  final_layer_norm: bool = False
  activation_fn: Callable[Ellipsis, Array] = functools.partial(
      nn.gelu, approximate=True)


class PositionEmbeddings(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs):
    """Applies PositionEmbeddings module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.
    Args:
      inputs: input data.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    config = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, "Number of dimensions should be 3, but it is: %d" % inputs.ndim
    length = inputs.shape[1]
    pos_emb_shape = (1, config.max_len, inputs.shape[-1])

    pos_embedding = self.param("pos_embedding", config.posemb_init,
                               pos_emb_shape)

    pe = pos_embedding[:, :length, :]
    return inputs + pe


class MLP(nn.Module):
  """Transformer MLP / feed-forward block.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      out_dim: optionally specify out dimension.
  """

  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self,
               inputs,
               deterministic = True):
    """Applies Transformer MLP module."""
    config = self.config
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    if config.inner_dim is None:
      inner_dim = config.hidden_size * 4
    else:
      inner_dim = config.inner_dim
    x = nn.Dense(
        inner_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
    )(
        inputs)
    x = config.activation_fn(x)
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
    )(
        x)
    output = nn.Dropout(rate=config.dropout_rate)(
        output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(
      self,
      inputs,
      mask = None,
      deterministic = None,
      return_attention = False,
  ):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      mask: self attention mask.
      deterministic: if true dropout is applied otherwise not.
      return_attention: returns attention weights.

    Returns:
      output after transformer encoder block.
    """
    config = self.config

    # Attention block.
    assert inputs.ndim == 3, "Number of dimensions should be 3, but it is: %d" % inputs.ndim
    assert (config.hidden_size % config.num_heads == 0
           ), "hidden size %d should be divisible by num_heads %d" % (
               config.hidden_size,
               config.num_heads,
           )
    x = inputs

    if config.norm_first and not config.disable_layer_norms:
      x = nn.LayerNorm(
          dtype=config.dtype,
          epsilon=1e-5,
      )(
          x)

    y, attn_weights = fpatch.SelfAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_features=config.hidden_size // config.num_heads,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
    )(x,
      mask=mask,
      deterministic=deterministic,
      return_attention=return_attention)

    if not config.norm_first and not config.disable_layer_norms:
      x = nn.LayerNorm(
          dtype=config.dtype,
          epsilon=1e-5,
      )(
          x)

    x = x + y  # + (x * y).sum(axis=-1, keepdims=True)

    if config.norm_first and not config.disable_layer_norms:
      x = nn.LayerNorm(
          dtype=config.dtype,
          epsilon=1e-5,
      )(
          x)

    y = MLP(config=config)(x, deterministic=deterministic)

    if not config.norm_first and not config.disable_layer_norms:
      x = nn.LayerNorm(
          dtype=config.dtype,
          epsilon=1e-5,
      )(
          x)

    x = x + y  # + (x * y).sum(axis=-1, keepdims=True)
    return x, attn_weights


class Transformer(nn.Module):
  """Transformer Model for sequence tagging."""

  config: TransformerConfig

  @nn.compact
  def __call__(self,
               *,
               inputs,
               mask,
               train,
               return_attention = False):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      mask: mask tensor
      train: if it is training.
      return_attention: return attention weights.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 3  # (batch, len, hidden_size)

    config = self.config

    # x = inputs.astype("int32")
    x = inputs
    x = nn.Dense(
        config.hidden_size,
        kernel_init=config.linear_w_init,
        bias_init=config.linear_bias_init,
    )(
        x)
    x = PositionEmbeddings(config)(x)
    outs = [x]
    attns = []
    for _ in range(config.num_layers):
      x, attn_weights = Encoder1DBlock(config)(
          x,
          mask=mask,
          deterministic=not train,
          return_attention=return_attention)
      outs.append(x)
      attns.append(attn_weights)

    if config.final_layer_norm:
      x = nn.LayerNorm(
          dtype=config.dtype,
          epsilon=1e-5,
      )(
          x)
    return x, outs, attns


def create_learning_rate_scheduler(
    *,
    num_warmup_steps,
    num_training_steps,
    num_cycles = 0.5,
    base_learning_rate = 0.001,
):
  """creates learning rate schedule.

  Args:
    num_warmup_steps: number of warmup steps.
    num_training_steps: number of total steps.
    num_cycles: num cosine cycles.
    base_learning_rate: float, the starting constant for the lr schedule.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """

  def step_fn(current_step):
    """Step to learning rate function."""
    in_linear_phase = jnp.less_equal(current_step, num_warmup_steps)

    linear_scale = jnp.array(current_step, float) / max(1.0, num_warmup_steps)

    after_progress = jnp.array(current_step - num_warmup_steps, float) / max(
        1.0, num_training_steps - num_warmup_steps)
    after_scale = jnp.maximum(
        0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * num_cycles * 2.0 * after_progress)))
    scale = in_linear_phase * linear_scale + (1.0 -
                                              in_linear_phase) * after_scale
    return scale * base_learning_rate

  return step_fn


def create_learning_rate_scheduler_v2(
    factors = "constant * linear_warmup",
    base_learning_rate = 0.5,
    warmup_steps = 1000,
    decay_factor = 0.5,
    steps_per_decay = 20000,
    steps_per_cycle = 100000,
    step_offset = 0,
    min_learning_rate = 1e-8,
):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * linear_decay: linear decay from warmup_steps with decay_factor slope. Note
      this option implies 'constant * linear_warmup', and should not be used
      in
      in conjunction with `constant` or `linear_warmup` factors.
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.
  Args:
    factors: string, factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.
    step_offset: int, an offset that the step parameters to this function are
      relative to.
    min_learning_rate: float, minimum learning rate to output. Useful for cases
      when a decay function is (mis)configured to decay to non-positive values.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split("*")]

  def step_fn(step):
    """Step to learning rate function."""
    step = jnp.maximum(0, step - step_offset)
    ret = 1.0
    for name in factors:
      if name == "constant":
        ret *= base_learning_rate
      elif name == "linear_warmup":
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == "linear_decay":
        ret *= base_learning_rate * jnp.minimum(
            step / warmup_steps, 1.0 + decay_factor * (warmup_steps - step))
      elif name == "rsqrt_decay":
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "rsqrt_normalized_decay":
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "decay_every":
        ret *= decay_factor**(step // steps_per_decay)
      elif name == "cosine_decay":
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError("Unknown factor %s." % name)
    ret = jnp.maximum(ret, min_learning_rate)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn
