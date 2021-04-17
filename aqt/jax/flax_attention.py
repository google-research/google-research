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

"""Fork of flax.nn.attention modules with optional quantization."""

import typing
from typing import Any, Callable, Iterable, Optional, Type, TypeVar

import dataclasses
from flax import linen as nn
from flax.linen import initializers
# TODO(malmaud): Remove reliance on these 'legacy' nn.attention methods
from flax.nn.attention import _make_causal_mask
from flax.nn.attention import make_padding_mask
import jax
from jax import lax
from jax import random
from jax._src.numpy import lax_numpy
import jax.numpy as jnp
import numpy as onp

from aqt.jax import flax_layers
from aqt.jax import get_bounds
from aqt.jax import quant_config
from aqt.jax import quantization
from aqt.jax import shape_utils
from aqt.jax import stats_tag
from aqt.jax.flax import struct as flax_struct
from aqt.jax.flax_layers import default_kernel_init
from aqt.jax.flax_layers import InitializerType
from aqt.jax.quantization import quantized_dynamic_dot_general
from aqt.jax.quantization import QuantOps
from aqt.jax.quantization import QuantType

T = TypeVar('T')

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass


@dataclass
class ExpHParams:
  """Hparams to customize exponential."""
  # Limit the total accumulated value of the sum of exponentials
  sum_high_bound: float
  # Defines low bound for (x-m) where x is an element of the incoming vector
  # and m is the max value in the vector.
  low_bound: float
  # Substract exponential evaluated a low bound (low_bound) from calculations so
  # that exponential calculations evaluated at (x-m = low_bound) converges to 0.
  clip_and_subtract: bool
  # Use this gradient as linear approximation of exponential.
  linear_gradient: float


@dataclass
class ReciprocalHParams:
  """Hparams to customize reciprocal."""
  # Use this gradient as linear approximation of exponential.
  linear_gradient: float
  # Defines low bound for input to reciprocal.
  low_bound: float


@dataclass
class SoftmaxQuantHParams:
  """Hyperparameters for quantized softmax operation used in attention."""

  # Floating-point precision to quantize intermediate computations to.
  # If None, intermediate computations will not be quantized.
  prec: Optional[QuantOps.FloatQuant.FloatPrec]

  # Floating-point precision to accumulate the sum reduction in softmax to. If
  # None, sums are accumulated based on the underlying device VPU (float32 for
  # TPU).
  reduction_prec: Optional[QuantOps.FloatQuant.FloatPrec]


@dataclass
class SoftmaxHParams:
  """Hparams to customize softmax."""
  exp_hparams: Optional[ExpHParams]
  reciprocal_hparams: Optional[ReciprocalHParams]
  quant_hparams: Optional[SoftmaxQuantHParams]


@dataclass
class DotProductAttnHParams:
  """HParams class to quantize DotProductAttnAqts."""
  # QuantOps hyperparameter to quantize query for attention weights
  # (i.e. quantizing Q for Q * K).
  attn_act_q: Optional[QuantOps.ActHParams]

  # QuantOps hyperparameter to quantize key for attention weights
  # (i.e. quantizing K for Q * K).
  attn_act_k: Optional[QuantOps.ActHParams]

  # QuantOps hyperparameter to quantize product Q*K (attn_weights).
  attn_act_probs: Optional[QuantOps.ActHParams]

  # QuantOps hyperparameter to quantize value.
  attn_act_v: Optional[QuantOps.ActHParams]

  # Quantization strategy, one of `fake_quant` or `aqt`.
  quant_type: QuantType

  # Custom softmax. Currently we support a linear approximation to exp and
  # reciprocal.
  # We also support downcast intermediate activations to a floating-point
  # format.
  softmax: Optional[SoftmaxHParams]
  # TODO(shivaniagrawal): Changed the strategy to AQT if quant_type is aqt.


def reciprocal(tensor, dtype, recip_hparams):
  """Generates a reciprocal function based on recip hyper params."""
  if recip_hparams is not None and recip_hparams.linear_gradient != 0:
    # Want: max(low_bound, -a*x+b) such that (-a*x+b) goes through
    # (1, 1)
    # Solution: max(low_bound, a+1- a*x) for arbitrary a>0.
    afull = jnp.full(tensor.shape,
                     recip_hparams.linear_gradient).astype(dtype)
    aplus1full = jnp.full(tensor.shape,
                          1 + recip_hparams.linear_gradient).astype(dtype)
    arecip = jnp.clip(
        lax.sub(aplus1full, lax.mul(afull, tensor)),
        recip_hparams.low_bound, 1.).astype(dtype)
  else:
    arecip = lax.reciprocal(tensor)
  return arecip


def exponential(tensor, dtype, exp_hparams):
  """Calculates an exponential approximation based on exp hyper params."""
  # If low_bound defined, it clips x-M.
  if exp_hparams.low_bound != 0:
    tensor = jnp.clip(tensor, exp_hparams.low_bound, 0.)

  # TODO(luispazos) Use standard calls to top level jnp functions.
  # pylint: disable=protected-access
  def make_constant(c):
    return lax_numpy._constant_like(tensor, c).astype(dtype)

  # If clip_and_subtract, replace exp(clip(x-M,low_bound)) term with
  # exp(clip(x-M,low_bound))-exp(low_bound).'
  if exp_hparams.clip_and_subtract:
    tensor = lax.sub(tensor, make_constant(onp.exp(exp_hparams.low_bound)))
  # If linear_gradient: use this gradient as linear approximation of
  # exponential.
  if exp_hparams.linear_gradient is not None and exp_hparams.linear_gradient != 0:
    # Want: max(0, a*x+b) such that a*x+b goes through (0, 1).
    #
    # This comes out to: max(0, a*x+1), for arbitrary a>0.
    one = jnp.full(tensor.shape, 1.).astype(dtype)
    gradient = jnp.full(tensor.shape,
                        exp_hparams.linear_gradient).astype(dtype)
    approx_exp = jnp.clip(lax.add(lax.mul(tensor, gradient), one), 0, 1)

  else:
    approx_exp = lax.exp(tensor)

  return approx_exp


def softmax(attn_weights, norm_dims, dtype, softmax_hparams,
            quant_context):
  """Normalizes attention."""
  a = attn_weights

  def unquantized_softmax(a):
    a = lax.exp(a -
                jax.scipy.special.logsumexp(a, axis=norm_dims, keepdims=True))
    return a.astype(dtype)

  # Quantize intermediate activations with QuantOps.
  # Currently only supports unscaled floating-point formats.
  def quantized_softmax(a):
    # We compute softmax as exp(x-max(x))/sum_i(exp(x_i-max(x))), quantizing
    # intermediate values. Note this differs from the log-domain
    # implementation of softmax used above.
    quant_hparams = softmax_hparams.quant_hparams
    fp_quant_config = QuantOps.FloatQuant(
        is_scaled=False, fp_spec=quant_hparams.prec)
    quant_ops = QuantOps.create_symmetric_fp(
        fp_quant=fp_quant_config, bounds=None)

    a = quant_ops.to_quantized(a, dtype=dtype)
    # Note that the max of a quantized vector is necessarily also quantized to
    # the same precision since the max of a vector must be an existing element
    # of the vector, so we don't need to explicitly insert a quantization
    # operator to the output of the max reduction.
    a_max = jnp.max(a, axis=norm_dims, keepdims=True)
    a_minus_max = quant_ops.to_quantized(a - a_max, dtype=dtype)
    a_exp = quant_ops.to_quantized(jnp.exp(a_minus_max), dtype=dtype)

    sum_exp_quantized_reduction = quantization.quantized_sum(
        a_exp, axis=norm_dims, keepdims=True, prec=quant_hparams.reduction_prec)
    sum_exp = quant_ops.to_quantized(sum_exp_quantized_reduction, dtype=dtype)

    inv_sum_exp = quant_ops.to_quantized(jnp.reciprocal(sum_exp), dtype=dtype)
    a_softmax = quant_ops.to_quantized(a_exp * inv_sum_exp, dtype=dtype)

    return a_softmax.astype(dtype)

  # If no params, return accurate Softmax.
  if softmax_hparams == SoftmaxHParams(None, None,
                                       None) or softmax_hparams is None:
    return unquantized_softmax(a)

  # TODO(shivaniagrawal): Partial sum quantization (if enabled) will happen for
  # the entire training run, even before the global activation start step.
  if softmax_hparams.quant_hparams is not None:
    return lax.cond(quant_context.quantize_acts, quantized_softmax,
                    unquantized_softmax, a)

  # Approximated Softmax
  exp_hparams = softmax_hparams.exp_hparams
  recip_hparams = softmax_hparams.reciprocal_hparams

  # Substract max value from dimensions to be normalized.
  shape = jax.util.subvals(onp.shape(a), zip(norm_dims, (1,) * len(norm_dims)))
  dimadd = lambda x: lax.reshape(x, shape)
  # pylint: disable=protected-access
  amax = lax.reduce(a, lax_numpy._constant_like(a, -onp.inf), lax.max,
                    norm_dims)
  amax = lax.select(lax.is_finite(amax), amax, lax.full_like(amax, 0))
  amax_singletons = dimadd(amax)
  asubmax = lax.sub(a, amax_singletons)

  # Calculate approximated exponential
  approx_exp = exponential(asubmax, dtype, exp_hparams)

  # If sum_high_bound: Upper clip bound for sum(exp(x-M)).
  asumexp = dimadd(
      lax.reduce(approx_exp, lax_numpy._constant_like(a, 0), lax.add,
                 norm_dims))

  if exp_hparams.sum_high_bound is not None and exp_hparams.sum_high_bound != 0:
    sum_low_bound = 1.
    if (exp_hparams.low_bound != 0) and exp_hparams.clip_and_subtract:
      sum_low_bound = 1 - onp.exp(exp_hparams.low_bound)
    asumexp = jnp.clip(asumexp, sum_low_bound, exp_hparams.sum_high_bound)

  # Approximation of reciprocal.
  arecip = reciprocal(asumexp, dtype, recip_hparams)
  return lax.mul(approx_exp, arecip).astype(dtype)


# Forked from flax.nn.attention.dot_product_attention
# https://github.com/google/flax/blob/65061e6128f6695eed441acf2bfffc3b1badd318/flax/nn/attention.py#L206
def dot_product_attention(query,
                          key,
                          value,
                          hparams,
                          quant_context,
                          paxis_name,
                          train,
                          key_padding_mask,
                          query_padding_mask,
                          attn_mask,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          broadcast_dropout=True,
                          dropout_rng=None,
                          dropout_rate=0.,
                          deterministic=False,
                          precision=None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    query: queries for calculating attention with shape of `[batch_size,
      sequence_length, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size,
      sequence_length, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size,
      sequence_length, num_heads, value_channels]`.
    hparams: hyperparameters used for quantization.
    quant_context: context for quantization.
    paxis_name: axis_name to which a user `pmaps` the parent module (model),
      refer to jax.pmap() for more documentation. This arg is used for
      get_bounds acts quantization (QuantOps.create_input_fake_quant)
    train: Whether model is training.
    key_padding_mask: boolean mask indicating which elements in 'key' and
      'value' are padding. Must have a shape compatible with 'key' and 'value'.
    query_padding_mask: boolean mask indicating which elements in `query` are
      padding (True means not padding).
    attn_mask: boolean mask indicating which elements of the calculated
      attention weight matrix should be used for collecting activation
      statistics. Should have a shape broadcast-compatible with '[bs,
      sequence_length, sequence_length]'. Must have a shape broadcast-compatible
      'query'.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout.
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, sequence_length, num_heads, value_channels]`.
  """
  batch_size, query_sequence_length, num_heads, channel_size = query.shape
  key_sequence_length = key.shape[1]
  shape_utils.assert_shapes_equal(
      key.shape, (batch_size, key_sequence_length, num_heads, channel_size))
  shape_utils.assert_shapes_equal(
      value.shape, (batch_size, key_sequence_length, num_heads, channel_size))
  if key_padding_mask is not None:
    shape_utils.assert_shapes_equal(key_padding_mask.shape,
                                    (batch_size, key_sequence_length, 1, 1))
  if query_padding_mask is not None:
    shape_utils.assert_shapes_equal(query_padding_mask.shape,
                                    (batch_size, query_sequence_length, 1, 1))

  if attn_mask is not None:
    shape_utils.assert_shapes_compatible(
        attn_mask.shape,
        (batch_size, 1, query_sequence_length, key_sequence_length))

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)

  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError('Attention axis must be between the batch '
                       'axis and the last-two axes.')
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(onp.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)

  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  shape_utils.assert_shapes_equal(
      key.shape, (batch_size, num_heads, key_sequence_length, channel_size))

  key_padding_mask_transposed = None
  query_padding_mask_transposed = None
  if key_padding_mask is not None:
    key_padding_mask_transposed = key_padding_mask.transpose(qk_perm)
    shape_utils.assert_shapes_equal(key_padding_mask_transposed.shape,
                                    (batch_size, 1, key_sequence_length, 1))

  if quant_context.collect_acts_stats:
    stats_tag.StatsTag(
        channel_axis=None, name='attn_act_k', update_stats=train)(
            key, mask=key_padding_mask_transposed)

  if query_padding_mask is not None:
    query_padding_mask_transposed = query_padding_mask.transpose(qk_perm)
    shape_utils.assert_shapes_equal(query_padding_mask_transposed.shape,
                                    (batch_size, 1, query_sequence_length, 1))

  key_get_bounds_params = get_bounds.GetBounds.Params(
      update_bounds=quant_context.update_bounds,
      update_stats=train,
      paxis_name=paxis_name,
      mask=key_padding_mask_transposed,
      module_name='K')

  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)
  shape_utils.assert_shapes_equal(
      value.shape, (batch_size, num_heads, channel_size, key_sequence_length))
  value_padding_mask_transposed = None
  if key_padding_mask is not None:
    value_padding_mask_transposed = key_padding_mask.transpose(v_perm)
    shape_utils.assert_shapes_equal(value_padding_mask_transposed.shape,
                                    (batch_size, 1, 1, key_sequence_length))

  if quant_context.collect_acts_stats:
    stats_tag.StatsTag(
        channel_axis=None, name='attn_act_v', update_stats=train)(
            value, mask=value_padding_mask_transposed)

  value_get_bounds_params = get_bounds.GetBounds.Params(
      update_bounds=quant_context.update_bounds,
      update_stats=train,
      paxis_name=paxis_name,
      mask=value_padding_mask_transposed,
      module_name='V')

  query = query / jnp.sqrt(depth).astype(dtype)
  query = query.transpose(qk_perm)
  shape_utils.assert_shapes_equal(
      query.shape, (batch_size, num_heads, query_sequence_length, channel_size))

  if quant_context.collect_acts_stats:
    stats_tag.StatsTag(
        channel_axis=None, name='attn_act_q', update_stats=train)(
            query, mask=query_padding_mask_transposed)

  query_get_bounds_params = get_bounds.GetBounds.Params(
      update_bounds=quant_context.update_bounds,
      update_stats=train,
      paxis_name=paxis_name,
      mask=query_padding_mask_transposed,
      module_name='Q')

  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = quantized_dynamic_dot_general(
      lhs_act=query,
      rhs_act=key,
      dot_dimension_numbers=(((n - 1,), (n - 1,)), (batch_dims_t,
                                                    batch_dims_t)),
      dot_precision=precision,
      quant_type=hparams.quant_type,
      lhs_act_hparams=hparams.attn_act_q,
      lhs_get_bounds_params=query_get_bounds_params,
      rhs_act_hparams=hparams.attn_act_k,
      rhs_get_bounds_params=key_get_bounds_params,
  )
  # NOTE(shivaniagrawal): we do per-layer quantization here since that's the
  # only way for activation*activation matmuls to be aqt compatible since we use
  # static scaling factors for activations.

  shape_utils.assert_shapes_equal(
      attn_weights.shape,
      (batch_size, num_heads, query_sequence_length, key_sequence_length))

  # apply attention bias: masking, dropout, proximity bias, ect.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = softmax(
      attn_weights,
      norm_dims,
      dtype,
      hparams.softmax,
      quant_context=quant_context)

  # apply dropout
  if not deterministic and dropout_rate > 0.0:
    if dropout_rng is None:
      raise ValueError('dropout_rng cannot be None if dropout is requested.')
    keep_prob = jax.lax.tie_in(attn_weights, 1.0 - dropout_rate)
    if broadcast_dropout:
      # dropout is broadcast across the batch+head+non-attention dimension
      dropout_dims = attn_weights.shape[-(2 * len(axis)):]
      dropout_shape = (tuple([1] * len(batch_dims_t)) + dropout_dims)
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  if quant_context.collect_acts_stats:
    stats_tag.StatsTag(
        channel_axis=None, name='attn_act_probs', update_stats=train)(
            attn_weights, mask=attn_mask)

  if hparams.attn_act_probs is not None:
    assert hparams.attn_act_probs.bounds == 1.0, (
        'act quantization bounds should '
        'be set to fix value 1.0 to '
        'match Softmax range.')
  probs_get_bounds_params = get_bounds.GetBounds.Params(
      update_bounds=quant_context.update_bounds,
      update_stats=train,
      paxis_name=paxis_name,
      mask=attn_mask,
      module_name='attn_probs')

  # compute the new values given the attention weights
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = quantized_dynamic_dot_general(
      lhs_act=attn_weights,
      rhs_act=value,
      dot_dimension_numbers=(wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      dot_precision=precision,
      quant_type=hparams.quant_type,
      lhs_act_hparams=hparams.attn_act_probs,
      lhs_get_bounds_params=probs_get_bounds_params,
      rhs_act_hparams=hparams.attn_act_v,
      rhs_get_bounds_params=value_get_bounds_params,
  )
  # NOTE(shivaniagrawal): we do per-layer quantization here since that's the
  # only way for activation*activation matmuls to be aqt compatible since we
  # use static scaling factors for activations.

  shape_utils.assert_shapes_equal(
      y.shape, (batch_size, num_heads, query_sequence_length, channel_size))
  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  shape_utils.assert_shapes_equal(
      y.shape, (batch_size, query_sequence_length, num_heads, channel_size))
  return y


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


class MultiHeadDotProductAttentionAqt(nn.Module):
  """Multi-head dot-product attention.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    hparams: hyperparameters
    update_bounds: Bool whether to update activation bounds.
    paxis_name: axis_name to which a user `pmaps` the parent module (model),
      refer to jax.pmap() for more documentation. This arg is used for
      get_bounds acts quantization (QuantOps.create_input_fake_quant)
    train: Whether model is training.
    collect_acts_stats: Whether to tag activations to record statistics.
    dtype: the dtype of the computation (default: float32)
    qkv_features: dimension of the key, query, and value.
    attention_axis: axes over which the attention is applied ( 'None' means
      attention over all axes, but batch, heads, and features).
    causal_mask: boolean specifying whether to apply a causal mask on the
      attention weights. If True, the output at timestep `t` will not depend on
      inputs at timesteps strictly greater than `t`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    bias: bool: whether pointwise QKVO dense transforms use bias.
    attention_fn: flax.nn.dot_product_attention or compatible function. Accepts
      query, key, value, and returns output of shape `[bs, dim1, dim2, ...,
      dimN,, num_heads, value_channels]``
  """

  @dataclass
  class HParams:
    """HParams class to quantize MultiHeadDotProductAttentionAqts."""
    # DenseAQT hyperparameter to quantize key, query and value dense
    # matmuls (i.e. xq * Wq, xk * Wk, xv * Wv) in Attention layer.
    dense_kqv: flax_layers.DenseAqt.HParams
    # DenseAQT hyperparameter to quantize dense_out matmul.
    dense_out: flax_layers.DenseAqt.HParams
    # QuantOps hyperparameter to quantize query, key and value for attention
    # act*act matmauls
    attn_acts: DotProductAttnHParams

  hparams: HParams
  num_heads: int
  paxis_name: Optional[str]
  train: bool
  quant_context: quant_config.QuantContext
  dtype: Type[Any]
  qkv_features: Optional[int]
  attention_axis: Optional[Iterable[int]]
  causal_mask: bool
  dropout_rate: float
  deterministic: bool
  decode: bool
  broadcast_dropout: bool = True
  kernel_init: InitializerType = default_kernel_init
  attention_fn: Callable[Ellipsis, jnp.ndarray] = dot_product_attention
  bias_init: InitializerType = initializers.zeros
  use_bias: bool = True

  @nn.compact
  def __call__(self,
               inputs_q,
               inputs_kv,
               *,
               padding_mask,
               key_padding_mask,
               segmentation = None,
               key_segmentation = None):
    """Applies multi-head dot product attention on the input data.

    If weight_prec is not None, scales and quantizes weights to signed int with
    weight_prec bits.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` or for self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
      inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]` or
        None for self-attention, inn which case key/values will be derived from
        inputs_q.
      padding_mask: boolean tensor specifying query tokens that are pad token.
      key_padding_mask: boolean tensor specifying key-value tokens that are pad
        token.
      segmentation: segment indices for packed inputs_q data.
      key_segmentation: segment indices for packed inputs_kv data.

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """
    batch_size, query_sequence_length, channel_size = inputs_q.shape
    hparams = self.hparams
    if inputs_kv is None:
      inputs_kv = inputs_q
      key_sequence_length = inputs_q.shape[1]
    else:
      key_sequence_length = inputs_kv.shape[1]
      shape_utils.assert_shapes_equal(
          inputs_kv.shape, (batch_size, key_sequence_length, channel_size))

    jax_precision = jax.lax.Precision.DEFAULT

    if padding_mask is not None:
      shape_utils.assert_shapes_equal(padding_mask.shape,
                                      (batch_size, query_sequence_length, 1))
    if key_padding_mask is None:
      key_padding_mask = padding_mask
    else:
      shape_utils.assert_shapes_equal(key_padding_mask.shape,
                                      (batch_size, key_sequence_length, 1))
    attention_axis = self.attention_axis
    if attention_axis is None:
      attention_axis = tuple(range(1, inputs_q.ndim - 1))

    qkv_features = self.qkv_features
    qkv_features = qkv_features or inputs_q.shape[-1]

    num_heads = self.num_heads
    assert qkv_features % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // num_heads

    paxis_name = self.paxis_name
    train = self.train
    kernel_init = self.kernel_init
    bias_init = self.bias_init
    use_bias = self.use_bias
    dtype = self.dtype

    def multi_batch_dense_aqt(inputs, *, name, padding_mask):
      batch_size, sequence_length, channel_size = inputs.shape
      inputs = inputs.reshape(batch_size * sequence_length, channel_size)
      if padding_mask is not None:
        padding_mask = padding_mask.reshape(batch_size * sequence_length, 1)
      out = flax_layers.DenseAqt(
          name=name,
          features=num_heads * head_dim,
          paxis_name=paxis_name,
          train=train,
          quant_context=self.quant_context,
          hparams=hparams.dense_kqv,
          kernel_init=kernel_init,
          bias_init=bias_init,
          use_bias=use_bias,
          dtype=dtype)(
              inputs, padding_mask=padding_mask)
      return out.reshape(batch_size, sequence_length, num_heads, head_dim)

    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, sequence_length, n_heads, n_features_per_head]
    query = multi_batch_dense_aqt(
        inputs_q, name='query', padding_mask=padding_mask)
    key = multi_batch_dense_aqt(
        inputs_kv, name='key', padding_mask=key_padding_mask)
    value = multi_batch_dense_aqt(
        inputs_kv, name='value', padding_mask=key_padding_mask)
    is_cache_initialized = False
    if self.decode:
      is_cache_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key', jnp.zeros, key.shape,
                                 key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_cache_initialized:
        expected_shape = list(cached_key.value.shape[:-2])
        for attn_dim in attention_axis:
          expected_shape[attn_dim] = 1
        expected_shape = tuple(expected_shape) + inputs_q.shape[-1:]
        if expected_shape != inputs_q.shape:
          raise ValueError('Invalid shape provided, '
                           'expected shape %s instead got %s.' %
                           (expected_shape, inputs_q.shape))

        cshape = cached_key.value.shape
        indices = [0] * len(cshape)
        i = cache_index.value
        attn_size = onp.prod(onp.take(cshape, attention_axis))

        *batch_dims, max_length, num_heads, depth_per_head = (  # pylint: disable=unused-variable
            cached_key.value.shape)
        indices = (0,) * len(batch_dims) + (i, 0, 0)

        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        one = jnp.array(1, jnp.uint32)
        cache_index.value = cache_index.value + one
        cached_key.value = key
        cached_value.value = value

        # TODO(levskaya): verify this is still needed in translation decoding.
        key_padding_mask = jnp.broadcast_to(
            (jnp.arange(max_length) < cache_index.value), cshape[:2])
        key_padding_mask = key_padding_mask.astype(jnp.float32)[Ellipsis, None]

    # create attention masks
    mask_components = []
    if self.causal_mask:
      if self.decode and is_cache_initialized:
        bias_pre_shape = (1,) * (key.ndim - 1)
        attn_shape = tuple(onp.take(key.shape, attention_axis))
        attn_size = onp.prod(attn_shape)
        ii = jnp.arange(attn_size, dtype=jnp.uint32)
        mask = ii < cache_index.value
        mask_components.append(mask.reshape(bias_pre_shape + attn_shape))
      else:
        mask_components.append(_make_causal_mask(key, attention_axis))
    if padding_mask is not None:
      if key_padding_mask is None:
        key_padding_mask = padding_mask
      attn_padding_mask = make_padding_mask(
          padding_mask_query=padding_mask,
          padding_mask_key=key_padding_mask,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis)
      mask_components.append(attn_padding_mask)
    if segmentation is not None:
      if key_segmentation is None:
        key_segmentation = segmentation
      segmentation_mask = make_padding_mask(
          padding_mask_query=segmentation,
          padding_mask_key=key_segmentation,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis,
          segmentation_mask=True)
      mask_components.append(segmentation_mask)
    attention_mask = None
    if mask_components:
      attention_mask = mask_components[0]
      for component in mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)
      attention_mask = attention_mask.astype(jnp.bool_)

      # attention mask in the form of attention bias
      attention_bias = jnp.where(
          attention_mask,
          jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    # Add an extra dimension to the mask corresponding to the head
    # dimension. eg, if inputs_q has shape [batch_size, sequence_length,
    # n_features], then padding_mask will have a shape
    # [batch_size, sequence_length, 1] and query will have shape
    # [batch_size, sequence_length, n_heads, n_features_per_head].
    # We create query_padding_mask with shape [batch_size, sequence_length,
    # 1, 1] to be broadcast-compatible with 'query'.
    if padding_mask is not None:
      padding_mask = padding_mask[Ellipsis, None]
      shape_utils.assert_shapes_equal(padding_mask.shape,
                                      (batch_size, query_sequence_length, 1, 1))
    if key_padding_mask is not None:
      key_padding_mask = key_padding_mask[Ellipsis, None]
      # During prediction, the key padding mask is only going to be
      # broadcast-compatible with the key.
      shape_utils.assert_shapes_compatible(
          key_padding_mask.shape, (batch_size, key_sequence_length, 1, 1))

    # apply attention
    attention_fn = self.attention_fn
    dropout_rate = self.dropout_rate
    broadcast_dropout = self.broadcast_dropout
    deterministic = self.deterministic
    if not deterministic and self.dropout_rate > 0.0:
      dropout_rng = self.make_rng('dropout')
    else:
      dropout_rng = None
    x = attention_fn(  # pylint: disable=redundant-keyword-arg
        query=query,
        key=key,
        value=value,
        hparams=hparams.attn_acts,
        paxis_name=paxis_name,
        train=train,
        quant_context=self.quant_context,
        dtype=dtype,
        axis=attention_axis,
        bias=attention_bias,
        precision=jax_precision,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic,
        query_padding_mask=padding_mask,
        key_padding_mask=key_padding_mask,
        attn_mask=attention_mask)
    shape_utils.assert_shapes_equal(
        x.shape, (batch_size, query_sequence_length, num_heads, head_dim))
    x = x.reshape(batch_size * query_sequence_length, num_heads * head_dim)
    if padding_mask is not None:
      padding_mask = padding_mask.reshape(batch_size * query_sequence_length, 1)
    # back to the original inputs dimensions
    out = flax_layers.DenseAqt(
        features=channel_size,
        hparams=hparams.dense_out,
        quant_context=self.quant_context,
        paxis_name=paxis_name,
        train=train,
        kernel_init=kernel_init,
        bias_init=bias_init,
        use_bias=use_bias,
        dtype=dtype,
        name='dense_out')(
            x, padding_mask=padding_mask)
    shape_utils.assert_shapes_equal(
        out.shape, (batch_size * query_sequence_length, channel_size))
    out = out.reshape(batch_size, query_sequence_length, channel_size)
    return out


class SelfAttentionAqt(MultiHeadDotProductAttentionAqt):
  """Self-attention."""

  @nn.compact
  def __call__(
      self,
      inputs_q,
      *,
      padding_mask,
      segmentation = None,
  ):
    return super().__call__(
        inputs_q=inputs_q,
        inputs_kv=inputs_q,
        padding_mask=padding_mask,
        key_padding_mask=padding_mask,
        segmentation=segmentation,
    )
