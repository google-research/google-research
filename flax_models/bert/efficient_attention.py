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

"""Attention Layers."""

from flax import nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from flax_models.bert import multihead


NEG_INFINITY = -1e9


class MultiHeadWrapper(nn.Module):
  """Wrapper for batching attention across examples and heads."""

  def apply(self, *args, wrapped_module,
            num_heads=1, num_parallel_heads=None, use_python_loop=False,
            **kwargs):
    # Re-use the same rng key across all examples and heads. This will result in
    # broadcasted dropout, which saves memory.
    # TODO(kitaev): options to swap broadcasted RNG on/off
    rng = nn.make_rng() if nn.is_stochastic() else None

    def init_single_head(init_rng, args, kwargs):
      if rng is None:
        _, head_params = wrapped_module.init(init_rng, *args, **kwargs)
      else:
        with nn.stochastic(rng):
          _, head_params = wrapped_module.init(init_rng, *args, **kwargs)
      return head_params

    def init_wrapped_module(rng, unused_shape):
      single_example_args = jax.tree_map(lambda x: x[:1], args)
      return multihead.chunked_multihead_map(
          init_single_head,
          in_has_batch_dim=(False, True, False),
          in_has_head_dim=(True, False, False),
          out_has_batch_dim=False,
          out_has_head_dim=True,
          use_python_loop=True,
          )(jax.random.split(rng, num_heads), single_example_args, kwargs)
    # TODO(kitaev): The original intent was to have this be a transparent module
    # but for some reason naming this parameter "0" and inheriting from
    # nn.base.TransparentModule is not enough to stop this parameter name from
    # explicitly showing up in the parameter tree.
    params = self.param("attn", None, init_wrapped_module)

    def run_single_example_and_head(params, args, kwargs):
      if rng is None:
        return wrapped_module.call(params, *args, **kwargs)
      else:
        with nn.stochastic(rng):
          return wrapped_module.call(params, *args, **kwargs)

    return multihead.chunked_multihead_map(
        run_single_example_and_head,
        in_has_batch_dim=(False, True, False),
        in_has_head_dim=(True, False, False),
        out_has_batch_dim=True,
        out_has_head_dim=False,
        num_parallel_heads=num_parallel_heads,
        use_python_loop=use_python_loop,
    )(params, args, kwargs)


def make_multihead(module_type):
  return MultiHeadWrapper.partial(wrapped_module=module_type)


@make_multihead
class BertSelfAttention(nn.Module):
  """Masked dot-product self-attention."""

  def apply(self,
            hidden_states, mask=None, *,
            d_qkv=64,
            attention_dropout_rate=0.0,
            output_dropout_rate=0.0,
            deterministic=False,
            kernel_init=nn.linear.default_kernel_init,
            output_kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            bias=True):
    """Applies attention for a single batch element and head."""
    d_model = hidden_states.shape[-1]
    dense = nn.DenseGeneral.partial(
        axis=-1,
        features=(d_qkv,),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias)
    query, key, value = (dense(hidden_states, name="query"),
                         dense(hidden_states, name="key"),
                         dense(hidden_states, name="value"))
    attention_scores = jnp.einsum("TN,FN->FT", key, query)
    attention_scores = attention_scores / jnp.sqrt(d_qkv)
    if mask is not None:
      padding_mask = (1.0 - mask[None, :]) * NEG_INFINITY
      attention_scores = attention_scores + padding_mask
    attention_scores = nn.softmax(attention_scores)
    attention_probs = nn.dropout(
        attention_scores, rate=attention_dropout_rate,
        deterministic=deterministic)
    hidden_states = jnp.einsum("FT,TH->FH", attention_probs, value)
    hidden_states = nn.linear.DenseGeneral(
        hidden_states,
        features=d_model,
        axis=(-1,),
        kernel_init=output_kernel_init,
        name="output")
    hidden_states = nn.dropout(
        hidden_states, rate=output_dropout_rate, deterministic=deterministic)
    return hidden_states


def look_adjacent(x, n_chunks_before, n_chunks_after):
  """Used to implement attention between consecutive chunks.

  Args:
    x: array of shape [n_chunks, chunk_len, ...]
    n_chunks_before: Number of previous chunks to attend to.
    n_chunks_after: Number of subsequent chunks to attend to.
  Returns:
    array of shape [n_chunks, N * chunk_len, ...], where
    N = (1 + n_chunks_before + n_chunks_after).
  """
  if n_chunks_before == 0 and n_chunks_after == 0:
    return x

  slices = []
  for i in range(-n_chunks_before, n_chunks_after + 1):
    if i == 0:
      slices.append(x)
    else:
      slices.append(jnp.concatenate([x[i:, Ellipsis], x[:i, Ellipsis]], axis=0))
  return jnp.concatenate(slices, axis=1)


def length_normalized(x, epsilon=1e-6):
  variance = jnp.mean(x**2, axis=-1, keepdims=True)
  norm_inputs = x / jnp.sqrt(variance + epsilon)
  return norm_inputs


def mask_self_attention(
    dots, q_info, kv_info, causal=True, exclude_self=True, masked=False):
  """Performs masking for self-attention."""
  if causal:
    mask = jax.lax.convert_element_type(
        jax.lax.lt(q_info, kv_info), jnp.float32)
    dots = dots - 1e9 * mask
  if exclude_self:
    mask = jax.lax.convert_element_type(
        jax.lax.eq(q_info, kv_info), jnp.float32)
    dots = dots - 1e5 * mask
  if masked:
    zeros_like_kv_info = jax.lax.tie_in(kv_info, jnp.zeros_like(kv_info))
    mask = jax.lax.convert_element_type(
        jax.lax.lt(kv_info, zeros_like_kv_info), jnp.float32)
    dots = dots - 1e9 * mask
  return dots


def attend(
    q, k=None, v=None,
    q_chunk_len=None, kv_chunk_len=None,
    n_chunks_before=0, n_chunks_after=0,
    mask_fn=None, q_info=None, kv_info=None,
    dropout=0.0, rng=None,
    ):
  """Dot-product attention, with optional chunking and/or masking.

  Args:
    q: Query vectors, shape [q_len, d_qk]
    k: Key vectors, shape [kv_len, d_qk]; or None
    v: Value vectors, shape [kv_len, d_v]
    q_chunk_len: Set to non-zero to enable chunking for query vectors
    kv_chunk_len: Set to non-zero to enable chunking for key/value vectors
    n_chunks_before: Number of adjacent previous chunks to attend to
    n_chunks_after: Number of adjacent subsequent chunks to attend to
    mask_fn: TODO(kitaev): doc
    q_info: Query-associated metadata for masking
    kv_info: Key-associated metadata for masking
    dropout: Dropout rate
    rng: RNG for dropout
  Returns:
    A tuple (output, dots_logsumexp). The output has shape [q_len, d_v], and
    dots_logsumexp has shape [q_len]. The logsumexp of the attention
    probabilities is useful for combining multiple rounds of attention (as in
    LSH attention).
  """
  assert v is not None
  share_qk = (k is None)

  if q_info is None:
    q_info = jnp.arange(q.shape[-2])

  if kv_info is None and not share_qk:
    kv_info = jnp.arange(v.shape[-2])

  # Split q/k/v into chunks along the time axis, if desired.
  if q_chunk_len is not None:
    q = jnp.reshape(q, (-1, q_chunk_len, q.shape[-1]))
    q_info = jnp.reshape(q_info, (-1, q_chunk_len))

  if share_qk:
    assert kv_chunk_len is None or kv_chunk_len == q_chunk_len
    k = q
    kv_chunk_len = q_chunk_len
    if kv_info is None:
      kv_info = q_info
    elif kv_chunk_len is not None:
      # kv_info is not None, but reshape as required.
      kv_info = jnp.reshape(kv_info, (-1, kv_chunk_len))
  elif kv_chunk_len is not None:
    k = jnp.reshape(k, (-1, kv_chunk_len, k.shape[-1]))
    kv_info = jnp.reshape(kv_info, (-1, kv_chunk_len))

  if kv_chunk_len is not None:
    v = jnp.reshape(v, (-1, kv_chunk_len, v.shape[-1]))

  if share_qk:
    k = length_normalized(k)
  k = k / jnp.sqrt(k.shape[-1])

  # Optionally include adjacent chunks.
  if q_chunk_len is not None or kv_chunk_len is not None:
    assert q_chunk_len is not None and kv_chunk_len is not None
  else:
    assert n_chunks_before == 0 and n_chunks_after == 0

  k = look_adjacent(k, n_chunks_before, n_chunks_after)
  v = look_adjacent(v, n_chunks_before, n_chunks_after)
  kv_info = look_adjacent(kv_info, n_chunks_before, n_chunks_after)

  # Dot-product attention.
  dots = jnp.matmul(q, jnp.swapaxes(k, -1, -2))

  # Masking
  if mask_fn is not None:
    dots = mask_fn(dots, q_info[Ellipsis, :, None], kv_info[Ellipsis, None, :])

  # Softmax.
  dots_logsumexp = logsumexp(dots, axis=-1, keepdims=True)
  dots = jnp.exp(dots - dots_logsumexp)

  if dropout > 0.0:
    assert rng is not None
    # Dropout is broadcast across the bin dimension
    dropout_shape = (dots.shape[-2], dots.shape[-1])
    # TODO(kitaev): verify that tie-in is safe to remove (in light of jax fix)
    keep_prob = jax.lax.tie_in(dots, 1.0 - dropout)
    keep = jax.random.bernoulli(rng, keep_prob, dropout_shape)
    multiplier = keep.astype(dots.dtype) / jax.lax.tie_in(keep, keep_prob)
    dots = dots * multiplier

  # The softmax normalizer (dots_logsumexp) is used by multi-round LSH attn.
  out = jnp.matmul(dots, v)
  out = jnp.reshape(out, (-1, out.shape[-1]))
  dots_logsumexp = jnp.reshape(dots_logsumexp, (-1,))
  return out, dots_logsumexp


def permute_via_gather(val, permutation, inverse_permutation, axis=0):
  """Permutation helper for LSH attention."""
  def permute_impl(p, unused_ip, val):
    return jnp.take(val, p, axis=axis)
  def permute_fwd(p, ip, val):
    return jnp.take(val, p, axis=axis), ip
  def permute_bwd(ip, permuted_grad):
    # JAX autodiff would synthesize a scatter operation because it doesn't
    # know that the indices are a permutation. However on TPU, gathers are
    # faster than scatters (at least in the regime the LSH attention uses).
    return (None, None, jnp.take(permuted_grad, ip, axis=axis))
  permute = jax.custom_vjp(permute_impl, permute_fwd, permute_bwd)
  return permute(permutation, inverse_permutation, val)


class MyDense(nn.Module):
  """Manually batched dense projection to produce query/key/value vectors."""

  def apply(self, hidden_states, axis, features, kernel_init, bias_init, bias):
    d_model = hidden_states.shape[-1]
    assert axis == -1
    num_heads, d_qkv = features
    def kernel_init_wrap(rng, shape, dtype=jnp.float32):
      del shape, dtype
      kernel = kernel_init(rng, (d_model, num_heads * d_qkv))
      kernel = jnp.reshape(kernel, (d_model, num_heads, d_qkv))
      return jnp.swapaxes(kernel, 0, 1)

    kernel = self.param("kernel", (num_heads, d_model, d_qkv), kernel_init_wrap)
    bias = self.param("bias", (num_heads, d_qkv), bias_init)
    return jnp.einsum("BFM,NMQ->BFNQ", hidden_states, kernel) + bias


class MyDenseOut(nn.Module):
  """Manually batched dense projection to produce attention output vectors."""

  def apply(self, o, num_heads, d_qkv, d_model, kernel_init, bias_init, bias):
    def kernel_init_wrap(rng, shape, dtype=jnp.float32):
      del shape, dtype
      return jnp.reshape(
          kernel_init(rng, (num_heads * d_qkv, d_model)),
          (num_heads, d_qkv, d_model))
    kernel = self.param("kernel", (num_heads, d_qkv, d_model), kernel_init_wrap)
    bias = self.param("bias", (num_heads, d_model), bias_init)
    hidden_states = jnp.einsum("BNFV,NVM->BFM", o, kernel) + jnp.sum(bias, 0)
    return hidden_states
