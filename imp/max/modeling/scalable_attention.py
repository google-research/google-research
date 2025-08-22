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

"""Core module supporting efficient non-autoregressive attention."""

from typing import Any

import jax
from jax import lax
import jax.numpy as jnp

from imp.max.modeling import embeddings
from imp.max.modeling import kernel_transformation as kt
from imp.max.modeling import stochastic
from imp.max.utils import typing


# ------------------------------------------------------------------------------
# Constants used in Performers' (https://arxiv.org/abs/2009.14794) modules:
# ------------------------------------------------------------------------------

# Seed used to generate random features for the approximate softmax kernel.
RANDOM_FEATURES_SEED = 873457891289
# Big constant.
BIG_CONSTANT = 10000000.0
# Seed used in Performers' variants applying RPEs.
PERFORMERS_RPE_SEED = 73829861893
# Maximum number of packed sequences in the packed Performer variant.
MAX_NB_PACKED_SEQS = 7

# ------------------------------------------------------------------------------
# Functions implementing the not-normalized attention (numerator) as well as
# the renormalizer (denominator) ensuring that the implicit attention matrix
# is row-stochastic. For each of them, two variants are provided: regular and
# the one supporting masking as in: https://arxiv.org/abs/2106.12566.
# ------------------------------------------------------------------------------


def noncausal_numerator(
    qs,
    ks,
    vs,
    dot_general = lax.dot_general,
    precision = None,
):
  """Computes not-normalized FAVOR+ noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    ks: key_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    vs: value tensor of the shape [B...,L,H,D].
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    Not-normalized FAVOR+ noncausal attention AV.
  """
  kvs = jnp.einsum('...lhm,...lhd->...hmd', ks, vs,
                   precision=precision,
                   _dot_general=dot_general)
  return jnp.einsum('...lhm,...hmd->...lhd', qs, kvs,
                    _dot_general=dot_general)


def noncausal_denominator(
    qs,
    ks,
    dot_general = lax.dot_general,
    precision = None,
):
  """Computes FAVOR+ normalizer in noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    ks: key_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    FAVOR+ normalizer in noncausal attention.
  """
  ks_sum = jnp.sum(ks, axis=-3)
  return jnp.einsum('...lhm,...hm->...lh', qs, ks_sum,
                    precision=precision,
                    _dot_general=dot_general)


def masked_numerator(
    qs,
    ks,
    vs,
    masker,
    mask,
    dot_general = lax.dot_general,
    precision = None,
):
  """Computes not-normalized FAVOR+ noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    ks: key_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    vs: value tensor of the shape [B...,L,H,D].
    masker: object of the type masks.Mask applying masking mechanism using given
      mask.
    mask: compact encoding of the masking mechanism.
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    Not-normalized masked FAVOR+ attention.
  """
  # See: Alg. 1 from https://arxiv.org/pdf/2107.07999.pdf.
  truncated_ks_shape = ks.shape[:-1]
  f1_tensor = jnp.reshape(
      jnp.einsum('...m,...d->...md', ks, vs,
                 precision=precision,
                 _dot_general=dot_general),
      truncated_ks_shape + (ks.shape[-1] * vs.shape[-1],),
  )
  d1_tensor = masker.act(mask, f1_tensor)
  truncated_d1_shape = d1_tensor.shape[:-1]
  d1_tensor_unflattened = jnp.reshape(
      d1_tensor, truncated_d1_shape + (ks.shape[-1], vs.shape[-1]),
  )
  return jnp.einsum('...m,...md->...d',
                    qs, d1_tensor_unflattened,
                    precision=precision,
                    _dot_general=dot_general)


def masked_denominator(
    qs,
    ks,
    masker,
    mask,
    dot_general = lax.dot_general,
    precision = None,
):
  """Computes masked FAVOR+ normalizer.

  Args:
    qs: query_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    ks: key_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    masker: object of the type masks.Mask applying masking mechanism using given
      mask.
    mask: compact encoding of the masking mechanism.
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    FAVOR+ normalizer in masked FAVOR+ attention.
  """
  d2_tensor = masker.act(mask, ks)
  return jnp.einsum('...m,...m->...', qs, d2_tensor,
                    precision=precision,
                    _dot_general=dot_general)

# ------------------------------------------------------------------------------
# Regular brute-force attention module.
# ------------------------------------------------------------------------------


def full_attn(
    query_matrix,
    key_matrix,
    value_matrix,
    attn_mask = None,
    dot_general = lax.dot_general,
    precision = None,
):
  """Applies kernel attention with query, key, value tensors.

  This function defines the computation inside `call` with projected
  multi-head Q, K, V inputs. Users can override this function for customized
  attention implementation.

  Args:
    query_matrix: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
    key_matrix: Projected key `Tensor` of shape `[B, S, N, key_dim]`.
    value_matrix: Projected value `Tensor` of shape `[B, S, N, value_dim]`.
    attn_mask: a boolean mask of shape `[B, S]`, that prevents attending to
      masked positions. Note that the mask is only appied to the keys. User may
      want to mask the output if query contains pads.
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    attention_output: Multi-headed outputs of attention computation.
  """

  full_product = jnp.einsum(
      'BFNH,BTNH->BFTN', query_matrix, key_matrix,
      precision=precision, _dot_general=dot_general
  )  # [B, F, T, N]
  if attn_mask is not None:
    attn_mask = attn_mask.astype(key_matrix.dtype)
    attn_mask = jnp.expand_dims(jnp.expand_dims(attn_mask, axis=1), axis=3)
    adder = (1.0 - attn_mask) * -10000.0
    full_product += adder
  full_product = jax.nn.softmax(full_product, axis=2)
  attention_output = jnp.einsum(
      'BFTN,BTNO->BFNO', full_product, value_matrix,
      precision=precision, _dot_general=dot_general
  )  # [B, F, N, O]
  return attention_output


# ------------------------------------------------------------------------------
# General bi-directional (spatial) efficient attention function.
# ------------------------------------------------------------------------------


def general_favor_attention(
    query,
    key,
    value,
    coords,
    kernel_transformation = kt.relu_kernel_transformation,
    num_kernel_features = 64,
    use_random_projections = False,
    simplex = False,
    inputs_mask = None,
    segment_ids = None,
    spe = False,
    spe_num_realizations = 64,
    spe_num_sines = 10,
    flt = False,
    flt_params = None,
    flt_num_blobs_per_head = 0,
    flt_num_rand_features = 0,
    grpe = False,
    grpe_params = None,
    bf_attention_global_size = 0,
    dot_general = lax.dot_general,
    precision = None):
  """Computes general bi-directional efficient attention.

  Computes general bi-directional attention function which is an extension
  of the efficient attention mechanism proposed in
  "Rethinking Attention with Performers": https://arxiv.org/abs/2009.14794).
  It supports in particular Fourier Learner Transformers (FLTs) attention
  (see: https://arxiv.org/abs/2302.01925).

  Args:
    query: query tensor of the shape [B, L, H, D_QK].
    key: key tensor of the shape [B, L, H, D_QK].
    value: value tensor of the shape [B, L, H, D_V].
    coords: coordinates of the points of the shape [L, d], where d
      stands for the number of coordinates defining each point.
    kernel_transformation: transformation used to get finite kernel features.
    num_kernel_features: number of kernel features to be used.
    use_random_projections: determines whether random or deterministic
      (canonical projections will be used).
    simplex: determines whether simplex (https://arxiv.org/abs/2301.13856) or
      the orthogonal QMC method will be applied.
    inputs_mask: <bool>[batch, length] array indicating True for non-padding
      tokens and False for padding.
    segment_ids: packing mask. The mask is the 2-dimensional tensor of the shape
      [B,L], where: B - batch didmension, L - length dimension. Each slice
      corresponding to the fixed index in the batch is of the form:
      [1,...1,2,...,2,...,N,...,N,0,...,0], where x...x corresponds to the
      tokens of a fixed sequence within the super-sequence of packed sequences,
      N is the total number of packed sequences in the slice and 0-tokens encode
      padding. Even though, we enumerate different sequences from left to right
      in the increasing order, the mechanism works for any enumeration.
    spe: determines whether stochastic positional encoding mechanism from
      https://arxiv.org/abs/2105.08399 will be applied.
    spe_num_realizations: number of samples for the SPE mechanism.
    spe_num_sines: number of sin waves for the SPE mechanism.
    flt: determines whether FourierLearner-Transformer (FLT) RPE mechanism from
      https://arxiv.org/abs/2302.01925 will be applied.
    flt_params: learnable parameters encoding RPE mechanism applied in FLT.
    flt_num_blobs_per_head: number of Gaussian blobs per head used to encode
      the Fourier Transform (FT) of the function defining RPE masking in the
      FLT model.
    flt_num_rand_features: number of random features used to approximate
      the function defining RPE masking in the FLT model.
    grpe: determines whether general time-efficient RPE masking mechanism from
      https://arxiv.org/abs/2106.12566 is applied.
    grpe_params: learnable parameters encoding general RPE masking mechanism.
    bf_attention_global_size: if not zero, use first <bf_attention_global_size>
      tokens for global brute-force (bf) full attention.
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the dot product is performed.

  Returns:
    bidirectional normalized general efficient attention.
  """
  projection_matrix = None
  # Apply positional encoding if this option is turned on:
  rng_key = jax.random.key(PERFORMERS_RPE_SEED)
  if spe:
    qbar, kbar = embeddings.sinespe(
        rng_key,
        query.shape,
        num_sines=spe_num_sines,
        num_realizations=spe_num_realizations,
        dot_general=dot_general,
        precision=precision,
    )
    qbar, kbar = embeddings.spegate(rng_key, (qbar, kbar))
    query = embeddings.apply_spe(query, qbar)
    key = embeddings.apply_spe(key, kbar)
  head_dim = query.shape[-1]
  if use_random_projections:
    extra_features = 0
    if flt:
      extra_features = flt_num_rand_features
    if not simplex:
      projection_matrix = stochastic.get_gaussian_orth_rand_mat(
          rng_key, num_kernel_features, head_dim + 2 * extra_features,
          dot_general=dot_general, precision=precision,
      )
    else:
      projection_matrix = stochastic.get_gaussian_simplex_rand_mat(
          rng_key, num_kernel_features, head_dim + 2 * extra_features,
          dot_general=dot_general, precision=precision,
      )
  global_full_attn_output = None
  if bf_attention_global_size > 0:
    global_full_attn_output = full_attn(
        query[:, :bf_attention_global_size, :, :], key, value, inputs_mask,
        precision=precision, dot_general=dot_general,
    )
    query = query[:, bf_attention_global_size:, :, :]

  if flt:
    if flt_params is None:
      raise ValueError(
          '`flt_params` should be provided. Instead, received None.')
    d = jnp.shape(query)[-1]
    snippet_shape = jnp.shape(query)[:-1] + (flt_num_rand_features,)
    snippet_shape_k = jnp.shape(key)[:-1] + (flt_num_rand_features,)
    coe = jnp.sqrt(jnp.sqrt(d))
    q_snippet = embeddings.create_flt_snippet(
        flt_params,
        coords,
        1.0,
        flt_num_blobs_per_head,
        flt_num_rand_features,
        einsum_dot_general=dot_general,
        einsum_precision=precision,
    )
    q_snippet_imag = coe * jnp.broadcast_to(
        jnp.imag(q_snippet), snippet_shape
    )
    q_snippet_real = coe * jnp.broadcast_to(
        jnp.real(q_snippet), snippet_shape
    )
    k_snippet = embeddings.create_flt_snippet(
        flt_params,
        coords,
        -1.0,
        flt_num_blobs_per_head,
        flt_num_rand_features,
        einsum_dot_general=dot_general,
        einsum_precision=precision,
    )
    k_snippet_imag = coe * jnp.broadcast_to(
        jnp.imag(k_snippet), snippet_shape_k
    )
    k_snippet_real = coe * jnp.broadcast_to(
        jnp.real(k_snippet), snippet_shape_k
    )
    query = jnp.concatenate([query, q_snippet_real, q_snippet_imag], axis=-1)
    key = jnp.concatenate([key, k_snippet_real, -k_snippet_imag], axis=-1)

  query_prime = kernel_transformation(query, key, True, projection_matrix,
                                      precision=precision,
                                      dot_general=dot_general)
  key_prime = kernel_transformation(key, query, False, projection_matrix,
                                    precision=precision,
                                    dot_general=dot_general)

  if segment_ids is None:
    if inputs_mask is not None:
      b, length, h, m = jnp.shape(key_prime)
      inputs_mask = jnp.tile(
          jnp.reshape(inputs_mask, [b, length, 1, 1]), [1, 1, h, m]
      )
      key_prime = jnp.where(inputs_mask, key_prime, 0)
  else:
    b, length, h, m = jnp.shape(key_prime)
    # Introducing extra dimension so that padding can be re-interpreted as
    # multi-packing with different packing masks corresponding to different
    # sequences in the super-sequence.
    packing_mask = jnp.arange(1, MAX_NB_PACKED_SEQS + 1, 1)
    packing_mask = jnp.tile(
        jnp.reshape(packing_mask, [MAX_NB_PACKED_SEQS, 1, 1, 1, 1]),
        [1, b, length, h, m],
    )
    segment_ids = jnp.tile(
        jnp.reshape(segment_ids, [1, b, length, 1, 1]),
        [MAX_NB_PACKED_SEQS, 1, 1, h, m],
    )
    padded_inputs_mask = segment_ids == packing_mask
    key_prime = jnp.tile(
        jnp.reshape(key_prime, [1, b, length, h, m]),
        [MAX_NB_PACKED_SEQS, 1, 1, 1, 1],
    )
    query_prime = jnp.tile(
        jnp.reshape(query_prime, [1, b, length, h, m]),
        [MAX_NB_PACKED_SEQS, 1, 1, 1, 1],
    )
    key_prime = jnp.where(padded_inputs_mask, key_prime, 0)
    query_prime = jnp.where(padded_inputs_mask, query_prime, 0)

  if not grpe:
    av_attention = noncausal_numerator(query_prime, key_prime, value,
                                       precision=precision,
                                       dot_general=dot_general)
    attention_normalizer = noncausal_denominator(query_prime, key_prime,
                                                 precision=precision,
                                                 dot_general=dot_general)
  else:
    if grpe_params is None:
      raise ValueError(
          '`grpe_params` should be provided. Instead, received None.')
    length = query.shape[-3]
    mask = (grpe_params[:, :length], grpe_params[:, length:])
    masker = embeddings.RPEMask(einsum_precision=precision,
                                einsum_dot_general=dot_general)
    av_attention = masked_numerator(query_prime, key_prime, value, masker, mask,
                                    precision=precision,
                                    dot_general=dot_general)
    attention_normalizer = masked_denominator(
        query_prime, key_prime, masker, mask,
        precision=precision, dot_general=dot_general,
    )

  if segment_ids is not None:
    av_attention = jnp.sum(av_attention, axis=0, keepdims=False)
    attention_normalizer = jnp.sum(attention_normalizer, axis=0, keepdims=False)

  attention_normalizer = jnp.expand_dims(
      attention_normalizer, len(attention_normalizer.shape)
  )
  attention_normalizer = jnp.where(
      attention_normalizer <= 0.0,
      jnp.ones(attention_normalizer.shape),
      attention_normalizer,
  )

  attention_output = av_attention / attention_normalizer
  if bf_attention_global_size > 0:
    attention_output = jnp.concatenate(
        [global_full_attn_output, attention_output], 1
    )
  return attention_output
