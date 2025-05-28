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

"""Kernel transformation functions for Performer attention."""

from typing import Any

import jax
from jax import lax
import jax.numpy as jnp

from imp.max.utils import typing

# pylint: disable=invalid-name

# ------------------------------------------------------------------------------
# Functions encoding various types of kernel transformations for efficient
# attention.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Kernel transformation for the Performer-ReLU kernel using FAVOR+ mechanism
# from https://arxiv.org/abs/2009.14794.
# ------------------------------------------------------------------------------
def relu_kernel_transformation(
    data,
    extra_data,
    is_query,
    projection_matrix = None,
    numerical_stabilizer = 0.000001,
    dot_general = lax.dot_general,
    precision = None):
  r"""Applies relu kernel transformation to input data."""
  # Delete extra args
  del extra_data, is_query, projection_matrix, dot_general, precision
  return jax.nn.relu(data) + numerical_stabilizer


# ------------------------------------------------------------------------------
# Kernel transformation for the softmax kernel using FAVOR+ mechanism from
# https://arxiv.org/abs/2009.14794.
# ------------------------------------------------------------------------------
def exp_softmax_kernel_transformation(
    data,
    extra_data,
    is_query,
    projection_matrix = None,
    numerical_stabilizer = 0.000001,
    normalize_data = True,
    numerator_denominator_stabilizer = True,
    dot_general = lax.dot_general,
    precision = None):
  r"""Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    extra_data: input extra data tensor of the shape [B...,T, H, D], where T
      stands for the number of the extra tokens; used if additional stats need
      to be collected to define kernel features.
    is_query: indicates whether input data is a query or key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
    normalize_data: whether queries/keys should \sqrt{d}-normalized.
    numerator_denominator_stabilizer: whether numerator and denominator in the
      normalized attention computation should be numerically stabilized.
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    Corresponding kernel feature map.
  """
  del extra_data
  if projection_matrix is None:
    raise ValueError('projection_matrix cannot be unspecified for softmax '
                     'kernel.')
  if normalize_data:
    data_normalizer = 1.0 / jnp.sqrt(jnp.sqrt(data.shape[-1]))
  else:
    data_normalizer = 1.0
    lengths = jnp.square(data)
    lengths = jnp.sum(lengths, axis=data.ndim - 1, keepdims=True)
    lengths = jnp.sqrt(lengths)
    data /= lengths
  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  data_dash = jnp.einsum('...lhd,md->...lhm',
                         data_normalizer * data, projection_matrix,
                         precision=precision,
                         _dot_general=dot_general)  # blhm
  diag_data = jnp.square(data)  # blhd
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)  # blh
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer  # blh
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)  # blh1

  if numerator_denominator_stabilizer:
    if is_query:
      last_dims_t = (len(data_dash.shape) - 1,)
      stab = jnp.max(data_dash, axis=last_dims_t, keepdims=True)  # blh1
    else:
      stab = jnp.max(data_dash, keepdims=True)  # 1111
    data_dash = ratio * (
        jnp.exp(data_dash - stab - diag_data) + numerical_stabilizer)  # blhm
  else:
    data_dash = ratio * (jnp.exp(data_dash) + numerical_stabilizer)  # blhm

  return data_dash  # blhm


def hyp_softmax_kernel_transformation(
    data,
    extra_data,
    is_query,
    projection_matrix = None,
    numerical_stabilizer = 0.000001,
    normalize_data = True,
    numerator_denominator_stabilizer = True,
    dot_general = lax.dot_general,
    precision = None):
  r"""Computes random features for the softmax kernel using hyperbolic mechanism.

  Args:
    data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    extra_data: input extra data tensor of the shape [B...,T, H, D], where T
      stands for the number of the extra tokens; used if additional stats need
      to be collected to define kernel features.
    is_query: indicates whether input data is a query or key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
    normalize_data: whether queries/keys should be sqrt{d}-normalized.
    numerator_denominator_stabilizer: whether numerator and denominator in the
      normalized attention computation should be numerically stabilized.
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    Corresponding kernel feature map.
  """
  del extra_data
  if projection_matrix is None:
    raise ValueError('projection_matrix cannot be unspecified for softmax '
                     'kernel.')
  if normalize_data:
    data_normalizer = 1.0 / jnp.sqrt(jnp.sqrt(data.shape[-1]))
  else:
    data_normalizer = 1.0
    lengths = jnp.square(data)
    lengths = jnp.sum(lengths, axis=data.ndim - 1, keepdims=True)
    lengths = jnp.sqrt(lengths)
    data /= lengths
  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  data_dash = jnp.einsum('...lhd,md->...lhm',
                         data_normalizer * data, projection_matrix,
                         precision=precision,
                         _dot_general=dot_general)  # blhm
  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)  # blh1

  a = 0.0
  b = 1.0
  d = 1.0

  diag_omega = jnp.square(projection_matrix)  # md
  diag_omega = jnp.sum(diag_omega, axis=projection_matrix.ndim - 1)  # m
  diag_omega = jnp.expand_dims(diag_omega, axis=0)  # 1m
  diag_omega = jnp.expand_dims(diag_omega, axis=0)  # 11m
  diag_omega = a * diag_omega

  if numerator_denominator_stabilizer:
    if is_query:
      last_dims_t = (len(data_dash.shape) - 1,)
      m = jnp.maximum(
          jnp.max(data_dash, axis=last_dims_t, keepdims=True),
          -jnp.min(data_dash, axis=last_dims_t, keepdims=True))  # blh1
    else:
      m = jnp.maximum(jnp.max(data_dash), -jnp.min(data_dash))  # 1
    stab = b * m  # blh1 or 1
    data_dash_plus = jnp.sqrt(0.5) * ratio * d * (
        jnp.exp(b * data_dash - stab - diag_data + diag_omega) +
        numerical_stabilizer)
    data_dash_minus = jnp.sqrt(0.5) * ratio * d * (
        jnp.exp(-b * data_dash - stab - diag_data + diag_omega) +
        numerical_stabilizer)
  else:
    data_dash_plus = jnp.sqrt(0.5) * ratio * d * (
        jnp.exp(b * data_dash - diag_data + diag_omega) + numerical_stabilizer)
    data_dash_minus = jnp.sqrt(0.5) * ratio * d * (
        jnp.exp(-b * data_dash - diag_data + diag_omega) + numerical_stabilizer)
  data_dash = jnp.concatenate((data_dash_plus, data_dash_minus), axis=-1)
  return data_dash


# ------------------------------------------------------------------------------
# General kernel transformation using deterministic features from
# https://arxiv.org/abs/2009.14794.
# ------------------------------------------------------------------------------
def generic_kernel_transformation(
    data,
    extra_data,
    is_query,
    projection_matrix = None,
    numerical_stabilizer = 0.001,
    normalize_data = True,
    numerator_denominator_stabilizer = True,
    activation_fn = jax.nn.relu,
    dot_general = lax.dot_general,
    precision = None
):
  r"""Computes features based on an activation (e.g.

  ReLU-kernel by default).

  By default, computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    extra_data: input extra data tensor of the shape [B...,T, H, D], where T
      stands for the number of the extra tokens; used if additional stats need
      to be collected to define kernel features.
    is_query: indicates whether input data is a query or key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
    normalize_data: whether queries/keys should \sqrt{d}-normalized.
    numerator_denominator_stabilizer: whether numerator and denominator in the
      normalized attention computation should be numerically stabilized.
    activation_fn: activation function to use for the kernel transformation.
      Defaults to relu.
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    Corresponding kernel feature map.
  """
  del extra_data
  del is_query
  del normalize_data
  del numerator_denominator_stabilizer
  if projection_matrix is None:
    return activation_fn(data) + numerical_stabilizer
  else:
    ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
    data_dash = ratio * jnp.einsum('...lhd,md->...lhm',
                                   data, projection_matrix,
                                   precision=precision,
                                   _dot_general=dot_general)
    kernel_feature_map = activation_fn(data_dash) + numerical_stabilizer
    return kernel_feature_map


# ------------------------------------------------------------------------------
# Kernel transformation the softmax kernel using FAVOR++ mechanism
# from https://arxiv.org/abs/2205.15317.
# ------------------------------------------------------------------------------
def expplus_softmax_kernel_transformation(
    base_data,
    extra_data,
    is_query,
    projection_matrix,
    numerical_stabilizer = 0.000001,
    normalize_data = True,
    numerator_denominator_stabilizer = True,
    dot_general = lax.dot_general,
    precision = None
):
  r"""Computes random features for the softmax kernel using FAVOR++ mechanism.

  Computes random features for the softmax kernel using FAVOR++ mechanism.

  Args:
    base_data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    extra_data: auxiliary data tensor of the. same shape as <base_data> for
      computing additional statistics to optimize the coefficients of the random
      maps.
    is_query: indicates whether input data is a query or key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
    normalize_data: whether queries/keys should \sqrt{d}-normalized.
    numerator_denominator_stabilizer: whether numerator and denominator in the
      normalized attention computation should be numerically stabilized.
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    Corresponding kernel feature map.
  """
  data = base_data
  if normalize_data:
    data_normalizer = 1.0 / jnp.sqrt(jnp.sqrt(data.shape[-1]))
  else:
    data_normalizer = 1.0
    lengths = jnp.square(data)
    lengths = jnp.sum(lengths, axis=data.ndim - 1, keepdims=True)
    lengths = jnp.sqrt(lengths)
    data /= lengths
    data *= jnp.sqrt(jnp.sqrt(data.shape[-1]))

  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  data_dash = jnp.einsum(
      '...lhd,md->...lhm',
      data_normalizer * data, projection_matrix,
      precision=precision,
      _dot_general=dot_general,
  )
  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)

  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)

  l = base_data.shape[-3]

  first_sum_of_squares = jnp.square(data)
  first_sum_of_squares = jnp.sum(
      first_sum_of_squares, axis=(1, -1), keepdims=True
  )
  first_sum_of_squares *= data_normalizer * data_normalizer
  first_sum_of_squares /= l
  second_sum_of_squares = jnp.square(extra_data)
  second_sum_of_squares = jnp.sum(
      second_sum_of_squares, axis=(1, -1), keepdims=True
  )
  second_sum_of_squares *= data_normalizer * data_normalizer
  second_sum_of_squares /= l
  data_sum = jnp.sum(data, axis=(1,), keepdims=True)
  extra_data_sum = jnp.sum(extra_data, axis=(1,), keepdims=True)
  d_prod = jnp.einsum('...lhd,...lhd->...lh',
                      data_sum, extra_data_sum,
                      precision=precision,
                      _dot_general=dot_general)
  d_prod = jnp.expand_dims(d_prod, axis=-1)
  d_prod *= data_normalizer * data_normalizer
  d_prod *= 2.0 / (l * l)
  ave = first_sum_of_squares + second_sum_of_squares + d_prod
  dim = projection_matrix.shape[-1]
  A = (1.0 / (4.0 * ave)) * (
      jnp.sqrt((2.0 * ave + dim) * (2.0 * ave + dim) + 8.0 * dim * ave)
      - 2.0 * ave
      - dim
  )
  A = (1.0 - 1.0 / A) / 8.0
  B = jnp.sqrt(1.0 - 4.0 * A)
  D = jnp.power(1.0 - 4.0 * A, dim / 4.0)

  diag_omega = jnp.square(projection_matrix)
  diag_omega = jnp.sum(diag_omega, axis=projection_matrix.ndim - 1)
  diag_omega = jnp.expand_dims(diag_omega, axis=0)
  diag_omega = jnp.expand_dims(diag_omega, axis=0)
  diag_omega = jnp.expand_dims(diag_omega, axis=0)
  diag_omega = A * diag_omega

  if numerator_denominator_stabilizer:
    if is_query:
      last_dims_t = (len(data_dash.shape) - 1,)
      stab = B * jnp.max(data_dash, axis=last_dims_t, keepdims=True)
    else:
      stab = B * jnp.max(data_dash, keepdims=True)
    data_dash = (
        ratio
        * D
        * (
            jnp.exp(B * data_dash - stab - diag_data + diag_omega)
            + numerical_stabilizer
        )
    )
  else:
    data_dash = (
        ratio
        * D
        * (
            jnp.exp(B * data_dash - diag_data + diag_omega)
            + numerical_stabilizer
        )
    )

  return data_dash


# ------------------------------------------------------------------------------
# Kernel transformation the softmax kernel using FAVOR# mechanism (asymmetric
# random features) from https://arxiv.org/abs/2302.00787.
# ------------------------------------------------------------------------------
def exparf_softmax_kernel_transformation(
    base_data,
    other_data,
    is_query,
    projection_matrix,
    numerical_stabilizer = 0.000001,
    normalize_data = True,
    numerator_denominator_stabilizer = True,
    dot_general = lax.dot_general,
    precision = None
):
  r"""Computes random features for the softmax kernel using FAVOR++ mechanism.

  Computes random features for the softmax kernel using FAVOR++ mechanism.

  Args:
    base_data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    other_data: auxiliary data tensor of the. same shape as <base_data> for
      computing additional statistics to optimize the coefficients of the random
      maps.
    is_query: indicates whether input data is a query or key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
    normalize_data: whether queries/keys should \sqrt{d}-normalized.
    numerator_denominator_stabilizer: whether numerator and denominator in the
      normalized attention computation should be numerically stabilized.
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    Corresponding kernel feature map.
  """
  #### Transforming data and other_data ########################################
  E = jnp.square(base_data)
  E = jnp.sum(E, axis=-3)
  F = jnp.square(other_data)
  F = jnp.sum(F, axis=-3)
  G = E / F
  G = jnp.sqrt(jnp.sqrt(G))
  H = 1.0 / G
  data = jnp.einsum('...lhd,...hd->...lhd',
                    base_data, G,
                    precision=precision,
                    _dot_general=dot_general)
  extra_data = jnp.einsum('...lhd,...hd->...lhd',
                          other_data, H,
                          _dot_general=dot_general)

  if normalize_data:
    data_normalizer = 1.0 / jnp.sqrt(jnp.sqrt(data.shape[-1]))
  else:
    data_normalizer = 1.0
    lengths = jnp.square(data)
    lengths = jnp.sum(lengths, axis=data.ndim - 1, keepdims=True)
    lengths = jnp.sqrt(lengths)
    data /= lengths
    data *= jnp.sqrt(jnp.sqrt(data.shape[-1]))

  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  data_dash = jnp.einsum(
      '...lhd,md->...lhm',
      data_normalizer * data, projection_matrix,
      precision=precision,
      _dot_general=dot_general,
  )
  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)

  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)

  l = base_data.shape[-3]

  first_sum_of_squares = jnp.square(data)
  first_sum_of_squares = jnp.sum(
      first_sum_of_squares, axis=(1, -1), keepdims=True
  )
  first_sum_of_squares *= data_normalizer * data_normalizer
  first_sum_of_squares /= l
  second_sum_of_squares = jnp.square(extra_data)
  second_sum_of_squares = jnp.sum(
      second_sum_of_squares, axis=(1, -1), keepdims=True
  )
  second_sum_of_squares *= data_normalizer * data_normalizer
  second_sum_of_squares /= l
  data_sum = jnp.sum(data, axis=(1,), keepdims=True)
  extra_data_sum = jnp.sum(extra_data, axis=(1,), keepdims=True)
  d_prod = jnp.einsum('...lhd,...lhd->...lh',
                      data_sum, extra_data_sum,
                      precision=precision,
                      _dot_general=dot_general)
  d_prod = jnp.expand_dims(d_prod, axis=-1)
  d_prod *= data_normalizer * data_normalizer
  d_prod *= 2.0 / (l * l)
  ave = first_sum_of_squares + second_sum_of_squares + d_prod
  dim = projection_matrix.shape[-1]
  A = (1.0 / (4.0 * ave)) * (
      jnp.sqrt((2.0 * ave + dim) * (2.0 * ave + dim) + 8.0 * dim * ave)
      - 2.0 * ave
      - dim
  )
  A = (1.0 - 1.0 / A) / 8.0
  B = jnp.sqrt(1.0 - 4.0 * A)
  D = jnp.power(1.0 - 4.0 * A, dim / 4.0)

  diag_omega = jnp.square(projection_matrix)
  diag_omega = jnp.sum(diag_omega, axis=projection_matrix.ndim - 1)
  diag_omega = jnp.expand_dims(diag_omega, axis=0)
  diag_omega = jnp.expand_dims(diag_omega, axis=0)
  diag_omega = jnp.expand_dims(diag_omega, axis=0)
  diag_omega = A * diag_omega

  if numerator_denominator_stabilizer:
    if is_query:
      last_dims_t = (len(data_dash.shape) - 1,)
      stab = B * jnp.max(data_dash, axis=last_dims_t, keepdims=True)
    else:
      stab = B * jnp.max(data_dash, keepdims=True)
    data_dash = (
        ratio
        * D
        * (
            jnp.exp(B * data_dash - stab - diag_data + diag_omega)
            + numerical_stabilizer
        )
    )
  else:
    data_dash = (
        ratio
        * D
        * (
            jnp.exp(B * data_dash - diag_data + diag_omega)
            + numerical_stabilizer
        )
    )

  return data_dash


# ------------------------------------------------------------------------------
# Kernel transformation the softmax kernel using FAVOR# mechanism (symmetric
# random features) from https://arxiv.org/abs/2302.00787.
# ------------------------------------------------------------------------------
def expsharp_softmax_kernel_transformation(
    data_orig,
    other_data,
    is_query,
    projection_matrix = None,
    numerical_stabilizer = 0.000001,
    normalize_data = True,
    numerical_renormalizer = True,
    dot_general = lax.dot_general,
    precision = None
):
  """FAVOR# mechanism from the FAVOR# paper: <TO BE PUBLISHED>.

  Args:
    data_orig: data tensor of shape [B,T,H,D] for which random features aree to
      be computed
    other_data: additional tensor of the shape [B,F,H,D] used to collect stats
      to determine the exact instantiation of the random feature mechanism
    is_query: boolean indicating whether <data_orig> tensor is a query tensor
    projection_matrix: tensor of the shape [M,D] encoding random projections for
      random features (M stands for the number of random features)
    numerical_stabilizer: numerical stabilizer for the kernel features
    normalize_data: whether to sqrt-d-normalize queries/keys as in the regular
      attention
    numerical_renormalizer: whether to apply additional renormalization for
      numerical stability
    dot_general: the function that performs dot product in the einsum.
    precision: the precision with which the einsum is performed.

  Returns:
    Random feature map tensor for the unbiased softmax-kernel estimation.
  """
  del numerical_renormalizer

  data = data_orig
  if projection_matrix is None:
    return data_orig
  if normalize_data:
    data_normalizer = 1.0 / jnp.sqrt(jnp.sqrt(data.shape[-1]))
  else:
    data_normalizer = 1.0
    lengths = jnp.square(data)
    lengths = jnp.sum(lengths, axis=len(data.shape) - 1)
    lengths = jnp.expand_dims(lengths, axis=len(data.shape) - 1)
    lengths = jnp.sqrt(lengths)
    data /= lengths

  ######################### Calculating: A, B, D ###############################
  l = data_orig.shape[-3]

  mu_one = (1.0 / l) * jnp.sum(data_orig, axis=-3)  # [B, H, D]
  mu_two = (1.0 / l) * jnp.sum(other_data, axis=-3)  # [B, H, D]

  M_one = (1.0 / l) * jnp.einsum(
      '...lhd,...lhx->...hdx',
      data_orig, data_orig,
      precision=precision,
      _dot_general=dot_general,
  )  # [B, H, D, D]
  M_two = (1.0 / l) * jnp.einsum(
      '...lhd,...lhx->...hdx',
      other_data, other_data,
      precision=precision,
      _dot_general=dot_general,
  )  # [B, H, D, D]

  P = (
      jnp.einsum('...hd,...hx->...hdx',
                 mu_one, mu_two,
                 precision=precision,
                 _dot_general=dot_general)
      + jnp.einsum('...hd,...hx->...hdx',
                   mu_two, mu_one,
                   precision=precision,
                   _dot_general=dot_general)
      + M_one
      + M_two
  )  # [B, H, D, D]

  Sigma, Q = jnp.linalg.eigh(P)  # Q: [B, H, D, D], Sigma = [B, H, D]

  A = jnp.sqrt((2.0 * Sigma + 1.0) * (2.0 * Sigma + 1.0) + 8.0 * Sigma)
  A = (1.0 / 16.0) * (1.0 - 2.0 * Sigma - A)  # [B, H, D]
  B = jnp.sqrt(1.0 - 4.0 * A)
  B = jnp.einsum('...hd,...hxd->...hdx',
                 B, Q,
                 precision=precision,
                 _dot_general=dot_general)  # [B, H, D, D]
  ##############################################################################

  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  b_data = jnp.einsum('...lhd, ...hed->...lhe',
                      data, B,
                      precision=precision,
                      _dot_general=dot_general)
  data_dash = jnp.einsum(
      '...lhd,md->...lhm',
      data_normalizer * b_data, projection_matrix,
      precision=precision,
      _dot_general=dot_general,
  )
  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=len(data.shape) - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=len(data.shape) - 1)

  diag_omega = jnp.square(projection_matrix)
  diag_omega = jnp.einsum('md,...hd->...hmd',
                          diag_omega, A,
                          precision=precision,
                          _dot_general=dot_general)
  diag_omega = jnp.sum(diag_omega, axis=-1)
  diag_omega = jnp.expand_dims(diag_omega, axis=-3)

  if is_query:
    last_dims_t = (len(data_dash.shape) - 1,)
    stab = jnp.max(data_dash, axis=last_dims_t, keepdims=True)
  else:
    stab = jnp.max(data_dash, keepdims=True)
  data_dash = ratio * (
      jnp.exp(data_dash - stab - diag_data + diag_omega) +
      numerical_stabilizer)
  return data_dash
