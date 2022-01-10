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

"""All style transfer functions in jax."""

import functools

import jax
import jax.numpy as jnp

from gift.utils import tensor_util


def get_covariance(x, mean_x):
  """computes the covariance matrix of a sample matrix x.

  Args:
    x: floatTensor; shape [batch_size, num_samples, dim].
    mean_x: floatTensor; shape [batch_size, dim,]

  Returns:
    cov_x: floatTensor; shape [batch_size, dim, dim].
  """

  num_samples = x.shape[1]
  centered_x = x - mean_x
  cov_x = jnp.matmul(jnp.transpose(centered_x, (0, 2, 1)), centered_x)
  return cov_x / jnp.float32(num_samples)


def get_variance(x, mean_x):
  """computes the variance diagonal of a sample matrix x.

  Args:
    x: floatTensor; shape [batch_size, num_samples, dim]
    mean_x: floatTensor; shape [batch_size, dim]

  Returns:
    var_x: floatTensor; shape [batch_size, dim, dim].
  """

  num_samples = x.shape[1]
  centered_x = x - mean_x
  var_x = jnp.sum(centered_x * centered_x, axis=1, keepdims=True)
  return var_x / jnp.float32(num_samples)


def get_diag_monge_map_fn(content_mean, content_var, style_mean, style_var):
  """Computes the Gaussian monge map from content to style."""

  a_c2s = jax.lax.rsqrt(content_var + 1.e-5) * jnp.sqrt(style_var)

  def monge_map(x):
    # all vectors are row vectors
    return style_mean + a_c2s * (x - content_mean)

  return monge_map


def get_monge_map_fn(content_mean, content_cov, style_mean, style_cov):
  """Computes the Gaussian monge map from content to style."""

  # u == v for symmetric matrix.
  u, s, _ = jax.lax.linalg.svd(content_cov, full_matrices=False)

  sqrt_s = jnp.sqrt(s)
  rsqrt_s = jax.lax.rsqrt(s + 1.e-5)
  u_t = jnp.transpose(u, (0, 2, 1))
  content_cov_sqrt = jnp.matmul(u * sqrt_s[:, None, :], u_t)
  content_cov_inv_sqrt = jnp.matmul(u * rsqrt_s[:, None, :], u_t)

  tmp = jnp.matmul(content_cov_sqrt, style_cov)
  tmp = jnp.matmul(tmp, content_cov_sqrt)

  u, s, _ = jax.lax.linalg.svd(tmp, full_matrices=False)
  u_t = jnp.transpose(u, (0, 2, 1))
  sqrt_s = jnp.sqrt(s)

  sqrt_tmp = jnp.matmul(u * sqrt_s[:, None, :], u_t)

  a_c2s = jnp.matmul(content_cov_inv_sqrt, sqrt_tmp)
  a_c2s = jnp.matmul(a_c2s, content_cov_inv_sqrt)

  def monge_map(x):
    a_c2s_t = jnp.transpose(a_c2s, (0, 2, 1))
    return style_mean + jnp.matmul(x - content_mean, a_c2s_t)

  return monge_map


def wasserstein(content, style, lmbda):
  """Computes wasserstein style transfer for two images or feature maps.

  Args:
    content: floatTensor; shape [batch_size, height, width, channels].
    style: floatTensor; shape [batch_size, height, width, channels].
    lmbda: floatTensor; in [0, 1], shape [batch_size, num_interpolations, 1].

  Returns:
    stylized: floatTensor; shape [batch_size, height, width, channels].
  """
  # Reshape to (num_samples, dim)
  batch_size, height, width, channels = content.shape
  content = jnp.reshape(content, (batch_size, height * width, channels))
  style = jnp.reshape(style, (batch_size, height * width, channels))

  # All computations on shapes arrays of shape [bs, num_samples, channels]
  # with num_samples = height * width
  content_mean = jnp.mean(content, axis=1, keepdims=True)
  style_mean = jnp.mean(style, axis=1, keepdims=True)
  content_cov = get_variance(content, content_mean)
  style_cov = get_variance(style, style_mean)

  monge_map_fn = get_diag_monge_map_fn(content_mean, content_cov, style_mean,
                                       style_cov)
  mapped_content = monge_map_fn(content)

  stylized = tensor_util.convex_interpolate(content, mapped_content, lmbda)

  # Reshape to (batch_size, height, width, channels)
  stylized = jnp.reshape(stylized, (batch_size, height, width, channels))
  return stylized


@functools.partial(jax.vmap, in_axes=(0, 0, 1))
def wct(content, style, lmbda, eps=1e-8):
  """Whiten-Color Transform.

  Taken from https://github.com/eridgd/WCT-TF/blob/master/ops.py

  Assume that content/style encodings have shape 1xHxWxC
  See p.4 of the Universal Style Transfer paper for corresponding
  equations:
  https://arxiv.org/pdf/1705.08086.pdf

  Args:
    content: floatTensor; shape (batch_size, height, width, channels).
    style: floatTensor; shape (batch_size, height, width, channels)
    lmbda: floatTensor; in [0, 1] with shape (batch_size,).
    eps: float; small float for numerical stability.

  Returns:
    blended image: floatTensor, shape (batch_size, height, width, channels)
  """
  # Reorder to CxHxW
  content_t = jnp.transpose(content, (2, 0, 1))
  style_t = jnp.transpose(style, (2, 0, 1))

  c_content, h_content, w_content = content_t.shape
  c_style, h_style, w_style = style_t.shape

  # CxHxW -> CxH*W
  content_flat = jnp.reshape(content_t, (c_content, h_content * w_content))
  style_flat = jnp.reshape(style_t, (c_style, h_style * w_style))

  # Content covariance
  mc = jnp.mean(content_flat, axis=1, keepdims=True)
  fc = content_flat - mc
  fc_t = jnp.transpose(fc, (1, 0))
  fcfc = jnp.matmul(fc, fc_t) / (jnp.float32(h_content * w_content) - 1.)
  fcfc += jnp.eye(c_content) * eps

  # Style covariance
  ms = jnp.mean(style_flat, axis=1, keepdims=True)
  fs = style_flat - ms
  fs_t = jnp.transpose(fs, (1, 0))
  fsfs = jnp.matmul(fs, fs_t) / (jnp.float32(h_style * w_style) - 1.)
  fsfs += jnp.eye(c_style) * eps

  # tf.linalg.svd is slower on GPU,
  # see https://github.com/tensorflow/tensorflow/issues/13603
  # TODO(samiraabnar): Do we have the same concerns in jax?
  u_content, s_content, _ = jax.lax.linalg.svd(fcfc, full_matrices=False)
  u_style, s_style, _ = jax.lax.linalg.svd(fsfs, full_matrices=False)

  # TODO(samiraabnar): Why do we need to do this and can this be done through
  # masking? (Jitted functions can't easily handel dynamic slicing).
  # Filter small singular values
  # k_c = jnp.sum(jnp.int32(s_content > 1e-5))
  # k_s = jnp.sum(jnp.int32(s_style > 1e-5))
  k_c = -1
  k_s = -1

  # Whiten content feature
  d_content = jnp.diag(jax.lax.pow(s_content[:k_c], -0.5))
  fc_hat = jnp.matmul(
      jnp.matmul(
          jnp.matmul(u_content[:, :k_c], d_content),
          jnp.transpose(u_content[:, :k_c], (1, 0))), fc)

  # Color content with style
  d_style = jnp.diag(jax.lax.pow(s_style[:k_s], 0.5))
  fcs_hat = jnp.matmul(
      jnp.matmul(
          jnp.matmul(u_style[:, :k_s], d_style),
          jnp.transpose(u_style[:, :k_s], (1, 0))), fc_hat)

  # Re-center with mean of style
  fcs_hat = fcs_hat + ms

  # Blend whiten-colored feature with original content feature
  blended = tensor_util.convex_interpolate(
      (fc + mc)[None, Ellipsis], fcs_hat[None, Ellipsis], lmbda[None, Ellipsis])[0]

  # CxH*W -> CxHxW
  blended = jnp.reshape(blended, (c_content, h_content, w_content))
  # CxHxW -> HxWxC
  blended = jnp.transpose(blended, (1, 2, 0))

  return blended
