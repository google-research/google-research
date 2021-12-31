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

"""Unet Utilities.

Taken from brain/diffusion/v3. Moved to remove dependencies and allow
modifications later.
"""

# pylint: disable=invalid-name

import functools
from typing import Iterable

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as onp


# for left-multiplication for RGB -> Y'PbPr
RGB_TO_YUV = onp.array([[0.29900, -0.16874, 0.50000],
                        [0.58700, -0.33126, -0.41869],
                        [0.11400, 0.50000, -0.08131]])


def normalize_data(x, mode=None):
  if mode is None or mode == 'rgb':
    return x / 127.5 - 1.
  elif mode == 'rgb_unit_var':
    return 2. * normalize_data(x, mode='rgb')
  elif mode == 'yuv':
    return (x / 127.5 - 1.).dot(RGB_TO_YUV)
  else:
    raise NotImplementedError(mode)


def unnormalize_data(x, mode=None):
  if mode is None or mode == 'rgb':
    return (x + 1.) * 127.5
  elif mode == 'rgb_unit_var':
    return unnormalize_data(0.5 * x, mode='rgb')
  elif mode == 'yuv':
    return (x.dot(onp.linalg.inv(RGB_TO_YUV)) + 1.) * 127.5
  else:
    raise NotImplementedError(mode)


def nearest_neighbor_upsample(x, k=2):
  B, H, W, C = x.shape
  x = x.reshape(B, H, 1, W, 1, C)
  x = jnp.broadcast_to(x, (B, H, k, W, k, C))
  return x.reshape(B, H * k, W * k, C)


def space_to_depth(x, k=2):
  B, H, W, C = x.shape
  assert H % k == 0 and W % k == 0
  x = x.reshape((B, H // k, k, W // k, k, C))
  x = x.transpose((0, 1, 3, 2, 4, 5))
  x = x.reshape((B, H // k, W // k, k * k * C))
  return x


def depth_to_space(x, k=2):
  B, h, w, c = x.shape
  x = x.reshape((B, h, w, k, k, c // (k * k)))
  x = x.transpose((0, 1, 3, 2, 4, 5))
  x = x.reshape((B, h * k, w * k, c // (k * k)))
  return x


def np_tile_imgs(imgs, *, pad_pixels=1, pad_val=255, num_col=0):
  """NumPy utility: tile a batch of images into a single image.

  Args:
    imgs: np.ndarray: a uint8 array of images of shape [n, h, w, c]
    pad_pixels: int: number of pixels of padding to add around each image
    pad_val: int: padding value
    num_col: int: number of columns in the tiling; defaults to a square

  Returns:
    np.ndarray: one tiled image: a uint8 array of shape [H, W, c]
  """
  if pad_pixels < 0:
    raise ValueError('Expected pad_pixels >= 0')
  if not 0 <= pad_val <= 255:
    raise ValueError('Expected pad_val in [0, 255]')

  imgs = onp.asarray(imgs)
  if imgs.dtype != onp.uint8:
    raise ValueError('Expected uint8 input')
  # if imgs.ndim == 3:
  #   imgs = imgs[..., None]
  n, h, w, c = imgs.shape
  if c not in [1, 3]:
    raise ValueError('Expected 1 or 3 channels')

  if num_col <= 0:
    # Make a square
    ceil_sqrt_n = int(onp.ceil(onp.sqrt(float(n))))
    num_row = ceil_sqrt_n
    num_col = ceil_sqrt_n
  else:
    # Make a B/num_per_row x num_per_row grid
    assert n % num_col == 0
    num_row = int(onp.ceil(n / num_col))

  imgs = onp.pad(
      imgs,
      pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels),
                 (pad_pixels, pad_pixels), (0, 0)),
      mode='constant',
      constant_values=pad_val
  )
  h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
  imgs = imgs.reshape(num_row, num_col, h, w, c)
  imgs = imgs.transpose(0, 2, 1, 3, 4)
  imgs = imgs.reshape(num_row * h, num_col * w, c)

  if pad_pixels > 0:
    imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
  if c == 1:
    imgs = imgs[Ellipsis, 0]
  return imgs


@functools.partial(jax.pmap, axis_name='batch')
def _check_synced(pytree):
  mins = jax.lax.pmin(pytree, axis_name='batch')
  equals = jax.tree_multimap(jnp.array_equal, pytree, mins)
  return jnp.all(jnp.asarray(jax.tree_leaves(equals)))


def assert_synced(pytree):
  """Check that `pytree` is the same across all replicas.

  Args:
    pytree: the pytree to check (should be replicated)

  Raises:
    RuntimeError: if sync check failed
  """
  equals = _check_synced(pytree)
  assert equals.shape == (jax.local_device_count(),)
  equals = all(jax.device_get(equals))  # no unreplicate
  logging.info('Sync check result: %d', equals)
  if not equals:
    raise RuntimeError('Sync check failed!')


@functools.partial(jax.pmap, axis_name='i')
def _barrier(x):
  return jax.lax.psum(x, axis_name='i')


def barrier():
  """MPI-like barrier."""
  jax.device_get(_barrier(jnp.ones((jax.local_device_count(),))))


def allgather_and_reshape(x, axis_name='batch'):
  """Allgather and merge the newly inserted axis w/ the original batch axis."""
  y = jax.lax.all_gather(x, axis_name=axis_name)
  assert y.shape[1:] == x.shape
  return y.reshape(y.shape[0] * x.shape[0], *x.shape[1:])


# def allgather_output(fn, axis_name='batch'):
#   """Process a function's return value using `allgather_and_reshape`."""

#   @functools.wraps(fn)
#   def wrapper(*args, **kwargs):
#     return jax.tree_map(
#         functools.partial(allgather_and_reshape, axis_name=axis_name),
#         fn(*args, **kwargs))

#   return wrapper


# def concat_unreplicate_map(fn, xs, callback=None):
#   """Map `fn` over `xs`, unreplicate results, then concatenate in NumPy."""
#   ys = []
#   for i, x in enumerate(xs):
#     ys.append(jax.device_get(flax.jax_utils.unreplicate(fn(x))))
#     if callback is not None:
#       callback(i)
#   return treecat(ys)


def np_treecat(xs):
  return jax.tree_multimap(lambda *zs: onp.concatenate(zs, axis=0), *xs)


def dist(fn, accumulate, axis_name='batch'):
  """Wrap a function in pmap and device_get(unreplicate(.)) its return value."""

  if accumulate == 'concat':
    accumulate_fn = functools.partial(
        allgather_and_reshape, axis_name=axis_name)
  elif accumulate == 'mean':
    accumulate_fn = functools.partial(
        jax.lax.pmean, axis_name=axis_name)
  elif accumulate == 'none':
    accumulate_fn = None
  else:
    raise NotImplementedError(accumulate)

  @functools.partial(jax.pmap, axis_name=axis_name)
  def pmapped_fn(*args, **kwargs):
    out = fn(*args, **kwargs)
    return out if accumulate_fn is None else jax.tree_map(accumulate_fn, out)

  def wrapper(*args, **kwargs):
    return jax.device_get(
        flax.jax_utils.unreplicate(pmapped_fn(*args, **kwargs)))

  return wrapper


def tf_to_numpy(tf_batch):
  """TF to NumPy, using ._numpy() to avoid copy."""
  # pylint: disable=protected-access,g-long-lambda
  return jax.tree_map(lambda x: (x._numpy()
                                 if hasattr(x, '_numpy') else x), tf_batch)


def numpy_iter(tf_dataset):
  return map(tf_to_numpy, iter(tf_dataset))


def sumflat(x):
  return x.sum(axis=tuple(range(1, len(x.shape))))


def meanflat(x):
  return x.mean(axis=tuple(range(1, len(x.shape))))


def flatten(x):
  return x.reshape(x.shape[0], -1)


def l2normalize(x, axis, eps=1e-8):
  return x * jax.lax.rsqrt(jnp.square(x).sum(axis=axis, keepdims=True) + eps)


def to_onehot(x, depth):
  return (x[Ellipsis, None] == jnp.arange(depth)).astype(x.dtype)


def reverse_fori_loop(lower, upper, body_fun, init_val):
  """Loop from upper-1 to lower."""
  if isinstance(lower, int) and isinstance(upper, int) and lower >= upper:
    raise ValueError('Expected lower < upper')

  def reverse_body_fun(i, val):
    return body_fun(upper + lower - 1 - i, val)

  return jax.lax.fori_loop(lower, upper, reverse_body_fun, init_val)


def normal_sample(*, rng, mean, logvar):
  mean, logvar = jnp.broadcast_arrays(mean, logvar)
  return mean + jnp.exp(0.5 * logvar) * jax.random.normal(
      rng, shape=logvar.shape)


def normal_kl(mean1, logvar1, mean2, logvar2):
  """KL divergence between normal distributions.

  Distributions parameterized by mean and log variance.

  Args:
    mean1: mean of the first distribution
    logvar1: log variance of the first distribution
    mean2: mean of the second distribution
    logvar2: log variance of the second distribution

  Returns:
    KL(N(mean1, exp(logvar1)) || N(mean2, exp(logvar2)))
  """
  return 0.5 * (-1.0 + logvar2 - logvar1 + jnp.exp(logvar1 - logvar2)
                + jnp.square(mean1 - mean2) * jnp.exp(-logvar2))


def approx_normal_cdf(x):
  return 0.5 * (1.0 + jnp.tanh(
      onp.sqrt(2.0 / onp.pi) * (x + 0.044715 * jnp.power(x, 3))))


def normal_cdf(x):
  return 0.5 * (1.0 + jax.lax.erf(x * (2.0 ** -0.5)))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
  """Log likelihood of a discretized Gaussian specialized for image data.

  Assumes data `x` consists of integers [0, 255] rescaled to [-1, 1].

  Args:
    x: where to evaluate the distribution
    means: the means of the distribution
    log_scales: log standard deviations (WARNING: not the log variance).

  Returns:
    log likelihoods
  """
  assert x.shape == means.shape == log_scales.shape

  centered_x = x - means
  inv_stdv = jnp.exp(-log_scales)
  cdf_plus = normal_cdf(inv_stdv * (centered_x + 1. / 255.))
  cdf_min = normal_cdf(inv_stdv * (centered_x - 1. / 255.))

  def safe_log(z):
    return jnp.log(jnp.maximum(z, 1e-12))

  log_cdf_plus = safe_log(cdf_plus)
  log_one_minus_cdf_min = safe_log(1. - cdf_min)
  log_cdf_delta = safe_log(cdf_plus - cdf_min)
  log_probs = jnp.where(
      x < -0.999, log_cdf_plus,
      jnp.where(x > 0.999, log_one_minus_cdf_min, log_cdf_delta))
  assert log_probs.shape == x.shape
  return log_probs


def count_params(pytree):
  return sum([x.size for x in jax.tree_leaves(pytree)])


def copy_pytree(pytree):
  return jax.tree_map(jnp.array, pytree)


def global_norm(pytree):
  return jnp.sqrt(jnp.sum(jnp.asarray(
      [jnp.sum(jnp.square(x)) for x in jax.tree_leaves(pytree)])))


def clip_by_global_norm(pytree, clip_norm, use_norm=None):
  if use_norm is None:
    use_norm = global_norm(pytree)
    assert use_norm.shape == ()  # pylint: disable=g-explicit-bool-comparison
  scale = clip_norm * jnp.minimum(1.0 / use_norm, 1.0 / clip_norm)
  return jax.tree_map(lambda x: x * scale, pytree), use_norm


def apply_ema(decay, avg, new):
  return jax.tree_multimap(lambda a, b: decay * a + (1. - decay) * b, avg, new)


def scale_init(scale, init_fn, dtype=jnp.float32):
  """Scale the output of an initializer."""

  def init(key, shape, dtype=dtype):
    return scale * init_fn(key, shape, dtype)

  return init


@functools.partial(jax.jit, static_argnums=(2,))
def _foldin_and_split(rng, foldin_data, num):
  return jax.random.split(jax.random.fold_in(rng, foldin_data), num)


class RngGen(object):
  """Random number generator state utility for Jax."""

  def __init__(self, init_rng):
    self._base_rng = init_rng
    self._counter = 0

  def __iter__(self):
    return self

  def __next__(self):
    return self.advance(1)

  def advance(self, count):
    self._counter += count
    return jax.random.fold_in(self._base_rng, self._counter)

  def split(self, num):
    self._counter += 1
    return _foldin_and_split(self._base_rng, self._counter, num)


def jax_randint(key, minval=0, maxval=2**20):
  return int(jax.random.randint(key, shape=(), minval=minval, maxval=maxval))


### Attention (Linen's version doesn't support arbitrary axes)


def dot_product_attention(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          # broadcast_dropout=True,
                          # dropout_rng=None,
                          # dropout_rate=0.,
                          # deterministic=False,
                          precision=None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
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
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  query = query / onp.sqrt(depth).astype(dtype)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = jax.lax.dot_general(
      query,
      key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
  attn_weights = attn_weights.astype(dtype)

  # # apply dropout
  # if not deterministic and dropout_rate > 0.:
  #   if dropout_rng is None:
  #     dropout_rng = stochastic.make_rng()
  #   keep_prob = jax.lax.tie_in(attn_weights, 1.0 - dropout_rate)
  #   if broadcast_dropout:
  #     # dropout is broadcast across the batch+head+non-attention dimension
  #     dropout_dims = attn_weights.shape[-(2 * len(axis)):]
  #     dropout_shape = (tuple([1] * len(batch_dims_t)) + dropout_dims)
  #     keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
  #   else:
  #     keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
  #   multiplier = (keep.astype(attn_weights.dtype) /
  #                 jnp.asarray(keep_prob, dtype=dtype))
  #   attn_weights = attn_weights * multiplier

  # compute the new values given the attention weights
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = jax.lax.dot_general(
      attn_weights,
      value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      precision=precision)

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  return y


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)
