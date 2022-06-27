# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Utilities."""

# pylint: disable=invalid-name

import functools

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as onp
import PIL


def normalize_data(x):
  return x / 127.5 - 1.


def unnormalize_data(x):
  return (x + 1.) * 127.5


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


def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255, num_col=0):
  PIL.Image.fromarray(
      np_tile_imgs(
          imgs, pad_pixels=pad_pixels, pad_val=pad_val,
          num_col=num_col)).save(filename)


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


def reverse_fori_loop(lower, upper, body_fun, init_val):
  """Loop from upper-1 to lower."""
  if isinstance(lower, int) and isinstance(upper, int) and lower >= upper:
    raise ValueError('Expected lower < upper')

  def reverse_body_fun(i, val):
    return body_fun(upper + lower - 1 - i, val)

  return jax.lax.fori_loop(lower, upper, reverse_body_fun, init_val)


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


@jax.custom_jvp
def log1mexp(x):
  """Accurate computation of log(1 - exp(-x)) for x > 0."""
  # From James Townsend's PixelCNN++ code
  # Method from
  # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  return jnp.where(x > jnp.log(2), jnp.log1p(-jnp.exp(-x)),
                   jnp.log(-jnp.expm1(-x)))


# log1mexp produces NAN gradients for small inputs because the derivative of the
# log1p(-exp(-eps)) branch has a zero divisor (1 + -jnp.exp(-eps)), and NANs in
# the derivative of one branch of a where cause NANs in the where's vjp, even
# when the NAN branch is not taken. See
# https://github.com/google/jax/issues/1052. We work around this by defining a
# custom jvp.
log1mexp.defjvps(lambda t, _, x: t / jnp.expm1(x))


def broadcast_from_left(x, shape):
  assert len(shape) >= x.ndim
  return jnp.broadcast_to(
      x.reshape(x.shape + (1,) * (len(shape) - x.ndim)),
      shape)

