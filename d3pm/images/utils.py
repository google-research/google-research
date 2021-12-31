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

"""Utilities for categorical diffusion and training loop."""

import functools
from absl import logging
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as onp
import PIL
import tensorflow.compat.v2 as tf

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


def log_min_exp(a, b, epsilon=1.e-6):
  """Computes the log(exp(a) - exp(b)) (b<a) in a numerically stable fashion."""
  y = a + jnp.log1p(-jnp.exp(b - a) + epsilon)
  return y


def sample_categorical(logits, uniform_noise):
  """Samples from a categorical distribution.

  Args:
    logits: logits that determine categorical distributions. Shape should be
      broadcastable under addition with noise shape, and of the form (...,
      num_classes).
    uniform_noise: uniform noise in range [0, 1). Shape: (..., num_classes).

  Returns:
    samples: samples.shape == noise.shape, with samples.shape[-1] equal to
      num_classes.
  """
  # For numerical precision clip the noise to a minimum value
  uniform_noise = jnp.clip(
      uniform_noise, a_min=jnp.finfo(uniform_noise.dtype).tiny, a_max=1.)
  gumbel_noise = -jnp.log(-jnp.log(uniform_noise))
  sample = jnp.argmax(logits + gumbel_noise, axis=-1)
  return jax.nn.one_hot(sample, num_classes=logits.shape[-1])


def categorical_kl_logits(logits1, logits2, eps=1.e-6):
  """KL divergence between categorical distributions.

  Distributions parameterized by logits.

  Args:
    logits1: logits of the first distribution. Last dim is class dim.
    logits2: logits of the second distribution. Last dim is class dim.
    eps: float small number to avoid numerical issues.

  Returns:
    KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
  """
  out = (
      jax.nn.softmax(logits1 + eps, axis=-1) *
      (jax.nn.log_softmax(logits1 + eps, axis=-1) -
       jax.nn.log_softmax(logits2 + eps, axis=-1)))
  return jnp.sum(out, axis=-1)


def categorical_kl_probs(probs1, probs2, eps=1.e-6):
  """KL divergence between categorical distributions.

  Distributions parameterized by logits.

  Args:
    probs1: probs of the first distribution. Last dim is class dim.
    probs2: probs of the second distribution. Last dim is class dim.
    eps: float small number to avoid numerical issues.

  Returns:
    KL(C(probs) || C(logits2)): shape: logits1.shape[:-1]
  """
  out = probs1 * (jnp.log(probs1 + eps) - jnp.log(probs2 + eps))
  return jnp.sum(out, axis=-1)


def categorical_log_likelihood(x, logits):
  """Log likelihood of a discretized Gaussian specialized for image data.

  Assumes data `x` consists of integers [0, num_classes-1].

  Args:
    x: where to evaluate the distribution. shape = (bs, ...), dtype=int32/int64
    logits: logits, shape = (bs, ..., num_classes)

  Returns:
    log likelihoods
  """
  log_probs = jax.nn.log_softmax(logits)
  x_onehot = jax.nn.one_hot(x, logits.shape[-1])
  return jnp.sum(log_probs * x_onehot, axis=-1)


def meanflat(x):
  """Take the mean over all axes except the first batch dimension."""
  return x.mean(axis=tuple(range(1, len(x.shape))))


def global_norm(pytree):
  return jnp.sqrt(
      jnp.sum(
          jnp.asarray([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(pytree)
                      ])))


@functools.partial(jax.jit, static_argnums=(2,))
def _foldin_and_split(rng, foldin_data, num):
  return jax.random.split(jax.random.fold_in(rng, foldin_data), num)


def jax_randint(key, minval=0, maxval=2**20):
  return int(jax.random.randint(key, shape=(), minval=minval, maxval=maxval))


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


def clip_by_global_norm(pytree, clip_norm, use_norm=None):
  if use_norm is None:
    use_norm = global_norm(pytree)
    # assert use_norm.shape == ()
    assert not use_norm.shape
  scale = clip_norm * jnp.minimum(1.0 / use_norm, 1.0 / clip_norm)
  return jax.tree_map(lambda x: x * scale, pytree), use_norm


def apply_ema(decay, avg, new):
  return jax.tree_multimap(lambda a, b: decay * a + (1. - decay) * b, avg, new)


def count_params(pytree):
  return sum([x.size for x in jax.tree_leaves(pytree)])


def copy_pytree(pytree):
  return jax.tree_map(jnp.array, pytree)


def dist(fn, accumulate, axis_name='batch'):
  """Wrap a function in pmap and device_get(unreplicate(.)) its return value."""

  if accumulate == 'concat':
    accumulate_fn = functools.partial(
        allgather_and_reshape, axis_name=axis_name)
  elif accumulate == 'mean':
    accumulate_fn = functools.partial(jax.lax.pmean, axis_name=axis_name)
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


def allgather_and_reshape(x, axis_name='batch'):
  """Allgather and merge the newly inserted axis w/ the original batch axis."""
  y = jax.lax.all_gather(x, axis_name=axis_name)
  assert y.shape[1:] == x.shape
  return y.reshape(y.shape[0] * x.shape[0], *x.shape[1:])


def write_config_json(config, path):
  if tf.io.gfile.exists(path):
    return
  with tf.io.gfile.GFile(path, 'w') as f:
    f.write(config.to_json_best_effort(sort_keys=True, indent=4) + '\n')


def tf_to_numpy(tf_batch):
  """TF to NumPy, using ._numpy() to avoid copy."""
  # pylint: disable=protected-access
  return jax.tree_map(
      lambda x: x._numpy() if hasattr(x, '_numpy') else x,
      tf_batch)


def numpy_iter(tf_dataset):
  return map(tf_to_numpy, iter(tf_dataset))


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
  # assert_synced.problem = pytree
  # raise NotImplementedError()
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
      constant_values=pad_val)
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
