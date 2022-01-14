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

"""Helper methods for Dream Fields."""

import functools
import operator
from typing import Any

from . import mipnerf
from . import scene

from absl import logging
import jax
import jax.numpy as np
import ml_collections
from scenic.projects.baselines.clip import model as clip
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer


PRNGKey = Any




@functools.partial(jax.jit, static_argnums=(2,))
def _foldin_and_split(rng, foldin_data, num):
  return jax.random.split(jax.random.fold_in(rng, foldin_data), num)


class RngGen:
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


def defragment():
  client = jax.lib.xla_bridge.get_backend()
  logging.info('starting defragment...')
  try:
    client.defragment()
    logging.info('finished defragment')
  except:  # pylint: disable=bare-except
    logging.info('defragmentation not implemented')


def matmul(a, b):
  """np.matmul defaults to bfloat16, but this helper function doesn't."""
  return np.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def normalize(x):
  """Normalize array along last dimension."""
  return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cosine_similarity(x, y):
  norm_x = np.linalg.norm(x, axis=-1)
  norm_y = np.linalg.norm(y, axis=-1)
  return np.sum(x * y, axis=-1) / (norm_x * norm_y)


def tree_norm(tree):
  """Compute the l2 norm of the leaves of a pytree."""
  reduce_fn = lambda s, a: s + np.sum(np.square(a))
  sum_sq = jax.tree_util.tree_reduce(reduce_fn, tree, 0)
  return np.sqrt(sum_sq)


def all_finite(array):
  return np.all(np.isfinite(array))


def all_finite_tree(tree):
  finite_tree = jax.tree_map(all_finite, tree)
  return all(jax.tree_leaves(finite_tree))


def state_to_variables(optimizer):
  """Convert an optimizer state to a variable dict."""
  variables = jax.tree_map(operator.itemgetter(0), optimizer)
  variables = jax.device_get(variables).target
  return variables


# CLIP
def load_image_text_model(model_name):
  if model_name.lower().startswith('clip_'):
    model_type = model_name[len('clip_'):]
    return load_clip(model_type)
  raise NotImplementedError


def load_clip(model_name):
  """Load CLIP model by name from the Scenic library."""
  model = clip.MODELS[model_name]()
  clip_vars = clip.load_model_vars(model_name)
  # pylint: disable=g-long-lambda
  encode_text = jax.jit(lambda texts: model.apply(
      clip_vars, text=texts, normalize=True, method=model.encode_text))
  encode_image = jax.jit(lambda x: model.apply(
      clip_vars, image=x, normalize=True, method=model.encode_image))
  # pylint: enable=g-long-lambda
  tokenize_fn = clip_tokenizer.build_tokenizer()
  return encode_image, encode_text, clip.normalize_image, tokenize_fn


def init_nerf_model(key, config):
  """Initialize NeRF MLP."""
  if config.parameterization == 'mipnerf':
    model = mipnerf.MipMLPLate(
        activation=config.mlp_activation,
        features_early=config.features_early,
        features_residual=config.features_residual,
        features_late=config.features_late,
        max_deg=config.posenc_deg,
        fourfeat=config.fourfeat,
        use_cov=config.mipnerf.get('use_cov', True),
        dropout_rate=config.mipnerf.get('dropout_rate', 0.))
    if config.viewdirs:
      x_late = scene.posenc(np.zeros([1, 3]), config.posenc_dirs_deg)
    else:
      x_late = None
    variables = model.init(
        key, np.zeros([1, 3]), np.zeros([1, 3, 3]), x_late, deterministic=True)

    render_rays = functools.partial(mipnerf.render_rays_mip, model=model)
  else:
    raise ValueError

  return variables, render_rays
