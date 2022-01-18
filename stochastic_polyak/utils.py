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

"""Utils for Stochastic Polyak solver. Including projections for pytress."""

import os

from typing import Any

import jax
import jax.numpy as jnp
from jaxopt import tree_util

import tensorflow as tf
import tensorflow_datasets as tfds


def create_dumpfile(config, solver_param_name, workdir, dataset):
  """Create directory to dump results."""
  dumpath = os.path.join(
      os.path.join(os.path.join(workdir, dataset), config.solver),
      solver_param_name)
  check_folder = tf.io.gfile.isdir(dumpath)
  if not check_folder:
    tf.io.gfile.makedirs(dumpath)
  return dumpath


def get_datasets(name):
  """Load train and test datasets into memory."""
  ds_builder = tfds.builder(name)
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds


def projection_hyperplane(a, b, x = None):
  r"""Projection onto a hyperplane defined by a pytree and scalar.

  The output is:
    ``argmin_{y, dot(a, y) = b} ||y - x||``.
  Which is equivalent to
     y = x - (<a,x>-b)/<a,a> a
  Args:
    x: pytree to project.
    hyperparams: tuple ``hyperparams = (a, b)``, where ``a`` is a pytree and
      ``b`` is a scalar.
  Returns:
    y: output array (same shape as ``x``)
  """
  if x is None:
    scale = b/tree_util.tree_vdot(a,a)
    return tree_util.tree_scalar_mul(scale, a)
  else:
    scale = (tree_util.tree_vdot(a,x) -b)/tree_util.tree_vdot(a,a)
    return tree_util.tree_add_scalar_mul(x, -scale, a)


def projection_halfspace(x,  a, b):
  r"""Projection onto a halfspace defined by a pytree and scalar.

  The output is:
    ``argmin_{y, dot(a, y) <= b} ||y - x||``.
  Args:
    x: pytree to project.
    a: pytree
    b: pytree

  Returns:
    y: output array (same shape as ``x``)
  """
  # a, b = hyperparams
  scale = jax.nn.relu(tree_util.tree_vdot(a, x) - b) / tree_util.tree_vdot(a, a)
  return tree_util.tree_add_scalar_mul(x, -scale, a)
