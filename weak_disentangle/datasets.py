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

# python3
"""Dataset-specific utilities."""

# pylint: disable=g-bad-import-order, unused-import, g-multiple-import
# pylint: disable=line-too-long, missing-docstring, g-importing-member
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.data.ground_truth import shapes3d
from disentanglement_lib.data.ground_truth import mpi3d
from disentanglement_lib.data.ground_truth import cars3d
from disentanglement_lib.data.ground_truth import norb
import numpy as np
import tensorflow.compat.v1 as tf
import gin

from weak_disentangle import utils as ut


def get_dlib_data(task):
  ut.log("Loading {}".format(task))
  if task == "dsprites":
    # 5 factors
    return dsprites.DSprites(list(range(1, 6)))
  elif task == "shapes3d":
    # 6 factors
    return shapes3d.Shapes3D()
  elif task == "norb":
    # 4 factors + 1 nuisance (which we'll handle via n_dim=2)
    return norb.SmallNORB()
  elif task == "cars3d":
    # 3 factors
    return cars3d.Cars3D()
  elif task == "mpi3d":
    # 7 factors
    return mpi3d.MPI3D()
  elif task == "scream":
    # 5 factors + 2 nuisance (handled as n_dim=2)
    return dsprites.ScreamDSprites(list(range(1, 6)))


@gin.configurable
def make_masks(string, s_dim, mask_type):
  strategy, factors = string.split("=")
  assert strategy in {"s", "c", "r", "cs", "l"}, "Only allow label, share, change, rank-types"

  # mask_type is only here to help sanity-check that I didn't accidentally
  # use an invalid (strategy , mask_type) pair
  if strategy == "r":
    assert mask_type == "rank", "mask_type must match data collection strategy"
    # Use factor indices as mask. Assumes single factor per comma
    return list(map(int, factors.split(",")))
  elif strategy in {"s", "c", "cs"}:
    assert mask_type == "match", "mask_type must match data collection strategy"
  elif strategy in {"l"}:
    assert mask_type == "label", "mask_type must match data collection strategy"

  if strategy == "cs":
    # Pre-process factors to add complement set
    idx = int(factors)
    l = list(range(s_dim))
    del l[idx]
    factors = "{},{}".format(idx, "".join(map(str, l)))

  factors = [list(map(int, l)) for l in map(list, factors.split(","))]
  masks = np.zeros((len(factors), s_dim), dtype=np.float32)
  for (i, f) in enumerate(factors):
    masks[i, f] = 1

  if strategy == "s":
    masks = 1 - masks
  elif strategy == "l":
    assert len(masks) == 1, "Only one mask allowed for label-strategy"

  ut.log("make_masks output:")
  ut.log(masks)
  return masks


def sample_match_factors(dset, batch_size, masks, random_state):
  factor1 = dset.sample_factors(batch_size, random_state)
  factor2 = dset.sample_factors(batch_size, random_state)
  mask_idx = np.random.choice(len(masks), batch_size)
  mask = masks[mask_idx]
  factor2 = factor2 * mask + factor1 * (1 - mask)
  factors = np.concatenate((factor1, factor2), 0)
  return factors, mask_idx


def sample_rank_factors(dset, batch_size, masks, random_state):
  # We assume for ranking that masks is just a list of indices
  factors = dset.sample_factors(2 * batch_size, random_state)
  factor1, factor2 = np.split(factors, 2)
  y = (factor1 > factor2)[:, masks].astype(np.float32)
  return factors, y


def sample_match_images(dset, batch_size, masks, random_state):
  factors, mask_idx = sample_match_factors(dset, batch_size, masks, random_state)
  images = dset.sample_observations_from_factors(factors, random_state)
  x1, x2 = np.split(images, 2)
  return x1, x2, mask_idx


def sample_rank_images(dset, batch_size, masks, random_state):
  factors, y = sample_rank_factors(dset, batch_size, masks, random_state)
  images = dset.sample_observations_from_factors(factors, random_state)
  x1, x2 = np.split(images, 2)
  return x1, x2, y


def sample_images(dset, batch_size, random_state):
  factors = dset.sample_factors(batch_size, random_state)
  return dset.sample_observations_from_factors(factors, random_state)


@gin.configurable
def paired_data_generator(dset, masks, random_seed=None, mask_type="match"):
  if mask_type == "match":
    return match_data_generator(dset, masks, random_seed)
  elif mask_type == "rank":
    return rank_data_generator(dset, masks, random_seed)
  elif mask_type == "label":
    return label_data_generator(dset, masks, random_seed)


def match_data_generator(dset, masks, random_seed=None):
  def generator():
    random_state = np.random.RandomState(random_seed)

    while True:
      x1, x2, idx = sample_match_images(dset, 1, masks, random_state)
      # Returning x1[0] and x2[0] removes batch dimension
      yield x1[0], x2[0], idx.item(0)

  return tf.data.Dataset.from_generator(
      generator,
      (tf.float32, tf.float32, tf.int32),
      output_shapes=(dset.observation_shape, dset.observation_shape, ()))


def rank_data_generator(dset, masks, random_seed=None):
  def generator():
    random_state = np.random.RandomState(random_seed)

    while True:
      # Note: remove batch dimension by returning x1[0], x2[0], y[0]
      x1, x2, y = sample_rank_images(dset, 1, masks, random_state)
      yield x1[0], x2[0], y[0]

  y_dim = len(masks)  # Remember, masks is just a list
  return tf.data.Dataset.from_generator(
      generator,
      (tf.float32, tf.float32, tf.float32),
      output_shapes=(dset.observation_shape, dset.observation_shape, (y_dim,)))


def label_data_generator(dset, masks, random_seed=None):
  # Normalize the factors using mean and stddev
  m, s = [], []
  for factor_size in dset.factors_num_values:
    factor_values = list(range(factor_size))
    m.append(np.mean(factor_values))
    s.append(np.std(factor_values))
  m = np.array(m)
  s = np.array(s)

  def generator():
    random_state = np.random.RandomState(random_seed)

    while True:
      # Note: remove batch dimension by returning x1[0], x2[0], y[0]
      factors = dset.sample_factors(1, random_state)
      x = dset.sample_observations_from_factors(factors, random_state)
      factors = (factors - m) / s  # normalize the factors
      y = factors * masks
      yield x[0], y[0]

  y_dim = masks.shape[-1]  # mask is 1-hot and equal in length to s_dim
  return tf.data.Dataset.from_generator(
      generator,
      (tf.float32, tf.float32),
      output_shapes=(dset.observation_shape, (y_dim,)))


@gin.configurable
def paired_randn(batch_size, z_dim, masks, mask_type="match"):
  if mask_type == "match":
    return match_randn(batch_size, z_dim, masks)
  elif mask_type == "rank":
    return rank_randn(batch_size, z_dim, masks)
  elif mask_type == "label":
    return label_randn(batch_size, z_dim, masks)


def match_randn(batch_size, z_dim, masks):
  # Note that masks.shape[-1] = s_dim and we assume s_dim <= z-dim
  n_dim = z_dim - masks.shape[-1]

  if n_dim == 0:
    z1 = tf.random_normal((batch_size, z_dim))
    z2 = tf.random_normal((batch_size, z_dim))
  else:
    # First sample the controllable latents
    z1 = tf.random_normal((batch_size, masks.shape[-1]))
    z2 = tf.random_normal((batch_size, masks.shape[-1]))

  # Do variable fixing here (controllable latents)
  mask_idx = tf.random_uniform((batch_size,), maxval=len(masks), dtype=tf.int32)
  mask = tf.gather(masks, mask_idx)
  z2 = z2 * mask + z1 * (1 - mask)

  # Add nuisance dims (uncontrollable latents)
  if n_dim > 0:
    z1_append = tf.random_normal((batch_size, n_dim))
    z2_append = tf.random_normal((batch_size, n_dim))
    z1 = tf.concat((z1, z1_append), axis=-1)
    z2 = tf.concat((z2, z2_append), axis=-1)

  return z1, z2, mask_idx


def rank_randn(batch_size, z_dim, masks):
  z1 = tf.random.normal((batch_size, z_dim))
  z2 = tf.random.normal((batch_size, z_dim))
  y = tf.gather(z1 > z2, masks, axis=-1)
  y = tf.cast(y, tf.float32)
  return z1, z2, y


# pylint: disable=unused-argument
def label_randn(batch_size, z_dim, masks):
  # Note that masks.shape[-1] = s_dim and we assume s_dim <= z-dim
  n_dim = z_dim - masks.shape[-1]

  if n_dim == 0:
    return tf.random.normal((batch_size, z_dim)) * (1 - masks)
  else:
    z = tf.random.normal((batch_size, masks.shape[-1])) * (1 - masks)
    n = tf.random.normal((batch_size, n_dim))
    z = tf.concat((z, n), axis=-1)
    return z
