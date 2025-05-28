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

"""Main train loop for DP-LS. File intended to be mostly self-contained."""

import functools

from clu import metric_writers
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import tensorflow as tf

from dp_transfer import data_utils
from dp_transfer import dataset
from dp_transfer import linear_regression_sanitizer
from dp_transfer import utils



def train_and_evaluate(config, workdir):
  """Top level training and eval loop."""
  tf.io.gfile.makedirs(workdir)
  start_step = 0

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)
  if start_step == 0:
    writer.write_hparams(dict(config))

  data_config = data_utils.get_data_config(config)
  train_ds, test_ds = dataset.get_datasets(
      config=config,
      data_config=data_config,
      batch_size=1024,
      repeat=False
  )

  test_xs = []
  test_labels = []
  for x in test_ds:
    test_xs.append(x['repr'])
    test_labels.append(x['label'])
  test_x_np_list = utils.to_flat_np(
      test_xs, test_labels, data_config.num_labels
  )

  train_xs = []
  train_labels = []
  for x in train_ds:
    train_xs.append(x['repr'])
    train_labels.append(x['label'])
  x_np_list = utils.to_flat_np(train_xs, train_labels, data_config.num_labels)

  xtx_list = []
  xty_list = []
  sanitizer = linear_regression_sanitizer.Sanitizer(
      max_norm=data_config.clip, random_seed=config.seed
  )

  @jax.jit
  def process(x):
    if config.is_private and config.epsilon > 0:
      x = sanitizer.clip(x)
    return x.T @ x, jnp.sum(x, axis=0)

  for x in x_np_list:
    x = jnp.array(x)
    y1, y2 = process(x)
    xtx_list.append(y1)
    xty_list.append(y2)

  if data_config.num_labels == 1000:
    num_devices = 8
  elif data_config.num_labels == 100:
    num_devices = 4
  elif data_config.num_labels == 10:
    num_devices = 2

  gramian = np.sum(np.array(xtx_list), axis=0)
  xtx = np.stack(xtx_list).reshape([
      data_config.num_labels // num_devices,
      num_devices,
      data_config.hidden_dims,
      data_config.hidden_dims,
  ])
  xty = np.stack(xty_list).reshape([
      data_config.num_labels // num_devices,
      num_devices,
      data_config.hidden_dims,
  ])

  if config.is_private and config.epsilon > 0:
    sanitizer.set_sigmas(
        target_epsilon=config.epsilon,
        target_delta=data_config.delta,
        sigma_ratio0=1.0,
        sigma_ratio1=1.0)
    gramian = sanitizer.apply_noise_gramian(gramian)
    sanitizer.refresh_key()

  def solve_theta(lhs, rhs, gramian):
    if config.is_private and config.epsilon > 0:
      lhs, rhs = sanitizer.apply_noise([lhs, rhs])

    # Add l2 reg
    lhs += jnp.identity(data_config.hidden_dims) * config.reg

    # Add gramian for negatives
    lhs += config.alpha * gramian

    return jnp.linalg.solve(lhs, rhs)

  solve_theta_pmap = jax.pmap(
      functools.partial(solve_theta, gramian=gramian),
      devices=jax.local_devices()[:num_devices])

  theta_list = []
  for xtxi, xtyi in zip(xtx, xty):
    thetai = np.array(solve_theta_pmap(xtxi, xtyi))
    theta_list.append(thetai)
  thetas = np.concatenate(theta_list, axis=0)

  nc = 0.0
  t = 0.0
  for l in range(data_config.num_labels):
    l_p = jnp.argmax(
        jnp.einsum('ld,nd->nl', thetas, jnp.array(test_x_np_list[l])), axis=1)
    t += len(test_x_np_list[l])
    nc += jnp.sum(l_p == l)
  print(f'Test Accuracy: {nc / t}')

  summary = {}
  summary['accuracy'] = nc / t
  with metric_writers.ensure_flushes(writer):
    writer.write_scalars(1, summary)

