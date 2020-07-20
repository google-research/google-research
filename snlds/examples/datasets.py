# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Dataset utility functions for CAVI SNLDS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.integrate import odeint
import tensorflow as tf


def generate_lorenz_attractor(
    num_steps,
    dt=0.1,
    sigma=10.,
    rho=28.,
    beta=2.667,
    mu=np.array([-0.5, -.5, 2.]),
    std=15.,
    init_position=None,
    noise_level=1e-3,
    burnin_steps=0,
    dtype=np.float32):
  """Adapted from the Lorenz attractor from Scott Linderman."""
  standardize = lambda state: (state - mu) / std
  unstandardize = lambda stdstate: std * stdstate + mu

  def lorenz_dynamics(stdstate, _):
    """Given the previous states, get the derivatives used by ODE."""
    x, y, z = unstandardize(stdstate)

    dx = sigma * (y-x)
    dy = (rho-z) * x - y
    dz = x*y - beta*z

    dstate = np.array([dx, dy, dz]) / std

    return dstate

  if init_position is None:
    init_position = np.random.randn(3) * 5.
  x0 = standardize(init_position)
  t = np.arange(0.0, dt*(num_steps+burnin_steps), dt)
  xfull = odeint(lorenz_dynamics, x0, t)[burnin_steps:, :]
  xfull += np.random.normal(scale=noise_level,
                            size=xfull.shape)

  return xfull.astype(dtype)


def simulate_lorenz(return_single_traj=False):
  """Simulate full dataset for Lorenz attractor."""
  num_paths = 100
  num_steps = 200
  multiples = 500
  burin_steps = 200
  dt = 0.1
  npdataset = np.array(
      [generate_lorenz_attractor(num_steps*multiples, dt=dt,
                                 init_position=None, burnin_steps=burin_steps)
       for _ in range(num_paths)])
  if return_single_traj:
    return npdataset.reshape([-1, 3])
  return npdataset.reshape([num_paths*multiples, num_steps, 3])


def create_lorenz_attractor_by_generator(
    batch_size,
    num_steps=200,
    random_seed=None):
  """Helper function creates a Lorenz Attractor dataset by continuous sampling.

  The dataset could sample the path on the fly. For example:
  ```python
  dataset = create_lorenz_attractor_by_generator(32, 200)
  dataset_iter = dataset.as_numpy_iterator()
  one_batch_data = dataset_iter.next()
  # one_batch_data will be a Tensor with shape [32, 200, 3] for Lorenz Attractor
  # where 3 indicates the (x, y, z) positional information.
  ```

  Args:
    batch_size: an `Int` indicates the number of trajectories per batch.
    num_steps: an `Int` indicates the number of time steps per trajectory.
    random_seed: an `Int` for as random seed passed to tf.random.set_seed().

  Returns:
    a tf.data.Dataset instance for dubins path generator.
    The dataset contains a float Tensor with shape [batch_size, num_steps, 3],
    genenerated by calling iter(dataset).next().
  """
  if random_seed is not None:
    tf.random.set_seed(random_seed)
  def lorenz_traj_generator():
    while True:
      dt = 0.1
      burin_steps = 200
      yield generate_lorenz_attractor(
          num_steps*batch_size, dt=dt,
          init_position=None, burnin_steps=burin_steps).reshape(
              [batch_size, num_steps, 3])
  dataset = tf.data.Dataset.from_generator(
      lorenz_traj_generator, (tf.float32))
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def get_vector_dataset_from_file(dataset_path,
                                 batch_size,
                                 num_repeat=None):
  """Helper function to load saved vector data."""
  with open(dataset_path, 'rb') as f:
    npdataset = np.load(f).astype(np.float32)
  assert npdataset.ndim == 3, (
      'SNLDS get_vector_dataset_from_file: Only support 3-D np.array.')
  dataset = tf.data.Dataset.from_tensor_slices(npdataset)
  dataset = dataset.repeat(count=num_repeat).batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  return dataset

