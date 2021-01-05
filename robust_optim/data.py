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

"""Functions for generating a logistic dataset.

  x ~ N(0, I_d)
  y ~ Bernoulli(sigmoid(-(1/temp) w^T x))
"""
import jax
from jax import numpy as jnp


def logistic_dataset_init_param(dim, r, rng_key):
  param0 = jax.random.normal(rng_key, (dim, 1))
  param0_norm = jnp.linalg.norm(param0)
  param = param0 / param0_norm * r
  return param


def logistic_dataset_gen_data(num, w, dim, temp, rng_key):
  """Samples data from a standard Gaussian with binary noisy labels.

  Args:
    num: An integer denoting the number of data points.
    w: An array of size dim x odim, the weight vector used to generate labels.
    dim: An integer denoting the number of input dimensions.
    temp: A float denoting the temperature parameter controlling label noise.
    rng_key: JAX random number generator key.

  Returns:
    x: An array of size dim x num denoting data points.
    y_pm: An array of size num x odim denoting +/-1 labels.
  """
  rng_subkey = jax.random.split(rng_key, 3)
  x = jax.random.normal(rng_subkey[0], (dim, num))
  prob = jax.nn.sigmoid(-(1 / temp) * w.T.dot(x))
  y = jax.random.bernoulli(rng_subkey[1], (prob))
  y_pm = 2. * y - 1
  return x, y_pm


def logistic_dataset_gen_train_test(config, rng_key):
  """Creates the train and test sets of a logistic dataset.

  Args:
    config: Dictionary of parameters.
      config.dim: A float denoting input dimensionality.
      config.r: A float denoting L2 norm of the true parameters.
      config.num_train: An integer denoting the number of training data.
      config.num_test: An integer denoting the number of test data.
    rng_key: JAX random number generator key.

  Returns:
    train_data: The tuple (input, label) of training data.
    test_data: The tuple (input, label) of test data.
  """

  dim = config.dim
  temp = config.temperature
  rng_subkey = jax.random.split(rng_key, 3)
  param = logistic_dataset_init_param(dim, config.r, rng_subkey[0])
  train_data = logistic_dataset_gen_data(config.num_train, param, dim, temp,
                                         rng_subkey[1])
  test_data = logistic_dataset_gen_data(config.num_test, param, dim, temp,
                                        rng_subkey[2])
  return train_data, test_data


def get_train_test_generator(dataset):
  if dataset == 'logistic':
    return logistic_dataset_gen_train_test
  raise NotImplementedError('Dataset not found.')
