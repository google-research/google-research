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

# python3
"""A family of tasks based on quadratic problems."""

import numpy as np

from task_set import registry
from task_set.tasks import quadratic_helper
from task_set.tasks import utils

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def _sample_distribution_over_matrix(rng):
  """Samples the config for a distribution over a matrix.

  Args:
    rng: np.random.RandomState

  Returns:
    A distribution over matrix config (containing a tuple with the name and
    extra args needed to create the distribution).
  """
  # At this point, the choices here are arbitrary. We know some things about
  # the spectrum of the hessian for various problems we care about but certainly
  # not all. For example see: https://arxiv.org/abs/1901.10159.
  # These selections define a distribution over spectrum. This distribution is
  # NOT like the ones found in prior work, but tries to be broader, and
  # (hopefully) will contain spectrum from real problems.
  choice = rng.choice(["normal", "linspace_eigen", "uniform", "logspace_eigen"])
  if choice == "normal":
    mean = rng.normal(0, 1)
    std = rng.uniform(0, 2)
    return choice, {"mean": mean, "std": std}
  elif choice == "uniform":
    min_v = rng.uniform(-3, 1)
    max_v = min_v + rng.uniform(0, 5)
    return choice, {"min": min_v, "max": max_v}
  elif choice == "linspace_eigen":
    min_v = rng.uniform(0.01, 50)
    max_v = min_v + rng.uniform(0, 100)
    return choice, {"min": min_v, "max": max_v}
  elif choice == "logspace_eigen":
    min_v = 1e-5
    max_v = rng.uniform(1, 1000)
    return choice, {"min": min_v, "max": max_v}


def _sample_matrix_noise_dist(rng):
  """Samples the config for a distribution over the matrix noise values.

  Args:
    rng: np.random.RandomState

  Returns:
    A distribution over matrix config (containing a tuple with the name and
    extra args needed to create the distribution).
  """
  # These noise choices at this point are arbitrary. They are scaled down from
  # the types of distributions sampled in `_sample_distribution_over_matrix`.
  choice = rng.choice(["normal", "linspace_eigen", "uniform", "logspace_eigen"])
  if choice == "normal":
    mean = rng.normal(0, 0.05)
    std = rng.uniform(0, 0.05)
    return choice, {"mean": mean, "std": std}
  elif choice == "uniform":
    min_v = rng.uniform(-0.15, 0.05)
    max_v = min_v + rng.uniform(0, 0.25)
    return choice, {"min": min_v, "max": max_v}
  elif choice == "linspace_eigen":
    min_v = rng.uniform(0.01, 2)
    max_v = min_v + rng.uniform(0, 10)
    return choice, {"min": min_v, "max": max_v}
  elif choice == "logspace_eigen":
    min_v = 1e-5
    max_v = rng.uniform(1, 10)
    return choice, {"min": min_v, "max": max_v}


def _get_distribution_over_matrix(dims, cfg):
  """Get the distribution over a matrix given a config.

  Args:
    dims: int dimensions of the matrix.
    cfg: Config generated from `get_distribution_over_matrix` or
      `sample_matrix_noise_dist`.

  Returns:
    The corresponding tfp.distribution.Distribution for the matrix.
  """
  name, params = cfg
  if name == "linspace_eigen":
    # Generates a random matrix whose spectrum is sampled uniformly from the
    # range (min, max).
    spectrum = np.linspace(params["min"], params["max"],
                           dims).astype(np.float32)
    return quadratic_helper.FixedEigenSpectrumMatrixDistribution(spectrum)
  elif name == "logspace_eigen":
    # Generates a random matrix whose spectrum is sampled logrithmically from
    # the range (min, max).
    spectrum = np.logspace(
        np.log10(params["min"]), np.log10(params["max"]),
        dims).astype(np.float32)
    return quadratic_helper.FixedEigenSpectrumMatrixDistribution(spectrum)
  elif name == "normal":
    # Generates a random matrix whose values are randomly sampled from a normal
    # with mean and std.
    return tfp.distributions.Normal(
        tf.ones([dims, dims]) * params["mean"],
        tf.ones([dims, dims]) * params["std"])
  elif name == "uniform":
    # Generates a random matrix whose values are randomly sampled from a uniform
    # with mean and std.
    return tfp.distributions.Uniform(
        tf.ones([dims, dims]) * params["min"],
        tf.ones([dims, dims]) * params["max"])
  else:
    raise ValueError("Name of matrix distribution [%s] not supported." % name)


def _sample_vector_dist(rng):
  """Sample a vector distribution config."""
  choice = rng.choice(["normal", "uniform"])
  if choice == "normal":
    mean = rng.normal(0, 1)
    std = rng.uniform(0, 2)
    return choice, {"mean": mean, "std": std}
  elif choice == "uniform":
    min_v = rng.uniform(-5, 2.5)
    max_v = min_v + rng.uniform(0, 5)
    return choice, {"min": min_v, "max": max_v}


def _sample_vector_noise_dist(rng):
  """Sample a vector distribution config for per step noise."""
  choice = rng.choice(["normal", "uniform"])
  if choice == "normal":
    mean = rng.normal(0, 0.1)
    std = rng.uniform(0, 0.2)
    return choice, {"mean": mean, "std": std}
  elif choice == "uniform":
    min_v = rng.uniform(-0.5, .25)
    max_v = min_v + rng.uniform(0, 0.5)
    return choice, {"min": min_v, "max": max_v}


def _get_vector_dist(dims, cfg):
  """Get a vector distribution given a config.

  Args:
    dims: int Length of vector distribution.
    cfg: Config generated from either `sample_vector_noise_dist` or
      `sample_vector_dist`.

  Returns:
    tf.distributions.Distribution
  """
  name, params = cfg
  if name == "normal":
    return tfp.distributions.Normal(
        tf.ones([dims]) * params["mean"],
        tf.ones([dims]) * params["std"])
  elif name == "uniform":
    return tfp.distributions.Uniform(
        tf.ones([dims]) * params["min"],
        tf.ones([dims]) * params["max"])
  else:
    raise ValueError("error")


def _get_output_fn(name, loss_scale):
  """Get the output transformation from the given name and loss_scale."""
  if name == "identity":
    return lambda x: x * loss_scale
  elif name == "log":
    return lambda x: tf.log(tf.maximum(x, 0.) + 1e-8) * loss_scale
  else:
    raise NotImplementedError("No implementation for [%s]" % name)


@registry.task_registry.register_sampler("quadratic_family")
def sample_quadratic_family_cfg(seed):
  """Sample a task config for a toy quadratic based problem.

  See QuadraticBasedTask for more information.
  These configs are nested python structures that provide enough information
  to create an instance of the problem.

  Args:
    seed: int Random seed to generate task from.

  Returns:
    A nested dictionary containing a configuration.
  """
  rng = np.random.RandomState(seed)
  cfg = {}
  cfg["A_dist"] = _sample_distribution_over_matrix(rng)

  cfg["initial_dist"] = _sample_vector_dist(rng)

  cfg["output_fn"] = rng.choice(["identity", "log"])

  cfg["dims"] = utils.sample_log_int(rng, 2, 3000)
  cfg["seed"] = rng.randint(0, 100000)

  cfg["loss_scale"] = utils.sample_log_float(rng, 1e-5, 1e3)

  if rng.choice([True, False]):
    cfg["noise"] = {}
    cfg["noise"]["A_noise"] = _sample_matrix_noise_dist(rng)
  else:
    cfg["noise"] = None
  return cfg


@registry.task_registry.register_getter("quadratic_family")
def get_quadratic_family(cfg, seed=None):
  """Get a task for the given cfg.

  Args:
    cfg: config specifying the model generated by `sample_quadratic_family_cfg`.
    seed: optional int Seed used to generate the instance of a given task. Note
      this is not the seed used to generate the cfg, but just an instance of the
      given cfg.

  Returns:
    base.BaseTask for the given config.
  """

  dims = cfg["dims"]
  # pylint: disable=invalid-name
  A_dist = _get_distribution_over_matrix(dims, cfg["A_dist"])

  initial_dist = _get_vector_dist(dims, cfg["initial_dist"])

  B_dist = quadratic_helper.ConstantDistribution(tf.zeros([dims]))
  C_dist = quadratic_helper.ConstantDistribution(0.)

  output_fn = _get_output_fn(cfg["output_fn"], cfg["loss_scale"])
  seed = cfg["seed"]

  if cfg["noise"]:
    A_noise_dist = _get_distribution_over_matrix(dims, cfg["noise"]["A_noise"])
  else:
    A_noise_dist = None

  return quadratic_helper.QuadraticBasedTask(
      dims=dims,
      initial_dist=initial_dist,
      A_dist=A_dist,
      A_noise_dist=A_noise_dist,
      B_dist=B_dist,
      C_dist=C_dist,
      seed=seed,
      output_fn=output_fn)
