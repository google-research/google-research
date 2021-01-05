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
# pylint: disable=line-too-long
"""Tasks built from "Learned Optimizers that Scale and Generalize"(losg) code.

paper: https://arxiv.org/pdf/1703.04813
code:
https://github.com/tensorflow/models/tree/master/research/learned_optimizer/problems

These tasks are similar to those used by the losg paper.
The main difference is that these contain different settings than the
original set.
"""
# pylint: enable=line-too-long

import collections
from typing import Any, Dict, Callable, Optional, Text, Tuple, Union, List

import numpy as np
import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
from task_set.tasks import utils
from task_set.tasks.losg_problems import datasets as losg_datasets
from task_set.tasks.losg_problems import problem_generator as pg
from task_set.tasks.losg_problems import problem_spec
import tensorflow.compat.v1 as tf

# (Problem spec from LOSG, Optional dataset object from LOSG task, batch size)
ProblemDefinition = Tuple[problem_spec.Spec, Optional[losg_datasets.Dataset],
                          Optional[int]]


class LOSGProblemTask(base.BaseTask):
  """Task built from a task in "Learned Optimizers that Scale and Generalize".

  Reference: https://arxiv.org/pdf/1703.04813
  """

  def __init__(self,
               problem_definition_fn,
               name = "LOSGProblemTask",
               seed = None,
               **kwargs):
    """Creates a Task from a ProblemDefinition.

    The problem definition consists of a tuple containing:
      A problem spec from LOSG
      An optional dataset object from LOSG task
      The batch size

    Args:
      problem_definition_fn: function that returns a problem definition.
      name: name of underlying sonnet module.
      seed: random seed used.
      **kwargs: args passed to BaseTask.
    """
    super(LOSGProblemTask, self).__init__(name=name, **kwargs)
    self._seed = seed

    with self._enter_variable_scope():
      spec, dataset, batch_size = problem_definition_fn()
      self._problem = spec.build()
      if dataset:
        examples = dataset.data.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.repeat().shuffle(examples)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        self._iterator = dataset.make_one_shot_iterator()
      else:
        self._iterator = None

      self._variables = self._problem.init_variables(seed=self._seed)

  def call_split(
      self,
      params,
      split,
      batch = None,
      with_metrics = False
  ):
    """Perform a forward pass of the task.

    Note: This changes the numpy global seed!

    Args:
      params: params to use for forward pass.
      split: split of data to use.
      batch: optional batch of data to compute over.
      with_metrics: flag to turn off and off extra metrics.

    Returns:
      Scalar loss computed over a batch of data.
    """
    if batch:
      data, label = batch
    else:
      if self._iterator:
        data, label = self._iterator.get_next()
      else:
        data, label = None, None

    # force random seed here before calling into the problem.
    np.random.seed(self._problem.random_seed)

    @tf.custom_gradient
    def fake_gradient(*params_values):
      loss = self._problem.objective(params_values, data, label)

      def grad(dy):
        grads = self._problem.gradients(loss, params_values)
        return [g * dy for g in grads]

      return loss, grad

    loss = fake_gradient(*list(params.values()))

    if with_metrics:
      return loss, {}
    else:
      return loss

  @snt.reuse_variables
  def current_params(self):
    """tf.Variables for the current parameters."""
    array = [(v.op.name, v) for v in self._variables]
    return collections.OrderedDict(array)

  @snt.reuse_variables
  def initial_params(self):
    """Initial values of parameters."""
    tensors = self._problem.init_tensors(seed=self._seed)
    array = [(v.op.name, t) for v, t in zip(self._variables, tensors)]
    return collections.OrderedDict(array)

  def get_batch(self, split):
    """Get a batch of data.

    Note the split is not used for lol problems.

    Args:
      split: split to take data from. This is not used by this function.

    Returns:
      A batch of data.
    """
    if self._iterator:
      return self._iterator.get_next()
    else:
      return None

  def get_variables(self):
    return self._variables


QuadraticConfig = Dict[Text, float]


def _sample_quadratic_problem(rng):
  """Sample a quadratic problem."""
  is_noise = utils.sample_bool(rng, 0.5)
  return {
      "dim":
          utils.sample_log_int(rng, 10, 1000),
      "noise_stdev":
          utils.sample_log_float(rng, 0.01, 10.0) if is_noise else 0.0,
  }


def _get_quadratic_problem(cfg):
  """Get a quadratic problem from the given config."""
  return problem_spec.Spec(pg.Quadratic, (cfg["dim"],),
                           {"noise_stdev": cfg["noise_stdev"]}), None, None


BowlConfig = Dict[Text, float]


def _sample_bowl_problems(rng):
  """Sample a bowl problem."""
  is_noise = utils.sample_bool(rng, 0.5)
  return {
      "cond":
          utils.sample_log_float(rng, 0.01, 100),
      "angle":
          rng.choice([0, 0, np.pi / 4., np.pi / 3]),
      "noise_stdev":
          utils.sample_log_float(rng, 0.01, 10.0) if is_noise else 0.0,
  }


def _get_bowl_problem(cfg):
  """Get a bowl problem from the given config."""
  return problem_spec.Spec(pg.Bowl, (cfg["cond"],), {
      "noise_stdev": cfg["noise_stdev"],
      "angle": cfg["angle"]
  }), None, None


SparseSoftmaxConfig = Dict[Text, Any]


def _sample_sparse_softmax_regression(
    rng):
  """Sample a sparse softmax regression problem."""
  is_noise = utils.sample_bool(rng, 0.5)
  return {
      "n_features":
          utils.sample_log_int(rng, 2, 100),
      "n_classes":
          2,
      "noise_stdev":
          utils.sample_log_float(rng, 0.01, 10.0) if is_noise else 0.0,
      "bs":
          utils.sample_log_int(rng, 1, 50),
      "n_samples":
          utils.sample_log_int(rng, 1, 30),
  }


def _get_sparse_softmax_regression(
    cfg):
  """Get a sparse softmax regression problem."""
  return (problem_spec.Spec(pg.SparseSoftmaxRegression,
                            (cfg["n_features"], cfg["n_classes"]),
                            {"noise_stdev": cfg["noise_stdev"]}),
          losg_datasets.noisy_parity_class(
              cfg["n_samples"], n_classes=cfg["n_classes"]), cfg["bs"])


_opt_test_problems = {
    "ackley": pg.Ackley,
    "beale": pg.Beale,
    "branin": pg.Branin,
    "logsumexp": pg.LogSumExp,
    "matyas": pg.Matyas,
    "michalewicz": pg.Michalewicz,
    "rosenbrock": pg.Rosenbrock,
    "StyblinskiTang": pg.StyblinskiTang,
}

OptimizationTestConfig = Dict[Text, Any]


def _sample_optimization_test_problems(
    rng):
  """Sample an optimization test function problem."""
  is_noise = utils.sample_bool(rng, 0.5)
  return {
      "problem":
          rng.choice(sorted(_opt_test_problems.keys())),
      "noise_stdev":
          utils.sample_log_float(rng, 0.01, 10.0) if is_noise else 0.0,
  }


def _get_optimization_test_problems(
    cfg):
  """Get an optimization test function problem form the given config."""
  return problem_spec.Spec(_opt_test_problems[cfg["problem"]], (),
                           {"noise_stdev": cfg["noise_stdev"]}), None, None


FullyConnectedConfig = Dict[Text, Any]


def _sample_fully_connected(rng):
  """Sample a fully connected problem."""
  n_layer = rng.choice([2, 3, 4, 5])
  fixed = utils.sample_bool(rng, 0.5)
  cfg = {
      "n_features": utils.sample_log_int(rng, 1, 16),
      "n_classes": 2,
      "activation": utils.sample_activation(rng),
      "bs": utils.sample_log_int(rng, 1, 200),
      "n_samples": utils.sample_log_int(rng, 1, 30),
  }
  if fixed:
    cfg["hidden_sizes"] = [utils.sample_log_int(rng, 4, 32)] * n_layer
  else:
    cfg["hidden_sizes"] = [
        utils.sample_log_int(rng, 4, 32) for _ in range(n_layer)
    ]

  return cfg


def _get_fully_connected(cfg):
  """Get a fully connected problem from the given config."""
  return (problem_spec.Spec(
      pg.FullyConnected, (cfg["n_features"], cfg["n_classes"]), {
          "hidden_sizes": tuple(cfg["hidden_sizes"]),
          "activation": utils.get_activation(cfg["activation"]),
      }), losg_datasets.random_mlp(cfg["n_features"],
                                   cfg["n_samples"]), cfg["bs"])


NormConfig = Dict[Text, Any]


def _sample_norm(rng):
  """Sample a norm problem."""
  return {
      "dim": utils.sample_log_int(rng, 3, 1000),
      "norm_power": rng.uniform(0.1, 5.0),
  }


def _get_norm(cfg):
  """Get a norm problem from the given config."""
  return (problem_spec.Spec(pg.Norm, (cfg["dim"],),
                            {"norm_power": cfg["norm_power"]}), None, None)


DependencyChainConfig = Dict[Text, Any]


def _sample_dependency_chain(
    rng):
  """Sample a dependency chain problem."""
  return {
      "dim": utils.sample_log_int(rng, 3, 100),
      "bs": utils.sample_log_int(rng, 1, 200),
      "n_samples": utils.sample_log_int(rng, 100, 20000),
  }


def _get_dependency_chain(cfg):
  """Get a dependency chain problem from the given config."""
  return (problem_spec.Spec(pg.DependencyChain, (cfg["dim"],), {}),
          losg_datasets.random_mlp(cfg["dim"], cfg["n_samples"]), cfg["bs"])


OutwardSnakeConfig = Dict[Text, Any]


def _sample_outward_snake(rng):
  """Sample an outward snake problem."""
  return {
      "dim": utils.sample_log_int(rng, 3, 100),
      "bs": utils.sample_log_int(rng, 1, 200),
      "n_samples": utils.sample_log_int(rng, 100, 20000),
  }


def _get_outward_snake(cfg):
  """Get an outward snake problem from the given config."""
  return (problem_spec.Spec(pg.OutwardSnake, (cfg["dim"],), {}),
          losg_datasets.random_mlp(cfg["dim"], cfg["n_samples"]), cfg["bs"])


MinMaxWellConfig = Dict[Text, Any]


def _sample_min_max_well(rng):
  """Sample a min max well problem."""
  is_noise = utils.sample_bool(rng, 0.5)
  return {
      "dim":
          utils.sample_log_int(rng, 10, 1000),
      "noise_stdev":
          utils.sample_log_float(rng, 0.01, 10.0) if is_noise else 0.0,
  }


def _get_min_max_well(cfg):
  """Get a min max well problem from the given config."""
  return problem_spec.Spec(pg.MinMaxWell, (cfg["dim"],),
                           {"noise_stdev": cfg["noise_stdev"]}), None, None


SumOfQuadraticsConfig = Dict[Text, Any]


def _sample_sum_of_quadratics(
    rng):
  """Sample a sum of quadratics problem."""
  return {
      "dim": utils.sample_log_int(rng, 3, 100),
      "bs": utils.sample_log_int(rng, 1, 200),
      "n_samples": utils.sample_log_int(rng, 100, 20000),
  }


def _get_sum_of_quadratics(cfg):
  """Get a sum of quadratics problem from the given config."""
  return (
      problem_spec.Spec(pg.SumOfQuadratics, (cfg["dim"],), {}),
      # dataset size must be divisible by 2.
      losg_datasets.random_symmetric(cfg["dim"],
                                     int(cfg["n_samples"] // 2) * 2),
      cfg["bs"])


ProjectedQuadraticConfig = Dict[Text, Any]


def _sample_projection_quadratic(
    rng):
  """Sample a projection quadratic problem."""
  return {
      "dim": utils.sample_log_int(rng, 3, 100),
      "bs": utils.sample_log_int(rng, 1, 200),
      "n_samples": utils.sample_log_int(rng, 100, 20000),
  }


def _get_projection_quadratic(
    cfg):
  """Get a projection quadratic problem from the given config."""
  return (problem_spec.Spec(pg.ProjectionQuadratic, (cfg["dim"],), {}),
          losg_datasets.random_symmetric(cfg["dim"],
                                         int(cfg["n_samples"] // 2) * 2),
          cfg["bs"])


_to_modify = [
    "quadratic", "bowl", "optimization_test_problems", "fully_connected",
    "norm", "dependency_chain", "outward_snake", "min_max_well",
    "sum_of_quadratics", "projection_quadratic"
]

SparseConfig = Dict[Text, Any]


def _sample_sparse_problem(rng):
  """Sample a sparse problem.

  This problem modifies a sampled base problem by setting some gradients to

  zero.

  Args:
    rng: Random state

  Returns:
    The sampled config.
  """
  is_noise = utils.sample_bool(rng, 0.5)
  base_config = rng.choice(_to_modify)
  return {
      "base": (base_config, _problem_sample_get[base_config][0](rng)),
      "zero_probability":
          rng.uniform(0.9, 0.99),
      "noise_stdev":
          utils.sample_log_float(rng, 0.01, 10.0) if is_noise else 0.0,
  }


def _get_sparse_problem(cfg):
  """Get a sparse problem from the given config."""
  name, cc = cfg["base"]
  base_spec, dataset, bs = _problem_sample_get[name][1](cc)
  return (problem_spec.Spec(
      pg.SparseProblem, [base_spec], {
          "zero_probability": cfg["zero_probability"],
          "noise_stdev": cfg["noise_stdev"]
      }), dataset, bs)


RescaleConfig = Dict[Text, Any]


def _sample_rescale_problem(rng):
  """Sample a rescale problem.

  This problem modifies a sampled base problem by rescaling the parameters.

  Args:
    rng: Random state

  Returns:
    The sampled config.
  """
  base_config = rng.choice(_to_modify)
  return {
      "base": (base_config, _problem_sample_get[base_config][0](rng)),
      "scale": utils.sample_log_float(rng, 0.001, 1000.0),
  }


def _get_rescale_problem(cfg):
  """Get a rescale problem from the given config."""
  name, cc = cfg["base"]
  base_spec, dataset, bs = _problem_sample_get[name][1](cc)
  return (problem_spec.Spec(pg.Rescale, [base_spec],
                            {"scale": cfg["scale"]}), dataset, bs)


LogObjectiveConfig = Dict[Text, Any]


def _sample_log_objective(rng):
  """Sample a log objective problem.

  This problem modifies a sampled base problem by taking the log of the loss.

  Args:
    rng: Random state.

  Returns:
    Config representing a losg task.
  """
  base_config = rng.choice(_to_modify)
  return {
      "base": (base_config, _problem_sample_get[base_config][0](rng)),
  }


def _get_log_objective(cfg):
  """Get a log objective problem fromt he given config."""
  name, cc = cfg["base"]
  base_spec, dataset, bs = _problem_sample_get[name][1](cc)
  return (problem_spec.Spec(pg.LogObjective, [base_spec], {}), dataset, bs)


_problem_sample_get = {
    "quadratic": (_sample_quadratic_problem, _get_quadratic_problem),
    "bowl": (_sample_bowl_problems, _get_bowl_problem),
    "sparse_softmax_regression":
        (_sample_sparse_softmax_regression, _get_sparse_softmax_regression),
    "optimization_test_problems":
        (_sample_optimization_test_problems, _get_optimization_test_problems),
    "fully_connected": (_sample_fully_connected, _get_fully_connected),
    "norm": (_sample_norm, _get_norm),
    "dependency_chain": (_sample_dependency_chain, _get_dependency_chain),
    "outward_snake": (_sample_outward_snake, _get_outward_snake),
    "min_max_well": (_sample_min_max_well, _get_min_max_well),
    "sum_of_quadratics": (_sample_sum_of_quadratics, _get_sum_of_quadratics),
    "projection_quadratic":
        (_sample_projection_quadratic, _get_projection_quadratic),
    "sparse_problems": (_sample_sparse_problem, _get_sparse_problem),
    "rescale_problems": (_sample_rescale_problem, _get_rescale_problem),
    "log_objective": (_sample_log_objective, _get_log_objective),
}

SampleProblemConfig = Tuple[Text, Any, int]


@registry.task_registry.register_sampler("losg_tasks_family")
def sample_losg_tasks_family_cfg(seed):
  """Samples a tasks based "Learned Optimizers that Scale and Generalize"(losg).

  These tasks are all build from components from losg but do not match exactly
  with that used to train this paper. The task suite here provides more
  variation across tasks.

  Args:
    seed: Random seed.

  Returns:
    Config representing a losg task.
  """
  rng = np.random.RandomState(seed)
  key = rng.choice(sorted(_problem_sample_get))
  # Add random seed at the end for consistent inits for better scaling.
  return (key, _problem_sample_get[key][0](rng), int(rng.uniform(0, 100000)))


@registry.task_registry.register_getter("losg_tasks_family")
def get_losg_tasks_family(cfg, seed=None):
  """Gets the task described by the given config.

  Args:
    cfg: Config of the task.
    seed: Random seed used for task creation.

  Returns:
    The task corresponding to the given config.
  """

  def get_problem_definition():
    orig_spec = _problem_sample_get[cfg[0]][1](cfg[1])
    if cfg[0] in ["rescale_problems", "log_objective"]:
      orig_spec[0].args[0].kwargs["random_seed"] = cfg[2]
    else:
      orig_spec[0].kwargs["random_seed"] = cfg[2]
    return orig_spec

  return LOSGProblemTask(get_problem_definition, seed=seed)
