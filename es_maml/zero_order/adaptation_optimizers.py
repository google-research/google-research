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

"""Collection of adapters for Zero-Order ES-MAML."""
# pylint: disable=g-doc-return-or-yield,missing-docstring,g-doc-args,line-too-long,invalid-name,pointless-string-statement, super-init-not-called, unused-argument
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
from absl import logging
import numpy as np
from es_maml.util.dpp.dpp import DPP
from es_maml.util.log_util import AlgorithmState


class Adaptation:
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def adaptation_step(self, policy_params, adaptation_params, task, **kwargs):
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def get_total_num_parameters(self):
    raise NotImplementedError("Abstract method")


class BlackboxAdaptation(Adaptation):

  def __init__(self, num_queries, adaptation_blackbox_optimizer,
               adaptation_precision_parameter, **kwargs):
    self.num_queries = num_queries
    self.adaptation_blackbox_optimizer = adaptation_blackbox_optimizer
    self.adaptation_precision_parameter = adaptation_precision_parameter
    self.perturbation_type = kwargs["perturbation_type"]
    self.dpp_rho = kwargs["dpp_rho"]

  def adaptation_step(self, policy_params, adaptation_params, task_value_fn,
                      **kwargs):
    dim = policy_params.shape[0]

    if self.perturbation_type == "Gaussian":
      perturbations = np.random.normal(
          size=(self.num_queries, dim)) * self.adaptation_precision_parameter
    elif self.perturbation_type == "DPP":
      perturbations = np.random.normal(
          size=(self.dpp_rho * self.num_queries,
                dim)) * self.adaptation_precision_parameter
      dpp = DPP(perturbations)
      dpp.compute_kernel(kernel_type="rbf")
      idx = dpp.sample_k(self.num_queries)
      perturbations = perturbations[idx]

    es_hess = np.zeros((dim, dim))

    pivot = task_value_fn(policy_params, **kwargs)

    function_values = []
    all_values = [pivot]
    for p in perturbations:
      temp_task_value = task_value_fn(policy_params + p, **kwargs)
      function_values.append(temp_task_value)
      all_values.append(temp_task_value)
      es_hess += temp_task_value * np.outer(p, p.T) / np.square(
          self.adaptation_precision_parameter)
    function_values = np.array(function_values)

    avg_fval = np.mean(function_values)

    for p in perturbations:
      es_hess -= avg_fval * np.outer(p, p.T) / np.square(
          self.adaptation_precision_parameter)
    es_hess /= len(perturbations)

    if np.std(all_values) == 0:  # in case function vals all equal
      return policy_params
    else:
      out = self.adaptation_blackbox_optimizer.run_step(
          perturbations=perturbations,
          function_values=function_values,
          current_input=policy_params,
          current_value=pivot)
      return out

  def get_total_num_parameters(self):
    return 0

  def get_initial(self):
    return np.array([])


class HillClimbAdaptation(Adaptation):

  def __init__(self, num_queries, adaptation_precision_parameter, **kwargs):
    self.num_queries = num_queries
    self.adaptation_precision_parameter = adaptation_precision_parameter
    self.parallel_evaluations = kwargs.get("parallel_evaluations", 1)
    self.parallel_alg = kwargs.get("parallel_alg", "batch")

  def adaptation_step(self,
                      policy_params,
                      adaptation_params,
                      task_value_fn,
                      loader=False,
                      **kwargs):

    parallel_alg = kwargs.get("parallel_alg", self.parallel_alg)
    parallel_evaluations = kwargs.get("parallel_evaluations",
                                      self.parallel_evaluations)
    dim = policy_params.shape[0]

    state = kwargs.pop("algorithm_state", AlgorithmState())

    if not state.meta_eval_passed:
      while len(state.single_values) < parallel_evaluations:
        state.single_values.append(task_value_fn(policy_params, **kwargs))
        state.best_params_so_far = policy_params
      state.pivot = np.average(state.single_values)
      state.meta_eval_passed = True

      if loader:
        logging.info("Average Objective of HillClimbing Iteration %d is: %f", 0,
                     np.average(state.single_values))

      state.query_index += 1
      state.single_values = []

    while state.query_index <= self.num_queries:
      if parallel_alg == "average":
        state.p = np.random.normal(
            size=(dim)) * self.adaptation_precision_parameter

        while len(state.single_values) < parallel_evaluations:
          single_value = task_value_fn(state.best_params_so_far + state.p,
                                       **kwargs)
          state.single_values.append(single_value)
        temp_task_value = np.average(state.single_values)
        state.single_values = []

        if temp_task_value > state.pivot:
          state.pivot = temp_task_value
          state.best_params_so_far = state.best_params_so_far + state.p

      elif parallel_alg == "batch":
        while len(state.single_values) < parallel_evaluations:
          p = np.random.normal(size=(dim)) * self.adaptation_precision_parameter
          state.temp_perturbations.append(p)
          single_value = task_value_fn(state.best_params_so_far + p, **kwargs)
          state.single_values.append(single_value)

        best_index = np.argmax(state.single_values)
        if state.single_values[best_index] > state.pivot:
          state.pivot = state.single_values[best_index]
          state.best_params_so_far = state.best_params_so_far + state.temp_perturbations[
              best_index]
        state.temp_perturbations = []

      if loader and len(state.single_values) == parallel_evaluations:
        logging.info("Average Objective of HillClimbing Iteration %d is: %f",
                     state.query_index, np.average(state.single_values))

      state.query_index += 1
      state.single_values = []

    return state.best_params_so_far

  def get_total_num_parameters(self):
    return 0

  def get_initial(self):
    return np.array([])
