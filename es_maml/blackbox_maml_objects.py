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

# pylint: disable=g-doc-return-or-yield,missing-docstring,g-doc-args,line-too-long,invalid-name,pointless-string-statement, super-init-not-called

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import math
import time

from absl import flags

import numpy as np

from es_maml.blackbox import blackbox_functions
from es_maml.blackbox import blackbox_objects

from es_maml.first_order import first_order_pb2
from es_maml.first_order import first_order_pb2_grpc

from es_maml.zero_order import zero_order_pb2
from es_maml.zero_order import zero_order_pb2_grpc

FLAGS = flags.FLAGS


class MAMLBlackboxObject(blackbox_objects.BlackboxObject):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def task_value(self, params, task, **kwargs):
    raise NotImplementedError("Abstract method")

  def use_adapter(self, adapter):
    self.adapter = adapter
    self.adapter_param_num = self.adapter.get_total_num_parameters()

  def get_initial(self):
    if self.config.algorithm == "first_order":
      return self.policy.get_initial()
    elif self.config.algorithm == "zero_order":
      return np.concatenate(
          ((self.policy).get_initial(), self.adapter.get_initial()))

  def adaptation_step(self, params, task, **kwargs):
    policy_params = params[:self.policy_param_num]
    adaptation_params = params[self.policy_param_num:]

    def task_value_fn(policy_params, **kwargs):
      return self.task_value(policy_params, task, **kwargs)

    new_policy_input = self.adapter.adaptation_step(policy_params,
                                                    adaptation_params,
                                                    task_value_fn, **kwargs)
    return new_policy_input

  def adaptation_task_value(self, params, task, **kwargs):
    new_params = self.adaptation_step(params, task, **kwargs)
    return self.task_value(new_params, task, **kwargs)

  def maml_value(self, params, task_list, **kwargs):
    return np.mean([
        self.adaptation_task_value(params, task, **kwargs) for task in task_list
    ])


class RLMAMLBlackboxObject(MAMLBlackboxObject):

  def __init__(self, config, **kwargs):

    self.config = config
    self.policy = config.rl_policy_fn()
    self.num_queries = config.num_queries
    self.alpha = config.alpha
    self.horizon = config.horizon
    self.num_rollouts_per_parameter = config.num_rollouts_per_parameter
    self.policy_param_num = self.policy.get_total_num_parameters()

    self.state_dimensionality = self.config.state_dimensionality

    self.nb_steps = 0
    self.sum_state_vector = [0.0] * self.state_dimensionality
    self.squares_state_vector = [0.0] * self.state_dimensionality
    self.mean_state_vector = [0.0] * self.state_dimensionality
    self.std_state_vector = [1.0] * self.state_dimensionality

    self.worker_id = 0
    self.hyperparameters = []

  def task_value(self, params, task, **kwargs):
    self.policy.update(params)

    evaluation_stats = []
    reward_list = []

    if self.config.algorithm == "first_order":
      hparams = [self.worker_id
                ] + self.mean_state_vector + self.std_state_vector
    elif self.config.algorithm == "zero_order":
      hparams = kwargs["hyperparameters"]

    for _ in range(self.num_rollouts_per_parameter):
      rewards, evaluation_stat = blackbox_functions.rl_extended_rollout(
          self.policy, hparams, task, kwargs.pop("horizon", self.horizon))
      evaluation_stats.append(evaluation_stat)
      reward_list.append(rewards)

    if self.config.algorithm == "first_order":  # local, per-worker state norm
      self.state_normalize(evaluation_stats)

    elif self.config.algorithm == "zero_order":  # global state norm params
      self.hyperparameters = [sum(x) for x in zip(*evaluation_stats)]

    aggregate_reward = np.mean(reward_list)
    return aggregate_reward

  def adaptation_step(self, params, task, **kwargs):
    policy_params = params[:self.policy_param_num]
    adaptation_params = params[self.policy_param_num:]

    def task_value_fn(policy_params, **kwargs):
      val = self.task_value(policy_params, task, **kwargs)
      return val

    new_policy_input = self.adapter.adaptation_step(policy_params,
                                                    adaptation_params,
                                                    task_value_fn, **kwargs)
    return new_policy_input

  def sample_train_batch(self, **kwargs):
    size = self.config.task_batch_size
    return np.random.choice(
        range(self.config.train_set_size), size=size, replace=False)

  def estimate_query_gradient(self, params, task, **kwargs):
    sigma = self.config.es_precision_parameter
    dim = params.shape[0]
    perturbations = np.random.normal(size=(self.num_queries, dim))

    p_grads = []
    fval_sum = 0
    es_hess = np.zeros((dim, dim))
    f_0 = self.task_value(params, task, **kwargs)
    for p in perturbations:
      f_p = self.task_value(params + sigma * p, task, **kwargs)
      if self.config.antithetic:
        f_anti_p = self.task_value(params - sigma * p, task, **kwargs)
        p_grads.append((f_p - f_anti_p) * p / (2.0 * sigma))
        fval_sum += (f_p + f_anti_p) / 2.0
        p = p.reshape((dim, 1))
        es_hess += (f_p + f_anti_p) * np.outer(p, p.T) / (2.0 * sigma * sigma)
      else:
        p_grads.append((f_p - f_0) * p / sigma)
        fval_sum += f_p
        p = p.reshape((dim, 1))
        es_hess += f_p * np.outer(p, p.T) / (sigma * sigma)
    avg_fval = fval_sum / len(perturbations)
    for p in perturbations:
      es_hess -= avg_fval * np.outer(p, p.T) / (sigma * sigma)
    es_hess /= len(perturbations)
    p_grads = np.array(p_grads)
    es_grad = np.mean(p_grads, axis=0)
    return es_grad, es_hess

  def task_gradient(self, params, task, **kwargs):
    self.policy.update(params)
    maml_adaptation_gradient, hess = self.estimate_query_gradient(
        params, task, **kwargs)
    task_maml_point = params + self.alpha * maml_adaptation_gradient
    maml_gradient, _ = self.estimate_query_gradient(task_maml_point, task,
                                                    **kwargs)
    if self.config.use_hess:
      maml_gradient = maml_gradient + self.alpha * np.matmul(
          hess, maml_gradient)
    return maml_gradient

  def execute(self, params, task_list, **kwargs):
    if self.config.algorithm == "zero_order":
      return_val = [
          self.maml_value(params, task_list, **kwargs), self.hyperparameters
      ]
    elif self.config.algorithm == "first_order" and kwargs["eval_order"] == 0:
      return_val = [[
          self.task_value(params, task, **kwargs) for task in task_list
      ], []]

    elif self.config.algorithm == "first_order" and kwargs["eval_order"] == 1:
      return_val = [[
          self.task_gradient(params, task, **kwargs) for task in task_list
      ], []]
    return return_val

  def state_normalize(self, evaluation_stats):
    evaluation_stats = [sum(x) for x in zip(*evaluation_stats)]
    self.nb_steps += evaluation_stats[0]
    evaluation_stats = evaluation_stats[1:]
    first_half = evaluation_stats[:self.state_dimensionality]
    second_half = evaluation_stats[self.state_dimensionality:]
    self.sum_state_vector = [
        sum(x) for x in zip(self.sum_state_vector, first_half)
    ]
    self.squares_state_vector = [
        sum(x) for x in zip(self.squares_state_vector, second_half)
    ]
    self.mean_state_vector = [
        x / float(self.nb_steps) for x in self.sum_state_vector
    ]
    mean_squares_state_vector = [
        x / float(self.nb_steps) for x in self.squares_state_vector
    ]

    self.std_state_vector = [
        math.sqrt(max(a - b * b, 0.0))
        for a, b in zip(mean_squares_state_vector, self.mean_state_vector)
    ]

  def get_metaparams(self):
    return [self.state_dimensionality]


class LossTensorMAMLBlackboxObject(MAMLBlackboxObject):

  def __init__(self, config, **kwargs):

    self.config = config
    self.policy = config.sl_policy_fn()
    self.num_queries = config.num_queries
    self.alpha = config.alpha
    self.policy_param_num = self.policy.get_total_num_parameters()

  def task_value(self, params, task, **kwargs):
    self.policy.update(params)
    xs, ys = task.generate_samples()
    return self.policy.sess.run(
        self.policy.obj_tensor,
        feed_dict={
            self.policy.input_ph: xs.reshape((-1, 1)),
            self.policy.output_ph: ys.reshape((-1, 1))
        })

  def execute(self, params, task_list, **kwargs):
    self.policy.update(params)
    return [self.maml_value(params, task_list, **kwargs), []]

  def get_metaparams(self):
    return None


class GeneralMAMLBlackboxWorker(zero_order_pb2_grpc.EvaluationServicer):

  def __init__(self, worker_id, blackbox_object, task_ids, task_batch_size,
               worker_mode):
    self.blackbox_object = blackbox_object
    self.worker_id = worker_id
    self.task_ids = task_ids
    self.task_batch_size = task_batch_size
    self.worker_mode = worker_mode

  def EvaluateBlackboxInput(self, request, context):
    np.random.seed(self.worker_id + int(time.time()))

    if self.worker_mode == "Train":
      return self.TrainEvaluate(request, context)
    elif self.worker_mode == "Test":
      return self.TestEvaluate(request, context)

  def TrainEvaluate(self, request, context):
    current_input = []
    core_hyperparameters = []
    for i in range(len(request.current_input.values)):
      current_input.append(request.current_input.values[i])
    for i in range(len(request.hyperparameters)):
      core_hyperparameters.append(request.hyperparameters[i])
    hyperparameters = [self.worker_id] + core_hyperparameters
    current_input_reshaped = np.array(current_input)
    tag = request.tag

    proposed_perturbations = []
    for j in range(len(request.perturbations)):
      proposed_perturbation = []
      for k in range(len(request.perturbations[j].values)):
        proposed_perturbation.append(request.perturbations[j].values[k])
      proposed_perturbations.append(proposed_perturbation)

    perturbations = []
    function_values = []
    evaluation_stats = []

    for i in range(len(proposed_perturbations)):
      perturbation = np.array(proposed_perturbations[i])
      perturbations.append(zero_order_pb2.Vector(values=perturbation.tolist()))
      task_id_list = np.random.choice(
          self.task_ids, size=self.task_batch_size, replace=False)

      task_list = [
          self.blackbox_object.config.make_task_fn(task_id=task_id)
          for task_id in task_id_list
      ]

      function_value, evaluation_stat = self.blackbox_object.execute(
          current_input_reshaped + perturbation,
          task_list,
          hyperparameters=hyperparameters)

      evaluation_stats.append(evaluation_stat)
      function_values.append(function_value)

    evaluation_stats_reduced = [sum(x) for x in zip(*evaluation_stats)]
    if not proposed_perturbations:
      results = zero_order_pb2.EvaluationResponse(
          perturbations=perturbations,
          function_values=function_values,
          evaluation_stats=evaluation_stats_reduced,
          tag=tag)
    else:
      results = zero_order_pb2.EvaluationResponse(
          perturbations=[],
          function_values=function_values,
          evaluation_stats=evaluation_stats_reduced,
          tag=tag)
    return results

  def TestEvaluate(self, request, context):
    current_input = []
    core_hyperparameters = []
    for i in range(len(request.current_input.values)):
      current_input.append(request.current_input.values[i])
    for i in range(len(request.hyperparameters)):
      core_hyperparameters.append(request.hyperparameters[i])
    hyperparameters = [self.worker_id] + core_hyperparameters
    current_input = np.array(current_input)
    iteration = request.tag

    if iteration % self.blackbox_object.config.test_frequency == 0:
      task = self.blackbox_object.config.make_task_fn(
          task_id=self.task_ids[self.worker_id])

      mamlpt_value = self.blackbox_object.task_value(
          params=current_input,
          task=task,
          hyperparameters=hyperparameters,
          test_mode=True,
          horizon=self.blackbox_object.config.horizon)

      mamlpt_value_list = [mamlpt_value]
      for _ in range(self.blackbox_object.config.test_parallel_evals - 1):
        temp_mamlpt_value = self.blackbox_object.task_value(
            params=current_input,
            task=task,
            hyperparameters=hyperparameters,
            test_mode=True,
            horizon=self.blackbox_object.config.horizon)
        mamlpt_value_list.append(temp_mamlpt_value)

      adaptation_param = self.blackbox_object.adaptation_step(
          params=current_input,
          task=task,
          hyperparameters=hyperparameters,
          test_mode=True,
          horizon=self.blackbox_object.config.horizon)

      adaptation_value = self.blackbox_object.task_value(
          params=adaptation_param,
          task=task,
          hyperparameters=hyperparameters,
          test_mode=True,
          horizon=self.blackbox_object.config.horizon)

      adaptation_value_list = [adaptation_value]
      for _ in range(self.blackbox_object.config.test_parallel_evals - 1):
        temp_adaptation_value = self.blackbox_object.task_value(
            params=adaptation_param,
            task=task,
            hyperparameters=hyperparameters,
            test_mode=True,
            horizon=self.blackbox_object.config.horizon)

        adaptation_value_list.append(temp_adaptation_value)

      test_vals = mamlpt_value_list + adaptation_value_list

      results = zero_order_pb2.EvaluationResponse(
          perturbations=[],
          function_values=test_vals,
          evaluation_stats=[],
          tag=0)
    else:
      results = zero_order_pb2.EvaluationResponse(
          perturbations=[], function_values=[], evaluation_stats=[], tag=0)

    return results


class GradientMAMLWorker(first_order_pb2_grpc.EvaluationServicer):
  """This worker can return the zero-order evaluation or the MAML gradient."""

  def __init__(self, worker_id, blackbox_object, tasks, **kwargs):
    self.blackbox_object = blackbox_object
    self.blackbox_object.worker_id = worker_id
    self.worker_id = worker_id
    self.tasks = tasks

  def EvaluateBlackboxInput(self, request, context):
    task_idx = request.request_task_idx
    current_input = np.array(request.current_input)
    task_list = [self.tasks[task_idx]]

    evaluation_result, _ = self.blackbox_object.execute(
        np.array(current_input), task_list, eval_order=request.eval_order)

    function_value = []
    task_maml_gradient = []
    if request.eval_order == 0:
      function_value = [evaluation_result[0]]

    elif request.eval_order == 1:
      task_maml_gradient = evaluation_result[0].tolist()

    results = first_order_pb2.TaskEvaluationResponse(
        respond_task_idx=task_idx,
        input_idx=request.input_idx,
        function_value=function_value,
        gradient=task_maml_gradient)
    return results
