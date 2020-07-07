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

"""Main routine of the RPC client coordinating First-Order ES-MAML."""
# pylint: disable=g-doc-return-or-yield,missing-docstring,g-doc-args,line-too-long,invalid-name,pointless-string-statement, logging-not-lazy
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

from absl import logging
import grpc
import numpy as np
from es_maml.first_order import first_order_pb2
from es_maml.util import log_util

_TIMEOUT = 3600

TASK_GRADIENT_EVAL_ORDER = 1
TASK_VALUE_EVAL_ORDER = 0


def propose_task_requests(current_input, task_idxs):
  """Wraps a list of requests for task evaluation in a EvaluationRequest PB."""
  requests = []
  for t in task_idxs:
    requests.append(
        first_order_pb2.TaskEvaluationRequest(
            request_task_idx=t,
            input_idx=0,
            eval_order=TASK_GRADIENT_EVAL_ORDER,
            current_input=current_input.tolist()))
  return requests


def generate_perturbations_map(config, dim, task_idxs, num_directions):
  """Generates a list of perturbations for each task.

  Args:
    task_idxs: A list of tasks.
    num_directions: Number of perturbations.
    **kwargs:

  Returns:
    A dictionary with keys from task_idxs, mapping to a list of perturbations.
  """
  perturbations_map = {}
  for t in task_idxs:
    perturbations = np.random.normal(0, 1, (num_directions, dim))
    perturbations *= config.precision_parameter
    if config.antithetic:
      perturbations = np.concatenate((perturbations, -perturbations), axis=0)
    else:
      perturbations = np.concatenate((np.zeros((1, dim)), perturbations),
                                     axis=0)
    perturbations_map[t] = perturbations
  return perturbations_map


def propose_perturbation_requests(current_input, task_idx, perturbations):
  """Wraps requests for perturbations of one task in a EvaluationRequest PB.

  Generates one request for each perturbation, given by adding the perturbation
  to current_input.

  Args:
    current_input: the current policy weights
    task_idx: The index of the task to evaluate.
    perturbations: A list of perturbations.

  Returns:
    A list of requests, one for each perturbation.
  """
  requests = []
  for p_idx, p in enumerate(perturbations):
    perturbed_input = current_input + p
    requests.append(
        first_order_pb2.TaskEvaluationRequest(
            request_task_idx=task_idx,
            input_idx=p_idx,
            eval_order=TASK_VALUE_EVAL_ORDER,
            current_input=perturbed_input.tolist()))
  return requests


def evaluate_requests(requests, workers):
  """Sends and receives RPC requests for evaluation.

  Should satisfy len(requests) <= len(workers)
  Args:
    requests: list of requests
    workers: list of worker stubs

  Returns:
    list of results
  """
  results = []
  futures = []
  print('Sending requests. Workers: ' + str(len(workers)) + ' requests: ' +
        str(len(requests)))
  for stub, request in zip(workers, requests):
    future = stub.EvaluateBlackboxInput.future(request, timeout=_TIMEOUT)
    futures.append(future)
  start = time.time()
  for future in futures:
    try:
      results.append(future.result())
    except grpc.RpcError as err:
      print('RPC:' + str(err))
  end = time.time()
  print('Responds received in time: [in sec].')
  print(end - start)
  sys.stdout.flush()
  return results


def process_gradient_results(results):
  sample_task_grads = []
  for i in range(len(results)):
    sample_task_grads.append(results[i].gradient)
  return np.array(sample_task_grads)


def process_function_value_results(config, results, is_test):
  """Group function value results per task from results.

  Args:
    results: list of results
    is_test: bool, since number of queries is different for test time **kwargs

  Returns:
    A dictionary with keys given by task indexes, mapping to a list of
    function values. The list is indexed by the input_idx, usually corresponding
    to perturbations.
  """
  function_values_map = {}
  if not is_test:
    num_perturbations = config.num_perturbations
  else:
    num_perturbations = config.num_queries
  if config.antithetic:
    num_perturbations *= 2
  else:
    num_perturbations += 1
  for i in range(len(results)):
    task_idx = results[i].respond_task_idx
    input_idx = results[i].input_idx
    function_value = results[i].function_value[0]
    if task_idx in function_values_map:
      function_values_map[task_idx][input_idx] = function_value
    else:
      function_values_map[task_idx] = np.zeros(num_perturbations)
      function_values_map[task_idx][input_idx] = function_value
  return function_values_map


def per_task_function_normalization(function_values_map):
  """Perform function normalization for each task separately."""
  for t in function_values_map:
    task_values = function_values_map[t]
    task_mean = np.mean(task_values)
    task_sd = np.std(task_values)
    if task_sd == 0.0:
      function_values_map[t] = np.zeros_like(task_values)
    else:
      function_values_map[t] = (task_values - task_mean) / task_sd


def compute_task_gradients(config, dim, function_values_map, perturbations_map):
  """Calculates the gradient for each task.

  Args:
    function_values_map: A map of (task_idx) -> (function_values)
    perturbations_map: A map of (task_idx) -> (perturbations)
    **kwargs:

  Returns:
    A map (task_idx) -> (gradient).
  """
  precision_parameter = config.precision_parameter
  task_gradient_map = dict.fromkeys(perturbations_map.keys(), np.zeros(dim))
  for t in perturbations_map.keys():
    task_perturbations = perturbations_map[t]
    function_values = function_values_map.get(t, None)
    task_gradient = np.zeros(dim)
    if function_values is not None:
      if config.antithetic:
        for i, fval in enumerate(function_values):
          p = task_perturbations[i] / precision_parameter
          task_gradient += fval * p / precision_parameter
        task_gradient /= len(function_values)
      else:
        for i, fval in enumerate(function_values):
          p = task_perturbations[i] / precision_parameter
          task_gradient += (fval - function_values[0]) * p / precision_parameter
        task_gradient /= (len(function_values) - 1)
      task_gradient_map[t] = task_gradient
  return task_gradient_map


def compute_task_hessians(config, dim, function_values_map, perturbations_map):
  precision_parameter = config.precision_parameter
  task_hessian_map = dict.fromkeys(perturbations_map.keys(), np.zeros(
      (dim, dim)))
  for t in perturbations_map.keys():
    task_perturbations = perturbations_map[t]
    function_values = function_values_map.get(t, None)
    task_hessian = np.zeros((dim, dim))
    if function_values is not None:
      avg_function_value = np.mean(function_values)
      for i, fval in enumerate(function_values):
        p = task_perturbations[i].reshape((dim, 1)) / precision_parameter
        task_hessian += (fval - avg_function_value) * np.outer(p, p.T)
      task_hessian /= len(function_values)
      task_hessian /= np.power(precision_parameter, 2)
      task_hessian_map[t] = task_hessian
  return task_hessian_map


def evaluate_adaptation_gradients(config, current_input, task_idxs, workers,
                                  is_test):
  """Evaluates adaptation gradients for a list of tasks.

  Args:
    current_input:
    task_idxs: list of tasks
    workers: stubs for workers
    is_test: bool **kwargs

  Returns:
    A tuple (task_gradients, task_hessians). Each is a map (task_idx) -> array
    sending the task_idx to the ES gradient of task at current_input.
  """
  dim = current_input.shape[0]

  if not is_test:
    num_perturbations = config.num_perturbations
  else:
    num_perturbations = config.num_queries
  task_perturbations_map = generate_perturbations_map(config, dim, task_idxs,
                                                      num_perturbations)
  requests = []
  for t in task_idxs:
    task_perturbations = task_perturbations_map[t]
    requests.extend(
        propose_perturbation_requests(current_input, t, task_perturbations))
  results = evaluate_requests(requests, workers)
  function_values_map = process_function_value_results(config, results, is_test)
  if config.fvalues_normalization:
    per_task_function_normalization(function_values_map)
  task_gradients = compute_task_gradients(config, dim, function_values_map,
                                          task_perturbations_map)
  task_hessians = None

  if config.use_hess:
    task_hessians = compute_task_hessians(config, dim, function_values_map,
                                          task_perturbations_map)
  return task_gradients, task_hessians


def run_step_rpc_blackbox_optimizer(config, current_input, workers,
                                    train_tasks):
  """Conducts a single step of the RPC-based blackbox optimization."""

  dim = current_input.shape[0]
  train_object = train_tasks['object']
  task_idxs = train_object.sample_train_batch()
  work_split = config.work_split
  if work_split == 'task_per_worker':
    requests = propose_task_requests(current_input, task_idxs)
    results = evaluate_requests(requests, workers)
    task_grads = process_gradient_results(results)
    mean_grad = np.mean(task_grads, axis=0)
    next_input = current_input + config.es_step_size * mean_grad
  elif work_split == 'perturbation_per_worker':
    _g, _h = evaluate_adaptation_gradients(config, current_input, task_idxs,
                                           workers, False)
    first_task_gradients = _g
    first_task_hessians = _h
    second_task_perturbations_map = generate_perturbations_map(
        config, dim, task_idxs, config.num_perturbations)
    second_requests = []
    for t in task_idxs:
      task_perturbations = second_task_perturbations_map[t]
      task_gradient = first_task_gradients[t]
      task_maml_point = current_input + config.alpha * task_gradient
      second_requests.extend(
          propose_perturbation_requests(task_maml_point, t, task_perturbations))
    second_results = evaluate_requests(second_requests, workers)
    second_function_values_map = process_function_value_results(
        config, second_results, False)
    if config.fvalues_normalization:
      per_task_function_normalization(second_function_values_map)
    second_task_gradients = compute_task_gradients(
        config, dim, second_function_values_map, second_task_perturbations_map)
    maml_gradient = np.zeros_like(current_input)
    for task_idx in second_task_gradients:
      task_gradient = second_task_gradients[task_idx]
      if config.use_hess:
        task_gradient += config.alpha * np.matmul(first_task_hessians[task_idx],
                                                  task_gradient)
      maml_gradient += task_gradient
    maml_gradient /= len(task_idxs)
    next_input = current_input + config.es_step_size * maml_gradient
  return next_input


def run_test(config, current_input, test_task_idxs, workers):
  task_gradients, _ = evaluate_adaptation_gradients(config, current_input,
                                                    test_task_idxs, workers,
                                                    True)
  requests = []
  for test_idx in test_task_idxs:
    maml_gradient = config.alpha * task_gradients[test_idx]
    requests.extend(
        propose_perturbation_requests(current_input, test_idx, [maml_gradient]))
  test_results = evaluate_requests(requests, workers)
  test_vals = []
  for result in test_results:
    test_vals.append(result.function_value[0])
  return test_vals


def run_blackbox(config, train_tasks, test_tasks, init_current_input, stubs):
  """Runs blackbox optimization coordinator."""
  current_input = np.array(init_current_input)
  iteration = 0
  test_values = [[
      'Iter', 'Test mean',
      'Test median (size ' + str(len(test_tasks['tasks'])) + ')', 'Test min'
  ]]
  best_params = None
  best_value = -np.Inf

  while True:
    print(iteration)
    if iteration % config.test_frequency == 0:
      if config.work_split == 'task_per_worker':
        test_object = test_tasks['object']
        test_vals = [
            test_object.adaptation_task_value(params=current_input, task=t)
            for t in test_tasks['tasks']
        ]
      elif config.work_split == 'perturbation_per_worker':
        test_vals = run_test(config, current_input, test_tasks['ids'], stubs)

      logging.info('Test values')
      logging.info(test_vals)

      test_mean = np.mean(test_vals)
      test_values.append(
          [iteration, test_mean,
           np.median(test_vals),
           np.min(test_vals)])
      if test_mean > best_value:
        best_value = test_mean
        best_params = current_input

      # Print log info to console
      logging.info('Iteration: ' + str(iteration) + ' Test mean: ' +
                   str(test_mean) + ' Norm of policy: ' +
                   str(np.linalg.norm(current_input)))

      log_util.log_row(config.params_file, current_input)
      log_util.log_row(config.best_params_file, best_params)
      log_util.log_row(config.test_values_file, test_values)

    sys.stdout.flush()
    # take a step of the optimizer
    current_input = run_step_rpc_blackbox_optimizer(config, current_input,
                                                    stubs, train_tasks)
    iteration += 1
    if iteration == config.nb_iterations:
      break
