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

"""Main routine of the RPC client coordinating RPC blackbox optimization."""
# pylint: disable=g-doc-return-or-yield,missing-docstring,g-doc-args,line-too-long,invalid-name,pointless-string-statement, super-init-not-called, unused-argument
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import time
from absl import logging
import grpc
import numpy as np

import tensorflow.compat.v1 as tf
from es_maml.util import log_util
from es_maml.zero_order import zero_order_pb2

_TIMEOUT = 3600


def propose_queries_blackbox_optimizer(config, current_input,
                                       blackbox_optimizer, iteration):
  core_hyperparameters = blackbox_optimizer.get_hyperparameters()
  current_input_vectorized = zero_order_pb2.Vector(
      values=current_input.tolist())

  proposed_perturbations = []
  requests = []
  for _ in range(config.test_workers):
    # only hyperparameters and current input matter here
    null_perturbation = [0.0] * len(current_input)
    request = zero_order_pb2.EvaluationRequest(
        current_input=current_input_vectorized,
        hyperparameters=core_hyperparameters,
        perturbations=[zero_order_pb2.Vector(values=null_perturbation)],
        tag=iteration)
    requests.append(request)

  for i in range(config.total_num_perturbations):
    perturbation = np.random.normal(
        size=(len(current_input))) * config.es_precision_parameter
    proposed_perturbations.append(perturbation)
    tag = i + 1
    for _ in range(config.num_repeats):
      request = zero_order_pb2.EvaluationRequest(
          current_input=current_input_vectorized,
          hyperparameters=core_hyperparameters,
          perturbations=[zero_order_pb2.Vector(values=perturbation.tolist())],
          tag=tag)
      requests.append(request)

    if config.es_est_type == 'antithetic':
      antiperturbation = -perturbation
      proposed_perturbations.append(antiperturbation)
      for _ in range(config.num_repeats):
        request = zero_order_pb2.EvaluationRequest(
            current_input=current_input_vectorized,
            hyperparameters=core_hyperparameters,
            perturbations=[
                zero_order_pb2.Vector(values=antiperturbation.tolist())
            ],
            tag=-tag)
        requests.append(request)
  for _ in range(config.num_repeats + config.num_exact_evals):
    null_perturbation = [0.0] * len(current_input)
    request = zero_order_pb2.EvaluationRequest(
        current_input=current_input_vectorized,
        hyperparameters=core_hyperparameters,
        perturbations=[zero_order_pb2.Vector(values=null_perturbation)],
        tag=0)
    requests.append(request)

  return requests, proposed_perturbations


def run_step_pythia_blackbox_optimizer(config,
                                       current_input,
                                       blackbox_optimizer,
                                       proposed_perturbations,
                                       results,
                                       logging_data=None):

  train_results = results[config.test_workers:]
  test_results = results[:config.test_workers]

  core_hyperparameters = blackbox_optimizer.get_hyperparameters()
  function_values = [0.0] * len(proposed_perturbations)
  perturbations = proposed_perturbations
  evaluation_stats = []
  current_value = 0.0
  current_value_counter = 0
  current_value_exact = 0.0

  for i in range(len(train_results)):
    tag = train_results[i].tag
    index = 0
    if tag > 0:
      if config.es_est_type == 'antithetic':
        index = (tag - 1) * 2
        function_values[index] += train_results[i].function_values[0]
      else:
        index = tag - 1
        function_values[index] += train_results[i].function_values[0]
    if tag < 0:
      index = (-tag - 1) * 2 + 1
      function_values[index] += train_results[i].function_values[0]
    if tag == 0:
      current_value_counter += 1
      if current_value_counter <= config.num_repeats:
        current_value += train_results[i].function_values[0]
      else:
        current_value_exact += train_results[i].function_values[0]
  for i in range(len(function_values)):
    function_values[i] /= float(config.num_repeats)
  current_value /= float(config.num_repeats)
  current_value_exact /= float(config.num_exact_evals)

  for i in range(len(train_results)):
    evaluation_stat = []
    for j in range(len(train_results[i].evaluation_stats)):
      evaluation_stat.append(train_results[i].evaluation_stats[j])
    evaluation_stats.append(evaluation_stat)

  function_values_reshaped = np.array(function_values)
  perturbations_reshaped = np.array(perturbations)

  logging.info('LIST OF FUNCTION VALUES')
  logging.info(function_values_reshaped)

  logging.info('MAX VALUE SEEN CURRENTLY')
  logging.info(np.max(function_values_reshaped))

  logging.info('MEAN OF VALUES')
  logging.info(np.mean(function_values_reshaped))

  iteration = logging_data['iteration']
  test_vals = []
  mamlpt_vals = []
  task_id_to_states = {}

  test_all_vals = []
  mamlpt_all_vals = []

  if iteration % config.test_frequency == 0:
    for i in range(len(test_results)):
      test_data = [x for x in test_results[i].function_values]
      if len(test_data) == 2 * config.test_parallel_evals:
        mamlpt_parallel_vals = test_data[:config.test_parallel_evals]
        mamlpt_vals.append(np.mean(mamlpt_parallel_vals))
        test_parallel_vals = test_data[config.test_parallel_evals:]
        test_vals.append(np.mean(test_parallel_vals))

        test_all_vals.append(test_parallel_vals)
        mamlpt_all_vals.append(mamlpt_parallel_vals)
      else:
        test_all_vals.append(config.test_parallel_evals * [0.0])
        mamlpt_all_vals.append(config.test_parallel_evals * [0.0])

    logging.info('TEST VALS')
    logging.info(test_vals)
    logging.info('TEST MEAN')
    logging.info(np.mean(test_vals))

    logging.info('MAMLPT VALS')
    logging.info(mamlpt_vals)
    logging.info('MAMLPT MEAN')
    logging.info(np.mean(mamlpt_vals))

  if logging_data is not None:
    best_value = logging_data['best_value']
    iteration = logging_data['iteration']
    best_input = logging_data['best_input']
    best_core_hyperparameters = logging_data['best_core_hyperparameters']
    optimizer_state = blackbox_optimizer.get_state()

    if current_value > best_value[0]:
      best_value[0] = current_value
      best_input = current_input
      best_core_hyperparameters = core_hyperparameters

    # Writing logs.
    if iteration % config.log_frequency == 0:
      log_util.log_row(config.params_file, current_input)
      log_util.log_row(config.best_params_file, best_input)
      log_util.log_row(config.best_core_hyperparameters_file,
                       best_core_hyperparameters)
      log_util.log_row(config.best_value_file, best_value)
      log_util.log_row(config.optimizer_internal_state_file, optimizer_state)
      log_util.log_row(config.current_values_list_file, [current_value])
      log_util.log_row(config.best_values_list_file, [best_value[0]])
      log_util.log_row(config.fvalues_file, function_values_reshaped)
      log_util.log_row(config.iteration_file, [iteration])
      log_util.log_row(config.test_values_file, test_vals)
      log_util.log_row(config.mamlpt_values_file, mamlpt_vals)

      if iteration % config.test_frequency == 0:
        test_mamlpt_parallel_evals_data_file = os.path.join(
            config.test_mamlpt_parallel_vals_folder,
            'test_mamlpt_parallel_vals_' + str(iteration) + '.npz')

        np.savez(
            tf.gfile.GFile(test_mamlpt_parallel_evals_data_file, 'w'),
            test_all_vals=np.array(test_all_vals),
            mamlpt_all_vals=np.array(mamlpt_all_vals))

      if config.log_states and iteration % config.log_states_iter == 0:
        states_file = os.path.join(config.states_folder,
                                   'states_' + str(iteration) + '.json')
        dumped = json.dumps(task_id_to_states, cls=log_util.NumpyEncoder)
        with tf.gfile.Open(states_file, 'w') as stf:
          json.dump(dumped, stf)

    print('Current value estimate:')
    print(current_value)
    print('More precise current value estimate:')
    print(current_value_exact)
    sys.stdout.flush()

  if np.std(function_values_reshaped) == 0:
    new_current_input = current_input
  else:
    new_current_input = blackbox_optimizer.run_step(perturbations_reshaped,
                                                    function_values_reshaped,
                                                    current_input,
                                                    current_value)

  evaluation_stats_reduced = [sum(x) for x in zip(*evaluation_stats)]
  blackbox_optimizer.update_state(evaluation_stats_reduced)

  return [True, new_current_input]


def run_step_rpc_blackbox_optimizer(config,
                                    current_input,
                                    blackbox_optimizer,
                                    workers,
                                    iteration,
                                    best_input,
                                    best_core_hyperparameters,
                                    best_value,
                                    log_bool=False):

  requests, proposed_perturbations = propose_queries_blackbox_optimizer(
      config, current_input, blackbox_optimizer, iteration)

  results = []
  futures = []
  num_worker_failures = 0
  for stub, request in zip(workers, requests):
    future = stub.EvaluateBlackboxInput.future(request, timeout=_TIMEOUT)
    futures.append(future)
  start = time.time()
  for w, future in enumerate(futures):
    try:
      results.append(future.result())
    except grpc.RpcError as err:
      print('RPC error caught in collecting results !')
      num_worker_failures += 1
      logging.info(err)
      logging.info('worker failed ID: ')
      logging.info(w)
  end = time.time()
  print('Responds received in time: [in sec].')
  print(end - start)
  sys.stdout.flush()
  if float(num_worker_failures) > config.critical * float(len(workers)):
    return [False, current_input]

  if log_bool:
    logging_data = {
        'best_value': best_value,
        'iteration': iteration,
        'best_input': best_input,
        'best_core_hyperparameters': best_core_hyperparameters
    }
  else:
    logging_data = None

  return run_step_pythia_blackbox_optimizer(config, current_input,
                                            blackbox_optimizer,
                                            proposed_perturbations, results,
                                            logging_data)


def run_blackbox(config,
                 blackbox_optimizer,
                 init_current_input,
                 init_best_input,
                 init_best_core_hyperparameters,
                 init_best_value,
                 init_iteration,
                 stubs,
                 log_bool=False):

  current_input = init_current_input
  best_input = init_best_input
  best_core_hyperparameters = init_best_core_hyperparameters
  best_value = [init_best_value]
  iteration = init_iteration

  while True:
    print(iteration)
    sys.stdout.flush()
    success, current_input = run_step_rpc_blackbox_optimizer(
        config, current_input, blackbox_optimizer, stubs, iteration, best_input,
        best_core_hyperparameters, best_value, log_bool)
    if success:
      iteration += 1
    if iteration == config.nb_iterations:
      break
