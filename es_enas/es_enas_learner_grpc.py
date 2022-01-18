# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""GRPC for proposing perturbations and topologies, and receiving objectives."""
import sys
import time
from absl import logging
import numpy as np
import pyglove as pg

from es_enas import util


def propose_queries_blackbox_optimizer(config, current_input,
                                       blackbox_optimizer, iteration):
  """Proposes perturbations and topology_str's."""

  start_time = time.time()

  core_hyperparameters = blackbox_optimizer.get_hyperparameters()
  proposed_perturbations = []
  proposed_dnas = []
  requests = []

  for i in range(config.total_num_perturbations):
    perturbation = np.random.normal(
        size=(len(current_input))) * config.es_precision_parameter
    proposed_perturbations.append(perturbation)

    dna = config.controller.propose_dna()
    topology_str = pg.to_json(dna)
    proposed_dnas.append(dna)

    tag = i + 1

    request = {
        'current_input': current_input,
        'hyperparameters': core_hyperparameters,
        'perturbation': perturbation,
        'tag': tag,
        'topology_str': topology_str
    }

    requests.append(request)

    if config.est_type == 'antithetic':
      antiperturbation = -perturbation
      proposed_perturbations.append(antiperturbation)

      dna = config.controller.propose_dna()
      topology_str = pg.to_json(dna)
      proposed_dnas.append(dna)

      request = {
          'current_input': current_input,
          'hyperparameters': core_hyperparameters,
          'perturbation': antiperturbation,
          'tag': -tag,
          'topology_str': topology_str
      }

      requests.append(request)
  for _ in range(config.num_exact_evals):
    null_perturbation = np.zeros_like(current_input)
    dna = config.controller.propose_dna()
    topology_str = pg.to_json(dna)
    proposed_dnas.append(dna)

    request = {
        'current_input': current_input,
        'hyperparameters': core_hyperparameters,
        'perturbation': null_perturbation,
        'tag': 0,
        'topology_str': topology_str
    }
    requests.append(request)

  end_time = time.time()
  logging.info('Iteration %d, requests proposed in %f seconds', iteration,
               end_time - start_time)

  return requests, proposed_perturbations, proposed_dnas


def run_step_blackbox_optimizer(config,
                                current_input,
                                blackbox_optimizer,
                                proposed_perturbations,
                                finished_dnas,
                                results,
                                logging_data=None):
  """Runs training step after collecting result protos."""
  core_hyperparameters = blackbox_optimizer.get_hyperparameters()
  function_values = [0.0] * len(proposed_perturbations)
  rewards_for_controller = []
  perturbations = proposed_perturbations
  evaluation_stats = []
  current_value_exact = 0.0
  current_value_exact_counter = 0

  for i in range(len(results)):
    rewards_for_controller.append(results[i]['function_value'])
    tag = results[i]['tag']
    index = 0
    if tag > 0:
      if config.est_type == 'antithetic':
        index = (tag - 1) * 2
        function_values[index] += results[i]['function_value']
      else:
        index = tag - 1
        function_values[index] += results[i]['function_value']
    if tag < 0:
      index = (-tag - 1) * 2 + 1
      function_values[index] += results[i]['function_value']
    if tag == 0:
      current_value_exact += results[i]['function_value']
      current_value_exact_counter += 1
  current_value_exact /= float(current_value_exact_counter)

  for result in results:
    evaluation_stat = list(result['evaluation_stat'])
    evaluation_stats.append(evaluation_stat)

  function_values_reshaped = np.array(function_values)
  perturbations_reshaped = np.array(perturbations)

  logging.info('LIST OF FUNCTION VALUES')
  logging.info(function_values_reshaped)

  logging.info('MAX VALUE SEEN CURRENTLY')
  logging.info(np.max(function_values_reshaped))

  logging.info('MEAN OF VALUES')
  logging.info(np.mean(function_values_reshaped))

  if logging_data is not None:
    iteration = logging_data['iteration']
    best_value = logging_data['best_value']
    iteration = logging_data['iteration']
    best_input = logging_data['best_input']
    best_core_hyperparameters = logging_data['best_core_hyperparameters']
    optimizer_state = blackbox_optimizer.get_state()

    if current_value_exact > best_value[0]:
      best_value[0] = current_value_exact
      best_input = current_input
      best_core_hyperparameters = core_hyperparameters

    # Writing logs.
    if iteration % config.log_frequency == 0:
      util.log_row(config.params_file, current_input)
      util.log_row(config.best_params_file, best_input)
      util.log_row(config.best_core_hyperparameters_file,
                   best_core_hyperparameters)
      util.log_row(config.best_value_file, best_value)
      util.log_row(config.optimizer_internal_state_file, optimizer_state)
      util.log_row(config.current_values_list_file, [current_value_exact])
      util.log_row(config.best_values_list_file, [best_value[0]])
      util.log_row(config.fvalues_file, function_values_reshaped)
      util.log_row(config.iteration_file, [iteration])

    print('Current exact value estimate:')
    print(current_value_exact)
    sys.stdout.flush()

  new_current_input = blackbox_optimizer.run_step(perturbations_reshaped,
                                                  function_values_reshaped,
                                                  current_input,
                                                  current_value_exact)
  config.controller.collect_rewards_and_train(rewards_for_controller,
                                              finished_dnas)

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
  """Handles the RPC communication in collecting results."""
  requests, proposed_perturbations, proposed_dnas = propose_queries_blackbox_optimizer(
      config, current_input, blackbox_optimizer, iteration)

  finished_dnas = []

  results = []
  futures = []
  num_worker_failures = 0
  for stub, request in zip(workers, requests):
    future = stub.EvaluateBlackboxInput.future(request)
    futures.append(future)
  start = time.time()
  for w, future in enumerate(futures):
    try:
      results.append(future.result())
      finished_dnas.append(proposed_dnas[w])
    except:  # pylint: disable=bare-except
      print('RPC error caught in collecting results !')
      num_worker_failures += 1
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

  return run_step_blackbox_optimizer(config, current_input, blackbox_optimizer,
                                     proposed_perturbations, finished_dnas,
                                     results, logging_data)


def run_optimization(config,
                     blackbox_optimizer,
                     init_current_input,
                     init_best_input,
                     init_best_core_hyperparameters,
                     init_best_value,
                     init_iteration,
                     workers,
                     log_bool=False):
  """Runs entire optimization procedure."""
  current_input = init_current_input
  best_input = init_best_input
  best_core_hyperparameters = init_best_core_hyperparameters
  best_value = [init_best_value]
  iteration = init_iteration

  while True:
    print(iteration)
    sys.stdout.flush()
    success, current_input = run_step_rpc_blackbox_optimizer(
        config, current_input, blackbox_optimizer, workers, iteration,
        best_input, best_core_hyperparameters, best_value, log_bool)
    if success:
      iteration += 1
    if iteration == config.nb_iterations:
      break
