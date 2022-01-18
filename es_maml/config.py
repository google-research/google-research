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

# pylint: disable=g-doc-return-or-yield,missing-docstring,unused-import,line-too-long,invalid-name,pointless-string-statement
"""Global config used for synchronization in ES-MAML.

A class whose attributes contain hyperparameter values and functions based on
those hyperparameters.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import datetime
import os
import numpy as np

from es_maml import blackbox_maml_objects
from es_maml import policies
from es_maml import task
from es_maml.blackbox import blackbox_optimization_algorithms
from es_maml.blackbox import regression_optimizers
from es_maml.zero_order import adaptation_optimizers


class Config(object):

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


def get_config(**kwargs):
  config = Config(**kwargs)

  config.folder_name = "./es_maml_logs/"
  config.run_locally = False
  config.num_queries = 20
  config.task_batch_size = 1
  config.train_set_size = 50
  config.test_set_size = 50
  config.num_rollouts_per_parameter = 1
  config.es_step_size = 0.01
  config.alpha = 0.05
  config.hidden_layers = []
  config.nb_iterations = 10000

  config.fvalues_normalization = True

  config.algorithm = "zero_order"
  # "zeroth_order"
  # "first_order"

  if config.algorithm == "zero_order":
    config.total_num_perturbations = 150
    config.num_exact_evals = 100
    config.num_repeats = 1
    config.nb_perturbations_per_worker = 1
    config.es_precision_parameter = 0.1
    config.adaptation_precision_parameter = 0.1
    config.hyperparameters_update_method = "state_normalization"

    config.es_est_type = "antithetic"
    config.adaptation_est_type = "antithetic"
    if config.es_est_type == "forward_fd":
      config.train_workers = config.total_num_perturbations * config.num_repeats + config.num_repeats + config.num_exact_evals
    elif config.es_est_type == "antithetic":
      config.train_workers = 2 * config.total_num_perturbations * config.num_repeats + config.num_repeats + config.num_exact_evals

    config.test_workers = 50
    assert config.test_set_size == config.test_workers
    config.test_parallel_evals = 1
    config.num_servers = config.train_workers + config.test_workers
    config.adaptation_string = "MC"
    # "SKLRegression"
    # "MC"
    # "GeneralRegression"
    config.regression_optimizer_string = "l1_jacobian_decoder"
    config.regularizer = 0.0
    config.perturbation_type = "Gaussian"
    # "Gaussian"
    # "DPP"
    config.dpp_rho = 5
    config.adapter_fn_string = "BlackboxAdaptation"
    # "BlackboxAdaptation"
    # "HillClimbAdaptation"

    config.hillclimb_parallel_alg = "average"
    # "batch"
    # "average"

    config.hillclimb_parallel_evaluations = 1

  elif config.algorithm == "first_order":
    config.use_hess = False
    config.work_split = "perturbation_per_worker"
    config.num_perturbations = 300  # for per-perturbation work_split
    config.num_servers = config.task_batch_size * (config.num_perturbations + 1)
    config.rollout_repeats = 1
    config.antithetic = False
    config.precision_parameter = 0.1
    config.hyperparameters_update_method = "None"
    config.test_frequency = 5
    config.adapter_fn_string = "None"

  config.critical = 0.4

  config.task_name = "NavigationTask2d"

  config.combo_task_num_subset_goals = 1
  config.combo_task_num_goals = 4

  config.test_frequency = 10
  config.horizon = 200

  if config.task_name in ["NavigationTask2d", "NavigationTask4corner"]:
    config.horizon = 100
  elif config.task_name in ["NavigationTaskCombo"]:
    config.horizon = 100

  return config


def generate_config(config, **kwargs):
  current_time_string = kwargs.get("current_time_string", "NA")
  config.json_hparams = copy.deepcopy(config.__dict__)

  def make_task_fn(task_id):
    return getattr(task, config.task_name)(
        task_id,
        num_subset_goals=config.combo_task_num_subset_goals,
        num_goals=config.combo_task_num_goals)

  config.make_task_fn = make_task_fn

  if config.algorithm == "zero_order":

    def es_blackbox_optimizer_fn(metaparams):
      return blackbox_optimization_algorithms.MCBlackboxOptimizer(
          config.es_precision_parameter,
          config.es_est_type,
          config.fvalues_normalization,
          config.hyperparameters_update_method,
          metaparams,
          config.es_step_size,
          num_top_directions=0)

    config.es_blackbox_optimizer_fn = es_blackbox_optimizer_fn

    def adaptation_blackbox_optimizer_fn(metaparams):
      if config.adaptation_string == "MC":
        return blackbox_optimization_algorithms.MCBlackboxOptimizer(
            config.adaptation_precision_parameter,
            config.adaptation_est_type,
            config.fvalues_normalization,
            config.hyperparameters_update_method,
            metaparams,
            config.alpha,
            num_top_directions=0)
      elif config.adaptation_string == "SKLRegression":
        return blackbox_optimization_algorithms.SklearnRegressionBlackboxOptimizer(
            "lasso", config.regularizer, config.est_type,
            config.fvalues_normalization, config.hyperparameters_update_method,
            metaparams, config.alpha)

      elif config.adaptation_string == "GeneralRegression":
        return blackbox_optimization_algorithms.GeneralRegressionBlackboxOptimizer(
            regression_method=getattr(regression_optimizers,
                                      config.regression_optimizer_string),
            regularizer=config.regularizer,
            est_type=config.adaptation_est_type,
            normalize_fvalues=config.fvalues_normalization,
            hyperparameters_update_method=config.hyperparameters_update_method,
            extra_params=metaparams,
            step_size=config.alpha)

    config.adaptation_blackbox_optimizer_fn = adaptation_blackbox_optimizer_fn

  temp_env = make_task_fn(0)
  config.state_dimensionality = temp_env.state_dimensionality()
  config.action_dimensionality = temp_env.action_dimensionality()

  def rl_policy_fn():
    return policies.DeterministicNumpyPolicy(config.state_dimensionality,
                                             config.action_dimensionality,
                                             config.hidden_layers)

  config.rl_policy_fn = rl_policy_fn

  def RLMAMLBlackboxObject_fn():
    return blackbox_maml_objects.RLMAMLBlackboxObject(config)

  config.RLMAMLBlackboxObject_fn = RLMAMLBlackboxObject_fn

  def sl_policy_fn():
    return policies.Basic_TF_Policy(config.state_dimensionality,
                                    config.action_dimensionality,
                                    config.hidden_layers, "sl")

  config.sl_policy_fn = sl_policy_fn

  def LossTensorMAMLBlackboxObject_fn():
    return blackbox_maml_objects.LossTensorMAMLBlackboxObject(config)

  config.LossTensorMAMLBlackboxObject_fn = LossTensorMAMLBlackboxObject_fn

  def blackbox_object_fn():
    blackbox_object = config.RLMAMLBlackboxObject_fn()
    policy_param_dim = blackbox_object.policy_param_num

    if config.algorithm == "zero_order":
      adaptation_blackbox_optimizer = config.adaptation_blackbox_optimizer_fn(
          blackbox_object.get_metaparams())
      adapter_fn = getattr(adaptation_optimizers, config.adapter_fn_string)
      adapter = adapter_fn(
          num_queries=config.num_queries,
          adaptation_blackbox_optimizer=adaptation_blackbox_optimizer,
          adaptation_precision_parameter=config.adaptation_precision_parameter,
          policy_param_dim=policy_param_dim,
          perturbation_type=config.perturbation_type,
          dpp_rho=config.dpp_rho,
          parallel_alg=config.hillclimb_parallel_alg,
          parallel_evaluations=config.hillclimb_parallel_evaluations)
      blackbox_object.use_adapter(adapter)
    return blackbox_object

  config.blackbox_object_fn = blackbox_object_fn

  if config.hidden_layers:
    hidden_layers_name = "H" + "_".join([str(h) for h in config.hidden_layers])
  else:
    hidden_layers_name = "L"
  local_logfoldername = "_".join([
      config.task_name, hidden_layers_name, "Q" + str(config.num_queries),
      current_time_string, config.adapter_fn_string
  ])

  config.global_logfoldername = os.path.join(config.folder_name,
                                             local_logfoldername)

  config.hparams_file = os.path.join(config.global_logfoldername,
                                     "hparams.json")

  config.log_frequency = 1
  config.params_file = os.path.join(config.global_logfoldername, "params.csv")
  config.best_params_file = os.path.join(config.global_logfoldername,
                                         "best_params.csv")
  config.best_core_hyperparameters_file = os.path.join(
      config.global_logfoldername, "best_core_hyperparams.csv")
  config.best_value_file = os.path.join(config.global_logfoldername,
                                        "best_value.csv")
  config.optimizer_internal_state_file = os.path.join(
      config.global_logfoldername, "optimizer_internal_state.csv")
  config.current_values_list_file = os.path.join(config.global_logfoldername,
                                                 "current_values_list.csv")
  config.best_values_list_file = os.path.join(config.global_logfoldername,
                                              "best_values_list.csv")
  config.plot_file = os.path.join(config.global_logfoldername, "plot.csv")
  config.fvalues_file = os.path.join(config.global_logfoldername, "fvalues.csv")
  config.iteration_file = os.path.join(config.global_logfoldername,
                                       "iteration.csv")
  config.test_values_file = os.path.join(config.global_logfoldername,
                                         "test_values_file.csv")

  config.mamlpt_values_file = os.path.join(config.global_logfoldername,
                                           "mamlpt_values_file.csv")

  config.test_mamlpt_parallel_vals_folder = os.path.join(
      config.global_logfoldername, "test_mamlpt_parallel_evals_folder")

  return config
