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

"""Config file for both client and server."""

import os
import ml_collections
import pyglove as pg

from es_enas import objects
from es_enas import policies
from es_enas.controllers import policy_gradient_controller
from es_enas.controllers import random_controller
from es_enas.controllers import regularized_evolution_controller

from es_optimization import algorithms
from es_optimization.blackbox_functions import get_environment


def get_config():
  """ML Collections config with JSON primitives."""

  config = ml_collections.ConfigDict()
  config.seed = 0  # Random seed for experimentation.

  config.folder_name = "./es_enas_logs/"

  config.est_type = "antithetic"  # Type of estimation method used.
  # "antithetic"
  # "forward_fd"

  config.total_num_perturbations = 150  # Total number of perturbations used.
  config.num_exact_evals = 150  # Number rollouts for current input eval.

  config.controller_type_str = "policy_gradient"
  # "random"
  # "policy_gradient"
  # "regularized_evolution"

  config.policy_gradient_update_batch_size = 64

  if config.est_type == "forward_fd":
    config.num_workers = config.total_num_perturbations + config.num_exact_evals
  elif config.est_type == "antithetic":
    config.num_workers = 2 * config.total_num_perturbations + config.num_exact_evals

  config.es_precision_parameter = 0.1  # Epsilon parameter of FD.
  config.es_step_size = 0.01  # Step size of the GD procedure.
  config.nb_iterations = 100000  # Number of iterations.

  # Indicates whether fvalues normalization is turned on.
  config.fvalues_normalization = 1

  # Hyperparameter update method used.
  config.hyperparameters_update_method = "state_normalization"
  config.nb_perturbations_per_worker = 1
  config.horizon = 1000

  config.critical = 0.4  # Acceptable server failure rate.
  config.environment_name = "HalfCheetah"  # Which environment to use
  config.policy_fn_string = "NumpyWeightSharingPolicy"
  # "NumpyWeightSharingPolicy"
  # "NumpyEdgeSparsityPolicy"

  config.hidden_layers = []  #  list of hidden layers
  config.num_partitions = 17  # If using "NumpyWeightSharingPolicy"

  # Num. edges for each layer if using "NumpyEdgeSparsityPolicy";
  # len(hidden_layer_edge_num) = len(hidden_layers) + 1
  config.hidden_layer_edge_num = [40, 40]

  # Edges allowed for "NumpyEdgeSparsityPolicy";
  config.edge_policy_sample_mode = "aggregate_edges"
  # "aggregate_edges"
  # "independent_edges"
  # "residual_edges"

  config.log_frequency = 1  # Frequency of log printing.

  return config


def generate_config(base_config, **kwargs):
  """Materializes functions and other objects which cannot be saved in JSON."""

  config = ml_collections.ConfigDict(initial_dictionary=base_config)
  if config.is_locked:
    config.unlock()

  config.json_hparams = config.to_json_best_effort()

  config.environment_fn = lambda: get_environment(config.environment_name)
  temp_environment = config.environment_fn()
  config.state_dimensionality = temp_environment.state_dimensionality()
  config.action_dimensionality = temp_environment.action_dimensionality()

  policy_fn = getattr(policies, config.policy_fn_string)

  def policy_fn_for_object():
    return policy_fn(
        state_dimensionality=config.state_dimensionality,
        action_dimensionality=config.action_dimensionality,
        hidden_layers=config.hidden_layers,
        num_partitions=config.num_partitions,
        hidden_layer_edge_num=config.hidden_layer_edge_num,
        edge_policy_sample_mode=config.edge_policy_sample_mode)

  config.policy_fn_for_object = policy_fn_for_object
  config.example_policy = config.policy_fn_for_object()

  def make_blackbox_object_fn():
    return objects.GeneralTopologyBlackboxObject(config)

  config.blackbox_object_fn = make_blackbox_object_fn

  def es_blackbox_optimizer_fn(metaparams):
    return algorithms.MCOptimizer(
        config.es_precision_parameter,
        config.est_type,
        config.fvalues_normalization,
        config.hyperparameters_update_method,
        metaparams,
        config.es_step_size,
        num_top_directions=0)

  config.es_blackbox_optimizer_fn = es_blackbox_optimizer_fn

  def setup_controller_fn():
    if config.controller_type_str == "policy_gradient":
      config.controller_fn = policy_gradient_controller.PolicyGradientController
    elif config.controller_type_str == "random":
      config.controller_fn = random_controller.RandomController
    elif config.controller_type_str == "regularized_evolution":
      config.controller_fn = regularized_evolution_controller.RegularizedEvolutionController
    config.controller = config.controller_fn(config.example_policy.dna_spec,
                                             config.num_workers)

    def sample_topology_str():
      sample = config.controller.propose_topology_str()
      return pg.to_json_str(sample)

    config.sample_topology_str = sample_topology_str

  config.setup_controller_fn = setup_controller_fn

  config.hparams_file = os.path.join(config.folder_name, "hparams.json")

  config.params_file = os.path.join(config.folder_name, "params_file.csv")
  config.best_params_file = os.path.join(config.folder_name,
                                         "best_params_file.csv")
  config.best_core_hyperparameters_file = os.path.join(
      config.folder_name, "best_core_hyperparameters_file.csv")
  config.best_value_file = os.path.join(config.folder_name,
                                        "best_value_file.csv")
  config.optimizer_internal_state_file = os.path.join(
      config.folder_name, "optimizer_internal_state_file.csv")
  config.current_values_list_file = os.path.join(
      config.folder_name, "current_values_list_file.csv")
  config.best_values_list_file = os.path.join(config.folder_name,
                                              "best_values_list_file.csv")
  config.fvalues_file = os.path.join(config.folder_name, "fvalues_file.csv")
  config.iteration_file = os.path.join(config.folder_name, "iteration_file.csv")

  del kwargs
  return config
