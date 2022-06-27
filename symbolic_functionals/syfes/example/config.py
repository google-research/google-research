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

"""Config file for SyFES."""

from ml_collections import config_dict


def get_xc_config():
  """Gets the config for XC functional."""
  config = config_dict.ConfigDict()
  # The names of features for the exchange enhancement factor.
  # Comma-separated string.
  config.feature_names_x = 'x2,w'
  # The names of features for the same-spin correlation enhancement factor.
  # Comma-separated string.
  config.feature_names_css = 'x2,w'
  # The names of features for the opposite-spin correlation enhancement factor.
  # Comma-separated string.
  config.feature_names_cos = 'x2,w'
  # The functional form from which mutations take place. Can be either a
  # functional name defined in xc_functionals, or a path to a json file that
  # includes the definition of a functional form.
  config.mutation_base = ''
  # The training, validation and test losses of the mutation base.
  # Comma separated string of 3 integers.
  config.mutation_base_losses = ''
  # The number of shared parameters in the functional form.
  # Comma-separated string of 1 integer or 3 integers (for 3
  # enhancement_factors). Defaults to those of the mutation_base.
  config.num_shared_parameters = ''
  # The number of temporary variables in the functional form.
  # Comma-separated string of 1 integer or 3 integers (for 3
  # enhancement_factors). Defaults to those of the mutation_base.
  config.num_variables = ''
  # The path to the specification of instruction pool for the experiment.
  config.instruction_pool = (
      '/namespace/gas/primary/california/instruction_pool'
      '/arithmetic_power_transform_functional.json')
  # The specification of mutation pool for the experiment.
  # Comma separated string of mutation rule and probabilities.
  config.mutation_pool = (
      'insert_instruction,0.25,remove_instruction,0.25,'
      'replace_instruction,0.25,change_argument,0.25')
  # The mutation probabilities for exchange, same-spin correlation and
  # opposite-spin correlation enhancement factors.
  # Comma separated string of 3 floats.
  config.component_mutation_probabilities = (
      '0.333333333333,0.333333333333,0.333333333333')
  # The maximum number of bound parameters.  If less than zero, no constraint
  # is applied to the number of bound parameters.
  config.max_num_bound_parameters = 2
  # The maximum number of instructions per enhancement factor. If less than
  # zero, no constraint is applied to the number of instructions.
  config.max_num_instructions = -1
  # The number of fixed instructions per enhancement factor.
  config.num_fixed_instructions = 0
  return config


def get_re_config():
  """Gets the config for regularized evolution."""
  config = config_dict.ConfigDict()
  # The number of individuals to keep in the population.
  config.population_size = 100
  # The number of individuals that should participate in each tournament.
  config.tournament_size = 10
  # The probability of mutation.
  config.mutation_probability = 0.9
  # The coefficient for train loss in the calculation of fitness.
  config.train_loss_coeff = 0.
  # The coefficient for validation loss in the calculation of fitness.
  # If train_loss_coeff = 0.227 & validation_loss_coeff = 0.773, the fitness
  # is equivalent to the WRMSD of training + validation set, as used in
  # the wB97M-V paper.
  config.validation_loss_coeff = 1.
  # The fitness penalty for each used parameters in functional forms.
  config.penalty_per_parameter = 0.
  return config


def get_infra_config():
  """Gets the config for infrastructure."""
  config = config_dict.ConfigDict()
  # The name of the population server.
  config.population_server_name_prefix = 'population_server'
  # The name of the fingerprint server.
  config.fingerprint_server_name_prefix = 'fingerprint_server'
  # The name of the data to store the history.
  config.history_data_name = ''
  # Whether to perform functional equivalence checking. If True, cache the
  # fingerprints of functionals in a memo server.
  config.cache_fingerprints = True
  # The number of feature samples used to evaluate functional fingerprints.
  config.fec_num_feature_samples = 10
  # The number of parameter samples used to evaluate functional fingerprints.
  config.fec_num_parameter_samples = 10
  # The number of decimals used to evaluate functional fingerprints.
  config.fec_num_decimals = 5
  return config


def get_dataset_config():
  """Gets the config for dataset."""
  config = config_dict.ConfigDict()
  # The path to the specification of grid evaluator.
  # If not specified, normal evaluator will be used.
  config.grid_evaluator_spec = ''
  # The directory of saved mgcdb84 dataset.
  config.dataset_directory = ''
  # The data types of MGCDB84 dataset to use. If specified, the training and
  # validation set will be obtained by partition data with specified type.
  # If not specified, training and validation set will be those of MCGDB84.
  config.mgcdb84_types = ''
  # The fraction of training, validation and set sets.
  # Only used if mgcdb84_types is not None. Comma separated string of 3 floats.
  config.train_validation_test_split = '0.6,0.2,0.2'
  # The targets used for training. Defaults to mgcdb84_ref, which uses target
  # values from reference values given by MCGDB84. targets can also be set to
  # the exchange-correlation energies of a certain functional, which can be
  # specified by an existing functional name in xc_functionals or the path to
  # a json file specifying the functional form and parameters.
  config.targets = 'mgcdb84_ref'
  # The number of targets used for training. Default to 0 (use all targets).
  config.num_targets = 0
  # If True, only spin unpolarized molecules are used.
  config.spin_singlet = False
  # The evaluation mode for training, validation and test sets. Possible values
  # are jit, onp and jnp. Comma separated string.
  config.eval_modes = 'jit,onp,onp'
  return config


def get_opt_config():
  """Gets the config for optimizer."""
  config = config_dict.ConfigDict()
  # The number of trials for optimizing each functional form. The minimum loss
  # of all trials will be taken as the training loss of the functional form.
  config.num_opt_trials = 1
  # The mean of random initial guess.
  config.initial_parameters_mean = 0.
  # The standard deviation of random initial guess.
  config.initial_parameters_std = 1.
  # The initial sigma for ES optimization.
  config.sigma0 = 1.
  # The population size for ES optimization. Defaults to 0 (automatically
  # determined by CMA-ES package)
  config.popsize = 0
  # The maximum number of function evaluations.
  config.maxfevals = 100000
  # The convergence tolerance on function value.
  config.tolfun = 1e-5
  # The following three flags determins the early termination behavior of
  # CMA-ES optimizations. CMA-ES optimizations will be terminated early if one
  # of the following happens:
  # 1. At any step all function values are larger than
  #    early_termination_abnormal_wrmsd.
  # 2. After early_termination_num_fevals evaluations, the best function value
  #    obtained is still larger than early_termination_wrmsd.
  config.early_termination_abnormal_wrmsd = 1e5
  config.early_termination_wrmsd = 50.
  config.early_termination_num_fevals = 5000
  # The lower and upper bounds of parameters for optimization.
  # Comma-separated string of 2 floats. Defaults to unbounded.
  config.bounds = ''
  # The L1 penalty for optimization.
  config.l1_penalty = 0.
  # The L2 penalty for optimization.
  config.l2_penalty = 0.
  return config


def get_config():
  """Gets the config."""
  config = config_dict.ConfigDict()
  config.xc = get_xc_config()
  config.re = get_re_config()
  config.infra = get_infra_config()
  config.dataset = get_dataset_config()
  config.opt = get_opt_config()
  return config
