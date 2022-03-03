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

"""Helper functions for searching symbolic forms of density functionals."""

import time

from absl import logging
from jax.interpreters import xla
import numpy as np

from symbolic_functionals.syfes.dataset import dataset
from symbolic_functionals.syfes.symbolic import enhancement_factors
from symbolic_functionals.syfes.symbolic import evaluators
from symbolic_functionals.syfes.symbolic import mutators
from symbolic_functionals.syfes.symbolic import xc_functionals


def make_mutator(instruction_pool,
                 mutation_pool,
                 max_num_instructions,
                 max_num_bound_parameters,
                 num_fixed_instructions,
                 component_mutation_probabilities=None,
                 seed=None):
  """Constructs mutator for functional forms.

  Args:
    instruction_pool: Dict, the pool of possible instructions.
    mutation_pool: Dict, the pool of possible mutation rules.
    max_num_instructions: Integer, the maximum number of instructions.
    max_num_bound_parameters: Integer, the maximum number of bound parameters.
    num_fixed_instructions: Integer, the number of fixed instructions.
    component_mutation_probabilities: Sequence of 3 floats, the probabilities
      for mutating exchange, same-spin or opposite-spin component of the
      functional.
    seed: Integer, random seed.

  Returns:
    Instance of mutators.XCFunctionalMutator, the resulting mutator.
  """
  return mutators.XCFunctionalMutator(
      mutator_x=mutators.EnhancementFactorMutator(
          instruction_pool=instruction_pool,
          mutation_pool=mutation_pool,
          max_num_instructions=max_num_instructions,
          num_fixed_instructions=num_fixed_instructions,
          max_num_bound_parameters=max_num_bound_parameters),
      mutator_css=mutators.EnhancementFactorMutator(
          instruction_pool=instruction_pool,
          mutation_pool=mutation_pool,
          max_num_instructions=max_num_instructions,
          num_fixed_instructions=num_fixed_instructions,
          max_num_bound_parameters=max_num_bound_parameters),
      mutator_cos=mutators.EnhancementFactorMutator(
          instruction_pool=instruction_pool,
          mutation_pool=mutation_pool,
          max_num_instructions=max_num_instructions,
          num_fixed_instructions=num_fixed_instructions,
          max_num_bound_parameters=max_num_bound_parameters),
      component_mutation_probabilities=component_mutation_probabilities,
      seed=seed)


def make_evaluators_with_mgcdb84_partitioning(
    dataset_directory,
    feature_names_x,
    feature_names_css,
    feature_names_cos,
    spin_singlet=False,
    targets='mgcdb84_ref',
    num_targets=None,
    omega=0.3,
    alpha=1.0,
    beta=-0.85,
    eval_modes=('jit', 'onp', 'onp')):
  """Constructs evaluators based on mgcdb84 training and validation set.

  Args:
    dataset_directory: String, the directory to dataset.
    feature_names_x: List of strings, the features for exchange enhancement
      factor.
    feature_names_css: List of strings, the features for same-spin correlation
      enhancement factor.
    feature_names_cos: List of strings, the features for opposite-spin
      correlation enhancement factor.
    spin_singlet: Boolean, if True, only spin unpolarized molecules will
      be included in dataset.
    targets: String, the targets used for evaluating WRMSD. Defaults to
      'mgcdb84_ref', which computes target values from reference values given
      by MCGDB84. Other supported values are:
        * B97X: target values are exchange-correlation energies evaluated by
                B97 exchange functional.
        * B97: target values are exchange-correlation energies evaluated by
                B97 functional.
    num_targets: Integer, the number of targets used to construct train
      or validation evaluator. Defaults to use all targets in MGCDB84 training
      set and all targets in validation set.
    omega: Float, RSH parameter for functional used in SCF calculations.
    alpha: Float, RSH parameter for functional used in SCF calculations.
    beta: Float, RSH parameter for functional used in SCF calculations.
      Default values of omega, alpha, beta are those of wB97M-V functional
      obtained with pyscf.dft.libxc.rsh_coeff('wb97m_v')
    eval_modes: Sequence of 3 strings, evaluation mode for training, validation
      and test evaluators. Possible values are onp, jnp and jit.

  Returns:
    List of 3 instances of evaluators.Evaluator, the evaluator for training,
      validation and test losses.
  """
  evaluator_list = []
  for paritition_index, partition in enumerate(['train', 'validation', 'test']):
    evaluator = evaluators.Evaluator.from_dataset(
        subset=dataset.Dataset.load_mcgdb84_subset(
            dataset_directory=dataset_directory,
            mgcdb84_set=partition,
            spin_singlet=spin_singlet,
            nrow_property=num_targets),
        feature_names_x=feature_names_x,
        feature_names_css=feature_names_css,
        feature_names_cos=feature_names_cos,
        targets=targets,
        omega=omega,
        alpha=alpha,
        beta=beta,
        eval_mode=eval_modes[paritition_index])
    logging.info('Evaluator on %s set constructed: %s', partition, evaluator)
    evaluator_list.append(evaluator)

  return evaluator_list


def make_evaluators_with_mgcdb84_type(
    dataset_directory,
    mcgdb84_types,
    feature_names_x,
    feature_names_css,
    feature_names_cos,
    train_validation_test_split=(0.6, 0.2, 0.2),
    spin_singlet=False,
    targets='mgcdb84_ref',
    num_targets=None,
    omega=0.3,
    alpha=1.0,
    beta=-0.85,
    eval_modes=('jit', 'onp', 'onp')):
  """Constructs evaluators using given type of MGCDB84.

  Data in given type will be combined and split into training, validation
  and test sets, resulting in 3 evaluators.

  Args:
    dataset_directory: String, the directory to dataset.
    mcgdb84_types: List of strings, the mgcdb84 types.
    feature_names_x: List of strings, the features for exchange enhancement
      factor.
    feature_names_css: List of strings, the features for same-spin correlation
      enhancement factor.
    feature_names_cos: List of strings, the features for opposite-spin
      correlation enhancement factor.
    train_validation_test_split: Sequence of 3 floats, the fraction of training,
      validation and test set.
    spin_singlet: Boolean, if True, only spin unpolarized molecules will
      be included in dataset.
    targets: String, the targets used for evaluating WRMSD. Defaults to
        'mgcdb84_ref', which computes target values from reference values given
        by MCGDB84. Other supported values are:
          * B97X: target values are exchange-correlation energies evaluated by
                  B97 exchange.
    num_targets: Integer, the total number of targets used to construct
      train, validation and test set evaluator. Defaults to use all targets with
      specified data types.
    omega: Float, RSH parameter for functional used in SCF calculations.
    alpha: Float, RSH parameter for functional used in SCF calculations.
    beta: Float, RSH parameter for functional used in SCF calculations.
      Default values of omega, alpha, beta are those of wB97M-V functional
      obtained with pyscf.dft.libxc.rsh_coeff('wb97m_v')
    eval_modes: Sequence of 3 strings, evaluation mode for training, validation
      and test evaluators. Possible values are onp, jnp and jit.

  Returns:
    List of 3 instances of evaluators.Evaluator, the evaluator for training,
      validation and test losses.

  Raises:
    ValueError, if train_validation_test_split has wrong length
      or train_validation_test_split contains negative values
      or train_validation_test_split do not sum to 1.
  """
  if (len(train_validation_test_split) != 3
      or any(frac < 0. for frac in train_validation_test_split)
      or abs(sum(train_validation_test_split) - 1.) > 1e-8):
    raise ValueError(
        'Invalid train_validation_test_split: ', train_validation_test_split)

  subset = dataset.Dataset.load_mcgdb84_subset(
      dataset_directory=dataset_directory,
      mgcdb84_types=mcgdb84_types,
      spin_singlet=spin_singlet,
      nrow_property=num_targets)

  property_dfs = np.split(
      subset.property_df.sample(frac=1, random_state=0), [
          int(train_validation_test_split[0] * subset.nrow_property),
          int(sum(train_validation_test_split[:2]) * subset.nrow_property)
      ])

  evaluator_list = []
  for paritition_index, partition in enumerate(['train', 'validation', 'test']):
    evaluator = evaluators.Evaluator.from_dataset(
        subset=subset.get_subset(
            property_df_subset=property_dfs[paritition_index]),
        feature_names_x=feature_names_x,
        feature_names_css=feature_names_css,
        feature_names_cos=feature_names_cos,
        targets=targets,
        omega=omega,
        alpha=alpha,
        beta=beta,
        eval_mode=eval_modes[paritition_index])
    logging.info('Evaluator on %s set constructed: %s', partition, evaluator)
    evaluator_list.append(evaluator)

  return evaluator_list


def make_grid_evaluators(
    features,
    weights,
    targets,
    e_lda_x,
    e_lda_css,
    e_lda_cos,
    signature,
    train_validation_test_split=(0.6, 0.2, 0.2),
    eval_modes=('jit', 'onp', 'onp')):
  """Constructs grid evaluators."""
  if (len(train_validation_test_split) != 3
      or any(frac < 0. for frac in train_validation_test_split)
      or abs(sum(train_validation_test_split) - 1.) > 1e-8):
    raise ValueError(
        'Invalid train_validation_test_split: ', train_validation_test_split)

  features = {feature_name: np.array(feature)
              for feature_name, feature in features.items()}
  weights = np.array(weights)
  targets = np.array(targets)
  e_lda_x = np.array(e_lda_x)
  e_lda_css = np.array(e_lda_css)
  e_lda_cos = np.array(e_lda_cos)

  num_grids = len(weights)
  grid_indices_partition = np.split(
      np.random.RandomState(0).permutation(num_grids), [
          int(train_validation_test_split[0] * num_grids),
          int(sum(train_validation_test_split[:2]) * num_grids)
      ])

  evaluator_list = []
  for paritition_index, partition in enumerate(['train', 'validation', 'test']):
    grid_indices = grid_indices_partition[paritition_index]
    evaluator = evaluators.GridEvaluator(
        # make copies to ensure memory layout is contiguous
        features={feature_name: feature[grid_indices].copy()
                  for feature_name, feature in features.items()},
        weights=weights[grid_indices].copy(),
        targets=targets[grid_indices].copy(),
        e_lda_x=e_lda_x[grid_indices].copy(),
        e_lda_css=e_lda_css[grid_indices].copy(),
        e_lda_cos=e_lda_cos[grid_indices].copy(),
        signature=signature,
        eval_mode=eval_modes[paritition_index])
    logging.info(
        'GridEvaluator on %s set constructed: %s', partition, evaluator)
    evaluator_list.append(evaluator)

  return evaluator_list


def make_random_functional(
    mutator,
    feature_names_x,
    feature_names_css,
    feature_names_cos,
    num_shared_parameters,
    num_variables,
    num_instructions):
  """Makes a random functional with given specifications.

  Args:
    mutator: Instance of mutators.XCFunctionalMutator, the mutator used to
      generate random instruction lists.
    feature_names_x: Sequence of strings, the feature names for evaluating
      exchange enhancement factor.
    feature_names_css: Sequence of strings, the feature names for evaluating
      same-spin correlation enhancement factor.
    feature_names_cos: Sequence of strings, the feature names for evaluating
      opposite-spin correlation enhancement factor.
    num_shared_parameters: Integer or sequence of 3 integers, the number of
      shared parameters for each enhancement factor. Defaults to None, which
      uses the number of shared parameters of current enhancement factors.
    num_variables: Integer or sequence of 3 integers, the number of variables
      for each enhancement factor. Defaults to None, which uses the number
      of variables of current enhancement factors.
    num_instructions: Integer, the number of instructions for each enhancement
      factor.

  Returns:
    Instance of xc_functionals.XCFunctional, the random functional.
  """
  if num_shared_parameters is None or isinstance(num_shared_parameters, int):
    num_shared_parameters_x = num_shared_parameters
    num_shared_parameters_css = num_shared_parameters
    num_shared_parameters_cos = num_shared_parameters
  else:
    (num_shared_parameters_x, num_shared_parameters_css,
     num_shared_parameters_cos) = num_shared_parameters

  if num_variables is None or isinstance(num_variables, int):
    num_variables_x = num_variables
    num_variables_css = num_variables
    num_variables_cos = num_variables
  else:
    num_variables_x, num_variables_css, num_variables_cos = num_variables

  f_x_base = enhancement_factors.f_empty.make_isomorphic_copy(
      feature_names=feature_names_x,
      num_shared_parameters=num_shared_parameters_x,
      num_variables=num_variables_x)
  f_css_base = enhancement_factors.f_empty.make_isomorphic_copy(
      feature_names=feature_names_css,
      num_shared_parameters=num_shared_parameters_css,
      num_variables=num_variables_css)
  f_cos_base = enhancement_factors.f_empty.make_isomorphic_copy(
      feature_names=feature_names_cos,
      num_shared_parameters=num_shared_parameters_cos,
      num_variables=num_variables_cos)

  return xc_functionals.XCFunctional(
      f_x=enhancement_factors.EnhancementFactor(
          feature_names=f_x_base.feature_names,
          shared_parameter_names=f_x_base.shared_parameter_names,
          variable_names=f_x_base.variable_names,
          instruction_list=mutator.mutator_x.randomize_instruction_list(
              f_x_base, num_instructions=num_instructions)[0]),
      f_css=enhancement_factors.EnhancementFactor(
          feature_names=f_css_base.feature_names,
          shared_parameter_names=f_css_base.shared_parameter_names,
          variable_names=f_css_base.variable_names,
          instruction_list=mutator.mutator_css.randomize_instruction_list(
              f_css_base, num_instructions=num_instructions)[0]),
      f_cos=enhancement_factors.EnhancementFactor(
          feature_names=f_cos_base.feature_names,
          shared_parameter_names=f_cos_base.shared_parameter_names,
          variable_names=f_cos_base.variable_names,
          instruction_list=mutator.mutator_cos.randomize_instruction_list(
              f_cos_base, num_instructions=num_instructions)[0]),
      )


def train_functional(functional,
                     optimizer,
                     num_opt_trials,
                     parameters_init=None,
                     evaluator_validation=None,
                     evaluator_test=None,
                     clear_xla_cache=True):
  """Trains a given functional form.

  Args:
    functional: Instance of xc_functionals.XCFunctional, the functional form
      with parameters to be determined by optimization.
    optimizer: Instance of optimizers.CMAESOptimizer, the optimizer.
    num_opt_trials: Integer, the number of trials for optimization. The final
      results will be determined by the trial with minimum training loss.
    parameters_init: Dict, initial parameters. If not specified, random initial
      parameters will be used.
    evaluator_validation: Instance of evaluators.Evaluator, the evaluator
      of validation loss. If present, the validation loss will be computed.
    evaluator_test: Instance of evaluators.Evaluator, the evaluator
      of test loss. If present, the test loss will be computed.
    clear_xla_cache: Boolean, if True, the XLA cache will be cleared. Only
      relevant if jax.jit is used for evaluations.

  Returns:
    Dict, the results of optimization, see CMAESOptimizer.run_optimization.
      If evaluator_validation is specified, the dict will include an additional
      key 'validation_loss' for validation loss.
  """
  if clear_xla_cache:
    # NOTE(htm): jax.jit will cache compiled functions with different static
    # arguments. Currently, one has to call the following protected function
    # to clear jit cache.
    xla._xla_callable.cache_clear()  # pylint: disable=protected-access

  # optimize the functional form with training set
  start = time.time()
  results = optimizer.run_optimization(
      functional,
      num_trials=num_opt_trials,
      parameters_init=parameters_init)
  results['train_loss'] = results.pop('fbest')
  results['train_time'] = time.time() - start

  evaluator_list = []
  if evaluator_validation is not None:
    evaluator_list.append(('validation', evaluator_validation))
  if evaluator_test is not None:
    evaluator_list.append(('test', evaluator_test))

  for prefix, evaluator in evaluator_list:
    start = time.time()

    if results['parameters'] is not None:
      loss = evaluator.get_eval_wrmsd(functional)(**results['parameters'])
    else:
      loss = np.nan

    results[f'{prefix}_loss'] = loss
    results[f'{prefix}_time'] = time.time() - start

    logging.info('%s WRMSD (kcal/mol): %s', prefix.capitalize(), loss)
    logging.info('Evaluation time for %s WRMSD: %s',
                 prefix, results[f'{prefix}_time'])

  return results
