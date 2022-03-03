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

"""Optimizer for parameters in exchange-correlation functionals."""

import tempfile
import time

from absl import logging
import cma
import jax
import numpy as np


class CMAESOptimizer:
  """CMA-ES optimizer."""

  _default_hyperparameters = {
      'initial_parameters_mean': 0.,
      'initial_parameters_std': 1.,
      'sigma0': 1.0,
      'popsize': None,
      'maxfevals': 50000,
      'tolfun': None,
      'early_termination_abnormal_wrmsd': float('inf'),
      'early_termination_num_fevals': float('inf'),
      'early_termination_wrmsd': float('inf'),
      'bounds': None,
      'l1_penalty': 0.,
      'l2_penalty': 0.,
      'seed': None
  }

  def __init__(self, evaluator, **kwargs):
    """Initializes CMA-ES optimizer.

    Args:
      evaluator: Instance of evaluators.Evaluator, the evaluator for the
        exchange-correlation functional.
      **kwargs: Dict, the hyperparameters for optimization. Possible keys are:
        * initial_parameters_mean: mean value for initial guess.
        * initial_parameters_std: standard deviation for initial guess. Initial
          guess is draw from normal distribution.
        * sigma0: the initial standard deviation for CMA-ES.
        * popsize: population size, defaults to be automatically determined.
        * maxfevals: maximum number of function evaluation.
        * tolfun: convergence threshold on function values.
        * early_termination_abnormal_wrmsd: early termination will be triggered
          if wrmsd is greater than early_termination_abnormal_wrmsd.
        * early_termination_num_fevals, early_termination_wrmsd: early
          termination will be triggered if wrmsd is greater than
          early_termination_wrmsd after early_termination_num_fevals
          function evaluations.
        * seed: random seed.
        Their default values are included in self._default_hyperparameters and
        will be overridden by **kwargs.

    Raises:
      ValueError: if unknown hyperparameters are specified in kwargs.
    """
    self.evaluator = evaluator

    for hyperparameter in kwargs:
      if hyperparameter not in self._default_hyperparameters:
        raise ValueError(f'Unknown hyperparameter: {hyperparameter}')

    hyperparameters = self._default_hyperparameters.copy()
    hyperparameters.update(kwargs)

    self.random_state = np.random.RandomState(hyperparameters.pop('seed'))

    self.hyperparameters = {key: value for key, value in hyperparameters.items()
                            if value is not None}
    logging.info(
        'CMAESOptimizer: constructing CMAESOptimizer with hyperparameters %s',
        self.hyperparameters)

    # direct the output files from CMA-ES calculations to a temporary folder
    self.hyperparameters['verb_filenameprefix'] = tempfile.mkdtemp()

    self.hyperparameters['termination_callback'] = (
        self.make_termination_callback(
            early_termination_abnormal_wrmsd=self.hyperparameters.pop(
                'early_termination_abnormal_wrmsd'),
            early_termination_num_fevals=self.hyperparameters.pop(
                'early_termination_num_fevals'),
            early_termination_wrmsd=self.hyperparameters.pop(
                'early_termination_wrmsd'))
        )

    self.sigma0 = self.hyperparameters.pop('sigma0')
    self.initial_parameters_mean = self.hyperparameters.pop(
        'initial_parameters_mean')
    self.initial_parameters_std = self.hyperparameters.pop(
        'initial_parameters_std')

    self.l1_penalty = self.hyperparameters.pop('l1_penalty')
    self.l2_penalty = self.hyperparameters.pop('l2_penalty')

  @classmethod
  def get_hyperparameter_names(cls):
    return list(cls._default_hyperparameters.keys())

  @staticmethod
  def make_termination_callback(early_termination_abnormal_wrmsd,
                                early_termination_num_fevals,
                                early_termination_wrmsd):
    """Makes a callback function for early termination of optimization."""

    def termination_callback(cma_es):
      """Helper function to terminate optimization for abnormal function values.

      Args:
        cma_es: Instance of cma.evolution_strategy.CMAEvolutionStrategy class,
          the CMAEvolutionStrategy object that carries the current status of
          optimization.

      Returns:
        Boolean, True if all function values at current step are infinite or
          larger than early_termination_abnormal_wrmsd, or the best function
          value obtained after early_termination_num_fevals evaluations is still
          larger than early_termination_wrmsd.
      """
      if not np.any(cma_es.fit.fit < early_termination_abnormal_wrmsd):
        logging.info(
            'Optimization terminated due to abnormal function values: %s',
            cma_es.fit.fit)
        return True
      elif (cma_es.countevals >= early_termination_num_fevals
            and cma_es.fit.histbest[0] > early_termination_wrmsd):
        logging.info(
            'Optimization terminated at step %s with best WRMSD %s > %s',
            cma_es.countevals,
            cma_es.fit.histbest[0],
            early_termination_wrmsd)
        return True
      else:
        return False

    return termination_callback

  @staticmethod
  def parse_cma_es_results(cma_es_results):
    """Parses results from CMA-ES minimization into dictionary.

    Args:
      cma_es_results: Instance of
        cma.evolution_strategy.CMAEvolutionStrategyResult, the results of CMA-ES
        minimization.

    Returns:
       Dict, the results of CMA-ES minimization in JSON-serializable format.
    """
    parsed_results = {}
    for i, key in enumerate([
        'xbest', 'fbest', 'evals_best', 'evaluations', 'iterations',
        'xfavorite', 'stds', 'stop'
    ]):
      value = cma_es_results[i]
      if isinstance(value, np.ndarray):
        parsed_results[key] = list(map(float, value))
      elif isinstance(value, np.integer):
        parsed_results[key] = int(value)
      else:
        parsed_results[key] = value

    return parsed_results

  def get_objective(self, functional):
    """Constructs the objective function."""
    eval_wrmsd = self.evaluator.get_eval_wrmsd(functional)

    def objective(parameters_vec):
      """Objective function for minimization.

      Args:
        parameters_vec: Float numpy array with shape (num_parameters,), the
          parameters for exchange-correlation functional.

      Returns:
        Float, the weighted root mean square deviation (WRMSD).
      """
      loss = float(
          eval_wrmsd(**jax.tree_unflatten(
              functional.parameters_spec, parameters_vec)))
      if self.l1_penalty > 1e-8:
        loss += self.l1_penalty * np.sum(np.abs(parameters_vec))
      if self.l2_penalty > 1e-8:
        loss += self.l2_penalty * np.sum(parameters_vec ** 2)
      return loss

    return objective

  def run_optimization(self,
                       functional,
                       num_trials=1,
                       parameters_init=None):
    """Performs multiple trials of optimization and returns the best solution.

    Args:
      functional: Instance of xc_functionals.XCFunctional, the exchange-
        correlation functional to be optimized.
      num_trials: Integer, the number of trials for optimization.
      parameters_init: Dict, initial parameters. If not specified, initial
        parameters will be drawn from a normal distribution defined by
        self.initial_parameters_mean and self.initial_parameters_std.

    Returns:
      Dict, the best solution of all trials of CMA-ES minimization.
    """
    logging.info(
        'CMAESOptimizer: optimization of %s-parameter function with '
        'specification: %s', functional.num_parameters,
        functional.parameters_spec)

    objective = self.get_objective(functional)

    if parameters_init:
      parameters_vec_init = jax.tree_flatten(parameters_init)[0]

    start = time.time()
    wrmsd_best = float('inf')
    wrmsds = []
    results = None
    for trial in range(num_trials):
      logging.info('CMAESOptimizer: trial #%s', trial)

      if parameters_init:
        initial_guess = parameters_vec_init
      else:
        initial_guess = self.random_state.normal(
            self.initial_parameters_mean,
            self.initial_parameters_std,
            size=functional.num_parameters)
      logging.info('CMAESOptimizer: initial parameters = %s', initial_guess)

      results_trial = self.optimize(objective, initial_guess=initial_guess)

      wrmsd = results_trial['fbest']
      wrmsds.append(wrmsd)
      logging.info(
          'CMAESOptimizer: trial #%s WRMSD (kcal/mol): %s', trial, wrmsd)
      if results is None or wrmsd < wrmsd_best:
        wrmsd_best = wrmsd
        results = results_trial
        results['parameters'] = (
            None if results['xbest'] is None else jax.tree_unflatten(
                functional.parameters_spec, results['xbest']))

    results['wrmsd_trials'] = wrmsds
    logging.info('CMAESOptimizer: best WRMSD (kcal/mol): %s', wrmsd_best)
    logging.info('CMAESOptimizer: best solution: %s', results)
    logging.info('CMAESOptimizer: time for %s optimizations: %s s',
                 num_trials, time.time() - start)
    return results

  def optimize(self, objective, initial_guess):
    """Performs CMA-ES minimization of the objective function.

    Args:
      objective: Function, the objective function.
      initial_guess: Float numpy array with shape (num_parameters,),
        the initial guess for optimization.

    Returns:
      Dict, the results of CMA-ES minimization.
    """
    cma_evolution_strategy = cma.CMAEvolutionStrategy(
        initial_guess, self.sigma0, self.hyperparameters)
    cma_evolution_strategy.optimize(objective)
    return self.parse_cma_es_results(cma_evolution_strategy.result)
