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

r"""Search symbolic functionals on a single machine.

"""


import json
import time

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags
import numpy as np
import tensorflow.compat.v1 as tf

from symbolic_functionals.syfes import loss
from symbolic_functionals.syfes.evolution import common
from symbolic_functionals.syfes.evolution import regularized_evolution
from symbolic_functionals.syfes.symbolic import optimizers
from symbolic_functionals.syfes.symbolic import search_utils
from symbolic_functionals.syfes.symbolic import xc_functionals


config_flags.DEFINE_config_file('config', 'config.py')

flags.DEFINE_integer('xid', 0, 'The experiment id.')
flags.DEFINE_integer('wid', 0, 'The work unit id.')

FLAGS = flags.FLAGS


class XCFunctionalPopulation(regularized_evolution.Population):
  """Population of functional forms."""

  def create_initial_population(self):
    """Creates the initial population with mutation base."""
    if 'json' in self._cfg.xc.mutation_base:
      with tf.io.gfile.GFile(self._cfg.xc.mutation_base, 'r') as f:
        functional_base = xc_functionals.XCFunctional.from_dict(json.load(f))
    else:
      functional_base = getattr(xc_functionals, self._cfg.xc.mutation_base)

    feature_names_x = self._cfg.xc.feature_names_x.split(',')
    feature_names_css = self._cfg.xc.feature_names_css.split(',')
    feature_names_cos = self._cfg.xc.feature_names_cos.split(',')

    if self._cfg.xc.num_shared_parameters:
      num_shared_parameters = list(
          map(int, self._cfg.xc.num_shared_parameters.split(',')))
      if len(num_shared_parameters) == 1:
        num_shared_parameters = num_shared_parameters[0]
    else:
      num_shared_parameters = None

    if self._cfg.xc.num_variables:
      num_variables = list(map(int, self._cfg.xc.num_variables.split(',')))
      if len(num_variables) == 1:
        num_variables = num_variables[0]
    else:
      num_variables = None

    logging.info(
        'Create initial population based on functional form: %s, '
        'using specifications: %s', self._cfg.xc.mutation_base,
        {'feature_names_x': feature_names_x,
         'feature_names_css': feature_names_css,
         'feature_names_cos': feature_names_cos,
         'num_shared_parameters': num_shared_parameters,
         'num_variables': num_variables})

    functional = functional_base.make_isomorphic_copy(
        feature_names_x=feature_names_x,
        feature_names_css=feature_names_css,
        feature_names_cos=feature_names_cos,
        num_shared_parameters=num_shared_parameters,
        num_variables=num_variables)

    train_loss, validation_loss, test_loss = map(
        float, self._cfg.xc.mutation_base_losses.split(','))
    unregularized_fitness = loss.combine_wrmsd(
        coeff_1=self._cfg.re.train_loss_coeff,
        coeff_2=self._cfg.re.validation_loss_coeff,
        wrmsd_1=train_loss,
        wrmsd_2=validation_loss)
    penalty = functional.eval_penalty(self._cfg.re.penalty_per_parameter)

    self.add_to_population(
        gene=str(functional),
        fitness=unregularized_fitness + penalty,
        train_loss=train_loss,
        validation_loss=validation_loss,
        test_loss=test_loss,
        unregularized_fitness=unregularized_fitness,
        penalty=penalty)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  cfg = FLAGS.config

  # =============================
  # Setup the fingerprint server.
  # =============================
  if cfg.infra.cache_fingerprints:
    fingerprints_client = common.start_fingerprint_client(
        cfg.infra.fingerprint_server_name_prefix,
        xid=FLAGS.xid,
        wid=FLAGS.wid,
        fingerprint_server={})
  else:
    fingerprints_client = None

  # =============================
  # Setup the population server.
  # =============================
  population = XCFunctionalPopulation(
      population_size=cfg.re.population_size,
      tournament_size=cfg.re.tournament_size,
      mutation_probability=cfg.re.mutation_probability,
      history_writer=common.create_writer(cfg.infra.history_data_name),
      other_config=cfg)

  # =============================
  # Setup the worker.
  # =============================
  client = common.start_population_client(
      cfg.infra.population_server_name_prefix,
      xid=FLAGS.xid,
      wid=FLAGS.wid,
      population=population)

  # construct mutator
  with tf.io.gfile.GFile(cfg.xc.instruction_pool, 'r') as f:
    instruction_pool = json.load(f)
  logging.info('Instruction pool: %s', instruction_pool)

  mutation_pool_tokens = cfg.xc.mutation_pool.split(',')
  if len(mutation_pool_tokens) % 2 != 0:
    raise ValueError('Wrong mutation pool: the mutation_pool flag should '
                     'contain pairs of mutation rule and probability.')
  mutation_pool = {
      mutation_pool_tokens[2 * i]: float(mutation_pool_tokens[2 * i + 1])
      for i in range(len(mutation_pool_tokens) // 2)}
  logging.info('Mutation pool: %s', mutation_pool)

  mutator = search_utils.make_mutator(
      instruction_pool=instruction_pool,
      mutation_pool=mutation_pool,
      max_num_instructions=(
          cfg.xc.max_num_instructions
          if cfg.xc.max_num_instructions >= 0 else None),
      max_num_bound_parameters=(
          cfg.xc.max_num_bound_parameters
          if cfg.xc.max_num_bound_parameters >= 0 else None),
      num_fixed_instructions=cfg.xc.num_fixed_instructions,
      component_mutation_probabilities=list(map(
          float, cfg.xc.component_mutation_probabilities.split(',')))
  )

  # construct evaluators for training and validation error
  feature_names_x = cfg.xc.feature_names_x.split(',')
  feature_names_css = cfg.xc.feature_names_css.split(',')
  feature_names_cos = cfg.xc.feature_names_cos.split(',')

  if not cfg.dataset.grid_evaluator_spec:
    if cfg.dataset.mgcdb84_types:
      # create training and validation set with specified data types in MGCDB84
      evaluator_train, evaluator_validation, evaluator_test = (
          search_utils.make_evaluators_with_mgcdb84_type(
              dataset_directory=cfg.dataset.dataset_directory,
              mcgdb84_types=cfg.dataset.mgcdb84_types.split(','),
              train_validation_test_split=list(map(
                  float, cfg.dataset.train_validation_test_split.split(','))),
              feature_names_x=feature_names_x,
              feature_names_css=feature_names_css,
              feature_names_cos=feature_names_cos,
              spin_singlet=cfg.dataset.spin_singlet,
              targets=cfg.dataset.targets,
              num_targets=cfg.dataset.num_targets,
              omega=0.3,
              alpha=1.0,
              beta=-0.85,
              eval_modes=cfg.dataset.eval_modes.split(','))
          )
    else:
      # use training and validation set of MGCDB84
      evaluator_train, evaluator_validation, evaluator_test = (
          search_utils.make_evaluators_with_mgcdb84_partitioning(
              dataset_directory=cfg.dataset.dataset_directory,
              feature_names_x=feature_names_x,
              feature_names_css=feature_names_css,
              feature_names_cos=feature_names_cos,
              spin_singlet=cfg.dataset.spin_singlet,
              targets=cfg.dataset.targets,
              num_targets=cfg.dataset.num_targets,
              omega=0.3,
              alpha=1.0,
              beta=-0.85,
              eval_modes=cfg.dataset.eval_modes.split(','))
          )
  else:
    with tf.io.gfile.GFile(cfg.dataset.grid_evaluator_spec, 'r') as f:
      grid_evaluator_spec = json.load(f)
    evaluator_train, evaluator_validation, evaluator_test = (
        search_utils.make_grid_evaluators(
            **grid_evaluator_spec, eval_modes=cfg.dataset.eval_modes.split(','))
        )

  # construct optimizer
  optimizer_hyperparameters = {}
  for hyperparameter in optimizers.CMAESOptimizer.get_hyperparameter_names():
    if hyperparameter in cfg.opt:
      optimizer_hyperparameters[hyperparameter] = getattr(
          cfg.opt, hyperparameter)
  if not optimizer_hyperparameters['popsize']:
    optimizer_hyperparameters['popsize'] = None
  if optimizer_hyperparameters['bounds']:
    optimizer_hyperparameters['bounds'] = list(map(
        float, optimizer_hyperparameters['bounds'].split(',')))
  else:
    optimizer_hyperparameters['bounds'] = None
  optimizer = optimizers.CMAESOptimizer(
      evaluator_train, **optimizer_hyperparameters)

  # regularized evolution
  step = 0
  while True:
    time.sleep(0.5)

    logging.info('Worker %s: Step #%s', FLAGS.wid, step)

    # get a child functional by mutating a parent functional fetched
    # from server
    parent_functional = xc_functionals.XCFunctional.from_dict(
        json.loads(client.get_parent()))
    child_functional = mutator.mutate(functional=parent_functional)[0]
    penalty = child_functional.eval_penalty(cfg.re.penalty_per_parameter)
    logging.info('Generated child functional with penalty %s', penalty)

    if fingerprints_client is not None:
      query = child_functional.get_fingerprint(
          num_feature_samples=cfg.infra.fec_num_feature_samples,
          num_parameter_samples=cfg.infra.fec_num_parameter_samples,
          num_decimals=cfg.infra.fec_num_decimals)
      unregularized_fitness = common.functional_equivalence_checking(
          query, fingerprints_client)
      if unregularized_fitness is not None:
        fitness = unregularized_fitness + penalty
        client.add_to_population(gene=str(child_functional), fitness=fitness)
        logging.info(
            'Found equivalence of child functional in fingerprints server '
            'with unregularized fitness %s, added to population with '
            'fitness %s', unregularized_fitness, fitness)
        step += 1
        continue

    # train the child functional
    results = search_utils.train_functional(
        functional=child_functional,
        optimizer=optimizer,
        num_opt_trials=cfg.opt.num_opt_trials,
        parameters_init=None,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test,
        clear_xla_cache=False)

    unregularized_fitness = loss.combine_wrmsd(
        coeff_1=cfg.re.train_loss_coeff,
        coeff_2=cfg.re.validation_loss_coeff,
        wrmsd_1=results['train_loss'],
        wrmsd_2=results['validation_loss'])

    # if the child functional has finite unregularized fitness, send
    # it to server
    if np.isfinite(unregularized_fitness):
      fitness = unregularized_fitness + penalty
      client.add_to_population(
          gene=str(child_functional),
          fitness=fitness,
          **{key: results[key] for key in [
              'train_loss', 'validation_loss', 'test_loss',
              'train_time', 'validation_time', 'test_time']},
          penalty=penalty,
          unregularized_fitness=unregularized_fitness,
          parameters=json.dumps(results['parameters']),
      )
      logging.info(
          'Added child functional to population with fitness %s', fitness)
    else:
      logging.info('Dropped child functional due to infinite fitness.')

    if fingerprints_client is not None:
      # NOTE(leeley): Record the fingerprint even if validation loss is not
      # finite. It avoids wasting computational power in similar functional
      # form.
      fingerprints_client[query] = unregularized_fitness
    step += 1


if __name__ == '__main__':
  app.run(main)
