# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Logic for running a single active learning cycle."""

import functools
import glob
import os
import shutil
from typing import List

import ml_collections
from modAL import models
import pandas as pd

from al_for_fep.utils import utils

_TRAINING_EXAMPLE_KEY = 'Training Example'
_BAYESIAN_MODELS = ['gp']


class MakitaCycle:
  """Coordinator object for a single active learning cycle."""

  def __init__(self, cycle_config):
    """Initialization.

    Args:
      cycle_config: ConfigDict with all the hyperparameters for this cycle.
    """
    self._cycle_config = cycle_config

  def _get_virtual_library(self, model_config,
                           selection_columns):
    """Helper function to determine train/selection split."""
    feature_column = model_config.features.params['feature_column']
    target_column = model_config.targets.params['feature_column']

    virtual_lib = pd.read_csv(self._cycle_config.virtual_library)

    training_pool_ids = []
    for fileglob in self._cycle_config.training_pool.split(','):
      for filename in glob.glob(fileglob):
        training_pool_ids.append(pd.read_csv(filename)[[feature_column]])
    training_pool_ids = pd.concat(training_pool_ids)

    columns_to_keep = list(
        set(selection_columns + [feature_column, target_column]))
    virtual_lib = virtual_lib[columns_to_keep].drop_duplicates()

    virtual_lib[_TRAINING_EXAMPLE_KEY] = virtual_lib[feature_column].isin(
        training_pool_ids[feature_column].values
    ) & ~virtual_lib[target_column].isna()

    return virtual_lib

  def _get_train_features_and_targets(self,
                                      model_config,
                                      training_pool):
    """Helper to parse and calculate feature and target values training data."""
    train_features = utils.DATA_PARSERS[model_config.features.feature_type](
        training_pool, **model_config.features.params)
    train_targets = utils.DATA_PARSERS[model_config.targets.feature_type](
        training_pool, **model_config.targets.params)

    return train_features, train_targets

  def _get_selection_pool_features(self,
                                   model_config,
                                   selection_pool):
    """Helper to parse and calculate feature values of a full selection pool."""
    selection_pool_features = utils.DATA_PARSERS[
        model_config.features.feature_type](selection_pool,
                                            **model_config.features.params)

    return selection_pool_features

  def run_cycle(self):
    """Driver for running a complete cycle."""

    model_config = self._cycle_config.model_config
    cycle_dir = self._cycle_config.cycle_dir

    selection_filename = os.path.join(cycle_dir, 'selection.csv')
    if os.path.exists(selection_filename):
      try:
        pd.read_csv(selection_filename)
        # The selection file exists and is valid. This cycle has been completed
        # previously and will be skipped. Delete the data to rerun the cycle.
        return
      except pd.errors.EmptyDataError:
        # The selection file exists, but there's an issue reading it.
        # We delete the existing file and restart the cycle.
        os.remove(selection_filename)

    if os.path.exists(cycle_dir):
      shutil.rmtree(cycle_dir)

    os.mkdir(cycle_dir)

    metadata_str = self._cycle_config.get('metadata', default=None)
    if metadata_str is None or not metadata_str:
      raise ValueError(
          'Expected non-empty "metadata" key with description of cycle.')

    with open(os.path.join(cycle_dir, 'metadata.txt'), 'w') as fout:
      fout.write(metadata_str)

    with open(os.path.join(cycle_dir, 'cycle_config.json'), 'w') as fout:
      fout.write(self._cycle_config.to_json())

    selection_columns = self._cycle_config.selection_config.selection_columns
    virtual_library = self._get_virtual_library(model_config, selection_columns)

    train_features, train_targets = self._get_train_features_and_targets(
        model_config, virtual_library[virtual_library[_TRAINING_EXAMPLE_KEY]])

    library_features = self._get_selection_pool_features(
        model_config, virtual_library)

    selection_pool = virtual_library[~virtual_library[_TRAINING_EXAMPLE_KEY]]
    selection_pool_features = self._get_selection_pool_features(
        model_config, selection_pool)

    estimator = utils.MODELS[model_config.model_type](
        model_config.hyperparameters, model_config.tuning_hyperparameters)

    if 'halfsample_log2_shards' in model_config:
      estimator = utils.HALF_SAMPLE_WRAPPER(
          subestimator=estimator.get_model(),
          shards_log2=model_config.halfsample_log2_shards,
          add_estimators=model_config.model_type in ['rf', 'gbm'])

    selection_config = self._cycle_config.selection_config
    query_strategy = functools.partial(
        utils.QUERY_STRATEGIES[selection_config.selection_type],
        n_instances=selection_config.num_elements,
        **selection_config.hyperparameters)

    target_multiplier = 1
    if selection_config.selection_type in ['thompson', 'EI', 'PI', 'UCB']:
      target_multiplier = -1

    train_targets = train_targets * target_multiplier

    if model_config.model_type in _BAYESIAN_MODELS:
      learner = models.BayesianOptimizer(
          estimator=estimator.get_model(),
          X_training=train_features,
          y_training=train_targets,
          query_strategy=query_strategy)
    else:
      learner = models.ActiveLearner(
          estimator=estimator.get_model(),
          X_training=train_features,
          y_training=train_targets,
          query_strategy=query_strategy)

    inference = learner.predict(library_features) * target_multiplier

    virtual_library['regression'] = inference.T.tolist()

    virtual_library.to_csv(
        os.path.join(cycle_dir, 'virtual_library_with_predictions.csv'),
        index=False)

    selection_idx, _ = learner.query(selection_pool_features)

    selection_pool.iloc[selection_idx][selection_columns].to_csv(
        selection_filename, index=False)
