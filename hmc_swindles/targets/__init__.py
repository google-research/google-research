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

# python3
"""Target distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from typing import Text

import numpy as np

from hmc_swindles.targets import banana as banana_lib
from hmc_swindles.targets import data
from hmc_swindles.targets import ill_conditioned_gaussian as ill_conditioned_gaussian_lib
from hmc_swindles.targets import item_response_theory
from hmc_swindles.targets import logistic_regression
from hmc_swindles.targets import neals_funnel as neals_funnel_lib
from hmc_swindles.targets import probit_regression
from hmc_swindles.targets import sparse_logistic_regression
from hmc_swindles.targets import target_spec

__all__ = [
    'banana',
    'german_credit_numeric_logistic_regression',
    'german_credit_numeric_probit_regression',
    'german_credit_numeric_sparse_logistic_regression',
    'german_credit_numeric_with_test_logistic_regression',
    'german_credit_numeric_with_test_probit_regression',
    'german_credit_numeric_with_test_sparse_logistic_regression',
    'ill_conditioned_gaussian',
    'neals_funnel',
    'stan_item_response_theory',
]


def _populate_ground_truth(
    target_name,
    spec):
  """Populates the ground truth values in a spec."""
  # We do a delayed import to avoid parsing sometimes huge ground truth files.
  # Also, this lets us do the import programmatically based on the passed name.
  module_name = 'hmc_swindles.targets.ground_truth.' + target_name
  module = importlib.import_module(module_name)
  for name in spec.expectations.keys():
    mean_name = f'{name.upper()}_MEAN'
    sem_name = f'{name.upper()}_MEAN_STANDARD_ERROR'
    sd_name = f'{name.upper()}_STANDARD_DEVIATION'
    spec.expectations[name] = spec.expectations[name]._replace(
        ground_truth_mean=getattr(module, mean_name, None),
        ground_truth_mean_standard_error=getattr(module, sem_name, None),
        ground_truth_standard_deviation=getattr(module, sd_name, None),
    )
  return spec


ill_conditioned_gaussian = ill_conditioned_gaussian_lib.ill_conditioned_gaussian
banana = banana_lib.banana
neals_funnel = neals_funnel_lib.neals_funnel


def german_credit_numeric_logistic_regression():
  """German credit (numeric) logistic regression.

  This requires setting the `german_credit_numeric_path` flag.

  Returns:
    target: TargetDensity.
  """
  spec = logistic_regression.logistic_regression(data.german_credit_numeric)
  return _populate_ground_truth(
      'german_credit_numeric_logistic_regression',
      spec,
  )


def german_credit_numeric_with_test_logistic_regression(
):
  """German credit (numeric) logistic regression, with a test set.

  This uses a 75/25 split of the data. This requires setting the
  `german_credit_numeric_path` flag.

  This requires setting the `german_credit_numeric_path` flag.

  Returns:
    target: TargetDensity.
  """
  spec = logistic_regression.logistic_regression(
      lambda: data.german_credit_numeric(train_fraction=0.75))
  return _populate_ground_truth(
      'german_credit_numeric_with_test_logistic_regression',
      spec,
  )


def german_credit_numeric_sparse_logistic_regression(
):
  """German credit (numeric) sparse logistic regression.

  This requires setting the `german_credit_numeric_path` flag.

  Returns:
    target: TargetDensity.
  """
  spec = sparse_logistic_regression.sparse_logistic_regression(
      data.german_credit_numeric)
  return _populate_ground_truth(
      'german_credit_numeric_sparse_logistic_regression',
      spec,
  )


def german_credit_numeric_with_test_sparse_logistic_regression(
):
  """German credit (numeric) sparse logistic regression, with a test set.

  This uses a 75/25 split of the data. This requires setting the
  `german_credit_numeric_path` flag.

  Returns:
    target: TargetDensity.
  """
  spec = sparse_logistic_regression.sparse_logistic_regression(
      lambda: data.german_credit_numeric(train_fraction=0.75))
  return _populate_ground_truth(
      'german_credit_numeric_with_test_sparse_logistic_regression',
      spec,
  )


def german_credit_numeric_probit_regression():
  """German credit (numeric) probit regression.

  This requires setting the `german_credit_numeric_path` flag.

  Returns:
    target: TargetDensity.
  """
  spec = probit_regression.probit_regression(data.german_credit_numeric)
  return _populate_ground_truth(
      'german_credit_numeric_probit_regression',
      spec,
  )


def german_credit_numeric_with_test_probit_regression(
):
  """German credit (numeric) probit regression, with a test set.

  This uses a 75/25 split of the data. This requires setting the
  `german_credit_numeric_path` flag.

  Returns:
    target: TargetDensity.
  """
  spec = probit_regression.probit_regression(
      lambda: data.german_credit_numeric(train_fraction=0.75))
  return _populate_ground_truth(
      'german_credit_numeric_with_test_probit_regression',
      spec,
  )


def stan_item_response_theory():
  """Stan synthetic one-parameter logistic item-response theory problem.

  This requires setting the `stan_item_response_theory_path` flag.

  Returns:
    target: TargetDensity.
  """
  spec = item_response_theory.item_response_theory(
      data.stan_item_response_theory)
  return _populate_ground_truth(
      'stan_item_response_theory',
      spec,
  )
