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

"""Gin-configurable wrappers for different data sources.

If we want to use in our code a fetcher, for example, for seismic waves (and
configure the stations, window size, etc. we want to use), this is the place to
put it.
"""

from typing import Optional, Sequence

import gin
import numpy as np
import pandas as pd

from eq_mag_prediction.utilities import catalog_filters
from eq_mag_prediction.utilities import data_utils


@gin.configurable
def target_catalog(
    catalog,
    earthquake_criterion,
    separate_repeating_times_in_catalog = True,
    smear_binned_magnitudes = False,
):
  """Returns a catalog of earthquakes that we want to forecast in some way.

  Args:
    catalog: A dataframe of earthquakes. Should be sorted by time.
    earthquake_criterion: A function that accepts an earthquake catalog and
      returns a boolean index array of the same length, marking which
      earthquakes to keep.
    separate_repeating_times_in_catalog: eliminate duplicate timestamps by
      spreading them in a short time interval.
    smear_binned_magnitudes: smear magnitudes across their bin in case of
      discretized magnitudes.

  Returns:
    A sub-catalog of target earthquakes (used to calculate labels).
  """
  assert np.all(np.diff(catalog.time.values) >= 0)
  return_catalog = catalog[earthquake_criterion(catalog)].copy()
  if separate_repeating_times_in_catalog:
    return_catalog = data_utils.separate_repeating_times_in_catalog(
        return_catalog
    )
  if smear_binned_magnitudes:
    return_catalog = data_utils.smear_binned_magnitudes(return_catalog)
  return return_catalog


@gin.configurable
def feature_catalog(
    catalog,
    earthquake_criterion,
    add_noise = True,
):
  """Returns a catalog of earthquakes that we can use in our model.

  Args:
    catalog: A dataframe of earthquakes. Should be sorted by time.
    earthquake_criterion: A function that accepts an earthquake catalog and
      returns a boolean index array of the same length, marking which
      earthquakes to keep.
    add_noise: Whether to add some noise to the time, longitude and latitude
      columns. This could be used to avoid duplicates or to weaken correlations
      between the precision of these columns and others.

  Returns:
    A sub-catalog of feature earthquakes (used to calculate features).
  """
  assert np.all(np.diff(catalog.time.values) >= 0)
  result = catalog[earthquake_criterion(catalog)].copy()
  if add_noise:
    n_eqs = len(result)
    result['longitude'] += np.random.uniform(low=-0.01, high=0.01, size=n_eqs)
    result['latitude'] += np.random.uniform(low=-0.01, high=0.01, size=n_eqs)
    result['time'] += np.random.uniform(low=-0.1, high=0.1, size=n_eqs)
  return result.sort_values('time', ascending=True).copy()


@gin.configurable
def cmt_catalog(
    catalog,
    earthquake_criterion,
):
  """Returns a catalog of earthquakes with focal mechanisms."""
  if catalog is None:
    catalog = data_utils.global_cmt_dataframe()
  assert np.all(np.diff(catalog.time.values) >= 0)
  return catalog[earthquake_criterion(catalog)].copy()


@gin.configurable
def permuted_catalog(
    catalog,
    earthquake_criterion,
    columns_to_permute = ('magnitude',),
    permute_earthquake_criterion = tuple(),
    separate_permute = False,
    random_seed = 1905,
):
  """Returns a copied df with randomly permuted values in the specified columns.

  Args:
    catalog: The catalog to randomly permute.
    earthquake_criterion: A function that accepts an earthquake catalog and
      returns a boolean index array of the same length, marking which
      earthquakes to keep.
    columns_to_permute: A sequence of column names to be permuted. All other
      columns remain in their original order. Default to the 'magnitude' column
      only.
    permute_earthquake_criterion: A list of earthquake_criterions, used to
      determine the subset of rows among which to perform the permutation.
    separate_permute: Whether to use the same permutation for all columns in
      columns_to_permute or shuffle between them. Defaults to False.
    random_seed:  An int to initialize the random number generator.

  Returns: A copy of the catalog permuted according to user specifications.
  """
  rng = np.random.default_rng(random_seed)
  original_catalog = target_catalog(catalog, earthquake_criterion)
  if len(permute_earthquake_criterion) == 0:  # pylint: disable=g-explicit-length-test
    permute_earthquake_criterion = [earthquake_criterion]
  catalog_copy = original_catalog.copy()
  columns_to_permute = list(columns_to_permute)

  rows_to_permute_list = [
      eq_crit(original_catalog) for eq_crit in permute_earthquake_criterion
  ]
  if len(rows_to_permute_list) != 1:
    assert np.all(
        ~np.logical_and.reduce(np.array(rows_to_permute_list))
    ), 'Earthquake criteria overlap!'
  for rows_to_permute in rows_to_permute_list:
    if separate_permute:
      for col in columns_to_permute:
        catalog_copy.loc[rows_to_permute, col] = rng.permutation(
            catalog_copy.loc[rows_to_permute, col]
        )
    else:
      catalog_copy.loc[rows_to_permute, columns_to_permute] = rng.permutation(
          original_catalog.loc[rows_to_permute, columns_to_permute]
      )
  return catalog_copy
