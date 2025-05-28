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

"""A module to create filters and statistics of entire catalogs.

E.g. calculating parameters of a moving Gutenberg Richter distribution and
saving them to cache.
"""

import hashlib
import numbers
import os
from typing import Callable, Optional, Sequence
import joblib
import numpy as np
import pandas as pd
from tensorflow.io import gfile
import pathlib
from eq_mag_prediction.utilities import catalog_analysis

CACHE_PATH = os.path.join(
    os.path.dirname(__file__), '../..', 'results/cached_benchmarks'
)
COMPLETENESS_VEC_TXT = 'completeness_vec.txt'
BETA_VEC_TXT = 'beta_vec.txt'
KDE_LIST_TXT = 'kde_list.joblib'
DAYS_TO_SECONDS = 60 * 60 * 24
KM_TO_DEG = 1 / 111.11


def _cache_file_path(filename, cache_folder=CACHE_PATH):
  return os.path.join(cache_folder, filename)


def hash_pandas_object(catalog):
  return hashlib.sha256(
      pd.util.hash_pandas_object(catalog, index=True).values
  ).hexdigest()


def hash_catalog_and_estimate_times(catalog, estimate_times):
  catalog_sha = hash_pandas_object(catalog)
  estimate_times_sha = hash_pandas_object(pd.Series(estimate_times))
  return hashlib.sha1(
      f'{catalog_sha}_{estimate_times_sha}'.encode('utf-8')
  ).hexdigest()


def hash_catalog_and_estimate_coors(catalog, estimate_coors):
  catalog_sha = hash_pandas_object(catalog)
  estimate_coors_sha = hash_pandas_object(pd.DataFrame(estimate_coors))
  return hashlib.sha1(
      f'{catalog_sha}_{estimate_coors_sha}'.encode('utf-8')
  ).hexdigest()


def _dict_to_path_name(d):
  s = ''
  for k, v in d.items():
    if isinstance(v, numbers.Number):
      v_s = f'{round(v, 4):n}'
    else:
      v_s = str(v)
    s += f'{str(k)}_{str(v_s)}_'
  return s[:-1]


def _folder_exists_in_dir(folder_name, parent_path=CACHE_PATH):
  return folder_name in os.listdir(parent_path)


_var_is_none = lambda v: v is None or np.isnan(v)


def _replace_none_with_nan(vector):
  vector[np.array([_var_is_none(v) for v in vector])] = np.nan


###############################################################################
# catalog_analysis.gr_moving_window_constant_time related functions
###############################################################################


def _gr_moving_window_constant_time_vars_to_str(
    estimate_times,
    catalog,
    **kwargs,
):
  """Converts gr_moving_window_constant_time's arguments into a unique str."""
  arguments_dict = {
      'catalog_estimate_times': hash_catalog_and_estimate_times(
          catalog, estimate_times
      ),
  }
  arguments_dict.update(kwargs)
  if callable(arguments_dict['completeness_calculator']):
    arguments_dict['completeness_calculator'] = arguments_dict[
        'completeness_calculator'
    ].__name__.replace('_', '')
  keys = list(arguments_dict.keys())
  for k in keys:
    arguments_dict[k.replace('_', '')] = arguments_dict.pop(k)
  arguments_str = _dict_to_path_name(arguments_dict)
  arguments_str = 'gr_moving_window_constant_time_' + arguments_str
  return arguments_str


def _calculate_and_save_gr_moving_window_constant_time(
    estimate_times,
    catalog,
    cache_folder = CACHE_PATH,
    **kwargs,
):
  """Executes gr_moving_window_constant_time and save result to cache folder."""
  completeness_vec, beta_vec = catalog_analysis.gr_moving_window_constant_time(
      estimate_times,
      catalog,
      **kwargs,
  )
  _replace_none_with_nan(completeness_vec)
  _replace_none_with_nan(beta_vec)

  arguments_str = _gr_moving_window_constant_time_vars_to_str(
      estimate_times,
      catalog,
      **kwargs,
  )
  dir_path = pathlib.Path(
      _cache_file_path(arguments_str, cache_folder=cache_folder)
  )
  dir_path.mkdir(exist_ok=True)

  completeness_path = os.path.join(
      _cache_file_path(arguments_str, cache_folder=cache_folder),
      COMPLETENESS_VEC_TXT,
  )
  if os.path.exists(completeness_path):
    gfile.remove(completeness_path)
  with open(completeness_path, 'wt') as f:
    np.savetxt(f, completeness_vec)

  beta_path = os.path.join(
      _cache_file_path(arguments_str, cache_folder=cache_folder),
      BETA_VEC_TXT,
  )
  if os.path.exists(beta_path):
    gfile.remove(beta_path)
  with open(beta_path, 'wt') as f:
    np.savetxt(f, beta_vec)


def _load_gr_moving_window_constant_time_result_from_cache(
    estimate_times,
    catalog,
    cache_folder=CACHE_PATH,
    **kwargs,
):
  """Loads saved results of gr_moving_window_constant_time from cache folder."""

  arguments_str = _gr_moving_window_constant_time_vars_to_str(
      estimate_times,
      catalog,
      **kwargs,
  )

  with open(
      os.path.join(
          _cache_file_path(arguments_str, cache_folder=cache_folder),
          COMPLETENESS_VEC_TXT,
      ),
      'r',
  ) as f:
    completeness_vec = np.loadtxt(f)
  with open(
      os.path.join(
          _cache_file_path(arguments_str, cache_folder=cache_folder),
          BETA_VEC_TXT,
      ),
      'r',
  ) as f:
    beta_vec = np.loadtxt(f)
  return completeness_vec, beta_vec


def get_gr_moving_window_constant_time(
    estimate_times,
    catalog,
    window_time = 10 * DAYS_TO_SECONDS,
    m_minimal = -100,
    n_above_complete = 1,
    weight_on_past = 1,
    default_beta = np.nan,
    default_mc = None,
    completeness_calculator = None,
    force_recalculate = False,
    cache_folder = CACHE_PATH,
):
  """Gets cached results of gr_moving_window_constant_time or calculates if needed."""
  gr_moving_window_constant_time_kwargs = {
      'window_time': window_time,
      'm_minimal': m_minimal,
      'n_above_complete': n_above_complete,
      'weight_on_past': weight_on_past,
      'default_beta': default_beta,
      'default_mc': default_mc,
      'completeness_calculator': completeness_calculator,
  }
  arguments_str = _gr_moving_window_constant_time_vars_to_str(
      estimate_times=estimate_times,
      catalog=catalog,
      **gr_moving_window_constant_time_kwargs,
  )
  cache_exists = _folder_exists_in_dir(arguments_str, parent_path=cache_folder)
  if (not cache_exists) | force_recalculate:
    _calculate_and_save_gr_moving_window_constant_time(
        estimate_times,
        catalog,
        cache_folder=cache_folder,
        **gr_moving_window_constant_time_kwargs,
    )
  return _load_gr_moving_window_constant_time_result_from_cache(
      estimate_times,
      catalog,
      cache_folder=cache_folder,
      **gr_moving_window_constant_time_kwargs,
  )


###############################################################################
# catalog_analysis.gr_moving_window_n_events related functions
###############################################################################


def _gr_moving_window_n_events_vars_to_str(
    estimate_times,
    catalog,
    **kwargs,
):
  """Converts gr_moving_window_constant_time's arguments into a unique str."""
  arguments_dict = {
      'catalog_estimate_times': hash_catalog_and_estimate_times(
          catalog, estimate_times
      ),
  }
  arguments_dict.update(kwargs)
  if callable(arguments_dict['completeness_and_beta_calculator']):
    arguments_dict['completeness_and_beta_calculator'] = arguments_dict[
        'completeness_and_beta_calculator'
    ].__name__.replace('_', '')
  keys = list(arguments_dict.keys())
  for k in keys:
    arguments_dict[k.replace('_', '')] = arguments_dict.pop(k)
  arguments_str = _dict_to_path_name(arguments_dict)
  arguments_str = 'gr_moving_window_n_events_' + arguments_str
  return arguments_str


def _calculate_and_save_gr_moving_window_n_events(
    estimate_times,
    catalog,
    cache_folder = CACHE_PATH,
    **kwargs,
):
  """Executes gr_moving_window_n_events and saves the result to cache folder."""
  completeness_vec, beta_vec = catalog_analysis.gr_moving_window_n_events(
      estimate_times,
      catalog,
      **kwargs,
  )
  _replace_none_with_nan(completeness_vec)
  _replace_none_with_nan(beta_vec)

  arguments_str = _gr_moving_window_n_events_vars_to_str(
      estimate_times,
      catalog,
      **kwargs,
  )
  cache_folder_path = _cache_file_path(arguments_str, cache_folder=cache_folder)
  dir_path = pathlib.Path(cache_folder_path)
  dir_path.mkdir(exist_ok=True)

  completeness_path = os.path.join(cache_folder_path, COMPLETENESS_VEC_TXT)
  if os.path.exists(completeness_path):
    gfile.remove(completeness_path)
  with open(completeness_path, 'wt') as f:
    np.savetxt(f, completeness_vec)

  beta_path = os.path.join(cache_folder_path, BETA_VEC_TXT)
  if os.path.exists(beta_path):
    gfile.remove(beta_path)
  with open(beta_path, 'wt') as f:
    np.savetxt(f, beta_vec)


def _load_gr_moving_window_n_events_result_from_cache(
    estimate_times,
    catalog,
    cache_folder=CACHE_PATH,
    **kwargs,
):
  """Loads saved results of gr_moving_window_constant_time from cache folder."""

  arguments_str = _gr_moving_window_n_events_vars_to_str(
      estimate_times,
      catalog,
      **kwargs,
  )

  with open(
      os.path.join(
          _cache_file_path(arguments_str, cache_folder=cache_folder),
          COMPLETENESS_VEC_TXT,
      ),
      'r',
  ) as f:
    completeness_vec = np.loadtxt(f)
  with open(
      os.path.join(
          _cache_file_path(arguments_str, cache_folder=cache_folder),
          BETA_VEC_TXT,
      ),
      'r',
  ) as f:
    beta_vec = np.loadtxt(f)
  return completeness_vec, beta_vec


def get_gr_moving_window_n_events(
    estimate_times,
    catalog,
    n_events = 250,
    m_minimal = -100,
    n_above_complete = 1,
    weight_on_past = 1,
    completeness_and_beta_calculator = None,
    force_recalculate=False,
    cache_folder = CACHE_PATH,
):
  """Gets cached results of gr_moving_window_n_events or calculates if needed."""
  gr_moving_window_n_events_kwargs = {
      'n_events': n_events,
      'm_minimal': m_minimal,
      'n_above_complete': n_above_complete,
      'weight_on_past': weight_on_past,
      'completeness_and_beta_calculator': completeness_and_beta_calculator,
  }
  arguments_str = _gr_moving_window_n_events_vars_to_str(
      estimate_times,
      catalog,
      **gr_moving_window_n_events_kwargs,
  )
  cache_exists = _folder_exists_in_dir(arguments_str, parent_path=cache_folder)
  if (not cache_exists) | force_recalculate:
    _calculate_and_save_gr_moving_window_n_events(
        estimate_times,
        catalog,
        cache_folder=cache_folder,
        **gr_moving_window_n_events_kwargs,
    )
  return _load_gr_moving_window_n_events_result_from_cache(
      estimate_times,
      catalog,
      cache_folder=cache_folder,
      **gr_moving_window_n_events_kwargs,
  )


###############################################################################
# catalog_analysis.kde_moving_window_n_events related functions
###############################################################################


def _kde_moving_window_n_events_vars_to_str(
    estimate_times,
    catalog,
    **kwargs,
):
  """Converts kde_moving_window_n_events's arguments into a unique str."""
  arguments_dict = {
      'catalog_estimate_times': hash_catalog_and_estimate_times(
          catalog, estimate_times
      ),
  }
  arguments_dict.update(kwargs)
  if callable(arguments_dict['completeness_calculator']):
    arguments_dict['completeness_calculator'] = arguments_dict[
        'completeness_calculator'
    ].__name__
  keys = list(arguments_dict.keys())
  for k in keys:
    arguments_dict[k.replace('_', '')] = arguments_dict.pop(k)
  arguments_str = _dict_to_path_name(arguments_dict)
  arguments_str = 'kde_moving_window_n_events_' + arguments_str
  return arguments_str


def _calculate_and_save_kde_moving_window_n_events(
    estimate_times,
    catalog,
    cache_folder = CACHE_PATH,
    **kwargs,
):
  """Executes kde_moving_window_n_events and saves the result to cache folder."""
  completeness_vec, kde_list = catalog_analysis.kde_moving_window_n_events(
      estimate_times,
      catalog,
      **kwargs,
  )
  _replace_none_with_nan(completeness_vec)

  arguments_str = _kde_moving_window_n_events_vars_to_str(
      estimate_times,
      catalog,
      **kwargs,
  )
  cache_folder_path = _cache_file_path(arguments_str, cache_folder=cache_folder)
  dir_path = pathlib.Path(cache_folder_path)
  dir_path.mkdir(exist_ok=True)

  completeness_path = os.path.join(cache_folder_path, COMPLETENESS_VEC_TXT)
  if os.path.exists(completeness_path):
    gfile.remove(completeness_path)
  with open(completeness_path, 'wt') as f:
    np.savetxt(f, completeness_vec)

  kde_list_path = os.path.join(cache_folder_path, KDE_LIST_TXT)
  if os.path.exists(kde_list_path):
    gfile.remove(kde_list_path)
  with open(kde_list_path, 'wb') as f:
    joblib.dump(kde_list, f)


def _load_kde_moving_window_n_events_result_from_cache(
    estimate_times,
    catalog,
    cache_folder=CACHE_PATH,
    **kwargs,
):
  """Loads saved results of kde_moving_window_constant_time from cache folder."""

  arguments_str = _kde_moving_window_n_events_vars_to_str(
      estimate_times,
      catalog,
      **kwargs,
  )

  with open(
      os.path.join(
          _cache_file_path(arguments_str, cache_folder=cache_folder),
          COMPLETENESS_VEC_TXT,
      ),
      'r',
  ) as f:
    completeness_vec = np.loadtxt(f)
  with open(
      os.path.join(
          _cache_file_path(arguments_str, cache_folder=cache_folder),
          KDE_LIST_TXT,
      ),
      'rb',
  ) as f:
    kde_list = joblib.load(f)
  return completeness_vec, kde_list


def get_kde_moving_window_n_events(
    estimate_times,
    catalog,
    n_events = 250,
    m_minimal = -100,
    n_above_complete = 1,
    weight_on_past = 1,
    completeness_calculator = None,
    force_recalculate = False,
    cache_folder = CACHE_PATH,
):
  """Gets cached results of kde_moving_window_n_events or calculates if needed."""
  kde_moving_window_n_events_kwargs = {
      'n_events': n_events,
      'm_minimal': m_minimal,
      'n_above_complete': n_above_complete,
      'weight_on_past': weight_on_past,
      'completeness_calculator': completeness_calculator,
  }
  arguments_str = _kde_moving_window_n_events_vars_to_str(
      estimate_times,
      catalog,
      **kde_moving_window_n_events_kwargs,
  )
  cache_exists = _folder_exists_in_dir(arguments_str, parent_path=cache_folder)
  if (not cache_exists) | force_recalculate:
    _calculate_and_save_kde_moving_window_n_events(
        estimate_times,
        catalog,
        cache_folder=cache_folder,
        **kde_moving_window_n_events_kwargs,
    )
  return _load_kde_moving_window_n_events_result_from_cache(
      estimate_times,
      catalog,
      cache_folder=cache_folder,
      **kde_moving_window_n_events_kwargs,
  )


###############################################################################
# catalog_analysis.SpatialBetaCalculator related functions
###############################################################################


def _gr_spatial_vars_to_str(
    estimate_coors,
    catalog,
    **kwargs,
):
  """Converts gr_spatial_beta_const_mc's arguments into a unique str."""
  arguments_dict = {
      'catalog_estimate_coors': hash_catalog_and_estimate_coors(
          catalog, estimate_coors
      ),
  }
  arguments_dict.update(kwargs)
  keys = list(arguments_dict.keys())
  for k in keys:
    arguments_dict[k.replace('_', '')] = arguments_dict.pop(k)
  arguments_str = _dict_to_path_name(arguments_dict)
  arguments_str = 'gr_spatial_const_mc_' + arguments_str
  return arguments_str


def _calculate_and_save_gr_spatial(
    estimate_coors,
    catalog,
    cache_folder = CACHE_PATH,
    **kwargs,
):
  """Executes spatial GR calculation and saves the result to cache folder."""
  completeness_vec, beta_vec = catalog_analysis.gr_spatial_beta_const_mc(
      estimate_coors=estimate_coors,
      catalog=catalog,
      **kwargs,
  )
  _replace_none_with_nan(completeness_vec)
  _replace_none_with_nan(beta_vec)

  arguments_str = _gr_spatial_vars_to_str(
      estimate_coors,
      catalog,
      **kwargs,
  )

  cache_folder_path = _cache_file_path(arguments_str, cache_folder=cache_folder)
  dir_path = pathlib.Path(cache_folder_path)
  dir_path.mkdir(exist_ok=True)

  completeness_path = os.path.join(cache_folder_path, COMPLETENESS_VEC_TXT)
  if os.path.exists(completeness_path):
    gfile.remove(completeness_path)
  with open(completeness_path, 'wt') as f:
    np.savetxt(f, completeness_vec)

  beta_path = os.path.join(cache_folder_path, BETA_VEC_TXT)
  if os.path.exists(beta_path):
    gfile.remove(beta_path)
  with open(beta_path, 'wt') as f:
    np.savetxt(f, beta_vec)


def _load_gr_spatial_result_from_cache(
    estimate_coors,
    catalog,
    cache_folder=CACHE_PATH,
    **kwargs,
):
  """Loads saved results of gr_spatial_beta_const_mc from cache folder."""
  arguments_str = _gr_spatial_vars_to_str(
      estimate_coors,
      catalog,
      **kwargs,
  )

  with open(
      os.path.join(
          _cache_file_path(arguments_str, cache_folder=cache_folder),
          COMPLETENESS_VEC_TXT,
      ),
      'r',
  ) as f:
    completeness_vec = np.loadtxt(f)
  with open(
      os.path.join(
          _cache_file_path(arguments_str, cache_folder=cache_folder),
          BETA_VEC_TXT,
      ),
      'r',
  ) as f:
    beta_vec = np.loadtxt(f)
  return completeness_vec, beta_vec


def get_gr_spatial(
    estimate_coors,
    catalog,
    completeness_magnitude = None,
    mc_calc_method = 'MAXC',
    grid_spacing = 0.1,
    smoothing_distance = 30 * KM_TO_DEG,
    discard_few_event_locations = 200,
    estimate_by_vicinity = True,
    force_recalculate = False,
    cache_folder = CACHE_PATH,
):
  """Gets cached results of gr_spatial_beta_const_mc or calculates if needed."""
  gr_spatial_kwargs = {
      'completeness_magnitude': completeness_magnitude,
      'mc_calc_method': mc_calc_method,
      'grid_spacing': grid_spacing,
      'smoothing_distance': smoothing_distance,
      'discard_few_event_locations': discard_few_event_locations,
      'estimate_by_vicinity': estimate_by_vicinity,
  }
  arguments_str = _gr_spatial_vars_to_str(
      estimate_coors,
      catalog,
      **gr_spatial_kwargs,
  )
  cache_exists = _folder_exists_in_dir(arguments_str, parent_path=cache_folder)
  if (not cache_exists) | force_recalculate:
    _calculate_and_save_gr_spatial(
        estimate_coors,
        catalog,
        cache_folder=cache_folder,
        **gr_spatial_kwargs,
    )
  return _load_gr_spatial_result_from_cache(
      estimate_coors,
      catalog,
      cache_folder=cache_folder,
      **gr_spatial_kwargs,
  )
