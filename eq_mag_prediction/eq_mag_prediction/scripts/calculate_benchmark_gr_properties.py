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

"""Calculate and cache properties of several variations of the GR benchmark."""

import collections
import os
from typing import Callable, Sequence

from absl import app
from absl import flags
from absl import logging
import joblib
import numpy as np

from eq_mag_prediction.forecasting import training_examples
from eq_mag_prediction.utilities import catalog_analysis
from eq_mag_prediction.utilities import catalog_processing
from eq_mag_prediction.utilities.geometry import Point


CatalogDomain = training_examples.CatalogDomain
DAY_TO_SECONDS = 60 * 60 * 24
N_DAYS_TO_COMPUTE = [10, 100, 1000]
N_KDE_EVENTS = [50, 300]
N_EVENTS = 300  # as in Gulia & Weimer, nature 2019

DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), '../..', 'results/cached_benchmarks'
)
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', DEFAULT_CACHE_PATH, 'Features cache directory.'
)

_FORCE_RECALCULATE = flags.DEFINE_boolean(
    'force_recalculate',
    False,
    'Force recomputation of properties even if existing.',
)

_ESTIMATE_BY_VICINITY = flags.DEFINE_boolean(
    'estimate_by_vicinity',
    True,
    'Calculate the spatial beta variability by spatial approximation for a'
    ' faster run.',
)
_DOMAIN_PATH = flags.DEFINE_string('domain_path', None, 'Path to the domain.')

_COMPUTE_BENCHMARK = flags.DEFINE_string(
    'compute_benchmark',
    '',
    "Benchmarks to compute. Defaults to True. format: '"
    " benchmarkname1=True,benchmarkname2=False,benchmarkname3=False' ",
)


def _load_domain(
    domain_path = None,
):
  with open(domain_path, 'rb') as fin:
    return joblib.load(fin)


def create_timestamps_dict(
    domain,
):
  test_timestamps = np.array(list(domain.test_examples.keys()))
  validation_timestamps = np.array(list(domain.validation_examples.keys()))
  train_timestamps = np.array(list(domain.train_examples.keys()))
  return {
      'train': train_timestamps,
      'validation': validation_timestamps,
      'test': test_timestamps,
  }


def create_coordinates_dict(
    domain,
):
  examples_to_list = lambda examples: [
      (v[0][0].lng, v[0][0].lat) for v in examples
  ]
  test_coors = examples_to_list(domain.test_examples.values())
  validation_coors = examples_to_list(domain.validation_examples.values())
  train_coors = examples_to_list(domain.train_examples.values())
  return {
      'train': train_coors,
      'validation': validation_coors,
      'test': test_coors,
  }


def create_coordinates_dict(
    domain,
):
  examples_to_list = lambda examples: [
      (v[0][0].lng, v[0][0].lat) for v in examples
  ]
  test_coors = examples_to_list(domain.test_examples.values())
  validation_coors = examples_to_list(domain.validation_examples.values())
  train_coors = examples_to_list(domain.train_examples.values())
  return {
      'train': train_coors,
      'validation': validation_coors,
      'test': test_coors,
  }


def _estimate_beta(domain, set_name):
  return catalog_analysis.estimate_beta(
      getattr(
          training_examples.magnitude_prediction_labels(domain),
          f'{set_name}_labels',
      ),
      None,
      'BPOS',
  )


def gr_of_moving_window_n_events(
    timestamps,
    domain,
    n_events_to_consider = 300,
    n_above_complete = 10,
    m_minimal = -100,
    weight_on_past = 0.50,
    completeness_and_beta_calculator = catalog_analysis.estimate_completeness_and_beta_by_maximal_curvature,
):
  """Calculates the beta and completeness magnitude of a window of n events."""
  completeness_vec, beta_vec = catalog_processing.get_gr_moving_window_n_events(
      timestamps,
      domain.earthquakes_catalog,
      n_events_to_consider,
      m_minimal,
      n_above_complete,
      weight_on_past=weight_on_past,
      completeness_and_beta_calculator=completeness_and_beta_calculator,
      force_recalculate=_FORCE_RECALCULATE.value,
      cache_folder=_CACHE_DIR.value,
  )
  return completeness_vec, beta_vec


def _constant_completeness_and_beta_function(
    mag_thresh,
):
  def _const_completeness(magnitudes):
    return catalog_analysis.estimate_completeness_and_beta_with_constant_mc(
        magnitudes, mag_thresh
    )

  return _const_completeness


def _constant_completeness_function(
    mag_thresh,
):
  return lambda magnitudes: catalog_analysis.return_constant_completeness(
      magnitudes, mag_thresh
  )


def _gr_of_n_events(
    timestamps_dict,
    domain,
    set_name,
    tense,
    mag_thresh,
):
  """Calculates the beta and completeness magnitude of a window of N_EVENTS events."""
  if tense == 'past':
    weight_on_past = 1
  elif tense == 'present':
    weight_on_past = 0.5
  else:
    assert False, 'Unknown tense'

  if mag_thresh is None:
    completeness_and_beta_calculator = (
        catalog_analysis.estimate_completeness_and_beta_by_maximal_curvature
    )
    mc_type = 'fitted'
  else:
    completeness_and_beta_calculator = _constant_completeness_and_beta_function(
        mag_thresh
    )
    mc_type = 'constant'
  data_name = f'n{N_EVENTS}_{tense}_events_{mc_type}_mc_{set_name}'
  completeness_vec, beta_vec = catalog_processing.get_gr_moving_window_n_events(
      timestamps_dict[set_name],
      domain.earthquakes_catalog,
      n_events=N_EVENTS,
      m_minimal=-100,
      n_above_complete=10,
      weight_on_past=weight_on_past,
      completeness_and_beta_calculator=completeness_and_beta_calculator,
      force_recalculate=_FORCE_RECALCULATE.value,
      cache_folder=_CACHE_DIR.value,
  )
  return completeness_vec, beta_vec, data_name


def _kde_of_n_events(
    timestamps_dict,
    domain,
    set_name,
    mag_thresh,
    n_events,
):
  """Calculates the beta and completeness magnitude of a window of n events."""
  weight_on_past = 1

  if mag_thresh is None:
    completeness_calculator = (
        catalog_analysis.estimate_completeness_by_maximal_curvature
    )
    mc_type = 'fitted'
  else:
    completeness_calculator = _constant_completeness_function(mag_thresh)
    mc_type = 'constant'
  data_name = f'n{n_events}_past_events_kde_{mc_type}_mc_{set_name}'
  completeness_vec, kde_list = (
      catalog_processing.get_kde_moving_window_n_events(
          timestamps_dict[set_name],
          domain.earthquakes_catalog,
          n_events=n_events,
          m_minimal=-100,
          n_above_complete=10,
          weight_on_past=weight_on_past,
          completeness_calculator=completeness_calculator,
          force_recalculate=_FORCE_RECALCULATE.value,
          cache_folder=_CACHE_DIR.value,
      )
  )
  return completeness_vec, kde_list, data_name


def _kde_of_n_events(
    timestamps_dict,
    domain,
    set_name,
    mag_thresh,
    n_events,
):
  """Calculates the beta and completeness magnitude of a window of n events."""
  weight_on_past = 1

  if mag_thresh is None:
    completeness_calculator = (
        catalog_analysis.estimate_completeness_by_maximal_curvature
    )
    mc_type = 'fitted'
  else:
    completeness_calculator = _constant_completeness_function(mag_thresh)
    mc_type = 'constant'
  data_name = f'n{n_events}_past_events_kde_{mc_type}_mc_{set_name}'
  completeness_vec, kde_list = (
      catalog_processing.get_kde_moving_window_n_events(
          timestamps_dict[set_name],
          domain.earthquakes_catalog,
          n_events=n_events,
          m_minimal=-100,
          n_above_complete=10,
          weight_on_past=weight_on_past,
          completeness_calculator=completeness_calculator,
          force_recalculate=_FORCE_RECALCULATE.value,
          cache_folder=_CACHE_DIR.value,
      )
  )
  return completeness_vec, kde_list, data_name


def _gr_of_constant_time(
    timestamps_dict,
    domain,
    set_name,
    n_days,
    beta_of_train_set,
    mag_thresh,
):
  """Calculates the beta and completeness magnitude of a window of constant time."""
  if mag_thresh is None:
    completeness_calculator = (
        catalog_analysis.estimate_completeness_by_maximal_curvature
    )
    mc_type = 'fitted'
  else:
    completeness_calculator = None
    mc_type = 'constant'

  data_name = f'gr_last_{n_days}_days_{mc_type}_mc_likelihood_{set_name}'
  completeness_vec, beta_vec = (
      catalog_processing.get_gr_moving_window_constant_time(
          estimate_times=timestamps_dict[set_name],
          catalog=domain.earthquakes_catalog,
          window_time=n_days * DAY_TO_SECONDS,
          m_minimal=-100,
          n_above_complete=1,
          weight_on_past=1,
          default_beta=beta_of_train_set,
          default_mc=mag_thresh,
          completeness_calculator=completeness_calculator,
          force_recalculate=_FORCE_RECALCULATE.value,
          cache_folder=_CACHE_DIR.value,
      )
  )
  return completeness_vec, beta_vec, data_name


def _local_spatial_gr(
    coordinates_dict,
    domain,
    set_name,
    estimation_set,
):
  """Calculates the beta and completeness magnitude of a spatial grid."""
  data_name = f'gr_spatial_on_{estimation_set}_likelihood_{set_name}'

  if estimation_set == 'all':
    estimation_catalog = domain.earthquakes_catalog
  else:
    estimation_examples = getattr(domain, f'{estimation_set}_examples')
    estimation_times = np.array(list(estimation_examples.keys()))
    min_estimation_times = estimation_times.min()
    max_estimation_times = estimation_times.max()
    time_slice = slice(
        np.searchsorted(
            domain.earthquakes_catalog.time.values,
            min_estimation_times,
            side='right',
        ),
        np.searchsorted(
            domain.earthquakes_catalog.time.values,
            max_estimation_times,
            side='right',
        ),
    )
    estimation_catalog = domain.earthquakes_catalog[time_slice]
  completeness_vec, beta_vec = catalog_processing.get_gr_spatial(
      estimate_coors=coordinates_dict[set_name],
      catalog=estimation_catalog,
      discard_few_event_locations=150,
      estimate_by_vicinity=_ESTIMATE_BY_VICINITY.value,
      force_recalculate=_FORCE_RECALCULATE.value,
      cache_folder=_CACHE_DIR.value,
  )
  return completeness_vec, beta_vec, data_name


def _get_compute_benchmark_dict():
  """Returns a dict of benchmark properties."""
  compute_benchmark = collections.defaultdict(lambda: True)
  benchmark_str = _COMPUTE_BENCHMARK.value
  if bool(benchmark_str):
    split_list = benchmark_str.replace(' ', '').split(',')
    if 'false' in [s.lower() for s in split_list]:
      compute_benchmark = collections.defaultdict(lambda: False)
      if 'False' in split_list:
        split_list.remove('False')
      if 'false' in split_list:
        split_list.remove('false')
    split_pairs = [s.split('=') for s in split_list]
    compute_benchmark.update(
        dict([(s[0], s[1].lower() == 'true') for s in split_pairs])
    )
  return compute_benchmark


def _local_spatial_gr(
    coordinates_dict,
    domain,
    set_name,
    estimation_set,
):
  """Calculates the beta and completeness magnitude of a spatial grid."""
  data_name = f'gr_spatial_on_{estimation_set}_likelihood_{set_name}'

  if estimation_set == 'all':
    estimation_catalog = domain.earthquakes_catalog
  else:
    estimation_examples = getattr(domain, f'{estimation_set}_examples')
    estimation_times = np.array(list(estimation_examples.keys()))
    min_estimation_times = estimation_times.min()
    max_estimation_times = estimation_times.max()
    time_slice = slice(
        np.searchsorted(
            domain.earthquakes_catalog.time.values,
            min_estimation_times,
            side='right',
        ),
        np.searchsorted(
            domain.earthquakes_catalog.time.values,
            max_estimation_times,
            side='right',
        ),
    )
    estimation_catalog = domain.earthquakes_catalog[time_slice]
  completeness_vec, beta_vec = catalog_processing.get_gr_spatial(
      estimate_coors=coordinates_dict[set_name],
      catalog=estimation_catalog,
      discard_few_event_locations=150,
      estimate_by_vicinity=_ESTIMATE_BY_VICINITY.value,
      force_recalculate=_FORCE_RECALCULATE.value,
      cache_folder=_CACHE_DIR.value,
  )
  return completeness_vec, beta_vec, data_name


def _get_compute_benchmark_dict():
  """Returns a dict of benchmark properties."""
  compute_benchmark = collections.defaultdict(lambda: True)
  benchmark_str = _COMPUTE_BENCHMARK.value
  if bool(benchmark_str):
    split_list = benchmark_str.replace(' ', '').split(',')
    if 'false' in [s.lower() for s in split_list]:
      compute_benchmark = collections.defaultdict(lambda: False)
      if 'False' in split_list:
        split_list.remove('False')
      if 'false' in split_list:
        split_list.remove('false')
    split_pairs = [s.split('=') for s in split_list]
    compute_benchmark.update(
        dict([(s[0], s[1].lower() == 'true') for s in split_pairs])
    )
  return compute_benchmark


def compute_and_assign_set_benchmarks(
    gr_models_beta,
    gr_models_mc,
    set_name,
    domain,
    timestamps_dict,
    coordinates_dict,
    beta_of_train_set,
    mag_thresh,
    compute_benchmark = None,
):
  """Computes GR benchmark properties and assigns them to relevant holding dictionaries."""
  logging.info('set: %s', set_name)

  if compute_benchmark is None:
    compute_benchmark = _get_compute_benchmark_dict()
  else:
    dd = collections.defaultdict(lambda: True)
    dd.update(compute_benchmark)
    compute_benchmark = dd


  if compute_benchmark is None:
    compute_benchmark = _get_compute_benchmark_dict()
  else:
    dd = collections.defaultdict(lambda: True)
    dd.update(compute_benchmark)
    compute_benchmark = dd

  #  train GR
  gr_models_beta[f'train_gr_likelihood_{set_name}'] = beta_of_train_set
  gr_models_mc[f'train_gr_likelihood_{set_name}'] = mag_thresh
  #  test GR
  gr_models_beta[f'test_gr_likelihood_{set_name}'] = _estimate_beta(
      domain, set_name
  )
  gr_models_mc[f'test_gr_likelihood_{set_name}'] = mag_thresh

  # Past time window:
  for mc in (mag_thresh, None):
    for n_days in N_DAYS_TO_COMPUTE:
      if compute_benchmark[f'{n_days}_past_days']:
        completeness_vec, beta_vec, data_name = _gr_of_constant_time(
            timestamps_dict, domain, set_name, n_days, beta_of_train_set, mc
        )
        gr_models_beta[data_name] = beta_vec
        gr_models_mc[data_name] = completeness_vec
        logging.info('Done with %s', data_name)
      if compute_benchmark[f'{n_days}_past_days']:
        completeness_vec, beta_vec, data_name = _gr_of_constant_time(
            timestamps_dict, domain, set_name, n_days, beta_of_train_set, mc
        )
        gr_models_beta[data_name] = beta_vec
        gr_models_mc[data_name] = completeness_vec
        logging.info('Done with %s', data_name)

  # N_EVENTS events window:
  for mc in (mag_thresh, None):
    for tense in ('past', 'present'):
      if compute_benchmark[f'{tense}_events']:
        completeness_vec, beta_vec, data_name = _gr_of_n_events(
            timestamps_dict, domain, set_name, tense, mc
        )
        gr_models_beta[data_name] = beta_vec
        gr_models_mc[data_name] = completeness_vec
        logging.info('Done with %s', data_name)

  if compute_benchmark['spatial_gr']:
    for estimation_set in ('all', 'train', 'test'):
      completeness_vec, beta_vec, data_name = _local_spatial_gr(
          coordinates_dict, domain, set_name, estimation_set

      )
      gr_models_beta[data_name] = beta_vec
      gr_models_mc[data_name] = completeness_vec
      logging.info('Done with %s', data_name)

  if compute_benchmark['n_past_events_kde']:
    for n_events in N_KDE_EVENTS:
      completeness_vec, kde_list, data_name = _kde_of_n_events(
          timestamps_dict, domain, set_name, mag_thresh, n_events

      )
      gr_models_beta[data_name] = kde_list
      gr_models_mc[data_name] = completeness_vec
      logging.info('Done with %s', data_name)

  return gr_models_beta, gr_models_mc


def compute_and_assign_benchmarks_all_sets(
    domain,
    timestamps_dict,
    coordinates_dict,
    beta_of_train_set,
    mag_thresh,
    compute_benchmark = None,
):
  """Iterate on train validation and test set to compute GR benchmark properties."""
  gr_models_beta, gr_models_mc = {}, {}
  for set_name in ['train', 'validation', 'test']:
    gr_models_beta, gr_models_mc = compute_and_assign_set_benchmarks(
        gr_models_beta,
        gr_models_mc,
        set_name,
        domain,
        timestamps_dict,
        coordinates_dict,
        beta_of_train_set,
        mag_thresh,
        compute_benchmark,
    )
  return gr_models_beta, gr_models_mc


def main(_):
  domain = _load_domain(
      domain_path=_DOMAIN_PATH.value,
  )
  mag_thresh = domain.magnitude_threshold
  beta_of_train_set = _estimate_beta(domain, 'train')
  assert beta_of_train_set is not None
  timestamps_dict = create_timestamps_dict(domain)
  coordinates_dict = create_coordinates_dict(domain)
  _ = compute_and_assign_benchmarks_all_sets(
      domain,
      timestamps_dict,
      coordinates_dict,
      beta_of_train_set,
      mag_thresh,
  )


if __name__ == '__main__':
  app.run(main)
