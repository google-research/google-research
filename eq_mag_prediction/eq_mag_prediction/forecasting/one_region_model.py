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

"""Methods that build features and encoders for a single region."""

import collections
import logging
import os
import pickle
from typing import Dict, Optional, Sequence, Union
import gin
import joblib
import numpy as np
import tensorflow as tf

from eq_mag_prediction.forecasting import data_sources
from eq_mag_prediction.forecasting import encoders
from eq_mag_prediction.forecasting import external_configurations

from eq_mag_prediction.forecasting import head_models
from eq_mag_prediction.forecasting import training_examples
from eq_mag_prediction.utilities import catalog_processing
from eq_mag_prediction.utilities import data_utils
from eq_mag_prediction.utilities import ml_utils
from tensorflow.io import gfile


FeaturesAndModels = collections.namedtuple(
    'FeaturesAndModels',
    [
        'features',
        'location_features',
        'spatially_independent_models',
        'spatially_dependent_models',
    ],
)

CACHED_FEATURES_DIR = os.path.join(
    os.path.dirname(__file__), '../..', 'results/cached_features'
)

CACHING_SUFFIXES = [
    '_train',
    '_validation',
    '_test',
    '_scalers',
    '_location_train',
    '_location_validation',
    '_location_test',
    '_location_scalers',
]


def _extend_filename(path, extension):
  dirname, basename = os.path.split(path)
  return os.path.join(dirname, basename + extension)


def _encoder_saving_path(domain, encoder, cache_dir):
  return _extend_filename(
      _features_full_file(domain, encoder, cache_dir),
      '.enc',
  )


def encoder_domain_id(
    domain,
    encoder,
):
  encoder_identifier = encoder.uuid()
  build_features_identifier = encoder.build_features_uuid()
  domain_id = domain.domain_examples_uuid()
  return f'encoder_{encoder_identifier}_build_features_{build_features_identifier}_domain_{domain_id}'


def id_files_exist_in_dir(
    file_name, parent_path = CACHED_FEATURES_DIR
):
  """Checks if file name of id w/ suffixes exists in path."""

  files_in_folder = os.listdir(parent_path)
  files_exist = [
      (file_name + suffix in files_in_folder) for suffix in CACHING_SUFFIXES
  ]
  files_exist.append(
      any([
          name.endswith('.enc') & name.startswith(file_name)
          for name in files_in_folder
      ])
  )
  relevant_suffixes = CACHING_SUFFIXES.copy()
  relevant_suffixes.append('.enc')
  return relevant_suffixes, files_exist


def _create_suffix_logical(encoder, suffixes_list, files_exist):
  if encoder.is_location_dependent:
    suffix_logical = np.array(
        [(not suffix.startswith('_location')) for suffix in suffixes_list]
    )
  else:
    suffix_logical = np.full_like(files_exist, True, dtype=bool)
  return suffix_logical


def _encoder_domain_files_exists_in_dir(
    file_name,
    encoder,
    parent_path = CACHED_FEATURES_DIR,
):
  """Returns a boolean indicating if file name of id w/ suffixes exists in path."""
  relevant_suffixes, files_exist = id_files_exist_in_dir(file_name, parent_path)
  files_exist = np.array(files_exist)
  suffix_logical = _create_suffix_logical(
      encoder, relevant_suffixes, files_exist
  )
  return np.all(files_exist[suffix_logical])


###############################################################################
########### Functions for building and caching feature and encoders  ###########
###############################################################################


def _features_full_file(domain, encoder, cache_dir):
  """Returns a path to save/look for the features, considering domain, encoder and cache_dir."""
  relevant_id = encoder_domain_id(domain, encoder)
  return os.path.join(cache_dir, relevant_id)


def get_features_scaler_encoder(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
    force_recalculate = False,
):
  """Returns the loaded features if exists, otherwise build features, saves, and returns."""
  relevant_id = encoder_domain_id(domain, encoder)
  cache_exists = _encoder_domain_files_exists_in_dir(
      relevant_id, encoder, parent_path=cache_dir
  )
  if (not cache_exists) | force_recalculate:
    build_and_cache_features_scaler_encoder(domain, encoder, cache_dir)
  return (
      _load_features(domain, encoder, cache_dir),
      _load_scaler(domain, encoder, cache_dir),
      _load_encoder(domain, encoder, cache_dir),
  )


def _load_features(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
):
  """Loads features from the dedicated unique path."""
  full_file_path = _features_full_file(domain, encoder, cache_dir)
  with open(_extend_filename(full_file_path, '_train'), 'rb') as f:
    flat_train = np.load(f)
  with open(_extend_filename(full_file_path, '_validation'), 'rb') as f:
    flat_validation = np.load(f)
  with open(_extend_filename(full_file_path, '_test'), 'rb') as f:
    flat_test = np.load(f)
  return (
      flat_train,
      flat_validation,
      flat_test,
  )


def _load_scaler(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
):
  """Loads scaler from the dedicated unique path."""
  full_file_path = _extend_filename(
      _features_full_file(domain, encoder, cache_dir), '_scalers'
  )
  with open(full_file_path, 'rb') as f:
    scaler = joblib.load(f)
  return scaler


def _load_encoder(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
):
  """Loads scaler from the dedicated unique path."""
  encoder_path = _encoder_saving_path(domain, encoder, cache_dir)
  with open(encoder_path, 'rb') as fin:
    loaded_encoder = pickle.load(fin)
  return loaded_encoder


def _load_features_and_encoder_save_scaler(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
    scaler_saving_dir = None,
):
  """Loads features and scaler, and saves scaler, from the dedicated unique path."""
  if scaler_saving_dir is not None:
    scaler = _load_scaler(domain, encoder, cache_dir)
    scaler.store(os.path.join(scaler_saving_dir, f'scalers/{encoder.name}'))
  return (
      _load_features(domain, encoder, cache_dir),
      _load_encoder(domain, encoder, cache_dir),
  )


def _get_features_and_encoder_save_scaler(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
    scaler_saving_dir = None,
):
  """Loads features and scaler if existing in cache, attempts to build of not, saves scaler.

  This function is meant for use in load_features_and_construct_models.

  Args:
    domain: Domain instance to be used.
    encoder: Encoder instance for creating the features.
    cache_dir: Directory for saving the computed features.
    scaler_saving_dir: Directory for re-saving the scalers. During the training
      process the scalers will be saved here to be read later while data
      analysis.

  Returns:
    tuple of features.
  """
  loaded_features, scaler, loaded_encoder = get_features_scaler_encoder(
      domain, encoder, cache_dir, force_recalculate=False
  )
  if scaler_saving_dir is not None:
    scaler.store(os.path.join(scaler_saving_dir, f'scalers/{encoder.name}'))
  return loaded_features, loaded_encoder


def _build_features_and_scaler(
    domain,
    encoder,
):
  """Builds, normalizes, and reshapes the features."""
  train_features = encoder.build_features(domain.train_examples)
  validation_features = encoder.build_features(domain.validation_examples)
  test_features = encoder.build_features(domain.test_examples)

  scaler = ml_utils.StandardScaler(encoder.scaling_axes)
  normalized_train = scaler.fit_transform(train_features)
  normalized_validation = scaler.transform(validation_features)
  normalized_test = scaler.transform(test_features)

  flat_train = encoder.flatten_features(normalized_train)
  flat_validation = encoder.flatten_features(normalized_validation)
  flat_test = encoder.flatten_features(normalized_test)

  return (flat_train, flat_validation, flat_test), scaler


@gin.configurable
def build_and_cache_features_scaler_encoder(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
):
  """Builds, normalizes, and reshapes the features. Saves scalers to folder."""
  full_file_path = _features_full_file(domain, encoder, cache_dir)
  (flat_train, flat_validation, flat_test), scaler = _build_features_and_scaler(
      domain,
      encoder,
  )
  with open(_extend_filename(full_file_path, '_train'), 'wb') as f:
    np.save(f, flat_train)
  with open(_extend_filename(full_file_path, '_validation'), 'wb') as f:
    np.save(f, flat_validation)
  with open(_extend_filename(full_file_path, '_test'), 'wb') as f:
    np.save(f, flat_test)
  scalers_path = _extend_filename(full_file_path, '_scalers')
  scaler.store(scalers_path)
  encoder.store(_encoder_saving_path(domain, encoder, cache_dir))


###############################################################################
###### Functions for building and caching LOCATION feature and encoders  ######
###############################################################################


def _location_features_full_file(domain, encoder, cache_dir):
  relevant_id = encoder_domain_id(domain, encoder)
  relevant_id += '_location'
  return os.path.join(cache_dir, relevant_id)


def get_location_features_scaler_encoder(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
    force_recalculate = False,
):
  """Returns location features, scaler, encoder cached. Computes if required."""
  relevant_id = encoder_domain_id(domain, encoder)
  cache_exists = _encoder_domain_files_exists_in_dir(
      relevant_id, encoder, parent_path=cache_dir
  )
  if (not cache_exists) | force_recalculate:
    build_and_cache_location_features_scaler_encoder(domain, encoder, cache_dir)
  return (
      _load_location_features(domain, encoder, cache_dir),
      _load_location_scaler(domain, encoder, cache_dir),
      _load_encoder(domain, encoder, cache_dir),
  )


def _load_location_features(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
):
  """Loads and returns location features from cache."""
  full_file_path = _location_features_full_file(domain, encoder, cache_dir)
  with open(_extend_filename(full_file_path, '_train'), 'rb') as f:
    normalized_train = np.load(f)
  with open(_extend_filename(full_file_path, '_validation'), 'rb') as f:
    normalized_validation = np.load(f)
  with open(_extend_filename(full_file_path, '_test'), 'rb') as f:
    normalized_test = np.load(f)
  return (
      normalized_train,
      normalized_validation,
      normalized_test,
  )


def _load_location_scaler(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
):
  full_file_path = _extend_filename(
      _location_features_full_file(domain, encoder, cache_dir), '_scalers'
  )
  with open(full_file_path, 'rb') as f:
    scaler = joblib.load(f)
  return scaler


def _load_location_features_and_encoder_save_scaler(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
    scaler_saving_dir = None,
):
  """Loads location features and encoder from cache. Re-saves scaler."""
  if scaler_saving_dir is None:
    scaler = _load_location_scaler(domain, encoder, cache_dir)
    scaler.store(
        os.path.join(scaler_saving_dir, f'location_scalers/{encoder.name}')
    )
  return (
      _load_location_features(domain, encoder, cache_dir),
      _load_encoder(domain, encoder, cache_dir),
  )


def _get_location_features_and_encoder_save_scaler(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
    scaler_saving_dir = None,
):
  """Returns location features and encoder cached. Computes if needed."""
  loaded_features, scaler, loaded_encoder = (
      get_location_features_scaler_encoder(
          domain,
          encoder,
          cache_dir,
          force_recalculate=False,
      )
  )
  if scaler_saving_dir is None:
    scaler.store(
        os.path.join(scaler_saving_dir, f'location_scalers/{encoder.name}')
    )
  return loaded_features, loaded_encoder


def _build_location_features_and_scaler(
    domain,
    encoder,
):
  """Builds training, validation and test features."""
  train_features = encoder.build_location_features(
      domain.train_examples,
      total_pixels=0,
      first_pixel_index=0,
      total_regions=0,
      region_index=0,
  )
  validation_features = encoder.build_location_features(
      domain.validation_examples,
      total_pixels=0,
      first_pixel_index=0,
      total_regions=0,
      region_index=0,
  )
  test_features = encoder.build_location_features(
      domain.test_examples,
      total_pixels=0,
      first_pixel_index=0,
      total_regions=0,
      region_index=0,
  )

  scaler = ml_utils.StandardScaler(encoder.location_scaling_axes)
  normalized_train = scaler.fit_transform(train_features)
  normalized_validation = scaler.transform(validation_features)
  normalized_test = scaler.transform(test_features)

  normalized_train = encoder.flatten_location_features(normalized_train)
  normalized_validation = encoder.flatten_location_features(
      normalized_validation
  )
  normalized_test = encoder.flatten_location_features(normalized_test)

  return (normalized_train, normalized_validation, normalized_test), scaler


@gin.configurable
def build_and_cache_location_features_scaler_encoder(
    domain,
    encoder,
    cache_dir = CACHED_FEATURES_DIR,
):
  """Builds location featres, caches them with scaler and encoder."""
  (normalized_train, normalized_validation, normalized_test), scaler = (
      _build_location_features_and_scaler(domain, encoder)
  )
  full_file_path = _location_features_full_file(domain, encoder, cache_dir)
  with open(_extend_filename(full_file_path, '_train'), 'wb') as f:
    np.save(f, normalized_train)
  with open(_extend_filename(full_file_path, '_validation'), 'wb') as f:
    np.save(f, normalized_validation)
  with open(_extend_filename(full_file_path, '_test'), 'wb') as f:
    np.save(f, normalized_test)
  scalers_path = _extend_filename(full_file_path, '_scalers')
  scaler.store(scalers_path)
  encoder.store(_encoder_saving_path(domain, encoder, cache_dir))


def _try_storing_object(
    obj, output_dir, name
):
  if obj is None:
    return
  path = os.path.join(output_dir, name)
  if os.path.exists(path):
    return
  with open(path, 'wb') as f:
    joblib.dump(obj, f)


def store_everything_in_folder(
    output_dir,
    model = None,
    history = None,
    encoders_dict = None,
    train_features = None,
    validation_features = None,
    test_features = None,
    loss_function = None,
    domain = None,
    epoch_number = None,
):
  """Store all of the training artifacts."""
  if model is not None:
    model_name = 'model' if epoch_number is None else f'model_{epoch_number}'
    ml_utils.save_model_to_directory(
        model, os.path.join(output_dir, model_name)
    )
  if history is not None:
    ml_utils.save_history_to_folder(history.history, output_dir)

  if encoders_dict is not None:
    for name, encoder in encoders_dict.items():
      encoder.store(os.path.join(output_dir, f'{name}.enc'))

  _try_storing_object(loss_function, output_dir, 'loss_function')
  _try_storing_object(domain, output_dir, 'domain')

  gin_target_path = os.path.join(output_dir, 'config.gin')
  os.makedirs(os.path.dirname(gin_target_path), exist_ok=True)
  with open(gin_target_path, 'w') as f:
    f.truncate(0)
    _ = f.write(gin.config_str())

  features_given = all([
      f is not None
      for f in [train_features, validation_features, test_features]
  ])
  if features_given and model is not None:
    for i, name in enumerate(['train', 'validation', 'test']):
      features = [train_features, validation_features, test_features][i]
      _try_storing_object(features, output_dir, f'{name}_features')

      forecast = model(features)
      if epoch_number is not None:
        forecast_path = os.path.join(
            output_dir, f'{name}_forecast_{epoch_number}'
        )
      else:
        forecast_path = os.path.join(output_dir, f'{name}_forecast')
      with open(forecast_path, 'wb') as f:
        np.save(f, forecast)


@gin.configurable
def build_encoders(
    domain = None,
    include_catalog_columns = True,
    include_recent_earthquakes = True,
    include_biggest_earthquakes = True,
    include_seismicity_grid = False,
    include_seismicity_rate = True,
    include_times = False,
):
  """A wrapper to choose which encoders we want to include in the model."""
  encoders_dict = {}
  if domain is not None:
    feature_catalog = domain.earthquakes_catalog
  else:
    feature_catalog = data_sources.feature_catalog()

  if include_catalog_columns:
    encoder = encoders.CatalogColumnsEncoder(feature_catalog)
    encoders_dict[encoder.name] = encoder
  if include_recent_earthquakes:
    if domain is not None:
      encoder = encoders.RecentEarthquakesEncoder(
          feature_catalog, domain.magnitude_threshold
      )
    else:
      encoder = encoders.RecentEarthquakesEncoder(feature_catalog)
    encoders_dict[encoder.name] = encoder
  if include_biggest_earthquakes:
    encoder = encoders.BiggestEarthquakesEncoder(feature_catalog)
    encoders_dict[encoder.name] = encoder
  if include_seismicity_grid:
    encoder = encoders.SeismicityGridEncoder(feature_catalog)
    encoders_dict[encoder.name] = encoder
  if include_seismicity_rate:
    encoder = encoders.SeismicityRateEncoder(feature_catalog)
    encoders_dict[encoder.name] = encoder
  if include_times:
    encoder = encoders.TimeEncoder()
    encoders_dict[encoder.name] = encoder
  return encoders_dict


@gin.configurable
def load_features_and_construct_models(
    domain,
    all_encoders,
    output_dir,
    cache_dir = CACHED_FEATURES_DIR,
):
  """Returns features and models for head model construction."""
  features = {}
  location_features = {}
  spatially_independent_models = {}
  spatially_dependent_models = {}
  for name, encoder in all_encoders.items():
    features[name], loaded_encoder = _load_features_and_encoder_save_scaler(
        domain,
        encoder,
        cache_dir,
        scaler_saving_dir=output_dir,
    )
    encoder = loaded_encoder
    if not encoder.is_location_dependent:
      location_features[name], loaded_encoder = (
          _load_location_features_and_encoder_save_scaler(
              domain,
              encoder,
              cache_dir,
              scaler_saving_dir=output_dir,
          )
      )
      encoder = loaded_encoder
      spatially_independent_models[name] = encoder.build_model()
    else:
      spatially_dependent_models[name] = encoder.build_model()
  return FeaturesAndModels(
      features,
      location_features,
      spatially_independent_models,
      spatially_dependent_models,
  )


@gin.configurable
def compute_and_cache_features_scaler_encoder(
    domain,
    all_encoders,
    cache_dir = CACHED_FEATURES_DIR,
    force_recalculate = False,
):
  """Returns features and models for head model construction."""
  for encoder in all_encoders.values():
    # build_and_cache_features_and_scaler(domain, encoder, cache_dir)
    _ = get_features_scaler_encoder(
        domain, encoder, cache_dir, force_recalculate
    )
    if not encoder.is_location_dependent:
      # build_and_cache_location_features_and_scaler(domain, encoder, cache_dir)
      _ = get_location_features_scaler_encoder(
          domain, encoder, cache_dir, force_recalculate
      )


def features_in_order(
    features_and_models, slice_index
):
  """Gets all features, in order, for a given slice (train/val/test)."""
  (
      features,
      location_features,
      spatially_independent_models,
      spatially_dependent_models,
  ) = features_and_models
  spatially_dependent_order, spatially_independent_order = (
      head_models.input_order(
          spatially_dependent_model_names=spatially_dependent_models.keys(),
          spatially_independent_model_names=spatially_independent_models.keys(),
      )
  )
  i = slice_index
  selected_features = [features[name][i] for name in spatially_dependent_order]
  selected_spatially_independent_features = [
      (features[name][i], location_features[name][i])
      for name in spatially_independent_order
  ]
  return selected_features + selected_spatially_independent_features
