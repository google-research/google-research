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

"""This module contains methods that extract forecasts from simulated catalogs.

We have a couple of simulators that store synthetic catalogs in SSTables. This
module contains methods that can read these tables and extract forecasts - such
as the distribution of the highest earthquake expected in some day, or the
probability of an earthquake occurring somewhere in the region.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from eq_mag_prediction.forecasting import encoders
from eq_mag_prediction.forecasting import head_models
from eq_mag_prediction.forecasting import one_region_model
from eq_mag_prediction.forecasting import training_examples
from eq_mag_prediction.utilities import geometry
from eq_mag_prediction.utilities import ml_utils


ExpectedEncoder = Union[
    encoders.LocationIndependentEncoder,
    encoders.RecentEarthquakesEncoder,
    encoders.SeismicityRateEncoder,
    encoders.BiggestEarthquakesEncoder,
    encoders.CatalogColumnsEncoder,
]


def _create_altered_features(
    altered_examples,
    altered_catalog,
    all_encoders,
    scalers,
    spatially_dependent_scalers,
):
  """Create features from input catalog using trained encoders and scalers."""
  alter_features = {}
  alter_spatial_feature = {}
  for enc_name, encoder in all_encoders.items():
    alt_feat = encoder.build_features(
        altered_examples, custom_catalog=altered_catalog
    )
    alter_feat_normalized = scalers[enc_name].transform(alt_feat)
    alter_feature_flat = encoder.flatten_features(alter_feat_normalized)
    alter_features[enc_name] = alter_feature_flat
    if not encoder.is_location_dependent:
      features_spatial = encoder.build_location_features(
          altered_examples,
          total_pixels=0,
          first_pixel_index=0,
          total_regions=0,
          region_index=0,
      )
      alter_spatial_feature[enc_name] = encoder.flatten_location_features(
          spatially_dependent_scalers[enc_name].transform(features_spatial)
      )

  spatially_dependent_order, spatially_independent_order = (
      head_models.input_order(
          spatially_dependent_model_names=alter_spatial_feature.keys(),
          spatially_independent_model_names=set(
              alter_features.keys()
          ).difference(set(alter_spatial_feature.keys())),
      )
  )

  selected_features = [
      alter_features[name] for name in spatially_independent_order
  ]
  selected_spatially_dependent_features = [
      (
          alter_features[name],
          alter_spatial_feature[name],
      )
      for name in spatially_dependent_order
  ]
  alter_all_features = selected_features + selected_spatially_dependent_features
  return alter_all_features


def magnitude_forecast_using_terminated_catalogs(
    loaded_model,
    starting_times_and_locations,
    catalog_domain,
    evaluation_times,
    scalers,
    spatially_dependent_scalers,
    eval_time_cutoff = np.inf,
):
  """Construct features and forecast using a trained model.

  Args:
    loaded_model: a trained magnitude prediction model.
    starting_times_and_locations: Sequence of tuples (time, coordinates). 'time'
      indicates the times at which the catalog will be terminated up to,
      'coordinates' indicates the location at which the model will be evaluated.
    catalog_domain: The catalog domain to be used to build encoders.
    evaluation_times: times at which the model will be evaluated at per entry
      from starting_times_and_locations. Locations will be taken from
      starting_times_and_locations.
    scalers: Dictionary of non spatial scalers fitted during model training.
    spatially_dependent_scalers: Dictionary of spatial scalers fitted during
      training.
    eval_time_cutoff: indicates how many seconds after the starting time of
      evaluation in each example shall be calculated.

  Returns:
    A list of prediction arrays per element in examples_as_starting_points. Each
    array is of size len(relevant_eval_times)x(model output example length)
  """
  all_encoders = one_region_model.build_encoders(catalog_domain)
  altered_forecasts_list = []
  for present_time, loc in starting_times_and_locations:
    relevant_eval_times = evaluation_times[
        (evaluation_times >= present_time)
        & (evaluation_times < (present_time + eval_time_cutoff))
    ]
    relevant_examples = dict(
        zip(
            relevant_eval_times,
            [[[loc]]] * len(relevant_eval_times),
        )
    )
    altered_catalog = catalog_domain.earthquakes_catalog[
        catalog_domain.earthquakes_catalog.time < present_time
    ]
    assert ~np.any(
        altered_catalog.time.values >= present_time
    ), 'catalog not filtered correctly'
    altered_features = _create_altered_features(
        relevant_examples,
        altered_catalog,
        all_encoders,
        scalers,
        spatially_dependent_scalers,
    )
    altered_forecasts = loaded_model.predict(altered_features, verbose=0)
    assert (
        altered_forecasts.shape[0] == relevant_eval_times.shape[0]
    ), 'prediction made for incorrect number of events'
    altered_forecasts = xr.DataArray(
        altered_forecasts,
        dims=['eval_time', 'model_parameter'],
        coords={'eval_time': relevant_eval_times},
    )
    altered_forecasts_list.append(altered_forecasts)

  assert len(altered_forecasts_list) == len(
      starting_times_and_locations
  ), 'not all times and locations have been evaluated'
  return altered_forecasts_list


def magnitude_forecast_on_spatial_surrounding(
    loaded_model,
    evaluation_time,
    evaluation_coordinates,
    catalog_domain,
    scalers,
    spatially_dependent_scalers,
):
  """Constructs features and forecasts using a trained model.

  Args:
    loaded_model: A trained magnitude prediction model.
    evaluation_time: The time at which the magnitude forecasting is evaluated.
    evaluation_coordinates: A list of N locations to evaluate the magnitude at.
    catalog_domain: The catalog domain to be used to build encoders.
    scalers: Dictionary of non spatial scalers fitted during model training.
    spatially_dependent_scalers: Dictionary of spatial scalers fitted during
      training.

  Returns:
    A list NxM arrays, where N is the number of cooredinates to be evaluated in
    evaluation_coordinates and M is the size of the model's output.
  """
  all_encoders = one_region_model.build_encoders(catalog_domain)
  n_model_parameters = loaded_model.layers[-1].output_shape[-1]
  altered_forecasts_array = xr.DataArray(
      np.empty((0, n_model_parameters)),
      dims=['coordinates', 'model_parameter'],
      coords={
          'coordinates': [],
          'model_parameter': list(range(n_model_parameters)),
      },
  )
  for loc in evaluation_coordinates:
    relevant_examples = {evaluation_time: [[loc]]}
    altered_catalog = catalog_domain.earthquakes_catalog[
        catalog_domain.earthquakes_catalog.time < evaluation_time
    ]
    altered_features = _create_altered_features(
        relevant_examples,
        altered_catalog,
        all_encoders,
        scalers,
        spatially_dependent_scalers,
    )
    altered_forecasts = xr.DataArray(
        loaded_model.predict(altered_features, verbose=0),
        dims=['coordinates', 'model_parameter'],
        coords={
            'model_parameter': list(range(n_model_parameters)),
            'coordinates': [str(loc)],
        },
    )

    altered_forecasts_array = xr.concat(
        (altered_forecasts_array, altered_forecasts), dim='coordinates'
    )

  assert altered_forecasts_array.shape == (
      len(evaluation_coordinates),
      n_model_parameters,
  )
  return altered_forecasts_array
