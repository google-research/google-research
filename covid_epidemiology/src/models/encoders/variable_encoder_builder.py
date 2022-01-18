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

# Lint as: python3
"""Constructs a variable encoder based on an encoder spec."""

from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from covid_epidemiology.src.models.encoders import gam_encoder
from covid_epidemiology.src.models.encoders import variable_encoders
from covid_epidemiology.src.models.shared import typedefs


def encoder_from_encoder_spec(
    encoder_spec,
    chosen_locations,
    num_known_timesteps,
    forecast_window_size,
    output_window_size,
    static_features = None,
    static_overrides = None,
    covariates = None,
    forecasted_covariates = None,
    covariate_overrides = None,
    ts_categorical_features = None,
    random_seed = 0,
    static_scalers = None,
    ts_scalers = None,
    trainable = True,
):
  """Returns a `FeatureEncoder` built as specified in the `encoder_spec`."""
  encoder_kwargs = encoder_spec.encoder_kwargs

  if encoder_spec.encoder_type == "gam":
    gam_kwargs = {}
    for kwarg in encoder_kwargs:
      if kwarg == "link_fn":
        gam_kwargs["link_fn"] = encoder_kwargs["link_fn"]
      elif kwarg == "distribution":
        gam_kwargs["distribution"] = encoder_kwargs["distribution"]
      elif kwarg == "initial_bias":
        gam_kwargs["initial_bias"] = encoder_kwargs["initial_bias"]
      elif kwarg == "location_dependent_bias":
        gam_kwargs["location_dependent_bias"] = encoder_kwargs[
            "location_dependent_bias"]
      elif kwarg == "lower_bound":
        gam_kwargs["lower_bound"] = encoder_kwargs["lower_bound"]
      elif kwarg == "upper_bound":
        gam_kwargs["upper_bound"] = encoder_kwargs["upper_bound"]
      elif kwarg == "use_fixed_covariate_mask":
        gam_kwargs["use_fixed_covariate_mask"] = encoder_kwargs[
            "use_fixed_covariate_mask"]
      else:
        raise ValueError(f"Unexpected kwarg: {kwarg} passed to encoder of type "
                         f"{encoder_spec.encoder_type}")
    return gam_encoder.GamEncoder(
        chosen_locations,
        num_known_timesteps,
        forecast_window_size=forecast_window_size,
        output_window_size=output_window_size,
        static_features=static_features,
        static_scalers=static_scalers,
        static_overrides=static_overrides,
        covariates=covariates,
        ts_scalers=ts_scalers,
        forecasted_covariates=forecasted_covariates,
        covariate_overrides=covariate_overrides,
        static_feature_specs=encoder_spec.static_feature_specs,
        covariate_feature_specs=encoder_spec.covariate_feature_specs,
        ts_categorical_features=ts_categorical_features,
        covariate_feature_time_offset=encoder_spec
        .covariate_feature_time_offset,
        covariate_feature_window=encoder_spec.covariate_feature_window,
        random_seed=random_seed,
        name=encoder_spec.encoder_name,
        trainable=trainable,
        **gam_kwargs)

  elif encoder_spec.encoder_type == "static":
    return variable_encoders.StaticEncoder()

  elif encoder_spec.encoder_type == "passthrough":
    return variable_encoders.PassThroughEncoder(
        chosen_locations,
        num_known_timesteps,
        forecast_window_size,
        covariates=covariates,
        forecasted_covariates=forecasted_covariates,
        covariate_overrides=covariate_overrides,
        covariate_feature_specs=encoder_spec.covariate_feature_specs,
        ts_categorical_features=ts_categorical_features,
        name=encoder_spec.encoder_name)

  elif encoder_spec.encoder_type == "vaccine":
    return variable_encoders.VaccineEncoder(
        chosen_locations,
        num_known_timesteps,
        forecast_window_size,
        covariates=covariates,
        forecasted_covariates=forecasted_covariates,
        covariate_overrides=covariate_overrides,
        covariate_feature_specs=encoder_spec.covariate_feature_specs,
        ts_categorical_features=ts_categorical_features,
        name=encoder_spec.encoder_name,
        vaccine_type=encoder_spec.vaccine_type)

  else:
    raise ValueError(f"encoder_spec passed in with invalid encoder_type: "
                     f"{encoder_spec.encoder_type}")
