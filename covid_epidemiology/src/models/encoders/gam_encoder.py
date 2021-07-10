# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Specifies Generalized Additive Model for encoding seir variables."""

import functools
import logging
import sys
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from covid_epidemiology.src import constants
from covid_epidemiology.src.models.encoders import covariates as covariate_lib
from covid_epidemiology.src.models.encoders import variable_encoders
from covid_epidemiology.src.models.shared import feature_utils
from covid_epidemiology.src.models.shared import model_spec as model_spec_lib
from covid_epidemiology.src.models.shared import typedefs

_GAM_LINK_FUNCTIONS = {"exp": tf.exp, "identity": tf.identity}

_GAM_DISTRIBUTION_FUNCTIONS = {"exp": tf.exp, "identity": tf.identity}

_DEFAULT_FEATURE_KERNEL_INITIALIZER_MEAN = 0.0
_DEFAULT_FEATURE_KERNEL_INITIALIZER_STD = 0.0


class GamEncoder(variable_encoders.VariableEncoder):
  """Time-dependent encoder to update parameters via GAM."""

  def __init__(
      self,
      chosen_locations,
      num_known_timesteps,
      forecast_window_size,
      output_window_size,
      static_features = None,
      static_scalers = None,
      static_overrides = None,
      covariates = None,
      ts_scalers = None,
      forecasted_covariates = None,
      covariate_overrides = None,
      static_feature_specs = None,
      covariate_feature_specs = None,
      ts_categorical_features = None,
      static_categorical_features = None,
      covariate_feature_time_offset = 0,
      covariate_feature_window = 1,
      use_fixed_covariate_mask = True,
      initial_bias = 0.0,
      lower_bound = -5.0,
      upper_bound = 5.0,
      location_dependent_bias = True,
      override_raw_features = True,
      random_seed = 0,
      name = "",
      link_fn = "identity",
      distribution = "linear",
      trainable = True,
  ):
    """Initialize GAM Encoder."""

    # This GAM encoder supports overriding the covariates for counterfactual
    # simulations. Covariate overrides are one of:
    # 1. A multiplicative factor for continuous features (not listed in
    #    ts_categorical_features).
    # 2. An absolute value for categorical features when they are to be
    #    overridden.
    # 3. A value of '-1' for categorical features when they are not to be
    #    overridden and the original value from the covariates tensor is to be
    #    used.
    # 4. The raw, unnormalized feature values for ground-truth overrides.
    self.name = name
    self.override_raw_features = override_raw_features
    np.random.seed(random_seed)
    self.random_seed = random_seed

    self.link_fn = link_fn
    self.distribution = distribution
    self.num_known_timesteps = num_known_timesteps
    if covariate_feature_time_offset > forecast_window_size:
      raise ValueError(
          "Covariate_feature_time_offset cannot be greater than forecast window"
          " size as there would be no weights to offset to.")
    self.covariate_feature_time_offset = covariate_feature_time_offset
    self.covariate_temporal_dim = covariate_feature_window
    self.covariate_feature_window = covariate_feature_window
    self.use_fixed_covariate_mask = use_fixed_covariate_mask

    # save scalers within GamEncoder object so that member methods can call
    # them. This is memory-inefficient since each encoder object instance will
    # then have its own copy of the scalers, but makes it easier to access them
    # within each method.
    self.static_scalers = static_scalers
    self.ts_scalers = ts_scalers
    self.covariate_feature_specs = covariate_feature_specs
    self.static_feature_specs = static_feature_specs

    self.static_feature_kernel = _init_feature_kernel(
        static_feature_specs or {},
        kernel_name=name + "StaticFeatureKernel",
        trainable=trainable,
    )
    self.covariate_feature_kernel = _init_feature_kernel(
        covariate_feature_specs or {},
        tile_size=self.covariate_temporal_dim,
        kernel_name=name + "CovariateFeatureKernel",
        trainable=trainable,
    )
    if static_overrides is not None:
      if self.override_raw_features and not self.static_scalers:
        raise ValueError(
            "You cannot override raw static features without static_scalers.")

      overridden_static_features = dict()
      for feature in static_features:
        if feature not in static_overrides:
          overridden_static_features[feature] = static_features[feature]
        else:
          overridden_static_features[feature] = dict()
          for location in static_features[feature]:
            if location in static_overrides[feature]:
              overridden_static_features[feature][
                  location] = self._apply_static_override(
                      feature, static_features[feature][location],
                      static_overrides[feature][location])
              logging.info(
                  "Overriding %s %s static feature %s at %s with %s (old: %s)",
                  type(self).__name__, self.name, feature, location,
                  overridden_static_features[feature][location],
                  static_features[feature][location])
            else:
              overridden_static_features[feature][location] = static_features[
                  feature][location]
      self.static_feature_values = feature_utils.static_feature_to_dense(
          overridden_static_features or {}, static_feature_specs or {},
          chosen_locations)
    else:
      self.static_feature_values = feature_utils.static_feature_to_dense(
          static_features or {}, static_feature_specs or {}, chosen_locations)

    self.static_feature_values = tf.constant(
        self.static_feature_values, dtype=tf.float32)

    if self.covariate_feature_kernel is not None:
      self.covariate_feature_values = feature_utils.covariate_features_to_dense(
          covariates or {}, covariate_feature_specs or {}, chosen_locations,
          num_known_timesteps)
      if covariate_overrides is not None:
        if self.override_raw_features and not self.ts_scalers:
          raise ValueError(
              "You cannot override raw ts features without ts_scalers.")
        self.covariate_feature_overrides = feature_utils.covariate_overrides_to_dense(
            covariate_overrides, covariate_feature_specs or {},
            chosen_locations, num_known_timesteps + forecast_window_size)
      else:
        self.covariate_feature_overrides = None
      self.covariate_feature_values = tf.constant(
          self.covariate_feature_values, dtype=tf.float32)
    else:
      self.covariate_feature_values = tf.constant([], dtype=tf.float32)

    # ts_categorical_features = constants.TS_CATEGORICAL_COUNTY_FEATURES
    # for the first version of the What-If Tool, we are filtering categorical
    # features by hardcoding a substring pattern as a filter.
    # for this we need to know the *positional* name of each feature override.
    # we get this by constructing a mask over the features tensor of shape
    # (num_locations)X(num_all_features).
    self.ts_categorical_mask = None
    self.ts_continuous_mask = None
    self.static_categorical_mask = None
    self.static_continuous_mask = None
    self.covariate_lasso_mask = None
    num_locations = len(chosen_locations)
    if covariate_overrides is not None:
      if ts_categorical_features and covariate_feature_specs:
        self.ts_categorical_mask = feature_utils.get_categorical_features_mask(
            covariate_feature_specs or {},
            ts_categorical_features,
            num_locations,
            is_static=False)
        self.ts_categorical_mask = self.ts_categorical_mask.astype(np.float32)
        self.ts_continuous_mask = np.ones_like(
            self.ts_categorical_mask) - self.ts_categorical_mask

      if static_categorical_features and static_feature_specs:
        self.static_categorical_mask = feature_utils.get_categorical_features_mask(
            static_feature_specs or {},
            static_categorical_features,
            num_locations,
            is_static=True)
        self.static_categorical_mask = self.static_categorical_mask.astype(
            np.float32)
        self.static_continuous_mask = np.ones_like(
            self.static_categorical_mask) - self.static_categorical_mask

    if self.covariate_feature_kernel is not None:
      # Extract covariates name for each encoder
      self.forecasted_feature_values = (
          feature_utils.extract_forecasted_features(forecasted_covariates,
                                                    covariate_feature_specs))
      self.forecasted_feature_values = tf.constant(
          np.array(self.forecasted_feature_values), dtype=tf.float32)
      self.covariate_lasso_mask = feature_utils.get_lasso_feature_mask(
          covariate_feature_specs)

    # Apply location dependent biasing.
    if location_dependent_bias:
      initial_location_bias = initial_bias * np.ones(len(chosen_locations))
    else:
      initial_location_bias = initial_bias

    self.location_bias = tf.Variable(
        initial_location_bias,
        dtype=tf.float32,
        name=name + "LocationBias",
        trainable=trainable)

  def encode_gam(self,
                 global_bias,
                 covariate_values=None,
                 covariate_weights=None):
    """Performs the encoding operation via a GAM."""

    if self.static_feature_kernel is not None:
      static_component = tf.linalg.matvec(self.static_feature_values,
                                          self.static_feature_kernel)
    else:
      static_component = tf.zeros_like(global_bias)

    if covariate_weights is not None and covariate_values is not None:
      # Aggregate the time-varying covariate encodings over multiple timesteps.

      covariate_component = tf.multiply(covariate_values, covariate_weights)
      covariate_component = tf.reduce_sum(covariate_component, axis=[1, 2])
    else:
      covariate_component = tf.zeros_like(global_bias)

    affine_prediction = (
        global_bias + static_component + covariate_component +
        self.location_bias)

    encoded_prediction = apply_gam_link_and_dist_fns(
        affine_prediction=affine_prediction,
        link_fn=self.link_fn,
        distribution=self.distribution)

    return encoded_prediction

  def encode(
      self,
      time_series_input,
      global_bias,
      timestep,
      prediction,
      scaler,  # pylint: disable=g-bare-generic
      is_training = False):
    """Returns time-dependent encoder to update parameters via GAM.

    Args:
      time_series_input: list of tensors of size [num_locations] for each
        previous timestep.
      global_bias: A tensor of shape [num_locations, output_dim] containing the
        global bias terms.
      timestep: Point in time relative to the beginning of time_series_input to
        predict from. Has to be <= len(time_series_input).
      prediction: A list of predicted tensors.
      scaler: normalization (scaling) dictionary.
      is_training: whether the model is currently training or performing
        inference.
    """
    if self.covariate_feature_kernel is not None:
      return self.encode_gam(
          global_bias=global_bias,
          covariate_values=self._covariate_terms_for_timestep(
              time_series_input, timestep, prediction, scaler),
          covariate_weights=self._masked_covariate_weights_for_timestep(
              timestep, is_training))
    else:
      return self.encode_gam(global_bias, None, None)

  def _covariate_terms_for_timestep(self,
                                    time_series_input,
                                    timestep,
                                    prediction,
                                    scaler,
                                    local_window=7,
                                    epsilon=1e-8):

    covariate_time_inputs = []
    for time_offset_value in range(self.covariate_temporal_dim):
      potential_timestep = timestep - time_offset_value - 1
      if (potential_timestep >= 0 and
          potential_timestep < self.num_known_timesteps):
        if self.covariate_feature_overrides is not None:
          covariate_values = self.get_overriden_covariate_values_gam(
              potential_timestep)
        else:
          covariate_values = self.covariate_feature_values[potential_timestep]

        covariate_time_inputs.append(covariate_values)
      elif (potential_timestep < 0 or prediction is None):
        covariate_time_inputs.append(
            tf.zeros(tf.shape(self.covariate_feature_values[0])))
      else:
        if self.covariate_feature_overrides is not None:
          covariate_values = self.get_overriden_covariate_values_gam(
              potential_timestep)
        else:
          covariate_values = self.forecasted_feature_values[
              potential_timestep - self.num_known_timesteps]
        covariate_values = covariate_values[:, 4:]

        # Normalize the predictions
        confirmed_pred = (prediction["confirmed"][potential_timestep] -
                          scaler["confirmed"]["min"]) / (
                              scaler["confirmed"]["max"] -
                              scaler["confirmed"]["min"] + epsilon)
        death_pred = (prediction["death"][potential_timestep] -
                      scaler["death"]["min"]) / (
                          scaler["death"]["max"] - scaler["death"]["min"] +
                          epsilon)
        # Construct rate features
        # TODO(chunliang) restructure the following for better readability.
        if potential_timestep - self.num_known_timesteps + 1 >= local_window:
          past_confirm = tf.stack(
              prediction["confirmed"][potential_timestep - local_window +
                                      1:potential_timestep + 1],
              axis=1)
          past_confirm = (past_confirm - scaler["confirmed"]["min"]) / (
              scaler["confirmed"]["max"] - scaler["confirmed"]["min"] + epsilon)
          past_death = tf.stack(
              prediction["death"][potential_timestep - local_window +
                                  1:potential_timestep + 1],
              axis=1)
          past_death = (past_death - scaler["death"]["min"]) / (
              scaler["death"]["max"] - scaler["death"]["min"] + epsilon)
        else:
          # how many days from the training data, the delta is defined as
          # local_window - potential_timestep + self.num_known_timesteps - 1
          train_confirm = [
              self.covariate_feature_values[i][:, 0] for i in range(
                  max(0, potential_timestep - local_window +
                      1), self.num_known_timesteps)
          ]
          train_confirm = tf.stack(train_confirm, axis=1)
          past_confirm = tf.stack(
              prediction["confirmed"]
              [self.num_known_timesteps:potential_timestep + 1],
              axis=1)
          past_confirm = (past_confirm - scaler["confirmed"]["min"]) / (
              scaler["confirmed"]["max"] - scaler["confirmed"]["min"] + epsilon)
          past_confirm = tf.concat((train_confirm, past_confirm), axis=1)

          train_death = [
              self.covariate_feature_values[i][:, 1] for i in range(
                  max(0, potential_timestep - local_window +
                      1), self.num_known_timesteps)
          ]
          train_death = tf.stack(train_death, axis=1)
          past_death = tf.stack(
              prediction["death"][self.num_known_timesteps:potential_timestep +
                                  1],
              axis=1)
          past_death = (past_death - scaler["death"]["min"]) / (
              scaler["death"]["max"] - scaler["death"]["min"] + epsilon)
          past_death = tf.concat((train_death, past_death), axis=1)

        confirmed_pred_rate = tf.reduce_max(
            past_confirm, axis=1) / (
                tf.reduce_sum(past_confirm, axis=1) + epsilon)
        death_pred_rate = tf.reduce_max(
            past_death, axis=1) / (
                tf.reduce_sum(past_death, axis=1) + epsilon)
        covariate_inputs = tf.concat([
            tf.transpose([
                confirmed_pred, death_pred, confirmed_pred_rate, death_pred_rate
            ]), covariate_values
        ],
                                     axis=1)
        covariate_time_inputs.append(covariate_inputs)

    if covariate_time_inputs:
      return tf.stack(covariate_time_inputs, axis=1)
    else:
      return None

  def _apply_static_override(self, feature_name, current_value,
                             override_scaler):
    """Apply a static override to a single feature and location.

    TODO(nyoder): Improve performance by using vectors.

    Args:
      feature_name: The name of the feature being transformed.
      current_value: The input value.
      override_scaler: The value to scale this by.

    Returns:
      The transformed value.
    """
    if self.override_raw_features:
      scaler = self.static_scalers.get(feature_name, None)
      if scaler:
        # Since this is a float we need to convert this to an array
        current_value = scaler.inverse_transform([[current_value]])
      scaled_value = current_value * override_scaler
      if scaler:
        # Convert back to a float.
        scaled_value = scaler.transform(scaled_value)[0, 0]
    else:
      scaled_value = current_value * override_scaler
    return scaled_value

  def _apply_overrides_to_unscaled_values(self,
                                          current_covariate_values,
                                          potential_timestep):
    """Applies overrides in unscaled space.

    Args:
      current_covariate_values: tensor of size [num locations x num covariates]
        with current covariate values for each location
      potential_timestep: time at which to apply the overrides.

    Returns:
      Tensor of overridden covariate values of size [num locations x
        num covariates]
    """
    if self.ts_continuous_mask is not None:
      covariate_values_continuous = (
          self.covariate_feature_overrides[potential_timestep] *
          current_covariate_values * self.ts_continuous_mask)
    else:
      covariate_values_continuous = self.covariate_feature_overrides[
          potential_timestep] * current_covariate_values
    # For categorical covariate overrides, legal values are >=0. So a value of
    # -1 is used to indicate that the corresponding value must be the original
    # value of the covariate instead of the override.
    # merge the categorical overrides with the covariate features accounting
    # for the '-1' flag.
    covariate_categorical_values_merged = np.where(
        self.covariate_feature_overrides[potential_timestep] == -1,
        current_covariate_values,
        self.covariate_feature_overrides[potential_timestep])

    if self.ts_categorical_mask is not None:
      covariate_values_categorical = (
          covariate_categorical_values_merged * self.ts_categorical_mask)
    else:
      # If None, there is no ts categorical covariate
      covariate_values_categorical = np.zeros_like(
          covariate_categorical_values_merged)
    # Aggregate them
    covariate_values = (
        covariate_values_continuous + covariate_values_categorical)

    # Log the overrides for debugging purposes.
    if self.ts_categorical_mask is not None:
      change_mask = np.where(
          self.covariate_feature_overrides[potential_timestep] != -1, 1,
          0) * self.ts_categorical_mask
      change_ixs = np.nonzero(change_mask)

      if change_ixs[0].size != 0:
        np.set_printoptions(threshold=sys.maxsize)
        current_covariate_values_np = np.asarray(current_covariate_values)
        covariate_values_categorical_np = np.asarray(
            covariate_values_categorical)
        indices = np.dstack(change_ixs)
        logging.debug(
            "Categorical covariate changes at t=%d"
            " for %s\n"
            "for [location index, covariate index] =\n %s:\n"
            "%s overriden by %s results in %s\n", potential_timestep, self.name,
            indices, current_covariate_values_np[change_ixs],
            self.covariate_feature_overrides[potential_timestep][change_ixs],
            covariate_values_categorical_np[change_ixs])

    if self.ts_continuous_mask is not None:
      change_mask = np.where(
          self.covariate_feature_overrides[potential_timestep] != 1, 1,
          0) * self.ts_continuous_mask
      change_ixs = np.nonzero(change_mask)

      if change_ixs[0].size != 0:
        np.set_printoptions(threshold=sys.maxsize)
        current_covariate_values_np = np.asarray(current_covariate_values)
        covariate_values_continuous_np = np.asarray(covariate_values_continuous)
        indices = np.dstack(change_ixs)
        logging.debug(
            "Continuous covariate changes at t=%d for %s"
            "for [location index, covariate index] =\n %s:\n"
            "%s overridden by %s results in\n%s", potential_timestep, self.name,
            indices, current_covariate_values_np[change_ixs],
            self.covariate_feature_overrides[potential_timestep][change_ixs],
            covariate_values_continuous_np[change_ixs])

    return covariate_values

  def _apply_overrides_to_unscaled_values_gam(self, current_covariate_values,
                                              potential_timestep):
    """This wrapper makes mocking this function possible.

    The function name must be unique among {GamEncoder, PassThroughEncoder}
    functions so that both GamEncoder and PassThroughEncoder functions can be
    mocked in the same test function.

    Look at test_actual_gt_overrides in
    what_if_pipeline_integration_test.py to see how the mock works.

    Args:
      current_covariate_values: tensor of size [num locations x num covariates]
        with current covariate values for each location
      potential_timestep: time at which to apply the overrides.

    Returns:
      Tensor of overridden covariate values of size [num locations x
        num covariates]
    """
    return self._apply_overrides_to_unscaled_values(
        current_covariate_values=current_covariate_values,
        potential_timestep=potential_timestep)

  def _get_overriden_covariate_values(self, potential_timestep):
    """Overrides the covariates depending on the type of covariate and timestep.

    Calculate the covariate override depending on whether it is continuous or
    categorical. Formula is: {override} * ({cat_mask} + {value} * {cont_mask}).

    Args:
      potential_timestep: integer. Timestep at which to override the covariate.

    Returns:
      Tensor with the computed overriden covariate at that timestep.
    """

    # The type of each feature (continuous or categorical) along axis=1
    # of the values and overrides tensor is given by the mask.
    # required behavior is:
    # 1. if continuous, return the product of the value and the override
    # 2. if categorical, return the override.
    #
    # Get each contribution separately by multiplying with the appropriate
    # mask.
    if 0 <= potential_timestep < self.num_known_timesteps:
      current_covariate_values = self.covariate_feature_values[
          potential_timestep]
    else:
      value_timestep = potential_timestep - self.num_known_timesteps
      current_covariate_values = self._get_forecasted_values(value_timestep)

    # Since overrides are to be performed in raw feature space, invert the
    # previously perfomed scaling operation. These are covariate values that
    # are being inverted, so no '-1' sentinel values are present.
    if self.override_raw_features:
      # Apply per-feature scaler inverse transform.
      current_covariate_values = self.apply_scaler_transforms(
          current_covariate_values, inverse_transform=True)

    covariate_values = self._apply_overrides_to_unscaled_values_gam(
        current_covariate_values=current_covariate_values,
        potential_timestep=potential_timestep)

    # Since overrides are to be performed in raw feature space, re-apply the
    # previously perfomed scaling operation.
    if self.override_raw_features:
      covariate_values = self.apply_scaler_transforms(
          covariate_values, inverse_transform=False)

    return covariate_values

  def _get_forecasted_values(self, value_timestep):
    return self.forecasted_feature_values[value_timestep]

  def get_overriden_covariate_values_gam(self, potential_timestep):
    """Wrapper to make mocking this function possible."""

    # The function name must be unique among {GamEncoder, PassThroughEncoder}
    # functions so that both GamEncoder and PassThroughEncoder functions can be
    # mocked in the same test function.
    new_covariate_values = self._get_overriden_covariate_values(
        potential_timestep)
    return new_covariate_values

  def _masked_covariate_weights_for_timestep(self, timestep, is_training):
    return covariate_lib.mask_covariate_weights_for_timestep(
        self.covariate_feature_kernel,
        timestep,
        self.num_known_timesteps,
        is_training,
        self.covariate_feature_time_offset,
        active_window_size=self.covariate_feature_window,
        use_fixed_covariate_mask=self.use_fixed_covariate_mask,
        seed=self.random_seed)

  def apply_feature_specific_operation(self, name, feature, **kwargs):
    """Apply an operation specific to a feature."""

    if name in constants.GOOGLE_MOBILITY_FEATURES:
      # Special case of Google Mobility, which is a feature referenced to a
      # baseline value, hence can take on both positive and negative values.
      # Apply an offset of 100 when applying an inverse scaling transform, and
      # an offset of -100 when applying a forward scaling transform.
      offset = kwargs["offset"]
      return feature + offset
    return feature

  def apply_scaler_transforms(self,
                              covariates_at_timestep,
                              inverse_transform = False):
    """Apply per-feature scaler transforms to covariates at current timestep.

    Args:
      covariates_at_timestep: covariate tensor at a timestep.
      inverse_transform: flag to perform an inverse scaling transform. Default
        is the forward transform.

    Returns:
      Transformed covariate tensor.
    """

    transformed_covariates_list = []
    for f, feature_spec in enumerate(self.covariate_feature_specs):
      feature_covariate_values = tf.reshape(covariates_at_timestep[:, f],
                                            [-1, 1])
      # If this feature was previously scaled, extract the fitted scaler
      scaler = self.ts_scalers.get(feature_spec.name, None)
      if scaler is not None:
        if inverse_transform:
          feature_specific_operation = functools.partial(
              self.apply_feature_specific_operation,
              name=feature_spec.name,
              offset=100)
          transform_operation = scaler.inverse_transform
        else:
          feature_specific_operation = functools.partial(
              self.apply_feature_specific_operation,
              name=feature_spec.name,
              offset=-100)
          transform_operation = scaler.transform
        if not inverse_transform:
          feature_covariate_values = feature_specific_operation(
              feature=feature_covariate_values)
        feature_covariate_values = transform_operation(feature_covariate_values)
        if inverse_transform:
          feature_covariate_values = feature_specific_operation(
              feature=feature_covariate_values)

      transformed_covariates_list.append(
          tf.squeeze(tf.cast(feature_covariate_values, dtype=tf.float32)))

    transformed_covariate_values = tf.stack(transformed_covariates_list, axis=1)
    return transformed_covariate_values

  @property
  def trainable_variables(self):
    # pylint: disable=g-complex-comprehension
    return [
        trainable_variable
        for trainable_variable in (self.static_feature_kernel,
                                   self.covariate_feature_kernel,
                                   self.location_bias)
        if trainable_variable is not None
    ]

  @property
  def direction(self):
    direction_coef = list()
    for feature_spec in self.covariate_feature_specs:
      if feature_spec.weight_sign_constraint == model_spec_lib.EncoderWeightSignConstraint.NONE:
        direction_coef.append(0)
      elif feature_spec.weight_sign_constraint == model_spec_lib.EncoderWeightSignConstraint.POSITIVE:
        direction_coef.append(1)
      elif feature_spec.weight_sign_constraint == model_spec_lib.EncoderWeightSignConstraint.NEGATIVE:
        direction_coef.append(-1)
    direction_coef = tf.reshape(
        tf.constant(direction_coef, dtype=tf.float32), [1, -1])
    return direction_coef

  @property
  def lasso_loss(self):
    return 0.0 if self.covariate_lasso_mask is None else tf.norm(
        self.covariate_feature_kernel * self.covariate_lasso_mask, ord=1)


def apply_gam_link_and_dist_fns(affine_prediction,
                                link_fn="identity",
                                distribution="linear",
                                **kwargs):
  """Returns the result of running a generalized additive model over features.

  Args:
    affine_prediction: scalar result of gam's affine layer prediction.
    link_fn: transformation function to apply to linear prediction to get
      distribution mean parameter.
    distribution: distribution to use for the prediction.
    **kwargs: Extra distribution specific keyword arguments.
  """
  if link_fn not in _GAM_LINK_FUNCTIONS:
    raise ValueError(f"Unsupported link_fn: {link_fn}, must be one of:"
                     f"{list[_GAM_LINK_FUNCTIONS.keys()]}")
  parameter_prediction = _GAM_LINK_FUNCTIONS[link_fn](affine_prediction)

  if distribution == "linear":
    return parameter_prediction
  elif distribution == "normal":
    if "stddev" not in kwargs:
      raise ValueError(
          "In order to use normal distribution glm, stddev must be passed in "
          "as a kwarg.")
    return parameter_prediction + tf.random.normal(
        shape=parameter_prediction.shape,
        mean=tf.zeros_like(parameter_prediction),
        stddev=kwargs["stddev"])
  else:
    raise ValueError(f"Unsupported distribution: {distribution}")


def _init_feature_kernel(feature_specs,
                         tile_size=0,
                         kernel_name="Kernel",
                         trainable=True):
  """Returns initialization of training kernel of size [num_features]."""
  if not feature_specs:
    return None

  feature_kernel_initializer = []
  for feature_spec in feature_specs:
    if feature_spec.initializer is not None:
      if isinstance(feature_spec.initializer, (int, float)):
        feature_kernel_initializer.append(feature_spec.initializer)
      else:
        raise ValueError(
            f"Non-numeric initializer found with value: "
            f"{feature_spec.initializer} for feature: {feature_spec.name}")
    else:
      sampled_kernel_value = np.random.normal(
          _DEFAULT_FEATURE_KERNEL_INITIALIZER_MEAN,
          _DEFAULT_FEATURE_KERNEL_INITIALIZER_STD)
      feature_kernel_initializer.append(sampled_kernel_value)

  if not tile_size:
    feature_kernel = tf.Variable(
        feature_kernel_initializer,
        dtype=tf.float32,
        name=kernel_name,
        trainable=trainable)

  else:
    feature_kernel_initializer_repeated_timesteps = np.expand_dims(
        feature_kernel_initializer, 0)
    feature_kernel_initializer_repeated_timesteps = np.tile(
        feature_kernel_initializer_repeated_timesteps, (tile_size, 1))
    feature_kernel = tf.Variable(
        feature_kernel_initializer_repeated_timesteps,
        dtype=tf.float32,
        name=kernel_name,
        trainable=trainable,
    )

  return feature_kernel
