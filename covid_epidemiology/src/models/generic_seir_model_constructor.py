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

"""Model constructor for Tensorflow models."""
import abc
from typing import Dict, List, Optional

from dataclasses import dataclass
import feature_preprocessing
import numpy as np
import pandas as pd
import tensorflow as tf

from covid_epidemiology.src import constants
from covid_epidemiology.src.models import generic_seir_model_constructor


@dataclass
class Compartment:
  name: str
  predictions: Dict[str, tf.Tensor]
  num_forecast_steps: int
  ground_truth: Optional[Dict[str, Dict[str, np.ndarray]]] = None
  use_quantiles: bool = True


class ModelConstructor(abc.ABC):
  """Constructs a Tensorflow model, to be used in generic_seir.

  This class will own SEIR variables such as average_contact_id and update
  them as it unrolls model dynamics over multiple timesteps.
  """

  def __init__(self, model_spec, random_seed):
    self.model_spec = model_spec
    self.encoder_specs = self.model_spec.encoder_specs
    self.random_seed = random_seed

  def set_random_seed(self, random_seed):
    self.random_seed = random_seed

  def preprocess_state_ground_truth_timeseries(self, ts_state_features,
                                               num_observed_timesteps,
                                               num_train_steps, smooth_coef):
    """Returns preprocessed state hospitalized ground truth."""
    pass

  def smooth_single_timeseries(self,
                               timeseries,
                               indicator,
                               num_train_steps,
                               smooth_coef,
                               valid_smooth_coef=0.05):
    """Smooth a single timeseries to reduce the bias of model selection."""
    # Make sure both variables a numpy arrays so numba can handle them
    if isinstance(timeseries, tf.Tensor):
      timeseries = timeseries.numpy()
    if isinstance(indicator, tf.Tensor):
      indicator = indicator.numpy()
    smooth_train = feature_preprocessing.smooth_period(
        timeseries[:num_train_steps], indicator[:num_train_steps], smooth_coef)
    smooth_valid = feature_preprocessing.smooth_period(
        timeseries[num_train_steps - 1:], indicator[num_train_steps - 1:],
        valid_smooth_coef)
    return np.concatenate((smooth_train, smooth_valid[1:]), axis=0)

  def smooth_ground_truth_timeseries(self, num_train_steps,
                                     ground_truth_timeseries, smooth_coef):
    """Smooth noisy ground truth observations."""
    (chosen_populations_list, gt_list, gt_indicator, gt_names,
     orig_gt) = ground_truth_timeseries

    smoothed_gt_list = dict()
    for gt_name in gt_names:
      if isinstance(gt_list[gt_name], tf.Tensor):
        cur_gt = gt_list[gt_name].numpy()
        indicator = gt_indicator[gt_name].numpy()
      else:
        cur_gt = gt_list[gt_name]
        indicator = gt_indicator[gt_name]

      if gt_name != "death":
        cur_gt = self.smooth_single_timeseries(cur_gt, indicator,
                                               num_train_steps, smooth_coef)

      # ffill, bfill and then median fill
      # This should not impact the actual training because the indicator remains
      # unchanged.
      if np.sum(indicator):
        overall_median = np.median(cur_gt[:num_train_steps, :].ravel())
        df = pd.DataFrame(cur_gt)
        df = df.mask(indicator == 0, np.nan)
        df = df.ffill(axis=0)
        # Subtract 1 because loc is inclusive
        df.loc[:num_train_steps - 1, :] = df.loc[:num_train_steps -
                                                 1, :].bfill(axis=0)
        df.loc[num_train_steps:, :] = df.loc[num_train_steps:, :].bfill(axis=0)
        df = df.fillna(overall_median)
        cur_gt = df.values

      smoothed_gt_list[gt_name] = tf.constant(cur_gt, dtype=tf.float32)

    smoothed_ground_truth_timeseries = (chosen_populations_list,
                                        smoothed_gt_list, gt_indicator,
                                        gt_names, orig_gt)
    return smoothed_ground_truth_timeseries

  def extract_prediction(self, all_states):
    """Extract the death and confirmed predictions."""
    pass

  def compute_coef(self,
                   ground_truth_timeseries,
                   ground_truth_state,
                   num_train_steps,
                   num_known_steps,
                   power=2.0):
    """Compute train/valid coefficients for loss computation."""
    pass

  def seir_dynamics(self, current_state, seir_variables):
    """Returns the derivatives of each state for SEIR dynamics."""
    pass

  def direction_losses(self, hparams, seir_encoders):
    """Direction loss of trainable coefficients."""
    direction_loss = 0
    for encoder in seir_encoders:
      encoder_trainable_vars = encoder.trainable_variables
      for var in encoder_trainable_vars:
        if "average_contact_id_rateCovariateFeatureKernel" in var.name:
          direction_loss += tf.reduce_mean(tf.nn.relu(-var * encoder.direction))
        elif "average_contact_iud_rateCovariateFeatureKernel" in var.name:
          direction_loss += tf.reduce_mean(tf.nn.relu(-var * encoder.direction))
    direction_loss *= hparams["direction_loss_coef"]
    return direction_loss

  def compute_losses(self,
                     hparams,
                     train_coefs,
                     valid_coefs,
                     propagated_states,
                     ground_truth_timeseries,
                     r_eff,
                     train_start_index,
                     train_end_index,
                     valid_start_index,
                     valid_end_index,
                     num_forecast_steps,
                     quantiles=None):
    """Calculates the loss between the propagates states and the ground truth."""
    pass

  def aggregation_penalty(self, hparams, train_coefs, valid_coefs,
                          propagated_states, chosen_location_list,
                          ground_truth_state, train_start_index,
                          train_end_index, valid_start_index, valid_end_index,
                          num_forecast_steps):
    """Calculates aggregation loss between state and county."""
    pass

  def generate_compartment_predictions(self, chosen_location_list,
                                       propagated_states, propagated_variables,
                                       num_forecast_steps,
                                       ground_truth_timeseries,
                                       quantile_regression):
    """Extracts and stores compartment predictions."""
    states = self.unpack_states(chosen_location_list, ground_truth_timeseries,
                                propagated_states, propagated_variables,
                                num_forecast_steps, quantile_regression)
    compartment_predictions = self.pack_compartments(states,
                                                     ground_truth_timeseries,
                                                     num_forecast_steps)
    return compartment_predictions

  @abc.abstractmethod
  def unpack_states(self, chosen_location_list, ground_truth_timeseries,
                    propagated_states, propagated_variables, num_forecast_steps,
                    quantile_regression):
    """Splits up propagated_states and variables into a list of dictionaries."""

    # Each element but one in the list will be a dictionary of values for one
    # feature (e.g., death_d_f_all_locations), keyed by location.
    # The final element is a dictionary containing the rates.

  @abc.abstractmethod
  def pack_compartments(self, states, ground_truth_timeseries,
                        num_forecast_steps):
    """Packs predictions into compartments with associated ground truth."""

  def get_encoder_by_name(self, encoder_specs, name):
    for encoder_spec in encoder_specs:
      if encoder_spec.encoder_name == name:
        return encoder_spec
    raise ValueError(f"No encoder spec for requested encoder with name: {name}")

  def extract_rate_list(self):
    """Return list of rates that correspond to 'propagated_variables' tensor.

    Args: None.

    Returns:
      List of rate names.
    """
    raise NotImplementedError(("Please call this method from an instance of",
                               " a child class of this parent class."))

  def apply_quantile_transform(self,
                               propagated_states,
                               quantile_kernel,
                               quantile_biases,
                               ground_truth_timeseries,
                               num_train_steps,
                               num_forecast_steps,
                               epsilon=1e-8,
                               is_training=True):
    """Transform predictions into vector representing different quantiles.

    Args:
      propagated_states: single value predictions, its dimensions represent
        timestep * states * location
      quantile_kernel: Quantile mapping kernel.
      quantile_biases: Global biases for quantiles.
      ground_truth_timeseries: Ground truth time series.
      num_train_steps: number of train steps.
      num_forecast_steps: number of forecasting steps.
      epsilon: Small number for 0 division.
      is_training: Whether the phase is training or inference.

    Returns:
      Vector value predictions of size
        timestep * states * location * num_quantiles
    """
    pass

  def lowerbound_postprocessing(self, compartments, groundtruth, location,
                                num_forecast_steps):
    """Lower bound of the cumulative quantiles are the last values.

    Args:
      compartments: cumulative compartments
      groundtruth: ground truth of the compartments
      location: Granularity location id
      num_forecast_steps: number of forecast steps

    Returns:
      compartments: post-processed cumulative compartments
    """
    num_seq, num_quantile = np.shape(compartments[location])
    # Set lower bound
    lowerbound = np.nan
    if groundtruth is not None:
      lowerbound = groundtruth[num_seq - num_forecast_steps - 1]
    # If groundtruth is None or groudtruth value is nan (missing)
    if np.isnan(lowerbound):
      lowerbound = compartments[location][-num_forecast_steps - 1, 0]
    lowerbound = np.tile(lowerbound, (num_forecast_steps, num_quantile))

    compartments[location][-num_forecast_steps:, :] = np.maximum(
        compartments[location][-num_forecast_steps:, :], lowerbound)
    return compartments

  def extract_rates(self, rates,
                    locations):
    """Extract rate name->location->tensor maps from rate tensor.

    Args:
      rates: tf.Tensor. Tensor containing fitted rates after training, of shape
        (locations, rate_names, times).
      locations: List of locations.

    Returns:
      Rate maps.
    """
    rates_np = rates.numpy()  # shape = (timesteps, rates, locations)
    rate_terms = self.extract_rate_list()
    rates_dict = {}
    rates_dict["R_eff"] = {}
    for name_idx, name in enumerate(rate_terms):
      rates_dict[name] = {}
      for loc_idx, location in enumerate(locations):
        rates_dict[name][location] = np.expand_dims(
            rates_np[:, name_idx, loc_idx], axis=1)
    # Add R_eff terms.
    for location in locations:
      name_tensor_dict = {
          name: rates_dict[name][location] for name in rate_terms
      }
      rates_dict["R_eff"][location] = self.calculate_r_eff(name_tensor_dict)
    return rates_dict

  def calculate_r_eff(self,
                      rates = None,
                      propagated_variables = None,
                      epsilon = 1e-8):
    """Calculate Basic Reproduction Number R_eff over time and locations.

    Args:
      rates: rate name->tensor maps.
      propagated_variables: single tensor of variables indexed by
        (time)x(variables)x(locations) (used in the training).
      epsilon: epsilon for avoiding numerical error.

    Returns:
      R_eff tensor.
    """
    raise NotImplementedError(("Please call this method from an instance of",
                               " a child class of this parent class."))

  def trend_following(self, chosen_location_list, death_gt, confirmed_gt,
                      hospitalized_gt, hospitalized_increase_gt,
                      num_known_steps, num_forecast_steps):
    """Simple copy baseline."""

    # Assign in the desired dictionary form.
    death_d_f_all_locations = {}
    death_horizon_ahead_d_f_all_locations = {}
    confirmed_f_all_locations = {}
    confirmed_horizon_ahead_d_f_all_locations = {}
    hospitalized_f_all_locations = {}
    hospitalized_increase_f_all_locations = {}

    for _, location in enumerate(chosen_location_list):
      # Previous deaths count
      death_past = death_gt[location][:num_known_steps]
      # Future death count
      # Future trends (number of increase) would be the same as
      # previous trends (number of increase)
      death_future = death_gt[location][num_known_steps - 1] + (
          death_gt[location][num_known_steps -
                             num_forecast_steps:num_known_steps] -
          death_gt[location][num_known_steps - num_forecast_steps - 1])
      death_all = np.concatenate((death_past, death_future), axis=0)
      death_all = np.expand_dims(np.nan_to_num(death_all, 0), -1)
      death_d_f_all_locations[location] = death_all
      # Death horizontal ahead
      death_horizon_ahead_d_f_all_locations[location] = (
          death_d_f_all_locations[location][num_forecast_steps - 1:] -
          death_d_f_all_locations[location][:-num_forecast_steps + 1])

      # Previous confirmed cases
      confirmed_past = confirmed_gt[location][:num_known_steps]
      # Future confirmed cases
      # Future trends (number of increase) would be the same as
      # previous trends (number of increase)
      confirmed_future = confirmed_gt[location][num_known_steps - 1] + (
          confirmed_gt[location][num_known_steps -
                                 num_forecast_steps:num_known_steps] -
          confirmed_gt[location][num_known_steps - num_forecast_steps - 1])
      confirmed_all = np.concatenate((confirmed_past, confirmed_future), axis=0)
      confirmed_all = np.expand_dims(np.nan_to_num(confirmed_all, 0), -1)
      confirmed_f_all_locations[location] = confirmed_all
      # Confirmed horizontal ahead
      confirmed_horizon_ahead_d_f_all_locations[location] = (
          confirmed_f_all_locations[location][num_forecast_steps - 1:] -
          confirmed_f_all_locations[location][:-num_forecast_steps + 1])

      # Hospitalized
      if hospitalized_gt is not None:
        # Future trends (counts) are the same as previous trends (counts)
        hospitalized_past = hospitalized_gt[location][:num_known_steps]
        hospitalized_future = hospitalized_gt[location][num_known_steps - 1] + (
            hospitalized_gt[location][num_known_steps -
                                      num_forecast_steps:num_known_steps] -
            hospitalized_gt[location][num_known_steps - num_forecast_steps - 1])
        hospitalized_future = tf.nn.relu(hospitalized_future)
        hospitalized_all = np.concatenate(
            (hospitalized_past, hospitalized_future), axis=0)
        hospitalized_all = np.expand_dims(
            np.nan_to_num(hospitalized_all, 0), -1)
        hospitalized_f_all_locations[location] = hospitalized_all
      else:
        hospitalized_f_all_locations = None

      # Hospitalized increase
      if hospitalized_increase_gt is not None:
        # Future trends (number of increase) would be the same as
        # previous trends (number of increase)
        hospitalized_increase_past = hospitalized_increase_gt[
            location][:num_known_steps]
        hospitalized_increase_future = hospitalized_increase_gt[location][
            num_known_steps - 1] + (
                hospitalized_increase_gt[location]
                [num_known_steps - num_forecast_steps:num_known_steps] -
                hospitalized_increase_gt[location][num_known_steps -
                                                   num_forecast_steps - 1])
        hospitalized_increase_all = np.concatenate(
            (hospitalized_increase_past, hospitalized_increase_future), axis=0)
        hospitalized_increase_all = np.expand_dims(
            np.nan_to_num(hospitalized_increase_all, 0), -1)
        hospitalized_increase_f_all_locations[
            location] = hospitalized_increase_all
      else:
        hospitalized_increase_f_all_locations = None

    states = (death_d_f_all_locations, death_horizon_ahead_d_f_all_locations,
              confirmed_f_all_locations,
              confirmed_horizon_ahead_d_f_all_locations,
              hospitalized_f_all_locations,
              hospitalized_increase_f_all_locations)
    orig_gt = {
        "death": death_gt,
        "confirmed": confirmed_gt,
        "hospitalized": hospitalized_gt,
        "hospitalized_increase": hospitalized_increase_gt,
    }
    compartments = self.trend_following_pack_compartments(
        states, orig_gt, num_forecast_steps)
    return compartments

  def trend_following_pack_compartments(self, states, orig_gt,
                                        num_forecast_steps):
    """Packs predictions into compartments with associated ground truth."""
    (death_d_f_all_locations, death_horizon_ahead_d_f_all_locations,
     confirmed_f_all_locations, confirmed_horizon_ahead_d_f_all_locations,
     hospitalized_f_all_locations,
     hospitalized_increase_f_all_locations) = states

    # pack all results in a list of compartment dataclasses.
    death_d_compartment = generic_seir_model_constructor.Compartment(
        name=constants.DEATH,
        predictions=death_d_f_all_locations,
        num_forecast_steps=num_forecast_steps,
        ground_truth=orig_gt["death"])
    confirmed_compartment = generic_seir_model_constructor.Compartment(
        name=constants.CONFIRMED,
        predictions=confirmed_f_all_locations,
        num_forecast_steps=num_forecast_steps,
        ground_truth=orig_gt["confirmed"])
    hospitalized_compartment = generic_seir_model_constructor.Compartment(
        name=constants.HOSPITALIZED,
        predictions=hospitalized_f_all_locations,
        num_forecast_steps=num_forecast_steps,
        ground_truth=orig_gt["hospitalized"])
    hospitalized_increase_compartment = (
        generic_seir_model_constructor.Compartment(
            name=constants.HOSPITALIZED_INCREASE,
            predictions=hospitalized_increase_f_all_locations,
            num_forecast_steps=num_forecast_steps))

    def create_horizon_ahead_gt(gt):
      """Creates incremental (1-day) ground truth values."""
      horizon_ahead_gt = {}
      for location in gt:
        horizon_ahead_gt[location] = (
            gt[location][num_forecast_steps - 1:] -
            gt[location][:-num_forecast_steps + 1])
      return horizon_ahead_gt

    death_horizon_ahead_d_compartment = generic_seir_model_constructor.Compartment(
        name=constants.HORIZON_AHEAD_DEATH,
        predictions=death_horizon_ahead_d_f_all_locations,
        num_forecast_steps=1,
        ground_truth=create_horizon_ahead_gt(orig_gt["death"]))
    confirmed_horizon_ahead_d_compartment = (
        generic_seir_model_constructor.Compartment(
            name=constants.HORIZON_AHEAD_CONFIRMED,
            predictions=confirmed_horizon_ahead_d_f_all_locations,
            num_forecast_steps=1,
            ground_truth=create_horizon_ahead_gt(orig_gt["confirmed"])))

    compartments = [
        death_d_compartment, death_horizon_ahead_d_compartment,
        confirmed_compartment, confirmed_horizon_ahead_d_compartment,
        hospitalized_compartment, hospitalized_increase_compartment
    ]
    return [
        compart for compart in compartments if compart.predictions is not None
    ]

  def acceleration_loss(self, x, k=1):
    """Computes the regularization of accelerations of the observation.

    Discards the early time steps and reshape it into
    [[x[loc, 0], ... x[loc, k-1] ],
     ...,
     [x[loc, t], ... x[loc, t+k-1]]
    ]
    then uses each column for caculating 2nd derivatives.

    Args:
      x: single tensor of observations indexed by (time)x(locations)
      k: a scaler of the interval for calculating the accelerations

    Returns:
      mean of the difference between accelerations of two consective time stamps
    """
    n, m = x.shape
    sample_x = x[(n % k):]
    sample_x = tf.reshape(sample_x, [-1, k, m])
    velocity_x = sample_x[1:] - sample_x[:-1]
    acc_x = velocity_x[1:] - velocity_x[:-1]
    loss = tf.math.reduce_mean(tf.square(acc_x[1:] - acc_x[:-1]))
    return loss
