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

"""Model constructor for Tensorflow state-level models."""
from typing import Dict, List
import numpy as np
import tensorflow as tf
from covid_epidemiology.src import constants
from covid_epidemiology.src.models import generic_seir_model_constructor
from covid_epidemiology.src.models import losses
from covid_epidemiology.src.models.shared import model_utils


class StateModelConstructor(generic_seir_model_constructor.ModelConstructor):
  """Constructs a state Tensorflow model, to be used in tf_seir."""

  def __init__(self, model_spec, random_seed=0):
    super(StateModelConstructor, self).__init__(model_spec, random_seed)
    self.num_states = 17

  def extract_prediction(self, all_states):
    """Extract the death and confirmed predictions."""

    confirmed_all = list()
    death_all = list()

    for curr_state in all_states:
      # pylint: disable=unused-variable
      (exposed_t, infected_d_t, infected_ud_t, recovered_d_t, recovered_ud_t,
       hospitalized_t, hospitalized_cumulative_t, hospitalized_increase_t,
       icu_t, ventilator_t, death_t, population_t, reinfectable_d_t,
       reinfectable_ud_t, reinfectable_vaccine_t, vaccine_immuned_t,
       infected_ud_increase_t) = tf.unstack(curr_state)

      # Include ICU and Ventilator since they are separate compartments.
      confirmed_t = (
          infected_d_t + recovered_d_t + hospitalized_t + icu_t + ventilator_t +
          death_t + reinfectable_d_t)

      confirmed_all.append(confirmed_t)
      death_all.append(death_t)

    return {"confirmed": confirmed_all, "death": death_all}

  def compute_coef(self,
                   ground_truth_timeseries,
                   ground_truth_state,
                   num_train_steps,
                   num_known_steps,
                   power=2.0):
    """Compute train/valid coefficients for loss computation.

    Args:
      ground_truth_timeseries: ground truth compartments
      ground_truth_state: ground truth state level compartments
      num_train_steps: number of timesteps for training
      num_known_steps: number of known timesteps
      power: 2 for MSE and 1 for MAE

    Returns:
      train_coefs: training coeffcients for each compartment
      valid_coefs: valid coeffcients for each compartment
    """
    (_, gt_list, gt_indicator, _, _) = ground_truth_timeseries

    # Recovered
    recovered_train, recovered_valid = model_utils.compartment_base(
        gt_list["recovered"], gt_indicator["recovered"], num_train_steps,
        num_known_steps)
    # Death
    death_train, death_valid = model_utils.compartment_base(
        gt_list["death"], gt_indicator["death"], num_train_steps,
        num_known_steps)
    # Confirmed
    confirmed_train, confirmed_valid = model_utils.compartment_base(
        gt_list["confirmed"], gt_indicator["confirmed"], num_train_steps,
        num_known_steps)
    # Hospitalized
    hospitalized_train, hospitalized_valid = model_utils.compartment_base(
        gt_list["hospitalized"], gt_indicator["hospitalized"], num_train_steps,
        num_known_steps)
    # Hospitalized cumulative
    hospitalized_cumulative_train, hospitalized_cumulative_valid = model_utils.compartment_base(
        gt_list["hospitalized_cumulative"],
        gt_indicator["hospitalized_cumulative"], num_train_steps,
        num_known_steps)
    # ICU
    icu_train, icu_valid = model_utils.compartment_base(gt_list["icu"],
                                                        gt_indicator["icu"],
                                                        num_train_steps,
                                                        num_known_steps)
    # Ventilator
    ventilator_train, ventilator_valid = model_utils.compartment_base(
        gt_list["ventilator"], gt_indicator["ventilator"], num_train_steps,
        num_known_steps)

    train_coefs = [
        0, (death_train / recovered_train)**power, 1,
        (death_train / confirmed_train)**power,
        (death_train / hospitalized_train)**power,
        (death_train / hospitalized_cumulative_train)**power,
        (death_train / icu_train)**power,
        (death_train / ventilator_train)**power
    ]

    valid_coefs = [
        0, (death_valid / recovered_valid)**power, 1,
        (death_valid / confirmed_valid)**power,
        (death_valid / hospitalized_valid)**power,
        (death_valid / hospitalized_cumulative_valid)**power,
        (death_valid / icu_valid)**power,
        (death_valid / ventilator_valid)**power
    ]

    train_coefs = np.nan_to_num(train_coefs).tolist()
    valid_coefs = np.nan_to_num(valid_coefs).tolist()

    return train_coefs, valid_coefs

  def seir_dynamics(self, current_state, seir_variables):
    """Model dynamics."""

    (first_dose_vaccine_ratio_per_day, second_dose_vaccine_ratio_per_day,
     average_contact_id, average_contact_iud, reinfectable_rate, alpha,
     diagnosis_rate, recovery_rate_id, recovery_rate_iud, recovery_rate_h,
     recovery_rate_i, recovery_rate_v, hospitalization_rate, icu_rate,
     ventilator_rate, death_rate_id, death_rate_h, death_rate_i,
     death_rate_v) = seir_variables

    # pylint: disable=unused-variable
    (exposed_t, infected_d_t, infected_ud_t, recovered_d_t, recovered_ud_t,
     hospitalized_t, hospitalized_cumulative_t, hospitalized_increase_t, icu_t,
     ventilator_t, death_t, population_t, reinfectable_d_t, reinfectable_ud_t,
     reinfectable_vaccine_t, vaccine_immuned_t,
     infected_ud_increase_t) = tf.unstack(current_state)

    # Setting the susceptible so that the population adds up to a constant.
    normalized_susceptible_t = 1.0 - (
        exposed_t + infected_d_t + infected_ud_t + recovered_d_t +
        recovered_ud_t + hospitalized_t + icu_t + ventilator_t + death_t +
        vaccine_immuned_t) / population_t
    normalized_susceptible_t = tf.nn.relu(normalized_susceptible_t)

    # Differential change on vaccine immuned.
    d_vaccine_immuned_dt = (
        first_dose_vaccine_ratio_per_day * population_t +
        second_dose_vaccine_ratio_per_day * population_t -
        reinfectable_vaccine_t - vaccine_immuned_t)

    # Differential change on reinfectable after vaccination.
    d_reinfectable_vaccine_dt = vaccine_immuned_t * 1.0 / constants.VACCINE_IMMUNITY_DURATION

    # Differential change on exposed
    d_exposed_dt = (average_contact_id * infected_d_t +
                    average_contact_iud * infected_ud_t
                   ) * normalized_susceptible_t - alpha * exposed_t

    # Differential change on infected, documented and undocumented
    d_infected_d_dt = (
        diagnosis_rate * infected_ud_t - recovery_rate_id * infected_d_t -
        death_rate_id * infected_d_t - hospitalization_rate * infected_d_t)

    d_infected_ud_dt = (
        alpha * exposed_t - diagnosis_rate * infected_ud_t -
        recovery_rate_iud * infected_ud_t)

    d_infected_ud_increase_dt = alpha * exposed_t - infected_ud_increase_t

    # Differential change on recovered, documented and undocumented
    d_recovered_d_dt = (
        recovery_rate_id * infected_d_t + recovery_rate_h * hospitalized_t -
        reinfectable_rate * recovered_d_t)

    d_recovered_ud_dt = (
        recovery_rate_iud * infected_ud_t - reinfectable_rate * recovered_ud_t)

    # Differential change on hospitalized
    d_hospitalized_d_dt = (
        hospitalization_rate * infected_d_t -
        (death_rate_h + recovery_rate_h + icu_rate) * hospitalized_t +
        recovery_rate_i * icu_t)

    d_hospitalized_cumulative_d_dt = (hospitalization_rate * infected_d_t)

    d_hospitalized_increase_d_dt = (
        hospitalization_rate * infected_d_t - hospitalized_increase_t)

    # Differential change on icu
    d_icu_d_dt = (
        icu_rate * hospitalized_t -
        (death_rate_i + recovery_rate_i + ventilator_rate) * icu_t +
        recovery_rate_v * ventilator_t)

    # Differential change on ventilator
    d_ventilator_d_dt = (
        ventilator_rate * icu_t -
        (death_rate_v + recovery_rate_v) * ventilator_t)

    # Differential change on death, documented
    d_death_d_dt = (
        death_rate_id * infected_d_t + death_rate_h * hospitalized_t +
        death_rate_i * icu_t + death_rate_v * ventilator_t)

    # Differential change on recovered, who may get the disease again.
    d_reinfectable_d_dt = reinfectable_rate * recovered_d_t

    d_reinfectable_ud_dt = reinfectable_rate * recovered_ud_t

    all_state_derivatives = [
        d_exposed_dt, d_infected_d_dt, d_infected_ud_dt, d_recovered_d_dt,
        d_recovered_ud_dt, d_hospitalized_d_dt, d_hospitalized_cumulative_d_dt,
        d_hospitalized_increase_d_dt, d_icu_d_dt, d_ventilator_d_dt,
        d_death_d_dt, -d_death_d_dt, d_reinfectable_d_dt, d_reinfectable_ud_dt,
        d_reinfectable_vaccine_dt, d_vaccine_immuned_dt,
        d_infected_ud_increase_dt
    ]

    return tf.stack(all_state_derivatives)

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

    train_loss_coefs = hparams["train_loss_coefs"]
    valid_loss_coefs = hparams["valid_loss_coefs"]
    time_scale_weight = hparams["time_scale_weight"]
    width_coef_train = hparams["width_coef_train"]
    width_coef_valid = hparams["width_coef_valid"]
    quantile_cum_viol_coef = hparams["quantile_cum_viol_coef"]
    increment_loss_weight = hparams["increment_loss_weight"]
    train_crps_weight = hparams["train_crps_weight"]
    valid_crps_weight = hparams["valid_crps_weight"]

    (_, gt_list, gt_indicator, _, _) = ground_truth_timeseries

    unstacked_propagated_states = tf.unstack(propagated_states, axis=1)
    pred_infected = unstacked_propagated_states[1]
    pred_recovered = unstacked_propagated_states[3]
    pred_hospitalized = unstacked_propagated_states[5]
    pred_hospitalized_cumulative = unstacked_propagated_states[6]
    pred_icu = unstacked_propagated_states[8]
    pred_ventilator = unstacked_propagated_states[9]
    pred_death = unstacked_propagated_states[10]
    pred_reinfected = unstacked_propagated_states[12]

    pred_confirmed = (
        pred_infected + pred_recovered + pred_death + pred_hospitalized +
        pred_icu + pred_ventilator + pred_reinfected)

    train_start_index = tf.identity(train_start_index)
    train_end_index = tf.identity(train_end_index)
    valid_start_index = tf.identity(valid_start_index)
    valid_end_index = tf.identity(valid_end_index)
    if quantiles is not None:
      quantiles = tf.constant(quantiles, dtype=tf.float32)

    # Use quantile loss if the value of quantiles are given
    def loss(pred_states,
             gt_list,
             gt_indicator,
             train_start_index,
             train_end_index,
             valid_start_index,
             valid_end_index,
             time_scale_weight=0,
             is_training=True):
      if quantiles is not None:

        if is_training:
          train_loss = losses.weighted_interval_loss(
              quantile_pred_states=pred_states,
              tau_list=quantiles,
              gt_list=gt_list,
              gt_indicator=gt_indicator,
              begin_timestep=train_start_index,
              end_timestep=train_end_index,
              time_scale_weight=time_scale_weight,
              width_coef=width_coef_train)
          valid_loss = losses.weighted_interval_loss(
              quantile_pred_states=pred_states,
              tau_list=quantiles,
              gt_list=gt_list,
              gt_indicator=gt_indicator,
              begin_timestep=valid_start_index,
              end_timestep=valid_end_index,
              time_scale_weight=time_scale_weight,
              width_coef=width_coef_train)
        else:
          train_loss = losses.weighted_interval_loss(
              quantile_pred_states=pred_states,
              tau_list=quantiles,
              gt_list=gt_list,
              gt_indicator=gt_indicator,
              begin_timestep=train_start_index,
              end_timestep=train_end_index,
              time_scale_weight=time_scale_weight,
              width_coef=width_coef_valid)
          valid_loss = losses.weighted_interval_loss(
              quantile_pred_states=pred_states,
              tau_list=quantiles,
              gt_list=gt_list,
              gt_indicator=gt_indicator,
              begin_timestep=valid_start_index,
              end_timestep=valid_end_index,
              time_scale_weight=time_scale_weight,
              width_coef=width_coef_valid)
        train_loss += train_crps_weight * losses.crps_loss(
            quantile_pred_states=pred_states,
            tau_list=quantiles,
            gt_list=gt_list,
            gt_indicator=gt_indicator,
            begin_timestep=train_start_index,
            end_timestep=train_end_index,
            time_scale_weight=time_scale_weight)
        valid_loss += valid_crps_weight * losses.crps_loss(
            quantile_pred_states=pred_states,
            tau_list=quantiles,
            gt_list=gt_list,
            gt_indicator=gt_indicator,
            begin_timestep=valid_start_index,
            end_timestep=valid_end_index,
            time_scale_weight=time_scale_weight)

      else:
        train_loss = losses.state_estimation_loss(
            pred_states=pred_states,
            gt_list=gt_list,
            gt_indicator=gt_indicator,
            begin_timestep=train_start_index,
            end_timestep=train_end_index,
            time_scale_weight=time_scale_weight,
            increment_loss_weight=increment_loss_weight,
            num_forecast_steps=num_forecast_steps)
        valid_loss = losses.state_estimation_loss(
            pred_states=pred_states,
            gt_list=gt_list,
            gt_indicator=gt_indicator,
            begin_timestep=valid_start_index,
            end_timestep=valid_end_index,
            time_scale_weight=time_scale_weight,
            increment_loss_weight=increment_loss_weight,
            num_forecast_steps=num_forecast_steps)
      return train_loss, valid_loss

    infected_doc_train_loss, infected_doc_valid_loss = loss(
        pred_infected,
        gt_list["infected"],
        gt_indicator["infected"],
        train_start_index,
        train_end_index,
        valid_start_index,
        valid_end_index,
        time_scale_weight=time_scale_weight)
    recovered_doc_train_loss, recovered_doc_valid_loss = loss(
        pred_recovered + pred_reinfected,
        gt_list["recovered"],
        gt_indicator["recovered"],
        train_start_index,
        train_end_index,
        valid_start_index,
        valid_end_index,
        time_scale_weight=time_scale_weight)
    death_train_loss, death_valid_loss = loss(
        pred_death,
        gt_list["death"],
        gt_indicator["death"],
        train_start_index,
        train_end_index,
        valid_start_index,
        valid_end_index,
        time_scale_weight=time_scale_weight)
    hospitalized_train_loss, hospitalized_valid_loss = loss(
        pred_hospitalized + pred_icu + pred_ventilator,
        gt_list["hospitalized"],
        gt_indicator["hospitalized"],
        train_start_index,
        train_end_index,
        valid_start_index,
        valid_end_index,
        time_scale_weight=time_scale_weight)
    hospitalized_cumulative_train_loss, hospitalized_cumulative_valid_loss = loss(
        pred_hospitalized_cumulative,
        gt_list["hospitalized_cumulative"],
        gt_indicator["hospitalized_cumulative"],
        train_start_index,
        train_end_index,
        valid_start_index,
        valid_end_index,
        time_scale_weight=time_scale_weight)
    icu_train_loss, icu_valid_loss = loss(
        pred_icu + pred_ventilator,
        gt_list["icu"],
        gt_indicator["icu"],
        train_start_index,
        train_end_index,
        valid_start_index,
        valid_end_index,
        time_scale_weight=time_scale_weight)
    ventilator_train_loss, ventilator_valid_loss = loss(
        pred_ventilator,
        gt_list["ventilator"],
        gt_indicator["ventilator"],
        train_start_index,
        train_end_index,
        valid_start_index,
        valid_end_index,
        time_scale_weight=time_scale_weight)
    confirmed_train_loss, confirmed_valid_loss = loss(
        pred_confirmed,
        gt_list["confirmed"],
        gt_indicator["confirmed"],
        train_start_index,
        train_end_index,
        valid_start_index,
        valid_end_index,
        time_scale_weight=time_scale_weight)

    train_loss_overall = (
        train_coefs[0] * train_loss_coefs[0] * infected_doc_train_loss +
        train_coefs[1] * train_loss_coefs[1] * recovered_doc_train_loss +
        train_coefs[2] * train_loss_coefs[2] * death_train_loss +
        train_coefs[3] * train_loss_coefs[3] * confirmed_train_loss +
        train_coefs[4] * train_loss_coefs[4] * hospitalized_train_loss +
        train_coefs[5] *
        (train_loss_coefs[5] * hospitalized_cumulative_train_loss) +
        train_coefs[6] * train_loss_coefs[6] * icu_train_loss +
        train_coefs[7] * train_loss_coefs[7] * ventilator_train_loss)

    valid_loss_overall = (
        valid_coefs[0] * valid_loss_coefs[0] * infected_doc_valid_loss +
        valid_coefs[1] * valid_loss_coefs[1] * recovered_doc_valid_loss +
        valid_coefs[2] * valid_loss_coefs[2] * death_valid_loss +
        valid_coefs[3] * valid_loss_coefs[3] * confirmed_valid_loss +
        valid_coefs[4] * valid_loss_coefs[4] * hospitalized_valid_loss +
        valid_coefs[5] *
        (valid_loss_coefs[5] * hospitalized_cumulative_valid_loss) +
        valid_coefs[6] * valid_loss_coefs[6] * icu_valid_loss +
        valid_coefs[7] * valid_loss_coefs[7] * ventilator_valid_loss)

    # Loss for r_eff. Penalize r_eff>5
    if quantiles is None:
      if r_eff is not None:
        train_loss_overall += (
            hparams["r_eff_penalty_coef"] * tf.math.reduce_mean(
                tf.math.softplus(r_eff - hparams["r_eff_penalty_cutoff"])))

      # Calculate accelration
      train_loss_overall += (
          hparams["acceleration_death_coef"] *
          self.acceleration_loss(pred_death, 3))

      train_loss_overall += (
          hparams["acceleration_confirm_coef"] *
          self.acceleration_loss(pred_confirmed, 3))

      train_loss_overall += (
          hparams["acceleration_hospital_coef"] *
          self.acceleration_loss(pred_hospitalized, 3))

    else:
      # Quantile cumulative violation penalty
      forecasting_horizon = valid_end_index - valid_start_index

      train_violation_confirmed = losses.quantile_viol_loss(
          forecasting_horizon, train_end_index, forecasting_horizon,
          gt_indicator["confirmed"], gt_list["confirmed"], pred_confirmed)
      train_violation_death = losses.quantile_viol_loss(
          forecasting_horizon, train_end_index, forecasting_horizon,
          gt_indicator["death"], gt_list["death"], pred_death)

      train_loss_overall += quantile_cum_viol_coef * tf.reduce_mean(
          train_violation_confirmed)
      train_loss_overall += quantile_cum_viol_coef * tf.reduce_mean(
          train_violation_death)

      valid_violation_confirmed = losses.quantile_viol_loss(
          valid_start_index, valid_end_index, forecasting_horizon,
          gt_indicator["confirmed"], gt_list["confirmed"], pred_confirmed)
      valid_violation_death = losses.quantile_viol_loss(
          valid_start_index, valid_end_index, forecasting_horizon,
          gt_indicator["death"], gt_list["death"], pred_death)

      valid_loss_overall += quantile_cum_viol_coef * tf.reduce_mean(
          valid_violation_confirmed)
      valid_loss_overall += quantile_cum_viol_coef * tf.reduce_mean(
          valid_violation_death)

    return train_loss_overall, valid_loss_overall

  def unpack_states(self,
                    chosen_location_list,
                    ground_truth_timeseries,
                    propagated_states,
                    propagated_variables,
                    num_forecast_steps,
                    quantile_regression=False):
    # Assign in the desired dictionary form.
    susceptible_f_all_locations = {}
    exposed_f_all_locations = {}
    infected_d_f_all_locations = {}
    infected_ud_f_all_locations = {}
    recovered_d_f_all_locations = {}
    recovered_ud_f_all_locations = {}
    death_d_f_all_locations = {}
    death_horizon_ahead_d_f_all_locations = {}
    confirmed_f_all_locations = {}
    confirmed_horizon_ahead_d_f_all_locations = {}
    hospitalized_f_all_locations = {}
    hospitalized_increase_f_all_locations = {}
    hospitalized_cumulative_f_all_locations = {}
    icu_f_all_locations = {}
    ventilator_f_all_locations = {}
    reinfectable_d_f_all_locations = {}
    reinfectable_ud_f_all_locations = {}
    population_f_all_locations = {}
    reinfectable_vaccine_f_all_locations = {}
    vaccine_immuned_t_f_all_locations = {}
    infected_ud_increase_f_all_locations = {}

    for location_index, location in enumerate(chosen_location_list):
      exposed_f_all_locations[
          location] = propagated_states[:, 0, location_index].numpy()
      infected_d_f_all_locations[
          location] = propagated_states[:, 1, location_index].numpy()
      infected_ud_f_all_locations[
          location] = propagated_states[:, 2, location_index].numpy()
      recovered_d_f_all_locations[location] = (
          propagated_states[:, 3, location_index].numpy())
      recovered_ud_f_all_locations[location] = (
          propagated_states[:, 4, location_index].numpy())
      hospitalized_f_all_locations[location] = (
          propagated_states[:, 5, location_index].numpy() +
          propagated_states[:, 8, location_index].numpy() +
          propagated_states[:, 9, location_index].numpy())
      hospitalized_increase_f_all_locations[
          location] = propagated_states[:, 7, location_index].numpy()
      hospitalized_cumulative_f_all_locations[
          location] = propagated_states[:, 6, location_index].numpy()
      icu_f_all_locations[location] = (
          propagated_states[:, 8, location_index].numpy() +
          propagated_states[:, 9, location_index].numpy())
      ventilator_f_all_locations[
          location] = propagated_states[:, 9, location_index].numpy()
      death_d_f_all_locations[
          location] = propagated_states[:, 10, location_index].numpy()
      death_horizon_ahead_d_f_all_locations[location] = (
          propagated_states[num_forecast_steps - 1:, 10,
                            location_index].numpy() -
          propagated_states[:-num_forecast_steps + 1, 10,
                            location_index].numpy())
      population_f_all_locations[
          location] = propagated_states[:, 11, location_index].numpy()
      reinfectable_d_f_all_locations[
          location] = propagated_states[:, 12, location_index].numpy()
      reinfectable_ud_f_all_locations[
          location] = propagated_states[:, 13, location_index].numpy()
      reinfectable_vaccine_f_all_locations[
          location] = propagated_states[:, 14, location_index].numpy()
      vaccine_immuned_t_f_all_locations[
          location] = propagated_states[:, 15, location_index].numpy()
      infected_ud_increase_f_all_locations[
          location] = propagated_states[:, 16, location_index].numpy()

      confirmed_f_all_locations[location] = (
          infected_d_f_all_locations[location] +
          recovered_d_f_all_locations[location] +
          death_d_f_all_locations[location] +
          hospitalized_f_all_locations[location])
      confirmed_horizon_ahead_d_f_all_locations[location] = (
          confirmed_f_all_locations[location][num_forecast_steps - 1:, :] -
          confirmed_f_all_locations[location][:-num_forecast_steps + 1, :])

      susceptible_f_all_locations[location] = np.maximum(
          0, (population_f_all_locations[location] -
              confirmed_f_all_locations[location] -
              exposed_f_all_locations[location] -
              recovered_ud_f_all_locations[location] -
              infected_ud_f_all_locations[location] -
              vaccine_immuned_t_f_all_locations[location]))

      recovered_d_f_all_locations[location] = (
          recovered_d_f_all_locations[location] +
          reinfectable_d_f_all_locations[location])

      recovered_ud_f_all_locations[location] = (
          recovered_ud_f_all_locations[location] +
          reinfectable_ud_f_all_locations[location])

      confirmed_f_all_locations[location] = (
          confirmed_f_all_locations[location] +
          reinfectable_d_f_all_locations[location])

      # Lower bound of the cumulative quantiles are the last values.
      # for all constructors.
      if quantile_regression:

        (_, gt_list, _, _, _) = ground_truth_timeseries

        death_d_f_all_locations = self.lowerbound_postprocessing(
            death_d_f_all_locations, gt_list["death"][:, location_index],
            location, num_forecast_steps)
        confirmed_f_all_locations = self.lowerbound_postprocessing(
            confirmed_f_all_locations, gt_list["confirmed"][:, location_index],
            location, num_forecast_steps)
        recovered_d_f_all_locations = self.lowerbound_postprocessing(
            recovered_d_f_all_locations, gt_list["recovered"][:,
                                                              location_index],
            location, num_forecast_steps)
        recovered_ud_f_all_locations = self.lowerbound_postprocessing(
            recovered_ud_f_all_locations, None, location, num_forecast_steps)
        reinfectable_d_f_all_locations = self.lowerbound_postprocessing(
            reinfectable_d_f_all_locations, None, location, num_forecast_steps)
        reinfectable_ud_f_all_locations = self.lowerbound_postprocessing(
            reinfectable_ud_f_all_locations, None, location, num_forecast_steps)

    rates = self.extract_rates(propagated_variables, chosen_location_list)

    return (susceptible_f_all_locations, exposed_f_all_locations,
            infected_d_f_all_locations, infected_ud_f_all_locations,
            recovered_d_f_all_locations, recovered_ud_f_all_locations,
            death_d_f_all_locations, death_horizon_ahead_d_f_all_locations,
            confirmed_f_all_locations,
            confirmed_horizon_ahead_d_f_all_locations,
            hospitalized_f_all_locations, hospitalized_increase_f_all_locations,
            hospitalized_cumulative_f_all_locations, icu_f_all_locations,
            ventilator_f_all_locations, infected_ud_increase_f_all_locations,
            rates)

  def pack_compartments(self, states, ground_truth_timeseries,
                        num_forecast_steps):
    """Packs predictions into compartments with associated ground truth."""
    (susceptible_f_all_locations, exposed_f_all_locations,
     infected_d_f_all_locations, infected_ud_f_all_locations,
     recovered_d_f_all_locations, recovered_ud_f_all_locations,
     death_d_f_all_locations, death_horizon_ahead_d_f_all_locations,
     confirmed_f_all_locations, confirmed_horizon_ahead_d_f_all_locations,
     hospitalized_f_all_locations, hospitalized_increase_f_all_locations,
     hospitalized_cumulative_f_all_locations, icu_f_all_locations,
     ventilator_f_all_locations, infected_ud_increase_f_all_locations,
     rates) = states

    (_, _, _, _, orig_gt) = ground_truth_timeseries

    # pack all results in a list of compartment dataclasses.
    susceptible_compartment = generic_seir_model_constructor.Compartment(
        name=constants.SUSCEPTIBLE,
        predictions=susceptible_f_all_locations,
        num_forecast_steps=num_forecast_steps)
    exposed_compartment = generic_seir_model_constructor.Compartment(
        name=constants.EXPOSED,
        predictions=exposed_f_all_locations,
        num_forecast_steps=num_forecast_steps)
    infected_d_compartment = generic_seir_model_constructor.Compartment(
        name=constants.INFECTED_DOC,
        predictions=infected_d_f_all_locations,
        num_forecast_steps=num_forecast_steps,
        ground_truth=orig_gt["infected"])
    infected_ud_compartment = generic_seir_model_constructor.Compartment(
        name=constants.INFECTED_UNDOC,
        predictions=infected_ud_f_all_locations,
        num_forecast_steps=num_forecast_steps)
    infected_ud_increase_compartment = generic_seir_model_constructor.Compartment(
        name=constants.INFECTED_UNDOC_INCREASE,
        predictions=infected_ud_increase_f_all_locations,
        num_forecast_steps=num_forecast_steps)
    recovered_d_compartment = generic_seir_model_constructor.Compartment(
        name=constants.RECOVERED_DOC,
        predictions=recovered_d_f_all_locations,
        num_forecast_steps=num_forecast_steps,
        ground_truth=orig_gt["recovered"])
    recovered_ud_compartment = generic_seir_model_constructor.Compartment(
        name=constants.RECOVERED_UNDOC,
        predictions=recovered_ud_f_all_locations,
        num_forecast_steps=num_forecast_steps)
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
    hospitalized_cumulative_compartment = (
        generic_seir_model_constructor.Compartment(
            name=constants.HOSPITALIZED_CUMULATIVE,
            predictions=hospitalized_cumulative_f_all_locations,
            num_forecast_steps=num_forecast_steps))
    icu_compartment = generic_seir_model_constructor.Compartment(
        name=constants.ICU,
        predictions=icu_f_all_locations,
        num_forecast_steps=num_forecast_steps,
        ground_truth=orig_gt["icu"])
    ventilator_compartment = generic_seir_model_constructor.Compartment(
        name=constants.VENTILATOR,
        predictions=ventilator_f_all_locations,
        num_forecast_steps=num_forecast_steps,
        ground_truth=orig_gt["ventilator"])

    def create_horizon_ahead_gt(gt):
      """Creates incremental (1-day) ground truth values."""
      horizon_ahead_gt = {}
      for location in gt:
        horizon_ahead_gt[location] = (
            gt[location][num_forecast_steps - 1:] -
            gt[location][:-num_forecast_steps + 1])
      return horizon_ahead_gt

    death_horizon_ahead_d_compartment = (
        generic_seir_model_constructor.Compartment(
            name=constants.HORIZON_AHEAD_DEATH,
            predictions=death_horizon_ahead_d_f_all_locations,
            num_forecast_steps=1,
            ground_truth=create_horizon_ahead_gt(orig_gt["death"])))
    confirmed_horizon_ahead_d_compartment = (
        generic_seir_model_constructor.Compartment(
            name=constants.HORIZON_AHEAD_CONFIRMED,
            predictions=confirmed_horizon_ahead_d_f_all_locations,
            num_forecast_steps=1,
            ground_truth=create_horizon_ahead_gt(orig_gt["confirmed"])))

    rates_compartments = []
    for name, predictions in rates.items():
      rates_compartments.append(
          generic_seir_model_constructor.Compartment(
              name=name,
              predictions=predictions,
              num_forecast_steps=num_forecast_steps,
              use_quantiles=False))

    compartments = [
        susceptible_compartment, exposed_compartment, infected_d_compartment,
        infected_ud_compartment, recovered_d_compartment,
        recovered_ud_compartment, death_d_compartment,
        death_horizon_ahead_d_compartment, confirmed_compartment,
        confirmed_horizon_ahead_d_compartment, hospitalized_compartment,
        hospitalized_increase_compartment, hospitalized_cumulative_compartment,
        icu_compartment, ventilator_compartment,
        infected_ud_increase_compartment
    ]
    compartments += rates_compartments
    return compartments

  def apply_quantile_transform(self,
                               hparams,
                               propagated_states,
                               quantile_kernel,
                               quantile_biases,
                               ground_truth_timeseries,
                               num_train_steps,
                               num_forecast_steps,
                               num_quantiles=23,
                               epsilon=1e-8,
                               is_training=True,
                               initial_quantile_step=0):
    """Transform predictions into vector representing different quantiles.

    Args:
      hparams: Hyperparameters.
      propagated_states: single value predictions, its dimensions represent
        timestep * states * location.
      quantile_kernel: Quantile mapping kernel.
      quantile_biases: Biases for quantiles.
      ground_truth_timeseries: Ground truth time series.
      num_train_steps: number of train steps
      num_forecast_steps: number of forecasting steps
      num_quantiles: Number of quantiles
      epsilon: A small number to avoid 0 division issues.
      is_training: Whether the phase is training or inference.
      initial_quantile_step: start index for quantile training

    Returns:
      Vector value predictions of size
        timestep * states * location * num_quantiles
    """
    (_, gt_list, gt_indicator, _, _) = ground_truth_timeseries

    unstacked_propagated_states = tf.unstack(propagated_states, axis=1)
    pred_infected = unstacked_propagated_states[1]
    pred_recovered = unstacked_propagated_states[3]
    pred_hospitalized = unstacked_propagated_states[5]
    pred_icu = unstacked_propagated_states[8]
    pred_ventilator = unstacked_propagated_states[9]
    pred_death = unstacked_propagated_states[10]
    pred_reinfected = unstacked_propagated_states[12]

    pred_confirmed = (
        pred_infected + pred_recovered + pred_death + pred_hospitalized +
        pred_icu + pred_ventilator + pred_reinfected)

    quantile_encoding_window = hparams["quantile_encoding_window"]
    smooth_coef = hparams["quantile_smooth_coef"]
    partial_mean_interval = hparams["partial_mean_interval"]

    quantile_mapping_kernel = tf.math.softplus(
        tf.expand_dims(quantile_kernel, 2))
    quantile_biases = tf.math.softplus(tf.expand_dims(quantile_biases, 1))

    propagated_states_quantiles = []
    state_quantiles_multiplier_prev = tf.ones_like(
        tf.expand_dims(propagated_states[0, :, :], 2))

    def gt_ratio_feature(gt_values,
                         predicted):
      """Creates the GT ratio feature."""

      # This uses the imputed values when the values are not valid.
      ratio_pred = (1 - (predicted[:num_train_steps, :] /
                         (epsilon + gt_values[:num_train_steps])))
      # Add 0 at the beginning
      ratio_pred = tf.concat([
          0 * ratio_pred[:(quantile_encoding_window + num_forecast_steps), :],
          ratio_pred
      ],
                             axis=0)
      ratio_pred = tf.expand_dims(ratio_pred, 1)
      ratio_pred = tf.tile(ratio_pred, [1, self.num_states, 1])
      return ratio_pred

    def indicator_feature(gt_indicator):
      """Creates the indicator feature."""

      indicator = 1. - gt_indicator
      # Add 0 at the beginning
      indicator = tf.concat([
          0 * indicator[:(quantile_encoding_window + num_forecast_steps), :],
          indicator
      ],
                            axis=0)
      indicator = tf.expand_dims(indicator, 1)
      indicator = tf.tile(indicator, [1, self.num_states, 1])
      return indicator

    # Propagated states features
    temp_propagated_states = tf.concat([
        0 * propagated_states[:quantile_encoding_window, :, :],
        propagated_states
    ],
                                       axis=0)

    # GT ratio features
    death_gt_ratio_feature = gt_ratio_feature(gt_list["death"], pred_death)
    confirmed_gt_ratio_feature = gt_ratio_feature(gt_list["confirmed"],
                                                  pred_confirmed)
    hospitalized_gt_ratio_feature = gt_ratio_feature(gt_list["hospitalized"],
                                                     pred_hospitalized)

    # Indicator features
    death_indicator_feature = indicator_feature(gt_indicator["death"])
    confirmed_indicator_feature = indicator_feature(gt_indicator["confirmed"])
    hospitalized_indicator_feature = indicator_feature(
        gt_indicator["hospitalized"])

    for ti in range(initial_quantile_step,
                    num_train_steps + num_forecast_steps):

      if ti < num_train_steps:
        state_quantiles_multiplier = tf.ones_like(
            tf.expand_dims(propagated_states[0, :, :], 2))
        state_quantiles_multiplier = tf.tile(state_quantiles_multiplier,
                                             [1, 1, num_quantiles])
      else:
        # Construct the input features to be used for quantile estimation.
        encoding_input = []

        # Features coming from the trend of the estimated.
        encoding_input.append(1 - (
            temp_propagated_states[ti:(ti + quantile_encoding_window), :, :] /
            (epsilon +
             temp_propagated_states[ti + quantile_encoding_window, :, :])))

        # Features coming from the ground truth ratio of death.
        encoding_input.append(
            death_gt_ratio_feature[ti:(ti + quantile_encoding_window), :, :])
        # Features coming from the ground truth ratio of confirmed.
        encoding_input.append(
            confirmed_gt_ratio_feature[ti:(ti +
                                           quantile_encoding_window), :, :])
        # Features coming from the ground truth ratio of hospitalized.
        encoding_input.append(
            hospitalized_gt_ratio_feature[ti:(ti +
                                              quantile_encoding_window), :, :])

        # Features coming from death indicator.
        encoding_input.append(
            death_indicator_feature[ti:(ti + quantile_encoding_window), :, :])
        # Features coming from confirmed indicator.
        encoding_input.append(
            confirmed_indicator_feature[ti:(ti +
                                            quantile_encoding_window), :, :])
        # Features coming from hospitalized indicator.
        encoding_input.append(
            hospitalized_indicator_feature[ti:(ti +
                                               quantile_encoding_window), :, :])

        encoding_input_t = tf.expand_dims(tf.concat(encoding_input, axis=0), 3)

        # Limit the range of features.
        encoding_input_t = model_utils.apply_relu_bounds(
            encoding_input_t,
            lower_bound=0.0,
            upper_bound=2.0,
            replace_nan=True)

        # Estimate the multipliers of quantiles
        state_quantiles_multiplier = quantile_biases + tf.math.reduce_mean(
            tf.multiply(encoding_input_t, quantile_mapping_kernel), 0)

        # Consider accumulation to guarantee monotonicity
        state_quantiles_multiplier = tf.math.cumsum(
            state_quantiles_multiplier, axis=-1)
        if partial_mean_interval == 0:
          # Normalize to match the median to point forecasts
          state_quantiles_multiplier /= (
              epsilon + tf.expand_dims(
                  state_quantiles_multiplier[:, :,
                                             (num_quantiles - 1) // 2], -1))
        else:
          # Normalize with major densities to approximate point forecast (mean)
          median_idx = (num_quantiles - 1) // 2
          normalize_start = median_idx - partial_mean_interval
          normalize_end = median_idx + partial_mean_interval
          normalizer = tf.reduce_mean(
              0.5 *
              (state_quantiles_multiplier[:, :, normalize_start:normalize_end] +
               state_quantiles_multiplier[:, :, normalize_start +
                                          1:normalize_end + 1]),
              axis=2,
              keepdims=True)
          state_quantiles_multiplier /= (epsilon + normalizer)

        state_quantiles_multiplier = (
            smooth_coef * state_quantiles_multiplier_prev +
            (1 - smooth_coef) * state_quantiles_multiplier)

      state_quantiles_multiplier_prev = state_quantiles_multiplier

      # Return the estimated quantiles
      propagated_states_quantiles_timestep = tf.multiply(
          tf.expand_dims(propagated_states[ti, :, :], 2),
          state_quantiles_multiplier)

      propagated_states_quantiles.append(propagated_states_quantiles_timestep)

    return tf.stack(propagated_states_quantiles)

  def extract_rate_list(self):
    """Return list of rates that correspond to 'propagated_variables' tensor.

    Args: None.

    Returns:
      List of rate names.
    """

    return constants.ICU_AND_VENTILATOR_RATE_LIST

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

    if rates is not None and propagated_variables is not None:
      raise ValueError("Only rates or seir_variables can be used.")
    elif rates is None and propagated_variables is None:
      raise ValueError("Have to specify one argument.")
    elif rates is not None:
      beta_d, beta_ud = rates["average_contact_id_rate"], rates[
          "average_contact_iud_rate"]
      rho_id, rho_iud = rates["recovery_id_rate"], rates["recovery_iud_rate"]
      gamma, h = rates["diagnosis_rate"], rates["hospitalization_rate"]
      kappa_id = rates["death_id_rate"]
      # equation is computed from the Next Generation Matrix Method.
      # If you are changing any of the parameters below, please make sure to
      # update the Next Generation Matrix derivation and parameters too.
      # LINT.IfChange
      r_eff = (beta_d * gamma + beta_ud *
               (rho_id + kappa_id + h)) / ((gamma + rho_iud) *
                                           (rho_id + kappa_id + h) + epsilon)
      return r_eff
    else:
      propagated_variables_list = tf.unstack(propagated_variables, axis=1)

      average_contact_id = propagated_variables_list[2]
      average_contact_iud = propagated_variables_list[3]
      diagnosis_rate = propagated_variables_list[6]
      recovery_rate_id = propagated_variables_list[7]
      recovery_rate_iud = propagated_variables_list[8]
      hospitalization_rate = propagated_variables_list[12]
      death_rate_id = propagated_variables_list[15]

      beta_d = average_contact_id
      beta_ud = average_contact_iud
      rho_id = recovery_rate_id
      rho_iud = recovery_rate_iud
      gamma = diagnosis_rate
      h = hospitalization_rate
      kappa_id = death_rate_id
      r_eff = (beta_d * gamma + beta_ud *
               (rho_id + kappa_id + h)) / ((gamma + rho_iud) *
                                           (rho_id + kappa_id + h) + epsilon)
      return r_eff
