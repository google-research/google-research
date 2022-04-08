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

"""Losses used in Tensorflow models."""
import logging

import tensorflow as tf


@tf.function
def state_estimation_loss(
    pred_states,
    gt_list,
    gt_indicator,
    begin_timestep = tf.constant(0, dtype=tf.int32),
    end_timestep = tf.constant(-1, dtype=tf.int32),
    epsilon = 1e-8,
    time_scale_weight = 0.0,
    num_forecast_steps = tf.constant(1, dtype=tf.int32),
    increment_loss_weight = 0.0,
    oneside = tf.constant(False, dtype=tf.bool)):
  """Estimates the loss between the propagates states and the ground truth.

  Args:
    pred_states: Propagated states tensor of size [num_timesteps,
      num_locations].
    gt_list: TF Tensor of ground truth for a particular state of size
      [num_timesteps, num_locations].
    gt_indicator: Binary TF Tensor to indicate whether the ground truth value
      exists for a particular location and time.
    begin_timestep: Start index for the loss.
    end_timestep: End index for the loss.
    epsilon: Small number of zero division.
    time_scale_weight: A coefficient to increase the emphasis on the latter
      timesteps. If time_scale_weight=0.0, all timesteps are weighed equally. As
      it increases, emphasis on the latter timesteps grows.
    num_forecast_steps: Number of forecasting steps.
    increment_loss_weight: Loss coefficient for the incremental term.
    oneside: Oneside loss or not.

  Returns:
    MSE loss across locations and timesteps.
  """
  logging.info('Tracing state_estimation_loss')
  time_amplification = tf.math.exp(
      time_scale_weight *
      tf.range(begin_timestep, end_timestep, dtype=tf.float32))
  time_amplification = tf.expand_dims(time_amplification, axis=1)

  mask = gt_indicator[begin_timestep:end_timestep]

  if oneside:
    squared_error = time_amplification * mask * tf.square(
        tf.nn.relu(pred_states[begin_timestep:end_timestep] -
                   gt_list[begin_timestep:end_timestep]))
  else:
    squared_error = time_amplification * mask * tf.square(
        pred_states[begin_timestep:end_timestep] -
        gt_list[begin_timestep:end_timestep])

  mean_squared_error = (
      tf.reduce_sum(squared_error) / (epsilon + tf.reduce_sum(mask)))

  time_amplification = tf.math.exp(time_scale_weight * tf.range(
      begin_timestep + num_forecast_steps, end_timestep, dtype=tf.float32))
  time_amplification = tf.expand_dims(time_amplification, axis=1)

  mask_increment = (
      gt_indicator[begin_timestep + num_forecast_steps:end_timestep] *
      gt_indicator[begin_timestep:end_timestep - num_forecast_steps])

  pred_increments = (
      pred_states[begin_timestep + num_forecast_steps:end_timestep] -
      pred_states[begin_timestep:end_timestep - num_forecast_steps])
  gt_increments = (
      gt_list[begin_timestep + num_forecast_steps:end_timestep] -
      gt_list[begin_timestep:end_timestep - num_forecast_steps])

  if oneside:
    incremental_squared_error = time_amplification * mask_increment * tf.square(
        tf.nn.relu(pred_increments - gt_increments))
  else:
    incremental_squared_error = time_amplification * mask_increment * tf.square(
        pred_increments - gt_increments)

  mean_squared_incremental_error = tf.reduce_sum(incremental_squared_error) / (
      epsilon + tf.reduce_sum(mask_increment))
  logging.info('Finished tracing state_estimation_loss')
  return (mean_squared_error +
          increment_loss_weight * mean_squared_incremental_error)


@tf.function
def weighted_interval_loss(
    quantile_pred_states,
    tau_list,
    gt_list,
    gt_indicator,
    begin_timestep = tf.constant(0, dtype=tf.int32),
    end_timestep = tf.constant(-1, dtype=tf.int32),
    epsilon = 1e-8,
    time_scale_weight = 0.0,
    width_coef = 1):
  """Estimates the weighted interval loss for the propagated states.

  Args:
    quantile_pred_states: Propagated states tensor of size [num_timesteps,
      num_locations, quantile].
    tau_list: List of quantile used for quantile regression
    gt_list: Numpy array for ground truth for a particular state of size
      [num_timesteps, num_locations].
    gt_indicator: Binary Numpy array to indicate whether the ground truth value
      exists for a particular location and time.
    begin_timestep: Start index for the loss.
    end_timestep: End index for the loss.
    epsilon: Small number of zero division.
    time_scale_weight: A coefficient to increase the emphasis on the latter
      timesteps. If time_scale_weight=0.0, all timesteps are weighed equally. As
      it increases, emphasis on the latter timesteps grows.
    width_coef: Coefficient for the width loss.

  Returns:
    Weighted interval loss across locations, timesteps and tau.
  """
  logging.info('Tracing weighted_interval_loss')
  loss = 0.0
  tau_list_len = (len(tau_list) // 2) + 1

  for index in tf.range(tau_list_len):
    loss += interval_loss(
        pred_states_upper=quantile_pred_states[:, :,
                                               len(tau_list) - 1 - index],
        pred_states_lower=quantile_pred_states[:, :, index],
        gt_list=gt_list,
        gt_indicator=gt_indicator,
        begin_timestep=begin_timestep,
        end_timestep=end_timestep,
        epsilon=epsilon,
        time_scale_weight=time_scale_weight,
        tau=tau_list[index],
        width_coef=width_coef) * tau_list[index]
  logging.info('Finished tracing weighted_interval_loss')
  return loss / tau_list_len


def interval_loss(pred_states_upper,
                  pred_states_lower,
                  gt_list,
                  gt_indicator,
                  begin_timestep=0,
                  end_timestep=-1,
                  epsilon=1e-8,
                  time_scale_weight=0.0,
                  tau=0.5,
                  width_coef=1):
  """Interval loss between the propagates states and the ground truth.

  Args:
    pred_states_upper: Propagated states tensor of size [num_timesteps,
      num_locations] with tau = 1-tau.
    pred_states_lower: Propagated states tensor of size [num_timesteps,
      num_locations] with tau = tau.
    gt_list: Numpy array for ground truth for a particular state of size
      [num_timesteps, num_locations].
    gt_indicator: Binary Numpy array to indicate whether the ground truth value
      exists for a particular location and time.
    begin_timestep: Start index for the loss.
    end_timestep: End index for the loss.
    epsilon: Small number of zero division.
    time_scale_weight: A coefficient to increase the emphasis on the latter
      timesteps. If time_scale_weight=0.0, all timesteps are weighed equally. As
      it increases, emphasis on the latter timesteps grows.
    tau: Quantile coefficient tau used for interval score (alpha = 2tau)
    width_coef: Coefficient for the width loss.

  Returns:
    interval loss across locations and timesteps.
  """

  time_amplification = tf.math.exp(
      time_scale_weight *
      tf.range(begin_timestep, end_timestep, dtype=tf.float32))
  time_amplification = tf.expand_dims(time_amplification, axis=1)

  mask = gt_indicator[begin_timestep:end_timestep]

  y_pred_upper = pred_states_upper[:end_timestep - begin_timestep]
  y_pred_lower = pred_states_lower[:end_timestep - begin_timestep]
  # Time horizon of the pred_states for quantile starts from begin_timestep
  y_true = gt_list[begin_timestep:end_timestep]

  width_loss = (y_pred_upper - y_pred_lower) * width_coef
  upper_loss = (1.0 / (tau + epsilon)) * tf.nn.relu(y_true - y_pred_upper)
  lower_loss = (1.0 / (tau + epsilon)) * tf.nn.relu(y_pred_lower - y_true)

  loss = time_amplification * mask * (width_loss + upper_loss + lower_loss)

  return tf.reduce_sum(loss) / (epsilon + tf.reduce_sum(mask))


@tf.function
def crps_loss(
    quantile_pred_states,
    tau_list,
    gt_list,
    gt_indicator,
    begin_timestep = tf.constant(0, dtype=tf.int32),
    end_timestep = tf.constant(-1, dtype=tf.int32),
    epsilon = 1e-8,
    time_scale_weight = 0.0,
):
  """Estimates the continuous ranked probability score.

  Compares the predicted states to the ground truth.

  Args:
    quantile_pred_states: Propagated states tensor of size [num_timesteps,
      num_locations, quantile].
    tau_list: List of quantile used for quantile regression
    gt_list: Numpy array for ground truth for a particular state of size
      [num_timesteps, num_locations].
    gt_indicator: Binary Numpy array to indicate whether the ground truth value
      exists for a particular location and time.
    begin_timestep: Start index for the loss.
    end_timestep: End index for the loss.
    epsilon: Small number of zero division.
    time_scale_weight: A coefficient to increase the emphasis on the latter
      timesteps. If time_scale_weight=0.0, all timesteps are weighed equally. As
      it increases, emphasis on the latter timesteps grows.

  Returns:
    Weighted CRPS loss across locations, timesteps and tau.
  """
  logging.info('Tracing crps_loss')
  loss = 0.0
  tau_list_len = len(tau_list)

  for index in tf.range(tau_list_len - 1):
    loss += interval_crps_loss(
        pred_states_upper=quantile_pred_states[:, :, index + 1],
        pred_states_lower=quantile_pred_states[:, :, index],
        gt_list=gt_list,
        gt_indicator=gt_indicator,
        begin_timestep=begin_timestep,
        end_timestep=end_timestep,
        epsilon=epsilon,
        time_scale_weight=time_scale_weight,
        tau_upper=tau_list[index + 1],
        tau_lower=tau_list[index])
  logging.info('Finished tracing crps_loss')
  return loss / tau_list_len


def interval_crps_loss(
    pred_states_upper,
    pred_states_lower,
    gt_list,
    gt_indicator,
    begin_timestep = tf.constant(0, dtype=tf.int32),
    end_timestep = tf.constant(-1, dtype=tf.int32),
    epsilon = 1e-8,
    time_scale_weight = 0.0,
    tau_upper = 0.5,
    tau_lower = 0.5):
  """CRPS loss between the propagates states and the ground truth for the given interval.

  Args:
    pred_states_upper: Propagated states tensor of size [num_timesteps,
      num_locations] of the interval upper bound.
    pred_states_lower: Propagated states tensor of size [num_timesteps,
      num_locations] of the interval lower bound.
    gt_list: Numpy array for ground truth for a particular state of size
      [num_timesteps, num_locations].
    gt_indicator: Binary Numpy array to indicate whether the ground truth value
      exists for a particular location and time.
    begin_timestep: Start index for the loss.
    end_timestep: End index for the loss.
    epsilon: Small number of zero division.
    time_scale_weight: A coefficient to increase the emphasis on the latter
      timesteps. If time_scale_weight=0.0, all timesteps are weighed equally. As
      it increases, emphasis on the latter timesteps grows.
    tau_upper: Upper bound of quantile coefficient tau used for interval score.
    tau_lower: Lower bound of quantile coefficient tau used for interval score.

  Returns:
    interval crps loss across locations and timesteps.
  """

  time_amplification = tf.math.exp(
      time_scale_weight *
      tf.range(begin_timestep, end_timestep, dtype=tf.float32))
  time_amplification = tf.expand_dims(time_amplification, axis=1)

  mask = gt_indicator[begin_timestep:end_timestep]

  y_pred_upper = pred_states_upper[:end_timestep - begin_timestep]
  y_pred_lower = pred_states_lower[:end_timestep - begin_timestep]
  y_true = gt_list[begin_timestep:end_timestep]

  # lower_loss, (partial) interval is smaller than y_true (y_true>y_pred_lower)
  lower_mask = tf.cast(y_true >= y_pred_lower, dtype=tf.float32)
  a_0 = tau_lower * tf.ones(tf.shape(y_true))
  a_1 = a_0 + tf.clip_by_value(
      (y_true - y_pred_lower) /
      (y_pred_upper - y_pred_lower + epsilon), 0, 1) * (
          tau_upper - tau_lower)
  q_0 = y_pred_lower
  q_1 = y_pred_lower + (a_1 - a_0) / (tau_upper - tau_lower + epsilon) * (
      y_pred_upper - y_pred_lower)
  lower_loss = (0.5 * ((a_1 - a_0) * y_true - a_1 * q_0 + a_0 * q_1) *
                (a_1 + a_0) - 1.0 / 3.0 * (q_1 - q_0) *
                (a_1**2 + a_1 * a_0 + a_0**2))
  lower_loss = lower_mask * lower_loss

  # upper_loss, (partial) interval is larger than y_true (y_true < y_pred_upper)
  upper_mask = tf.cast(y_true <= y_pred_upper, dtype=tf.float32)
  a_0 = tau_lower + tf.clip_by_value(
      (y_true - y_pred_lower) /
      (y_pred_upper - y_pred_lower + epsilon), 0, 1) * (
          tau_upper - tau_lower)
  a_1 = tau_upper * tf.ones(tf.shape(y_true))
  q_0 = y_pred_lower + (a_1 - a_0) / (tau_upper - tau_lower + epsilon) * (
      y_pred_upper - y_pred_lower)
  q_1 = y_pred_upper
  upper_loss = (0.5 * ((a_1 - a_0) * y_true - a_1 * q_0 + a_0 * q_1) *
                (a_1 + a_0) - 1.0 / 3.0 * (q_1 - q_0) *
                (a_1**2 + a_1 * a_0 + a_0**2) + (a_1 - a_0) *
                (y_true - 0.5 * q_0 - 0.5 * q_1))
  upper_loss = upper_mask * upper_loss
  loss = time_amplification * mask * (upper_loss + lower_loss)

  return tf.reduce_sum(loss) / (epsilon + tf.reduce_sum(mask))


def boundary_loss_term_states(propagated_states,
                              lower_bound=0,
                              upper_bound=1e10,
                              begin_timestep=0,
                              end_timestep=-1):
  """Applies the loss for states that violate boundary conditions.

  Args:
    propagated_states: Propagated states tensor of size [num_timesteps,
      num_states, num_locations].
    lower_bound: Lower bound value for each state.
    upper_bound: Upper bound value for each state.
    begin_timestep: Start index for the loss.
    end_timestep: End index for the loss.

  Returns:
    Average boundary violation loss across locations and timesteps.
  """

  fitted_states = propagated_states[begin_timestep:end_timestep, :, :]
  lower_loss = tf.square(tf.nn.relu(lower_bound - fitted_states))
  upper_loss = tf.square(tf.nn.relu(fitted_states - upper_bound))

  return tf.reduce_mean(upper_loss + lower_loss)


def boundary_loss_term_quantiles(propagated_states,
                                 lower_bound=0,
                                 upper_bound=1e10,
                                 begin_timestep=0,
                                 end_timestep=-1):
  """Applies the loss for states that violate boundary conditions.

  Args:
    propagated_states: Propagated states tensor of size [num_timesteps,
      num_states, num_locations, num_quantiles].
    lower_bound: Lower bound value for each state.
    upper_bound: Upper bound value for each state.
    begin_timestep: Start index for the loss.
    end_timestep: End index for the loss.

  Returns:
    Average boundary violation loss across locations and timesteps.
  """

  fitted_states = propagated_states[:end_timestep - begin_timestep, :, :, :]
  lower_loss = tf.square(tf.nn.relu(lower_bound - fitted_states))
  upper_loss = tf.square(tf.nn.relu(fitted_states - upper_bound))

  return tf.reduce_mean(upper_loss + lower_loss)


def boundary_loss_term_coefs(coefficients,
                             lower_bound=0,
                             upper_bound=1e10,
                             weight=1.0):
  """Applies the loss for coefficients that violate boundary conditions.

  Args:
    coefficients: Input coefficients.
    lower_bound: Lower bound value for each state.
    upper_bound: Upper bound value for each state.
    weight: Coefficient of the loss.

  Returns:
    Average boundary violation loss across locations and timesteps.
  """

  lower_loss = tf.square(tf.nn.relu(lower_bound - coefficients))
  upper_loss = tf.square(tf.nn.relu(coefficients - upper_bound))

  return tf.reduce_mean((upper_loss + lower_loss) * weight)


def quantile_viol_loss(start_ind, end_ind, forecasting_horizon, gt_indicator,
                       gt_list, pred):
  """Returns loss penalty.

  For the case when the predicted quantiles are larger than the value from
  forecasting_horizon days before.
  Args:
    start_ind: Start index.
    end_ind: End index.
    forecasting_horizon: Forecasting horizon.
    gt_indicator: GT indicator tensor.
    gt_list: GT value tensor.
    pred: Predictions tensor.

  Returns:
    Quantile violation loss across locations and timesteps.
  """

  mask = tf.expand_dims(
      gt_indicator[end_ind - 2 * forecasting_horizon:end_ind -
                   forecasting_horizon], -1)
  gt_bound = tf.expand_dims(
      gt_list[end_ind - 2 * forecasting_horizon:end_ind - forecasting_horizon],
      -1)
  pred_q = pred[:end_ind - start_ind]
  return tf.nn.relu(mask * (gt_bound - pred_q))
