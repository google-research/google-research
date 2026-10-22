# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-multiple-import,missing-function-docstring,g-doc-return-or-yield,g-doc-args

import datetime

from absl import app
import pandas as pd

from constant_defs import HORIZONS
from constant_defs import QUANTILES
from constant_defs import REQUIRED_CDC_LOCATIONS
from constant_defs import TARGET_STR
from epi_utils import compute_rolling_evaluation
from epi_utils import format_for_cdc
from epi_utils import get_most_recent_saturday_date_str
from epi_utils import get_next_saturday_date_str
from epi_utils import get_saturdays_between_dates
from plotting_utils import plot_season_forecasts

timedelta = datetime.timedelta


INPUT_DIR = ''
MODEL_NAME = 'Google_SAI-Adapted_13'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# --- Core Principles of the method ---
# - Semi-mechanistic Renewal Equation: The model's foundation is the renewal equation (I_t = Rt * S_t) where S_t is the convolution of past incidence with a generation interval distribution.
# - Bayesian MCMC for Current Rt: The effective reproduction number (Rt) for the most recent period is estimated using Metropolis-Hastings MCMC, sampling from a posterior distribution derived from a Poisson likelihood and a log-normal prior.
# - Log-Linear Rt Trend Projection: A log-linear trend is fitted to historical, smoothed Rt estimates over a recent window, and this trend is projected forward stochastically to model future Rt dynamics.
# - Stochastic Forward Simulation: Future daily hospitalizations are simulated iteratively using the projected Rt, the renewal equation, and Poisson noise, generating multiple trajectories for robust probabilistic forecasts.

from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import gamma, poisson, linregress, lognorm
from scipy.signal import savgol_filter
import warnings

# Constants (defined locally for this block, though assumed global in notebook context)
QUANTILES = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
]
TARGET_STR = 'Total COVID-19 Admissions'

# Model specific constants
GENERATION_INTERVAL_MEAN = 4.5  # Mean of the generation interval in days
GENERATION_INTERVAL_STD = (
    2.0  # Standard deviation of the generation interval in days
)
GENERATION_INTERVAL_MAX_DAYS = (
    21  # Max lag for generation interval convolution (L in renewal equation)
)

WEEKLY_SMOOTHING_WINDOW_SIZE = (
    5  # Window size in weeks for initial smoothing of weekly data, must be odd
)
SMOOTHING_WINDOW_SIZE = (
    7  # Daily window size for Savitzky-Golay filter, must be odd
)
SMOOTHING_POLY_ORDER = (
    2  # Polynomial order for Savitzky-Golay filter (must be < window_size)
)

RT_TREND_WINDOW_DAYS = (
    28  # Number of days for fitting the linear trend on log(Rt)
)
MCMC_RT_EST_WINDOW_DAYS = (
    14  # Number of recent days for MCMC estimation of current Rt
)
MCMC_BURN_IN = 500  # Number of MCMC steps to discard
MCMC_STEPS = 2000  # Total MCMC iterations, including burn-in
MCMC_PROPOSAL_SCALE = 0.15  # Standard deviation for log-normal proposal distribution for Rt in MCMC
RT_HISTORY_SMOOTHING_WINDOW = 28  # Rolling window size for smoothing historical Rt estimates for trend fitting
RT_RANDOM_WALK_STD = 0.02  # Daily standard deviation for log(Rt) random walk noise during simulation
RT_TREND_SLOPE_STDERR_FALLBACK = 0.005  # Fallback stderr for trend slope when estimation fails, for simulating uncertainty

SIMULATION_RUNS = (
    500  # Number of trajectories to simulate for probabilistic forecasts
)
EPSILON = 1e-6  # Small value to prevent log(0) or division by zero, ensure positive values
MAX_RT_FOR_TREND_ESTIMATION = 10.0  # Upper bound for Rt values when fitting trend, to avoid extreme outliers

# Pre-calculate generation interval PMF (probability mass function)
# Parameters for gamma distribution
k = (GENERATION_INTERVAL_MEAN**2) / (
    GENERATION_INTERVAL_STD**2
)  # Shape parameter
theta = (
    GENERATION_INTERVAL_STD**2
) / GENERATION_INTERVAL_MEAN  # Scale parameter
gen_interval_pmf_raw = np.array([
    gamma.cdf(d + 1, a=k, scale=theta) - gamma.cdf(d, a=k, scale=theta)
    for d in range(GENERATION_INTERVAL_MAX_DAYS + 1)
])
gen_interval_pmf = gen_interval_pmf_raw / np.sum(
    gen_interval_pmf_raw
)  # Normalize to sum to 1
gen_interval_pmf_s_gt_0 = gen_interval_pmf[
    1:
]  # Slice for s > 0, i.e., [g_1, g_2, ..., g_L]


# Helper function to apply Savitzky-Golay filter robustly with fallbacks
def _apply_savgol_filter_robustly(
    data_array, window_size, poly_order, context = ''
):
  n_points = len(data_array)

  # Handle empty or single-point series immediately
  if n_points == 0:
    return np.array([])
  if n_points == 1:
    return data_array

  # Calculate effective window length for Savitzky-Golay
  savgol_window_len = window_size
  if savgol_window_len % 2 == 0:
    savgol_window_len += 1  # Ensure odd

  # Minimum valid window for Savitzky-Golay is poly_order + 2 (and must be odd).
  min_savgol_valid_window = poly_order + 2
  if min_savgol_valid_window % 2 == 0:
    min_savgol_valid_window += 1

  savgol_window_len = max(savgol_window_len, min_savgol_valid_window)

  # If the adjusted window is still too large for the data, or invalid, fallback to rolling mean
  if savgol_window_len > n_points or savgol_window_len <= poly_order:
    warnings.warn(
        f'Data ({n_points} points, {context}) too short or window invalid for'
        f' Savitzky-Golay filter (polyorder {poly_order}, effective window'
        f' {savgol_window_len}). Falling back to rolling mean.'
    )

    # Determine rolling mean window size based on context and available data
    if context == 'weekly':
      target_rolling_window = 3  # Aim for up to 3 weeks
    else:  # daily context
      target_rolling_window = 7  # Aim for up to 7 days

    # Ensure rolling_window_size is valid for the number of points
    rolling_window_size = max(1, min(target_rolling_window, n_points))

    return (
        pd.Series(data_array)
        .rolling(window=rolling_window_size, min_periods=1, center=True)
        .mean()
        .values
    )
  else:
    return savgol_filter(
        data_array, window_length=savgol_window_len, polyorder=poly_order
    )


# Optimized convolution helper: S_t = sum(I_{t-s} * g_s)
def _calculate_convolved_incidence_potential(
    past_incidence_array, gen_interval_pmf_slice
):
  """Calculates the "infection potential" S_t = sum(I_{t-L}, ..., I_{t-1} * g_1, ..., g_L) for the renewal equation.

  Args:
      past_incidence_array: Array of daily incidence values for `I_{t-L}, ...,
        I_{t-1}`. The most recent incidence `I_{t-1}` should be at
        `past_incidence_array[-1]`.
      gen_interval_pmf_slice: Array of generation interval PMF values `[g_1,
        g_2, ..., g_L]`. `g_s` is the probability that an infection occurred `s`
        days ago.

  Returns:
      float: The convolved infection potential S_t.
  """
  L = len(
      gen_interval_pmf_slice
  )  # L is the length of the generation interval PMF slice
  if (
      len(past_incidence_array) < L
  ):  # Should not happen if slicing is correct (always L_gen)
    return 0.0
  # The dot product past_incidence_array[::-1] @ gen_interval_pmf_slice correctly computes:
  # I_{t-1}*g_1 + I_{t-2}*g_2 + ... + I_{t-L}*g_L
  return np.dot(past_incidence_array[::-1], gen_interval_pmf_slice)


# MCMC log likelihood for a given Rt over a window
def _log_likelihood_mcmc(
    current_rt,
    daily_admissions_array,
    potential_history_for_rt_calc,
    window_start_idx,
    window_length,
):
  """Calculates log-likelihood for a constant Rt over a specific window using Poisson distribution.

  I_t ~ Poisson(Rt * S_t) where S_t is the infection potential.

  Args:
      current_rt: The R_t value for which to calculate the likelihood.
      daily_admissions_array: The complete smoothed daily incidence array.
      potential_history_for_rt_calc: Precomputed S_t values for
        `daily_admissions_array`.
      window_start_idx: The global starting index of the MCMC window within
        `daily_admissions_array`.
      window_length: The length of the MCMC window.

  Returns:
      float: The log-likelihood. Returns -np.inf if an invalid state is
      encountered.
  """
  ll = 0.0
  # current_rt is already clamped by the MCMC loop proposal, but defensive here.
  current_rt_clamped = np.maximum(current_rt, EPSILON)

  for i in range(window_length):
    current_day_global_idx = window_start_idx + i

    potential = potential_history_for_rt_calc[current_day_global_idx]
    observed_incidence = daily_admissions_array[current_day_global_idx]

    if potential <= EPSILON:
      if (
          observed_incidence > EPSILON
      ):  # Observed incidence > 0 when no potential exists (impossible state)
        return -np.inf
      else:  # observed_incidence is 0 or near-zero, and potential is 0 or near-zero.
        # This implies lambda=0 and observation=0, which has log-pmf = 0 (P(0|0) = 1).
        ll += 0.0  # Contribution is 0, move to next day
        continue
    else:
      # If potential > EPSILON, then lambda_t must be positive.
      lambda_t = current_rt_clamped * potential
      # Ensure lambda_t is strictly positive for poisson.logpmf to avoid issues with mean=0 and k>0
      ll += poisson.logpmf(observed_incidence, np.maximum(lambda_t, EPSILON))
  return ll


# Log-prior for Rt (log-normal, centered at 1 on the linear scale)
def _log_prior_rt(rt):
  """Log-prior probability for R_t. Uses a log-normal distribution."""
  if rt <= 0:
    return -np.inf  # R_t must be positive
  # Parameters for lognorm.pdf: s (shape, std dev of log(x)), loc (offset), scale (exp(meanlog))
  # A prior centered at 1 (i.e. log(Rt)=0) with std_dev_log(Rt)=0.5
  return lognorm.logpdf(rt, s=0.5, loc=0, scale=1.0)


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Return quantiles.
  """

  # Store all predictions
  all_quantile_predictions = []

  # Group by location for individual forecasts
  for location_code in test_x['location'].unique():
    loc_train_x = train_x[
        train_x['location'] == location_code
    ].copy()  # Operate on a copy
    loc_train_y = train_y[loc_train_x.index]  # Ensure y matches x index
    loc_test_x = test_x[test_x['location'] == location_code].copy()

    # Ensure target_end_date is datetime for consistency
    loc_train_x['target_end_date'] = pd.to_datetime(
        loc_train_x['target_end_date']
    )

    # Minimum data needed for the model to reasonably run:
    min_daily_data_for_core_model = (
        max(GENERATION_INTERVAL_MAX_DAYS + 1, SMOOTHING_POLY_ORDER + 2)
        + MCMC_RT_EST_WINDOW_DAYS
    )

    min_weekly_data_points_required = (
        max(
            WEEKLY_SMOOTHING_WINDOW_SIZE,
            int(np.ceil(min_daily_data_for_core_model / 7.0)),
        )
        + 1
    )  # Add 1 week for robust context

    if loc_train_y.empty or len(loc_train_y) < min_weekly_data_points_required:
      warnings.warn(
          f'Insufficient training data ({len(loc_train_y)} weeks, required'
          f' {min_weekly_data_points_required}) for location {location_code}.'
          ' Returning zero predictions.'
      )
      predictions_loc = pd.DataFrame(
          np.zeros((len(loc_test_x), len(QUANTILES))),
          columns=[f'quantile_{q}' for q in QUANTILES],
          index=loc_test_x.index,
      )
      all_quantile_predictions.append(predictions_loc)
      continue

    # --- 1. Signal Processing (Two-stage Smoothing and Daily Disaggregation) ---
    # Combine train_x and train_y for time series processing
    weekly_data = (
        pd.DataFrame({
            'date': pd.to_datetime(loc_train_x['target_end_date']),
            'admissions_weekly_sum': loc_train_y.values,
        })
        .set_index('date')
        .sort_index()
    )

    # Stage 1: Smooth weekly data
    weekly_sums_positive = np.maximum(
        0, weekly_data['admissions_weekly_sum'].values
    )
    smoothed_weekly_sums = _apply_savgol_filter_robustly(
        weekly_sums_positive,
        WEEKLY_SMOOTHING_WINDOW_SIZE,
        SMOOTHING_POLY_ORDER,
        'weekly',
    )

    # Create a full daily index spanning from the start of the earliest week to the end of the latest week.
    first_day_of_earliest_week = weekly_data.index.min() - pd.Timedelta(
        days=6
    )  # Sunday of the earliest week
    last_day_of_latest_week = (
        weekly_data.index.max()
    )  # Saturday of the latest week
    full_daily_index = pd.date_range(
        start=first_day_of_earliest_week, end=last_day_of_latest_week, freq='D'
    )

    # Stage 2: Daily Disaggregation (Improved Interpolation and Normalization)
    # Create a sparse daily series by assigning smoothed weekly sums to the end of their respective weeks (Saturdays).
    daily_admissions_sparse = pd.Series(np.nan, index=full_daily_index)
    for i, week_end_date in enumerate(weekly_data.index):
      daily_admissions_sparse.loc[week_end_date] = smoothed_weekly_sums[i]

    # Linearly interpolate to fill in daily values between known weekly points.
    daily_admissions_interpolated = (
        daily_admissions_sparse.interpolate(
            method='linear', limit_area='inside'
        )
        .ffill()
        .bfill()
    )  # ffill/bfill to handle any NaNs at the very start or end if not Saturdays

    # Ensure interpolated values are non-negative
    daily_admissions_interpolated = np.maximum(0, daily_admissions_interpolated)

    # Normalize the interpolated daily series such that the sum over each week matches the smoothed weekly sum.
    # First, map each daily date to its corresponding week_end_date.
    daily_week_end_dates = (
        daily_admissions_interpolated.index.to_series().apply(
            lambda d: d
            + pd.Timedelta(
                days=(5 - d.weekday()) % 7
            )  # Get Saturday for current week
        )
    )

    # Create a DataFrame to hold interpolated daily admissions and their week_end_date
    daily_admissions_with_week_info = pd.DataFrame({
        'admissions_interp': daily_admissions_interpolated,
        'week_end_date': daily_week_end_dates,
    })

    # Calculate the daily ratios within each week based on the interpolated profile
    # Avoid division by zero if a week's sum is zero in the interpolated series
    daily_ratios_within_week = daily_admissions_with_week_info.groupby(
        'week_end_date'
    )['admissions_interp'].transform(
        lambda x: x / x.sum() if x.sum() > EPSILON else 0
    )

    # Map the smoothed_weekly_sums (targets) back to a Series indexed by week_end_date
    target_smoothed_weekly_series = pd.Series(
        smoothed_weekly_sums, index=weekly_data.index
    )

    # Multiply daily ratios by the target smoothed weekly sums to get normalized daily admissions
    daily_admissions_normalized = (
        daily_ratios_within_week
        * daily_admissions_with_week_info['week_end_date'].map(
            target_smoothed_weekly_series
        )
    )

    # Fill any remaining NaNs (e.g., if a whole week had zero interpolated sum AND zero target sum) with 0.
    daily_admissions_normalized = daily_admissions_normalized.fillna(0)

    # Apply a second Savitzky-Golay filter to the normalized and interpolated daily data
    daily_admissions_array = _apply_savgol_filter_robustly(
        daily_admissions_normalized.values,
        SMOOTHING_WINDOW_SIZE,
        SMOOTHING_POLY_ORDER,
        'daily',
    )
    # Ensure non-negative after smoothing
    daily_admissions_array = np.maximum(0, daily_admissions_array)

    # Precompute `potential_history_for_rt_calc` (S_t) for all historical days
    potential_history_for_rt_calc = np.zeros_like(
        daily_admissions_array, dtype=float
    )
    L_gen = GENERATION_INTERVAL_MAX_DAYS
    for t in range(L_gen, len(daily_admissions_array)):
      past_incidence_for_conv = daily_admissions_array[t - L_gen : t]
      potential_history_for_rt_calc[t] = (
          _calculate_convolved_incidence_potential(
              past_incidence_for_conv, gen_interval_pmf_s_gt_0
          )
      )

    end_idx_daily_array = len(daily_admissions_array)
    # MCMC window starts from max(L_gen, end_idx - MCMC_RT_EST_WINDOW_DAYS) to ensure enough history for S_t calculation
    mcmc_window_start_idx = max(
        L_gen, end_idx_daily_array - MCMC_RT_EST_WINDOW_DAYS
    )
    mcmc_incidence_window_length = end_idx_daily_array - mcmc_window_start_idx

    if (
        mcmc_incidence_window_length < (MCMC_RT_EST_WINDOW_DAYS // 2)
        or mcmc_incidence_window_length < 1
    ):
      warnings.warn(
          f'MCMC window too short ({mcmc_incidence_window_length} days, ideal'
          f' {MCMC_RT_EST_WINDOW_DAYS}) for location {location_code}. Using'
          ' fallback Rt=1.0 and default trend uncertainty for simulations.'
      )
      # Fallback: if MCMC window is too short, directly use a default Rt and skip MCMC.
      rt_samples_mcmc = [1.0] * SIMULATION_RUNS
      # Also set trend slope to 0 with default stderr
      rt_trend_slope = 0.0
      rt_trend_slope_stderr = RT_TREND_SLOPE_STDERR_FALLBACK
    else:
      # --- 2. Bayesian Inference (MCMC for latest Rt) ---
      rt_samples_mcmc = []
      current_rt = 1.0  # Initial guess for Rt, ensuring positive (log-normal prior median)

      # Metropolis-Hastings MCMC
      for step in range(MCMC_STEPS):
        # Propose new Rt from a log-normal distribution around current Rt
        proposed_rt = lognorm.rvs(
            s=MCMC_PROPOSAL_SCALE, loc=0, scale=current_rt
        )
        # Explicitly clip proposed_rt within reasonable bounds and strictly positive
        proposed_rt = np.clip(
            proposed_rt, EPSILON, MAX_RT_FOR_TREND_ESTIMATION * 2
        )

        # Calculate log posterior for current and proposed Rt
        log_posterior_current = _log_likelihood_mcmc(
            current_rt,
            daily_admissions_array,
            potential_history_for_rt_calc,
            mcmc_window_start_idx,
            mcmc_incidence_window_length,
        ) + _log_prior_rt(current_rt)
        log_posterior_proposed = _log_likelihood_mcmc(
            proposed_rt,
            daily_admissions_array,
            potential_history_for_rt_calc,
            mcmc_window_start_idx,
            mcmc_incidence_window_length,
        ) + _log_prior_rt(proposed_rt)

        # Correct Hastings ratio for log-normal proposal: log(q(current|proposed) / q(proposed|current))
        # q(x'|x) = lognorm.pdf(x', s=MCMC_PROPOSAL_SCALE, loc=0, scale=x)
        log_hastings_ratio = lognorm.logpdf(
            current_rt, s=MCMC_PROPOSAL_SCALE, loc=0, scale=proposed_rt
        ) - lognorm.logpdf(
            proposed_rt, s=MCMC_PROPOSAL_SCALE, loc=0, scale=current_rt
        )

        log_alpha = (
            log_posterior_proposed - log_posterior_current
        ) + log_hastings_ratio

        accept = False
        if (
            log_posterior_proposed == -np.inf
        ):  # Proposed state is impossible: Reject
          accept = False
        elif (
            log_posterior_current == -np.inf
        ):  # Current state is impossible, proposed is valid: Accept (to try to escape the impossible state)
          accept = True
        else:  # Standard M-H comparison in log-domain
          accept = np.log(np.random.rand()) < log_alpha

        if accept:
          current_rt = proposed_rt

        if step >= MCMC_BURN_IN:
          # Ensure samples are finite and positive before appending to rt_samples_mcmc
          if np.isfinite(current_rt) and current_rt > EPSILON:
            rt_samples_mcmc.append(current_rt)
          else:
            # If current_rt became invalid, reset to a valid neutral value to attempt recovery
            warnings.warn(
                'MCMC generated invalid or non-positive Rt sample'
                f' ({current_rt}) for location {location_code} at step {step}.'
                ' Resetting current_rt to 1.0 (clamped positive).'
            )
            current_rt = np.maximum(
                EPSILON, 1.0
            )  # Reset to a neutral, strictly positive value

      if (
          not rt_samples_mcmc
      ):  # Fallback if MCMC failed to generate valid samples (e.g., all steps resulted in invalid Rt)
        warnings.warn(
            'MCMC failed to generate valid Rt samples for location'
            f' {location_code}. Using default Rt=1.0 for simulations.'
        )
        rt_samples_mcmc = [
            1.0
        ] * SIMULATION_RUNS  # Use a neutral default for simulations

      # --- 3. Trend Estimation (for past Rt) ---
      raw_daily_rt_estimates = np.full_like(
          daily_admissions_array, np.nan, dtype=float
      )
      # Only calculate Rt where potential is positive to avoid division by zero or infinites
      valid_potential_indices = (
          np.where(potential_history_for_rt_calc[L_gen:] > EPSILON)[0] + L_gen
      )

      if len(valid_potential_indices) > 0:
        raw_rt_vals = (
            np.maximum(0, daily_admissions_array[valid_potential_indices])
            / potential_history_for_rt_calc[valid_potential_indices]
        )
        raw_rt_vals = np.clip(raw_rt_vals, EPSILON, MAX_RT_FOR_TREND_ESTIMATION)
        raw_daily_rt_estimates[valid_potential_indices] = raw_rt_vals

      # Smooth historical Rt estimates before fitting trend
      smoothed_daily_rt_for_trend = (
          pd.Series(raw_daily_rt_estimates)
          .rolling(
              window=RT_HISTORY_SMOOTHING_WINDOW, min_periods=1, center=False
          )
          .mean()
          .values
      )

      # REVISED TREND ESTIMATION WINDOW: End at the last observed daily data point
      # and use RT_TREND_WINDOW_DAYS for the lookback.
      trend_estimation_end_global_idx = end_idx_daily_array
      trend_estimation_start_global_idx = max(
          L_gen,  # Ensure enough history for convolution (L_gen days before the start of the window)
          trend_estimation_end_global_idx - RT_TREND_WINDOW_DAYS,
      )

      rt_history_for_trend_slice = smoothed_daily_rt_for_trend[
          trend_estimation_start_global_idx:trend_estimation_end_global_idx
      ]
      valid_indices_for_trend_fit = np.isfinite(rt_history_for_trend_slice) & (
          rt_history_for_trend_slice > EPSILON
      )
      valid_rt_trend_points = rt_history_for_trend_slice[
          valid_indices_for_trend_fit
      ]

      rt_trend_slope = 0.0
      rt_trend_slope_stderr = 0.0

      if len(valid_rt_trend_points) >= 2:
        x_days = np.arange(len(valid_rt_trend_points))
        log_rt_history_for_trend = np.log(valid_rt_trend_points)
        slope, _, _, _, stderr_slope = linregress(
            x_days, log_rt_history_for_trend
        )
        rt_trend_slope = slope
        rt_trend_slope_stderr = stderr_slope
      else:
        # Explicitly set fallback values for clarity and use the default stderr for trend uncertainty
        warnings.warn(
            'Not enough valid points for Rt trend estimation for location'
            f' {location_code}. Using flat trend with default uncertainty.'
        )
        rt_trend_slope = 0.0
        rt_trend_slope_stderr = RT_TREND_SLOPE_STDERR_FALLBACK  # Use non-zero fallback for better calibration

    # --- 4. & 5. Forward Simulation & Incidence Calculation ---
    simulated_weekly_admissions = {
        f'horizon_{h}': [] for h in loc_test_x['horizon'].unique()
    }

    # Determine the actual start date for the forecast period (day after last observed daily data)
    forecast_start_date_actual = full_daily_index[-1] + pd.Timedelta(days=1)

    # Calculate the maximum number of days to simulate to cover all forecast horizons
    max_forecast_target_date = loc_test_x['target_end_date'].max()
    # max_forecast_horizon_days = (max_forecast_target_date - forecast_start_date_actual).days + 1
    max_forecast_horizon_days = (
        pd.Timestamp(max_forecast_target_date) - forecast_start_date_actual
    ).days + 1  # Manually changed
    if max_forecast_horizon_days < 0:
      max_forecast_horizon_days = (
          0  # No days to simulate if forecast target is in the past
      )

    for _ in range(SIMULATION_RUNS):
      # Initialize current R_t for this simulation run by sampling from the MCMC posterior
      # If MCMC failed, rt_samples_mcmc will contain 1.0s.
      initial_rt_for_sim = np.random.choice(rt_samples_mcmc)
      current_log_rt_sim = np.log(np.maximum(EPSILON, initial_rt_for_sim))

      # Sample a trend slope for this simulation run to incorporate uncertainty
      sampled_trend_slope = np.random.normal(
          rt_trend_slope, rt_trend_slope_stderr
      )

      # Initialize daily admissions for simulation with recent historical data for convolution.
      # We need GENERATION_INTERVAL_MAX_DAYS historical daily admissions to start the convolution.
      historical_slice_for_sim_init_raw = daily_admissions_array[-L_gen:]
      if len(historical_slice_for_sim_init_raw) < L_gen:
        padded_zeros = np.zeros(L_gen - len(historical_slice_for_sim_init_raw))
        historical_slice_for_sim_init = np.concatenate(
            (padded_zeros, historical_slice_for_sim_init_raw)
        )
      else:
        historical_slice_for_sim_init = historical_slice_for_sim_init_raw

      sim_daily_admissions = list(historical_slice_for_sim_init)

      # Calculate the initial "convolution potential" (S_t) for the first simulated day
      # This potential corresponds to the first day to be *simulated*.
      current_convolution_potential = _calculate_convolved_incidence_potential(
          sim_daily_admissions[-L_gen:], gen_interval_pmf_s_gt_0
      )

      for d_forecast in range(
          max_forecast_horizon_days
      ):  # Project daily incidence
        # Project Rt based on the sampled log-linear trend and add random walk noise
        current_log_rt_sim += sampled_trend_slope + np.random.normal(
            0, RT_RANDOM_WALK_STD
        )
        projected_rt = np.exp(current_log_rt_sim)
        projected_rt = np.maximum(
            EPSILON, projected_rt
        )  # Ensure projected Rt is positive

        # Calculate expected lambda for next day using the renewal equation: I_t = Rt * S_t
        expected_lambda = projected_rt * current_convolution_potential

        # Simulate daily admissions from Poisson distribution
        simulated_daily_admission = poisson.rvs(
            np.maximum(expected_lambda, EPSILON)
        )  # Ensure non-zero lambda for Poisson mean
        sim_daily_admissions.append(simulated_daily_admission)

        # Update the convolution potential for the next day (S_{t+1})
        # This uses the most recent L_gen days of incidence, including the newly simulated day.
        current_convolution_potential = (
            _calculate_convolved_incidence_potential(
                sim_daily_admissions[-L_gen:], gen_interval_pmf_s_gt_0
            )
        )

      # Aggregate daily simulations to weekly sums for each forecast horizon
      # The full simulated daily series starts L_gen days before forecast_start_date_actual.
      sim_full_daily_start_date = forecast_start_date_actual - pd.Timedelta(
          days=L_gen
      )

      # Ensure the index is correctly created for the full simulated daily series
      sim_full_daily = pd.Series(
          sim_daily_admissions,
          index=pd.date_range(
              start=sim_full_daily_start_date, periods=len(sim_daily_admissions)
          ),
      )

      for horizon_week in loc_test_x['horizon'].unique():
        forecast_target_date = loc_test_x[
            loc_test_x['horizon'] == horizon_week
        ]['target_end_date'].iloc[0]
        # Sum admissions for the 7 days *ending* on forecast_target_date
        week_start_for_sum = forecast_target_date - pd.Timedelta(days=6)

        ## Manually added
        # Convert to Timestamp for consistent comparison with sim_full_daily.index
        week_start_for_sum_ts = pd.Timestamp(week_start_for_sum)
        forecast_target_date_ts = pd.Timestamp(forecast_target_date)
        ## Manually added

        # Robust check to ensure the target week is fully covered by simulated data
        if (week_start_for_sum_ts >= sim_full_daily.index.min()) and (
            forecast_target_date_ts <= sim_full_daily.index.max()
        ):  ## Manually updated
          weekly_sum = sim_full_daily.loc[
              week_start_for_sum:forecast_target_date
          ].sum()
        else:
          # If the simulation period doesn't fully cover the target week, return 0 and warn.
          weekly_sum = 0
          warnings.warn(
              f'Simulation for location {location_code}, horizon {horizon_week}'
              f' (target {forecast_target_date}) is out of simulated date'
              ' range. Setting weekly sum to 0. Simulated range:'
              f' {sim_full_daily.index.min()} to {sim_full_daily.index.max()}.'
          )
        simulated_weekly_admissions[f'horizon_{horizon_week}'].append(
            weekly_sum
        )

    # --- 6. Quantile Extraction ---
    predictions_loc = pd.DataFrame(
        index=loc_test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    for horizon_week in loc_test_x['horizon'].unique():
      col_data = np.array(
          simulated_weekly_admissions[f'horizon_{horizon_week}']
      )
      # Ensure non-negative and integer counts
      col_data = np.maximum(0, col_data).round().astype(int)

      if (
          len(col_data) == 0
      ):  # Handle cases where simulation might have failed entirely for a horizon
        quantiles_for_horizon = np.zeros(len(QUANTILES))
      else:
        quantiles_for_horizon = np.percentile(
            col_data, [q * 100 for q in QUANTILES]
        )

      # Ensure monotonicity for quantiles: each quantile must be >= the previous one
      quantiles_for_horizon = np.maximum.accumulate(quantiles_for_horizon)

      # Assign to the correct rows in the predictions DataFrame for this horizon
      rows_for_horizon = loc_test_x[loc_test_x['horizon'] == horizon_week].index
      for i, q_col in enumerate(predictions_loc.columns):
        predictions_loc.loc[rows_for_horizon, q_col] = quantiles_for_horizon[i]

    all_quantile_predictions.append(predictions_loc)

  final_predictions_df = pd.concat(all_quantile_predictions)
  final_predictions_df = final_predictions_df.astype(
      int
  )  # Ensure integer output as admissions are counts
  return final_predictions_df


def main(argv):
  del argv  # Unused.
  locations = locations[locations['location'].isin(REQUIRED_CDC_LOCATIONS)]
  locations['location'] = locations['location'].astype(int)
  location_codes = locations['location'].unique()

  print('Locations sample:')
  print(locations.head())

  dataset = pd.read_csv(f'{INPUT_DIR}/dataset.csv')
  dataset['target_end_date'] = pd.to_datetime(
      dataset['target_end_date']
  ).dt.date

  print('Dataset sample (check for existence of most recent data):')
  print(dataset.sort_values(by=['target_end_date'], ascending=False).head())

  dataset[TARGET_STR] = (
      pd.to_numeric(dataset[TARGET_STR], errors='coerce')
      .replace({np.nan: np.nan})
      .astype('Int64')
  )

  # --- Execute Validation Run ---
  print('--- Starting Validation Run ---')
  # Define validation and test periods

  validation_date_end = get_most_recent_saturday_date_str()
  validation_date_start = pd.to_datetime(validation_date_end) - pd.Timedelta(
      weeks=3
  )

  validation_reference_dates = get_saturdays_between_dates(
      validation_date_start, validation_date_end
  )
  print('validation_reference_dates:', validation_reference_dates)
  validation_forecasts, validation_score = compute_rolling_evaluation(
      observed_values=dataset.copy(),
      reference_dates=validation_reference_dates,
      fit_and_predict_fn=fit_and_predict_fn,
      horizons=HORIZONS,
      location_codes=location_codes,
      locations_df=locations,
  )

  print(f'\nValidation Score: {validation_score}')
  if not validation_forecasts.empty:
    validation_forecasts.to_csv('/tmp/validation_forecasts.csv', index=False)
    print("Validation forecasts saved to '/tmp/validation_forecasts.csv'")

  # Plot forecast and predictions on validation dates against observed data

  validation_forecasts['target_end_date'] = pd.to_datetime(
      validation_forecasts['target_end_date']
  )
  validation_forecasts['reference_date'] = pd.to_datetime(
      validation_forecasts['reference_date']
  )

  # Prepare the observed data
  national_observed_all = (
      dataset.groupby('target_end_date')[TARGET_STR].sum().reset_index()
  )
  national_observed_all['target_end_date'] = pd.to_datetime(
      national_observed_all['target_end_date']
  )

  dates_to_plot_validation = [
      {
          'start': pd.to_datetime(validation_date_start) - timedelta(weeks=2),
          'end': pd.to_datetime(validation_date_end) + timedelta(weeks=5),
          'name': 'validation',
      },
  ]

  for season in dates_to_plot_validation:
    print(f"--- Generating plot for {season['name']} dates ---")
    plot_season_forecasts(
        season_start=season['start'],
        season_end=season['end'],
        season_name=season['name'],
        all_forecasts_df=validation_forecasts,
        national_observed_df=national_observed_all,
        step_size=1,
    )

  submission_date_str = get_next_saturday_date_str()
  submission_date = pd.to_datetime(submission_date_str).date()

  test_forecasts, _ = compute_rolling_evaluation(
      observed_values=dataset.copy(),
      reference_dates=[submission_date],
      fit_and_predict_fn=fit_and_predict_fn,
      horizons=HORIZONS,
      location_codes=location_codes,
      locations_df=locations,
  )

  print('\n--- Creating the submission file ---')

  if not test_forecasts.empty:
    cdc_submission = format_for_cdc(test_forecasts, 'wk inc covid hosp')
    cdc_submission.to_csv(
        f'/tmp/{submission_date_str}_{MODEL_NAME}.csv', index=False
    )
    print(
        'Submission forecasts saved to'
        f" '/tmp/{submission_date_str}_{MODEL_NAME}.csv'"
    )

    print('Verify final submission file:')
    print(cdc_submission)

    # Convert dates in test_forecasts to Timestamp
    test_forecasts['target_end_date'] = pd.to_datetime(
        test_forecasts['target_end_date']
    )
    test_forecasts['reference_date'] = pd.to_datetime(
        test_forecasts['reference_date']
    )

    # Plot forecasts for submission (all horizons)
    cdc_submission['target_end_date'] = pd.to_datetime(
        cdc_submission['target_end_date']
    )
    cdc_submission['reference_date'] = pd.to_datetime(
        cdc_submission['reference_date']
    )

    dates_to_plot_submission = [
        {
            'start': pd.to_datetime(submission_date) - timedelta(weeks=1),
            'end': pd.to_datetime(submission_date) + timedelta(weeks=3),
            'name': f'{submission_date} forecast',
        },
    ]

    for season in dates_to_plot_submission:
      print(f"--- Generating plot for {season['name']} dates ---")
      plot_season_forecasts(
          season_start=season['start'],
          season_end=season['end'],
          season_name=season['name'],
          all_forecasts_df=test_forecasts,
          national_observed_df=None,
          step_size=1,
      )


if __name__ == '__main__':
  app.run(main)
