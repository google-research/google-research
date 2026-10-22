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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-multiple-import,missing-class-docstring,g-doc-return-or-yield,g-doc-args,g-short-docstring-punctuation,protected-access,broad-exception-caught,missing-function-docstring

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
MODEL_NAME = 'Google_SAI-Adapted_17'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit
from scipy.stats import poisson, nbinom
from sklearn.linear_model import LinearRegression  # For improved ILINet scaling

# Define QUANTILES from the notebook's helper functions section
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
TARGET_STR = 'Total Influenza Admissions'

# Constants for MCMC and MOA weighting - ADJUSTED FOR EFFICIENCY AND ROBUSTNESS
NUM_MCMC_STEPS = 250000  # Adjusted from 200,000 to allow more exploration
BURN_IN = 50000  # Adjusted from 40,000 for more exploration
THINNING = 100  # Kept at 100
MCMC_PROPOSAL_SCALE = 0.08  # Reverted to 0.08 from 0.05 for broader exploration
MOA_TRAINING_WEIGHT = 0.5  # Kept at 0.5
NUM_SIMULATIONS = 10000

# Parameters whose values are log-transformed for MCMC exploration
# Indices correspond to the order in `_log_likelihood`'s `params` tuple:
# 0: initial_immune_prop (direct)
# 1: initial_I_prop (log-transform)
# 2: rho_H (log-transform)
# 3: base_beta (log-transform)
# 4: beta_amplitude (log-transform)
# 5: t_onset_days (direct)
# 6: t_offset_days (direct)
# 7: sharpness_rise (log-transform)
# 8: sharpness_fall (log-transform)
# 9: log_r (direct, already log)
LOG_TRANSFORM_INDICES = [1, 2, 3, 4, 7, 8]


# Helper function to get epi-year based on CDC convention (e.g., season starts in week 40)
def get_flu_season_year(date):
  return date.year if date.month < 8 else date.year + 1


# --- SIRH-like Model Definition (Abstraction of Compartmental Dynamics) ---
# This class implements a simplified SIR[H]2 model that can be "fit" to hospitalization incidence
# profiles. It incorporates daily cadence, time-dependent beta, and conceptual H2/Vaccinated compartments.


class SIRH_Model:

  def __init__(
      self,
      N,
      generation_time_days = 4,  # Avg time from infection to infection (Tg)
      hosp_incubation_days = 5,  # Infection to H1 admission
      hosp_duration_days = 7,
  ):  # H1 to H2 to R (length of stay in H2)
    """Initializes a simplified SIR[H]2 model for a given population, with daily cadence.

    Args:
        N (int): Total population of the location.
        generation_time_days (int): Average generation time (infection to
          infection), ~Tg.
        hosp_incubation_days (int): Average days from infection to first
          hospitalization compartment (H1).
        hosp_duration_days (int): Average days for a patient to move through H1
          and H2.
    """
    self.N = float(N)  # Ensure population is float for divisions
    self.generation_time_days = generation_time_days
    self.hosp_incubation_days = hosp_incubation_days
    self.hosp_duration_days = hosp_duration_days
    self.days_per_week = 7

    # Daily rates for SIR-H1-H2-R
    self.gamma_I = 1 / self.generation_time_days  # Recovery rate from I
    self.gamma_H1 = 1 / self.hosp_incubation_days  # Rate from H1 to H2
    self.gamma_H2 = 1 / self.hosp_duration_days  # Rate from H2 to R

  # Time-dependent beta using two arctan terms to create a seasonal peak (rise and fall)
  def _time_dependent_beta(
      self,
      t_day,
      base_beta,
      beta_amplitude,
      t_onset_days,
      t_offset_days,
      sharpness_rise,
      sharpness_fall,
  ):
    """Calculates time-dependent beta, modeling seasonal flu rise and fall.

    This function uses a product of two sigmoid-like arctan functions to create
    a hump shape. Now with separate sharpness parameters for the rising and
    falling phases.
    """
    if sharpness_rise <= 1e-6:
      sharpness_rise = 1e-6
    if sharpness_fall <= 1e-6:
      sharpness_fall = 1e-6

    # Rise phase (sigmoid from 0 to 1 as t_day increases past t_onset_days)
    rise_factor = (
        0.5 + np.arctan((t_day - t_onset_days) / sharpness_rise) / np.pi
    )

    # Fall phase (sigmoid from 1 to 0 as t_day increases past t_offset_days)
    fall_factor = (
        0.5 - np.arctan((t_day - t_offset_days) / sharpness_fall) / np.pi
    )

    # beta_t is base_beta plus an amplitude-scaled seasonal component
    beta_t = base_beta + beta_amplitude * rise_factor * fall_factor

    return np.maximum(1e-9, beta_t)  # Ensure beta is positive and non-zero

  def _simulate_sirh2_daily(
      self,
      t_weeks_for_curve_fit,
      initial_immune_prop,
      initial_I_prop,
      rho_H,
      base_beta,
      beta_amplitude,
      t_onset_days,
      t_offset_days,
      sharpness_rise,
      sharpness_fall,
  ):
    """Simulates the SIR-H1-H2-R dynamics with daily cadence and time-dependent beta.

    Tracks daily admissions into the first hospitalization compartment (H1).
    initial_immune_prop (float): Proportion of population initially
    immune/vaccinated. t_weeks_for_curve_fit: relative week numbers (0, 1, 2,
    ...) for which to return predictions. t_onset_days, t_offset_days: are
    expected in *absolute days* from the start of the simulation (day 0).
    """
    if self.N <= 0 or len(t_weeks_for_curve_fit) == 0:
      return np.zeros_like(t_weeks_for_curve_fit, dtype=float)

    # Improved initial compartment sizes, ensuring they sum to N and I0 is positive
    # I0 must be at least 1 individual, or a small proportion of N, capped at 20% of N (relaxed from 10%)
    # MODIFICATION: Increased upper clip for initial_I_prop to N * 0.2 for consistency with _log_likelihood bounds
    I0 = np.clip(initial_I_prop * self.N, 1.0, self.N * 0.2)

    R0 = 0.0
    H1_0 = 0.0
    H2_0 = 0.0

    S0_base = self.N - I0 - R0 - H1_0 - H2_0
    S0 = np.clip(
        S0_base * (1.0 - initial_immune_prop), 0.0, self.N - I0
    )  # Scale S0 by immune_prop

    # Ensure that S + I + R + H1 + H2 sum to N
    current_total = S0 + I0 + R0 + H1_0 + H2_0
    # Always normalize initial state to ensure sum to N
    if current_total != self.N:
      scale_factor = self.N / current_total
      S = S0 * scale_factor
      I = I0 * scale_factor
      R = R0 * scale_factor
      H1 = H1_0 * scale_factor
      H2 = H2_0 * scale_factor
    else:
      S, I, R, H1, H2 = S0, I0, R0, H1_0, H2_0

    daily_admissions_to_h1_tracker = (
        []
    )  # Stores new admissions to H1 for each day

    # Calculate the maximum day needed for simulation to cover all target weeks
    max_target_week = (
        int(t_weeks_for_curve_fit[-1]) if len(t_weeks_for_curve_fit) > 0 else 0
    )
    max_sim_day_needed = (
        (max_target_week + 1) * self.days_per_week
        + self.hosp_duration_days
        + self.hosp_incubation_days
        + 90
    )  # Increased buffer from 60 to 90 days

    # Simulate day by day using Euler integration
    for d in range(max_sim_day_needed):
      beta_t = self._time_dependent_beta(
          d,
          base_beta,
          beta_amplitude,
          t_onset_days,
          t_offset_days,
          sharpness_rise,
          sharpness_fall,
      )

      # Calculate changes for each compartment
      dS = -beta_t * S * I / self.N
      dI = (
          beta_t * S * I / self.N - self.gamma_I * I - rho_H * I
      )  # rho_H * I represents rate of I moving to H1
      dH1 = rho_H * I - self.gamma_H1 * H1
      dH2 = self.gamma_H1 * H1 - self.gamma_H2 * H2
      dR = self.gamma_I * I + self.gamma_H2 * H2

      # Update compartments (ensure non-negative)
      S_new = S + dS
      I = max(0.0, I + dI)
      H1 = max(0.0, H1 + dH1)
      H2 = max(0.0, H2 + dH2)
      R = max(0.0, R + dR)
      S = max(0.0, S_new)  # Update S last, after dS calculation

      # Re-normalize if sum deviates too much due to Euler integration (tighter thresholds)
      current_sum = S + I + R + H1 + H2
      if current_sum > self.N * (1.0 + 1e-9) or current_sum < self.N * (
          1.0 - 1e-9
      ):  # TIGHTENED THRESHOLDS for numerical stability
        scale_factor = self.N / current_sum
        S *= scale_factor
        I *= scale_factor
        H1 *= scale_factor
        H2 *= scale_factor
        R *= scale_factor

      # Store new admissions to H1 for the current day
      daily_admissions_to_h1_tracker.append(max(0.0, rho_H * I))

    predicted_weekly_hosp_for_fit = []
    for week_num in t_weeks_for_curve_fit:
      weekly_hosp_sum = 0.0  # Initialize for each week
      start_day_of_week = int(week_num) * self.days_per_week
      end_day_of_week = start_day_of_week + self.days_per_week

      # Sum daily admissions to H1 over the week
      for d_in_week in range(start_day_of_week, end_day_of_week):
        if d_in_week >= 0 and d_in_week < len(daily_admissions_to_h1_tracker):
          weekly_hosp_sum += daily_admissions_to_h1_tracker[d_in_week]
      predicted_weekly_hosp_for_fit.append(weekly_hosp_sum)

    return np.maximum(0.0, np.array(predicted_weekly_hosp_for_fit))


class SIRH_Forecaster:

  def __init__(self, N):
    self.sirh_model = SIRH_Model(
        N=N
    )  # Initialize with default parameters for the model structure
    self.mcmc_samples = []  # Store MCMC samples directly
    self.last_t_index_weeks = 0  # Last relative week index in training data

  def _log_likelihood(
      self, params_raw, t_data_weeks, y_data_abs, is_moa_data_flags
  ):
    """Calculates the Negative Binomial log-likelihood of observing y_data_abs given model predictions

    from the SIRH model with the given parameters, applying weights for MOA
    data.
    Params_raw contains log-transformed values for specific parameters.
    """
    # Unpack parameters, applying np.exp() for log-transformed ones
    unpacked_params = list(params_raw)
    for idx in LOG_TRANSFORM_INDICES:
      unpacked_params[idx] = np.exp(params_raw[idx])

    (
        initial_immune_prop,
        initial_I_prop,
        rho_H,
        base_beta,
        beta_amplitude,
        t_onset_days,
        t_offset_days,
        sharpness_rise,
        sharpness_fall,
        log_r,
    ) = unpacked_params

    # --- MODIFICATION START ---
    # Refined parameter validity checks for t_onset_days and t_offset_days
    # Ensure onset is strictly before offset by at least 1 day for basic sanity
    # And ensure minimum epidemic duration is 3 weeks
    if not (
        0.0 <= initial_immune_prop <= 0.9
        and 1e-8 <= initial_I_prop <= 0.2
        and 1e-7 <= rho_H <= 0.3  # ADJUSTED rho_H upper bound from 0.2 to 0.3
        and 1e-7 <= base_beta <= 1.0
        and 1e-7
        <= beta_amplitude
        <= 7.0  # ADJUSTED beta_amplitude upper bound from 5.0 to 7.0
        and 0 <= t_onset_days
        and 0 <= t_offset_days
        and t_onset_days
        < t_offset_days - 1.0  # Onset must be at least 1 day before offset
        and (t_offset_days - t_onset_days)
        >= self.sirh_model.days_per_week
        * 3  # Minimum epidemic duration for validity (3 weeks)
        and 1.0 <= sharpness_rise <= 500.0  # WIDENED sharpness bounds
        and 1.0 <= sharpness_fall <= 500.0  # WIDENED sharpness bounds
        and np.log(0.001) <= log_r <= np.log(10000.0)
    ):
      return -np.inf  # Penalize out-of-bounds parameters
    # --- MODIFICATION END ---

    model_output = self.sirh_model._simulate_sirh2_daily(
        t_data_weeks,
        initial_immune_prop,
        initial_I_prop,
        rho_H,
        base_beta,
        beta_amplitude,
        t_onset_days,
        t_offset_days,
        sharpness_rise,
        sharpness_fall,
    )

    # Add a small epsilon to model_output to avoid log(0) and nbinom(k,0) issues
    model_output = np.maximum(1e-9, model_output)
    y_data_abs_safe = np.maximum(0, y_data_abs).astype(
        int
    )  # Observed can be 0, model can predict >0

    # Negative Binomial log-likelihood
    r_param = np.exp(log_r)

    # Prevent numerical issues if model_output is extremely large or r_param is very small
    p_param = np.clip(r_param / (r_param + model_output), 1e-9, 1.0 - 1e-9)

    with warnings.catch_warnings():
      warnings.simplefilter(
          'ignore', RuntimeWarning
      )  # Ignore nbinom.logpmf warnings for small values
      log_prob = nbinom.logpmf(k=y_data_abs_safe, n=r_param, p=p_param)

    # Apply reduced weight for MOA data
    weights = np.ones_like(log_prob, dtype=float)
    # Ensure is_moa_data_flags is a boolean array for proper indexing
    if is_moa_data_flags.dtype != bool:
      is_moa_data_flags = is_moa_data_flags.astype(bool)
    weights[is_moa_data_flags] = MOA_TRAINING_WEIGHT

    weighted_log_prob = log_prob * weights

    valid_log_prob = weighted_log_prob[np.isfinite(weighted_log_prob)]

    if len(valid_log_prob) == 0:
      return -np.inf  # All log_probs were invalid, reject proposal

    return np.sum(valid_log_prob)

  def fit(
      self, training_data_df
  ):  # Expects 'per_capita_admissions', 'relative_week', and 'is_moa_data'
    """Performs a simplified Metropolis-Hastings MCMC to infer SIR[H]2 model parameters.

    Uses curve_fit for initial parameter guesses.
    """
    y_data_raw = training_data_df['per_capita_admissions'].values
    y_data_abs = y_data_raw * self.sirh_model.N
    t_data_weeks = training_data_df['relative_week'].values
    is_moa_data_flags = training_data_df['is_moa_data'].values

    self.last_t_index_weeks = t_data_weeks[-1] if len(t_data_weeks) > 0 else 0

    # Fallback if no data or very low signal (ADJUSTED THRESHOLDS for more SIRH use)
    # Relaxed to sum < 0.1 or len < 2 (was < 0.5 and < 2)
    if np.sum(y_data_abs) < 0.1 or len(y_data_abs) < 2:
      self.mcmc_samples = (
          []
      )  # Ensure mcmc_samples is empty to trigger caller's fallback
      return

    # Robust initial guesses for SIRH parameters

    # Smooth y_data_abs to find a more robust peak and onset/offset
    y_data_smoothed = (
        pd.Series(y_data_abs)
        .rolling(window=5, center=True, min_periods=1)
        .mean()
        .values
    )
    peak_val = np.max(y_data_smoothed) if len(y_data_smoothed) > 0 else 0.0

    # --- Improved t_onset_days and t_offset_days initial guesses ---
    onset_detection_threshold = max(
        0.1, peak_val * 0.0005
    )  # Dynamic threshold for onset detection, slightly lower and more sensitive

    t_onset_guess_days = 0
    t_offset_guess_days = 0

    active_weeks_indices = np.where(
        y_data_smoothed >= onset_detection_threshold
    )[0]

    # Calculate maximum relative day in the full SIRH fit data window (actual + MOA)
    # This defines the full potential range for the epidemic curve
    max_relative_day_in_fit_data = (
        t_data_weeks.max() * self.sirh_model.days_per_week
        if len(t_data_weeks) > 0
        else 0
    )

    if len(active_weeks_indices) > 0:
      first_active_week = active_weeks_indices[0]
      last_active_week = active_weeks_indices[-1]

      # Extend guesses a bit beyond detected active period for robustness
      t_onset_guess_days = max(
          0, int((first_active_week - 4) * self.sirh_model.days_per_week)
      )  # Increased buffer
      t_offset_guess_days = int(
          (last_active_week + 12) * self.sirh_model.days_per_week
      )  # Increased offset buffer for curve_fit
    else:  # If no strong signal above threshold, use full season timing or a fixed long duration
      t_onset_guess_days = 0  # Start from beginning of the season window
      # Ensure t_offset_guess_days covers at least 30 weeks if no signal detected
      t_offset_guess_days = max(
          max_relative_day_in_fit_data, self.sirh_model.days_per_week * 30
      )

    # Ensure minimum duration for initial guesses, even if calculated period is short
    min_epidemic_duration_days_init_guess = (
        self.sirh_model.days_per_week * 8
    )  # At least 8 weeks
    if (
        t_offset_guess_days - t_onset_guess_days
        < min_epidemic_duration_days_init_guess
    ):
      t_offset_guess_days = (
          t_onset_guess_days + min_epidemic_duration_days_init_guess
      )
      # Clip to max_relative_day_in_fit_data + a buffer, but ensure sufficient length
      t_offset_guess_days = max(
          t_offset_guess_days,
          max_relative_day_in_fit_data + self.sirh_model.days_per_week * 4,
      )

    # Bounds for SIRH parameters (original scale for curve_fit) - UPDATED (WIDENED)
    bounds_sirh_lower = [
        0.0,
        1e-8,
        1e-7,
        1e-7,
        1e-7,
        0,
        0,
        1.0,
        1.0,
    ]  # Adjusted lower bounds for stability, initial_I_prop to 1e-8
    bounds_sirh_upper = [
        0.9,
        0.2,
        0.3,
        1.0,
        7.0,  # ADJUSTED rho_H upper bound to 0.3, beta_amplitude to 7.0
        max_relative_day_in_fit_data
        + self.sirh_model.days_per_week * 10,  # Extended max t_onset
        max_relative_day_in_fit_data
        + self.sirh_model.days_per_week * 12,  # Extended max t_offset
        500.0,
        500.0,
    ]  # WIDENED sharpness bounds

    # Ensure t_onset_guess and t_offset_guess are within initial bounds
    t_onset_guess_days = np.clip(
        t_onset_guess_days, bounds_sirh_lower[5], bounds_sirh_upper[5]
    )
    t_offset_guess_days = np.clip(
        t_offset_guess_days, bounds_sirh_lower[6], bounds_sirh_upper[6]
    )

    # Further ensure t_offset_guess is strictly after t_onset_guess with minimum gap
    # Relaxed minimum epidemic duration for curve_fit init guess to 3 weeks (was 0.1 week)
    min_epidemic_duration_days_init_guess_strict = (
        self.sirh_model.days_per_week * 3
    )
    if (
        t_offset_guess_days
        <= t_onset_guess_days + min_epidemic_duration_days_init_guess_strict
    ):
      t_offset_guess_days = (
          t_onset_guess_days + min_epidemic_duration_days_init_guess_strict
      )
      t_offset_guess_days = np.clip(
          t_offset_guess_days, bounds_sirh_lower[6], bounds_sirh_upper[6]
      )

    # --- Updated p0 to include initial_immune_prop and refined guesses (MORE DYNAMIC) ---
    p0_sirh = [
        0.2,  # initial_immune_prop
        max(
            bounds_sirh_lower[1], min(0.0001, 5.0 / self.sirh_model.N)
        ),  # initial_I_prop, at least 5 people / N, capped at 0.0001
        max(
            bounds_sirh_lower[2],
            min(
                bounds_sirh_upper[2],
                0.0001
                if peak_val < 100
                else (0.0005 if peak_val < 500 else 0.002),
            ),
        ),  # ADJUSTED rho_H initial guess based on peak
        0.005,  # base_beta
        max(
            bounds_sirh_lower[4],
            min(
                bounds_sirh_upper[4],
                0.2 if peak_val < 100 else (1.0 if peak_val < 1000 else 2.0),
            ),
        ),  # beta_amplitude more dynamic
        t_onset_guess_days,
        t_offset_guess_days,
        10.0,  # sharpness_rise
        10.0,  # sharpness_fall
    ]

    # 1. Use curve_fit to find a good starting point (popt) for SIRH parameters
    try:
      # Create a lambda function to pass the fixed arguments to _simulate_sirh2_daily
      popt_sirh, _ = curve_fit(
          self.sirh_model._simulate_sirh2_daily,
          t_data_weeks,
          y_data_abs,
          p0=p0_sirh,
          bounds=(bounds_sirh_lower, bounds_sirh_upper),
          maxfev=500000,  # Increased maxfev to 500,000
          method='trf',
          jac='3-point',
      )
    except Exception as e:
      warnings.warn(
          'Initial curve_fit failed for MCMC start for population'
          f' {self.sirh_model.N}: {e}. Using robust p0 instead.'
      )
      # Fallback for popt if curve_fit fails, use adjusted p0
      # Ensure fallback popt_sirh is also clipped to bounds for robustness
      popt_sirh = np.clip(
          np.array(p0_sirh), bounds_sirh_lower, bounds_sirh_upper
      )

    # Now extend with initial guess for log_r for Negative Binomial
    # Improved initial guess for Negative Binomial r parameter based on mean and variance
    valid_y_data_abs = y_data_abs[
        y_data_abs > 0
    ]  # Consider only positive observations for mean/var
    mean_y = np.mean(valid_y_data_abs) if len(valid_y_data_abs) > 0 else 0
    var_y = np.var(valid_y_data_abs) if len(valid_y_data_abs) > 1 else 0

    # MODIFICATION: Changed 10000.0 to 100.0 for more flexible initial 'r'
    r_param_init = (
        2.0  # Default 'r' (changed from 1000.0 for more spread initially)
    )
    if var_y > mean_y and mean_y > 0:
      # Moment-based estimate for r in Negative Binomial
      r_param_init = max(
          0.5, mean_y**2 / (var_y - mean_y)
      )  # Lower bound for r adjusted to 0.5
    elif mean_y > 0:
      r_param_init = 100.0  # ADJUSTED: If variance <= mean, use a slightly large 'r' to allow more variability than 10000.0
    else:
      r_param_init = 10000.0  # Very high 'r' if no signal, indicates low variance, pushing towards 0

    # Prepare initial MCMC parameters: apply log transform to specific parameters
    p0_full_untransformed = list(popt_sirh) + [np.log(r_param_init)]
    current_params = np.array(p0_full_untransformed)
    for idx in LOG_TRANSFORM_INDICES:
      # Ensure the value is positive before taking log, clip if necessary
      current_params[idx] = np.log(
          np.maximum(1e-7, current_params[idx])
      )  # Adjusted clip for log-transformed params

    # Define bounds for the MCMC, in the transformed space for log-transformed parameters - UPDATED (WIDENED)
    bounds_full_lower_mcmc = np.array(
        bounds_sirh_lower + [np.log(0.001)]
    )  # ADJUSTED log_r lower bound
    bounds_full_upper_mcmc = np.array(
        bounds_sirh_upper + [np.log(10000.0)]
    )  # ADJUSTED log_r upper bound to 10000.0
    for idx in LOG_TRANSFORM_INDICES:
      # Adjusted lower bound for log-transformed parameters in MCMC search space
      bounds_full_lower_mcmc[idx] = np.log(
          np.maximum(1e-8, bounds_sirh_lower[idx])
      )  # initial_I_prop lower bound 1e-8
      bounds_full_upper_mcmc[idx] = np.log(
          np.maximum(1e-7, bounds_sirh_upper[idx])
      )

    current_log_likelihood = self._log_likelihood(
        current_params, t_data_weeks, y_data_abs, is_moa_data_flags
    )

    # Robustify MCMC start: If initial likelihood is -inf, perturb parameters and retry
    if current_log_likelihood == -np.inf:
      warnings.warn(
          'Initial MCMC likelihood is -inf. Attempting to find a valid starting'
          ' point by perturbing.'
      )
      best_init_likelihood = -np.inf
      best_init_params = None
      # Try a few perturbations around the initial guess
      for _ in range(20):  # Try 20 times
        perturbed_params = np.array(current_params) + np.random.normal(
            0, MCMC_PROPOSAL_SCALE * 2, len(current_params)
        )  # Larger perturbation
        # Clip perturbed parameters to bounds
        for j in range(len(perturbed_params)):
          perturbed_params[j] = np.clip(
              perturbed_params[j],
              bounds_full_lower_mcmc[j],
              bounds_full_upper_mcmc[j],
          )

        # Check t_onset_days < t_offset_days constraint for direct parameters
        temp_params_unpacked_for_check = list(perturbed_params)
        for idx in LOG_TRANSFORM_INDICES:
          temp_params_unpacked_for_check[idx] = np.exp(perturbed_params[idx])

        # --- MODIFICATION START ---
        # Use simplified onset/offset check for initial perturbations
        if not (
            temp_params_unpacked_for_check[5]
            < temp_params_unpacked_for_check[6]
            - 1.0  # Onset must be at least 1 day before offset
            and (
                temp_params_unpacked_for_check[6]
                - temp_params_unpacked_for_check[5]
            )
            >= self.sirh_model.days_per_week * 3
        ):  # Minimum 3 weeks duration
          continue  # Skip if constraint violated
        # --- MODIFICATION END ---

        temp_likelihood = self._log_likelihood(
            perturbed_params, t_data_weeks, y_data_abs, is_moa_data_flags
        )
        if temp_likelihood > best_init_likelihood:
          best_init_likelihood = temp_likelihood
          best_init_params = perturbed_params

      if best_init_likelihood == -np.inf:
        warnings.warn(
            'Could not find a valid MCMC starting point after perturbations.'
            ' MCMC will likely fail.'
        )
        self.mcmc_samples = []  # Force fallback
        return
      else:
        current_params = best_init_params
        current_log_likelihood = best_init_likelihood

    mcmc_samples_raw = []

    # Simplified Metropolis-Hastings MCMC loop
    for i in range(NUM_MCMC_STEPS):
      proposed_params = np.array(current_params)

      # Propose new parameters by adding Gaussian noise scaled by proposal_scale
      # Noise applied in log-space for log-transformed parameters
      noise = np.random.normal(0, MCMC_PROPOSAL_SCALE, len(current_params))
      proposed_params += noise

      # Clip proposed parameters to stay within MCMC bounds (already in transformed space)
      for j in range(len(proposed_params)):
        proposed_params[j] = np.clip(
            proposed_params[j],
            bounds_full_lower_mcmc[j],
            bounds_full_upper_mcmc[j],
        )

      # Ensure t_onset_days < t_offset_days (with buffer) for direct parameters
      # Unpack direct parameters for this check (re-applying exp for log-transformed ones for the check)
      temp_params_unpacked_for_check = list(proposed_params)
      for idx in LOG_TRANSFORM_INDICES:
        temp_params_unpacked_for_check[idx] = np.exp(
            proposed_params[idx]
        )  # Convert back for logic

      # --- MODIFICATION START ---
      # Use simplified onset/offset check for MCMC proposals, matching _log_likelihood
      if not (
          temp_params_unpacked_for_check[5]
          < temp_params_unpacked_for_check[6]
          - 1.0  # Onset must be at least 1 day before offset
          and (
              temp_params_unpacked_for_check[6]
              - temp_params_unpacked_for_check[5]
          )
          >= self.sirh_model.days_per_week * 3
      ):  # Minimum 3 weeks duration
        # If constraint violated, reject proposal immediately
        proposed_log_likelihood = -np.inf
      # --- MODIFICATION END ---
      else:
        proposed_log_likelihood = self._log_likelihood(
            proposed_params, t_data_weeks, y_data_abs, is_moa_data_flags
        )

      # Metropolis-Hastings acceptance ratio
      if proposed_log_likelihood == -np.inf:
        alpha = 0.0  # Reject if proposed likelihood is invalid
      elif current_log_likelihood == -np.inf:
        # If current likelihood is invalid, but proposed is valid, always accept
        alpha = 1.0 if proposed_log_likelihood > -np.inf else 0.0
      else:
        alpha = np.exp(proposed_log_likelihood - current_log_likelihood)

      if np.random.rand() < alpha:
        current_params = proposed_params
        current_log_likelihood = proposed_log_likelihood

      # Store sample after burn-in and thinning
      if i >= BURN_IN and (i % THINNING == 0):
        mcmc_samples_raw.append(current_params.copy())

    if not mcmc_samples_raw:
      self.mcmc_samples = []
      warnings.warn(
          'MCMC sampling failed to produce valid samples for this fit. No SIRH'
          ' model output.'
      )
    else:
      self.mcmc_samples = mcmc_samples_raw

  def predict_stochastic(
      self, forecast_horizon_weeks, num_simulations = NUM_SIMULATIONS
  ):
    """Performs stochastic simulations using parameters sampled from the MCMC chain.

    Returns zeros if no MCMC samples were generated during fitting.
    """
    if not self.mcmc_samples:
      return np.zeros((num_simulations, forecast_horizon_weeks))

    # Sample from the stored MCMC chain (with replacement if num_simulations > len(mcmc_samples))
    indices = np.random.choice(
        len(self.mcmc_samples), size=num_simulations, replace=True
    )
    sampled_params_raw = np.array(self.mcmc_samples)[indices]

    simulated_hosp_trajectories = np.zeros(
        (num_simulations, forecast_horizon_weeks)
    )

    # Generate forecast time points (weeks), continuing from the end of training data's relative weeks
    forecast_t_indices_weeks = np.arange(
        self.last_t_index_weeks + 1,
        self.last_t_index_weeks + 1 + forecast_horizon_weeks,
    )

    for s in range(num_simulations):
      # Unpack parameters, applying np.exp() for log-transformed ones
      current_params_sim = list(sampled_params_raw[s])
      for idx in LOG_TRANSFORM_INDICES:
        current_params_sim[idx] = np.exp(sampled_params_raw[s][idx])

      (
          initial_immune_prop,
          initial_I_prop,
          rho_H,
          base_beta,
          beta_amplitude,
          t_onset_days,
          t_offset_days,
          sharpness_rise,
          sharpness_fall,
          log_r,
      ) = current_params_sim

      hosp_preds_mean = self.sirh_model._simulate_sirh2_daily(
          forecast_t_indices_weeks,
          initial_immune_prop,
          initial_I_prop,
          rho_H,
          base_beta,
          beta_amplitude,
          t_onset_days,
          t_offset_days,
          sharpness_rise,
          sharpness_fall,
      )

      # Stochasticity using Negative Binomial for prediction
      r_param_sample = np.exp(log_r)

      # Use nbinom.rvs to get count realizations
      hosp_preds_mean_safe = np.maximum(1e-9, hosp_preds_mean)
      p_for_nbinom = np.clip(
          r_param_sample / (r_param_sample + hosp_preds_mean_safe),
          1e-9,
          1.0 - 1e-9,
      )

      # Sample from Negative Binomial
      sampled_counts = nbinom.rvs(
          n=r_param_sample, p=p_for_nbinom, size=len(forecast_t_indices_weeks)
      )

      simulated_hosp_trajectories[s, :] = np.maximum(0, sampled_counts)

    return simulated_hosp_trajectories


# --- Global ILINet Preprocessing (moved out of function for efficiency) ---
ilinet_state_clean_global = None
ilinet_synthetic_history_df_global = None
global_median_ili_to_admissions_ratio = (
    None  # New global variable for robust fallback
)


def _prepare_ilinet_synthetic_history(
    ilinet_state_raw, locations_df, train_df_actual
):
  global ilinet_state_clean_global, ilinet_synthetic_history_df_global, global_median_ili_to_admissions_ratio

  if ilinet_synthetic_history_df_global is not None:
    return

  ilinet_state_clean_global = ilinet_state_raw[
      ilinet_state_raw['region_type'] == 'States'
  ].copy()
  ilinet_state_clean_global['week_start'] = pd.to_datetime(
      ilinet_state_clean_global['week_start']
  )

  location_name_to_fips = locations_df.set_index('location_name')[
      'location'
  ].to_dict()
  ilinet_state_clean_global['location'] = ilinet_state_clean_global[
      'region'
  ].map(location_name_to_fips)
  ilinet_state_clean_global = ilinet_state_clean_global.dropna(
      subset=['location']
  )
  ilinet_state_clean_global['location'] = ilinet_state_clean_global[
      'location'
  ].astype(int)

  ilinet_state_clean_global = ilinet_state_clean_global.merge(
      locations_df[['location', 'population']].drop_duplicates(),
      on='location',
      how='left',
  )

  # Temporary lists to store scaling info for each location and all ratios for global median calculation
  loc_scaling_results = {}
  all_valid_ratios_for_global = (
      []
  )  # Collect all robust local ratios/slopes here

  # First pass: Learn transformations and collect successful ratios for global median
  for loc_id in train_df_actual['location'].unique():
    loc_train_df = train_df_actual[train_df_actual['location'] == loc_id].copy()
    loc_ilinet_df = ilinet_state_clean_global[
        ilinet_state_clean_global['location'] == loc_id
    ].copy()

    # Fixed overlap period for consistency in learning global scaling
    overlap_start = pd.Timestamp('2020-09-01')
    overlap_end = pd.Timestamp('2022-10-15')

    overlap_train = loc_train_df[
        (loc_train_df['target_end_date'] >= overlap_start)
        & (loc_train_df['target_end_date'] <= overlap_end)
    ].copy()
    overlap_ilinet = loc_ilinet_df[
        (loc_ilinet_df['week_start'] >= overlap_start)
        & (loc_ilinet_df['week_start'] <= overlap_end)
    ].copy()

    overlap_train['per_capita_admissions'] = (
        overlap_train[TARGET_STR] / overlap_train['population']
    )

    merged_overlap = pd.merge(
        overlap_train[['target_end_date', 'per_capita_admissions']],
        overlap_ilinet[['week_start', 'unweighted_ili']],
        left_on='target_end_date',
        right_on='week_start',
        how='inner',
    ).dropna()

    scaling_model = None
    r_squared = 0.0
    local_mean_ratio = None

    if (
        not merged_overlap.empty
        and merged_overlap['unweighted_ili'].notna().any()
    ):
      X = merged_overlap[['unweighted_ili']]
      y = merged_overlap['per_capita_admissions']
      valid_idx = np.isfinite(X['unweighted_ili']) & np.isfinite(y)
      X_clean = X[valid_idx]
      y_clean = y[valid_idx]

      if (
          not X_clean.empty
          and len(X_clean) >= 5
          and X_clean['unweighted_ili'].nunique() > 1
      ):
        scaling_model_with_intercept = LinearRegression(
            positive=True, fit_intercept=True
        )
        r_squared_with_intercept = -np.inf
        try:
          scaling_model_with_intercept.fit(X_clean, y_clean)
          r_squared_with_intercept = scaling_model_with_intercept.score(
              X_clean, y_clean
          )
        except ValueError:
          pass

        scaling_model_no_intercept = LinearRegression(
            positive=True, fit_intercept=False
        )
        r_squared_no_intercept = -np.inf
        try:
          scaling_model_no_intercept.fit(X_clean, y_clean)
          r_squared_no_intercept = scaling_model_no_intercept.score(
              X_clean, y_clean
          )
        except ValueError:
          pass

        best_scaling_model = None
        best_r_squared = -np.inf

        # ADJUSTMENT: Increased r_squared threshold to 0.5 (from 0.6) for choosing best LR model
        if (
            r_squared_with_intercept > 0.5
            and scaling_model_with_intercept.coef_[0] > 0
        ):
          best_scaling_model = scaling_model_with_intercept
          best_r_squared = r_squared_with_intercept

        # ADJUSTMENT: Increased r_squared threshold to 0.5 (from 0.6) for choosing best LR model
        if (
            r_squared_no_intercept > 0.5
            and scaling_model_no_intercept.coef_[0] > 0
        ):
          if not best_scaling_model or r_squared_no_intercept > best_r_squared:
            best_scaling_model = scaling_model_no_intercept
            best_r_squared = r_squared_no_intercept

        scaling_model = best_scaling_model
        r_squared = best_r_squared if best_scaling_model else 0.0

      else:
        warnings.warn(
            'Insufficient variation in unweighted_ili for linear regression'
            f' for location {loc_id}.'
        )

      # Calculate local mean ratio regardless for robust fallback
      if y_clean.mean() > 0 and X_clean['unweighted_ili'].mean() > 0:
        local_mean_ratio = y_clean.mean() / X_clean['unweighted_ili'].mean()
      elif y_clean.mean() > 0:
        local_mean_ratio = (
            1e-5  # Small positive if ILI is zero but admissions are positive
        )
      elif X_clean['unweighted_ili'].mean() > 0:
        local_mean_ratio = (
            1e-6  # Even smaller if ILI is positive but admissions are zero
        )
      else:
        local_mean_ratio = 0.0  # Both are zero

    loc_scaling_results[loc_id] = {
        'model': scaling_model,
        'r_squared': r_squared,
        'local_mean_ratio': local_mean_ratio,
        'ilinet_df': loc_ilinet_df,
    }

    # Add any valid positive ratio (slope or mean ratio) to the global pool for median calculation
    # Removed the arbitrary local_mean_ratio < 0.1 filter.
    if scaling_model and scaling_model.coef_[0] > 0 and r_squared > 0.5:
      all_valid_ratios_for_global.append(scaling_model.coef_[0])
    elif (
        local_mean_ratio is not None and local_mean_ratio > 0
    ):  # Include any valid positive mean ratio
      all_valid_ratios_for_global.append(local_mean_ratio)

  # Calculate global median ILI to Admissions ratio from all robust local methods
  # ADJUSTED: If no valid ratios, use a more reasonable default (e.g., 5e-5)
  global_median_ili_to_admissions_ratio = (
      np.median(all_valid_ratios_for_global)
      if all_valid_ratios_for_global
      else 5e-5
  )

  # Second pass: Apply transformations using the best available scaling (LR, local ratio, or global fallback)
  augmented_data_list_for_scaling = []
  for loc_id in train_df_actual['location'].unique():
    res = loc_scaling_results[loc_id]
    loc_ilinet_df = res[
        'ilinet_df'
    ].copy()  # Use a copy to avoid SettingWithCopyWarning
    scaling_model = res['model']
    r_squared = res['r_squared']
    local_mean_ratio = res['local_mean_ratio']

    loc_ilinet_df['synthetic_per_capita_admissions'] = 0.0  # Initialize column
    valid_ili_idx = loc_ilinet_df['unweighted_ili'].notna() & (
        loc_ilinet_df['unweighted_ili'] >= 0
    )  # Ensure non-negative ILI

    # Apply hierarchical scaling based on robustness
    # Use LR more often when justified (r_squared threshold for use is now 0.5)
    if scaling_model and scaling_model.coef_[0] > 0 and r_squared > 0.5:
      loc_ilinet_df.loc[valid_ili_idx, 'synthetic_per_capita_admissions'] = (
          scaling_model.predict(
              np.maximum(
                  0, loc_ilinet_df.loc[valid_ili_idx, ['unweighted_ili']]
              )
          )
      )
    elif (
        local_mean_ratio is not None and local_mean_ratio > 0
    ):  # Fallback to local mean ratio if available
      loc_ilinet_df.loc[valid_ili_idx, 'synthetic_per_capita_admissions'] = (
          np.maximum(
              0,
              loc_ilinet_df.loc[valid_ili_idx, 'unweighted_ili']
              * local_mean_ratio,
          )
      )
    else:  # Ultimate fallback to global median ratio
      loc_ilinet_df.loc[valid_ili_idx, 'synthetic_per_capita_admissions'] = (
          np.maximum(
              0,
              loc_ilinet_df.loc[valid_ili_idx, 'unweighted_ili']
              * global_median_ili_to_admissions_ratio,
          )
      )

    loc_ilinet_df['synthetic_per_capita_admissions'] = np.maximum(
        0, loc_ilinet_df['synthetic_per_capita_admissions']
    )

    ilinet_augmented_loc = loc_ilinet_df[[
        'week_start',
        'location',
        'population',
        'synthetic_per_capita_admissions',
    ]].rename(
        columns={
            'week_start': 'target_end_date',
            'synthetic_per_capita_admissions': 'per_capita_admissions',
        }
    )
    ilinet_augmented_loc['is_moa_data'] = True

    augmented_data_list_for_scaling.append(ilinet_augmented_loc)

  ilinet_synthetic_history_df_global = pd.concat(
      augmented_data_list_for_scaling, ignore_index=True
  )
  ilinet_synthetic_history_df_global['target_end_date'] = pd.to_datetime(
      ilinet_synthetic_history_df_global['target_end_date']
  )


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x by modelling train_x to train_y and return quantiles.

  Implements a simplified SIR[H]2 model with historical ILINet augmentation
  (MOA-like data) and MCMC-based inference, adhering more closely to the method
  description.
  """
  # Global variables from notebook preamble (assuming these are loaded in the environment)
  # global ilinet_state, locations # These are not passed as args, assume accessible.

  # 1. Data Preparation and ILINet Augmentation (Historical Data)
  train_df = train_x.copy()
  train_df[TARGET_STR] = train_y.copy()

  # Ensure target_end_date is datetime objects for consistency
  train_df['target_end_date'] = pd.to_datetime(train_df['target_end_date'])
  test_x_original_index = test_x.index
  test_x = test_x.copy()
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])

  # Prepare the global ILINet synthetic history if not already done
  _prepare_ilinet_synthetic_history(ilinet_state, locations, train_df)

  train_df['per_capita_admissions'] = (
      train_df[TARGET_STR] / train_df['population']
  )
  train_df['is_moa_data'] = False  # Actual data is not MOA data

  # Prepare output DataFrame
  test_y_hat_quantiles = pd.DataFrame(
      index=test_x_original_index,
      columns=[f'quantile_{q}' for q in QUANTILES],
      dtype=float,
  )

  # 2. Model Training (SIRH with MCMC) and Prediction (per Location)
  unique_locations_to_predict = test_x['location'].unique()

  for loc_id in unique_locations_to_predict:
    loc_actual_train_df = train_df[train_df['location'] == loc_id].copy()
    loc_ilinet_history_scaled = ilinet_synthetic_history_df_global[
        ilinet_synthetic_history_df_global['location'] == loc_id
    ].copy()

    loc_test_x_rows = test_x[test_x['location'] == loc_id].copy()
    loc_test_x_indices = loc_test_x_rows.index
    forecast_horizon_weeks = len(loc_test_x_rows)

    if loc_actual_train_df.empty:  # No actual training data for this location
      for q_col in test_y_hat_quantiles.columns:
        test_y_hat_quantiles.loc[loc_test_x_indices, q_col] = 0
      continue

    loc_population = loc_actual_train_df['population'].iloc[
        0
    ]  # Population is assumed constant per loc

    # --- Define current season window for SIRH fit ---
    sirh_fit_end_date = loc_actual_train_df[
        'target_end_date'
    ].max()  # Last date of actual training data
    current_season_year = get_flu_season_year(sirh_fit_end_date)

    # Approximate season start date (e.g., first Saturday of October in the *previous* calendar year)
    # For 2023/24 season (current_season_year = 2024), season_start_date_base is Oct 2023.
    season_start_date_base = pd.Timestamp(
        year=current_season_year - 1, month=10, day=1
    )
    current_season_start_date = season_start_date_base.to_period(
        'W-SAT'
    ).start_time

    # Actual data for the current season up to train_end_date
    current_season_actual_data = loc_actual_train_df[
        (loc_actual_train_df['target_end_date'] >= current_season_start_date)
        & (loc_actual_train_df['target_end_date'] <= sirh_fit_end_date)
    ].copy()

    # --- Generate MOA-derived "future" data for the current season ---
    # This provides a full epidemic profile for SIRH model to fit, even with sparse actual data
    moa_forecast_start_date = sirh_fit_end_date + pd.Timedelta(weeks=1)

    # ADJUSTMENT: Dynamically set moa_forecast_end_date to cover a full season or forecast horizon + buffer
    moa_forecast_end_date_for_sirh_fit = max(
        current_season_start_date
        + pd.Timedelta(
            weeks=52
        ),  # Ensure minimum full season length (52 weeks)
        loc_test_x_rows['target_end_date'].max()
        + pd.Timedelta(weeks=8),  # Cover forecast plus buffer
    )

    moa_dates_range = pd.date_range(
        start=moa_forecast_start_date,
        end=moa_forecast_end_date_for_sirh_fit,
        freq='W-SAT',
    )
    moa_data_list = []

    # Find historical ILINet seasons that are *not* the current season to avoid data leakage
    past_ilinet_seasons = loc_ilinet_history_scaled[
        loc_ilinet_history_scaled['target_end_date'] < current_season_start_date
    ].copy()

    for moa_date in moa_dates_range:
      epi_week = moa_date.isocalendar().week

      # Median of synthetic per_capita_admissions for this epi_week from past ILINet seasons
      past_ilinet_for_epiweek = past_ilinet_seasons[
          past_ilinet_seasons['target_end_date'].dt.isocalendar().week
          == epi_week
      ]
      moa_value = (
          past_ilinet_for_epiweek['per_capita_admissions'].median()
          if not past_ilinet_for_epiweek.empty
          else 0.0
      )

      moa_data_list.append({
          'target_end_date': moa_date,
          'location': loc_id,
          'population': loc_population,
          'per_capita_admissions': moa_value,
          'is_moa_data': True,
      })
    moa_future_synthetic_df = pd.DataFrame(moa_data_list)

    # Combine actual data with MOA future data for SIRH fitting
    sirh_fit_data = pd.concat(
        [current_season_actual_data, moa_future_synthetic_df]
    )
    sirh_fit_data = (
        sirh_fit_data.drop_duplicates(subset=['target_end_date'])
        .sort_values('target_end_date')
        .reset_index(drop=True)
    )

    # Filter to the relevant season window
    sirh_fit_data = (
        sirh_fit_data[
            (sirh_fit_data['target_end_date'] >= current_season_start_date)
            & (
                sirh_fit_data['target_end_date']
                <= moa_forecast_end_date_for_sirh_fit
            )
        ]
        .sort_values('target_end_date')
        .reset_index(drop=True)
    )

    # Recalculate relative_week for this specific season window to start from 0
    sirh_fit_data['relative_week'] = np.arange(len(sirh_fit_data))

    # Initialize and fit SIRH_Forecaster (MCMC-based inference)
    sirh_forecaster = SIRH_Forecaster(N=loc_population)

    # Flag to determine if fallback is needed
    needs_fallback = False
    # ADJUSTMENT: Relaxed fallback thresholds for SIR[H]2 model to run more often
    if (
        sirh_fit_data.empty
        or sirh_fit_data['per_capita_admissions'].sum() < 0.1
        or len(sirh_fit_data) < 2
    ):  # Relaxed to sum < 0.1 and len < 2
      warnings.warn(
          'Insufficient total data/signal (actual + MOA) for SIR[H]2 model fit'
          f' for location {loc_id}. Falling back to statistical model.'
      )
      needs_fallback = True
    else:
      sirh_forecaster.fit(sirh_fit_data)  # Pass DataFrame including is_moa_data
      if (
          not sirh_forecaster.mcmc_samples
      ):  # MCMC failed to produce valid samples
        warnings.warn(
            f'SIR[H]2 MCMC fit failed for location {loc_id} after windowing and'
            ' MOA. Falling back to statistical model.'
        )
        needs_fallback = True

    if needs_fallback:
      # --- Improved Baseline statistical model for no epi signal (using BOTH actual and synthetic historical data) ---

      # New: Define how many recent seasons to consider for fallback
      RECENT_SEASON_COUNT = 3

      # Calculate the year for the start of the current flu season (based on reference_date)
      current_ref_date = loc_test_x_rows['reference_date'].iloc[0]
      current_ref_date_season_year = get_flu_season_year(current_ref_date)

      # Determine the earliest season year to consider for fallback data
      earliest_fallback_season_year = (
          current_ref_date_season_year - RECENT_SEASON_COUNT
      )

      # 1. Actual historical data for fallback
      actual_hist_for_fallback = loc_actual_train_df[
          (
              loc_actual_train_df['target_end_date']
              < loc_test_x_rows['target_end_date'].min()
          )
          & (
              loc_actual_train_df['target_end_date'].apply(get_flu_season_year)
              >= earliest_fallback_season_year
          )
      ].copy()
      actual_hist_for_fallback['absolute_admissions'] = (
          actual_hist_for_fallback['per_capita_admissions'] * loc_population
      )
      actual_hist_for_fallback = actual_hist_for_fallback[
          ['target_end_date', 'absolute_admissions']
      ]

      # 2. Synthetic ILINet historical data for fallback (ensure it's from the global preprocessed dataset)
      ilinet_hist_for_fallback = ilinet_synthetic_history_df_global[
          (ilinet_synthetic_history_df_global['location'] == loc_id)
          & (
              ilinet_synthetic_history_df_global['target_end_date']
              < loc_test_x_rows['target_end_date'].min()
          )
          & (
              ilinet_synthetic_history_df_global['target_end_date'].apply(
                  get_flu_season_year
              )
              >= earliest_fallback_season_year
          )
      ].copy()
      ilinet_hist_for_fallback['absolute_admissions'] = (
          ilinet_hist_for_fallback['per_capita_admissions'] * loc_population
      )
      ilinet_hist_for_fallback = ilinet_hist_for_fallback[
          ['target_end_date', 'absolute_admissions']
      ]

      # 3. Combine actual and synthetic ILINet historical data for fallback, actual takes precedence on overlap
      combined_fallback_historical_data = pd.concat(
          [actual_hist_for_fallback, ilinet_hist_for_fallback]
      ).drop_duplicates(subset=['target_end_date'], keep='first')
      combined_fallback_historical_data['target_end_date'] = pd.to_datetime(
          combined_fallback_historical_data['target_end_date']
      )

      num_dummy_samples = NUM_SIMULATIONS
      dummy_samples_abs = np.zeros((num_dummy_samples, forecast_horizon_weeks))

      # ADJUSTMENT: Wider epi-week window size for fallback
      EPI_WEEK_WINDOW_SIZE = 13

      for week_idx, test_idx in enumerate(loc_test_x_indices):
        current_target_epi_week = (
            loc_test_x_rows.loc[test_idx, 'target_end_date'].isocalendar().week
        )

        # Identify relevant epi-weeks for the window, handling wrap-around for week numbers
        relevant_epi_weeks = set()
        for offset in range(
            -EPI_WEEK_WINDOW_SIZE // 2, EPI_WEEK_WINDOW_SIZE // 2 + 1
        ):
          week_candidate = current_target_epi_week + offset
          if week_candidate > 52:
            week_candidate -= 52
          elif week_candidate < 1:
            week_candidate += 52
          relevant_epi_weeks.add(week_candidate)

        # Get historical data for the specific epi-week window from the combined data
        weekly_historical_data_window = combined_fallback_historical_data[(
            combined_fallback_historical_data['target_end_date']
            .dt.isocalendar()
            .week.isin(relevant_epi_weeks)
        )]['absolute_admissions']

        mean_weekly_admissions_abs = (
            weekly_historical_data_window.mean()
            if not weekly_historical_data_window.empty
            else 0.0
        )
        var_weekly_admissions_abs = (
            weekly_historical_data_window.var()
            if len(weekly_historical_data_window) > 1
            else 0.0
        )

        fallback_mean_abs = max(0.0, mean_weekly_admissions_abs)

        # --- MODIFICATION START ---
        # Improved Fallback Model Stochasticity
        if fallback_mean_abs == 0:
          # Explicitly predict zeros if historical data is all zeros for this epi-week window
          sampled_counts = np.zeros(num_dummy_samples)
        elif var_weekly_admissions_abs == 0.0:
          # If mean > 0 but variance is 0, use a reasonable 'r' to introduce spread
          # (e.g., r=1.0 or r=0.5) to better characterize uncertainty.
          # This prevents an overly sharp (Poisson-like) prediction.
          r_fallback = 1.0  # Set a fixed 'r' to ensure some spread
          p_fallback = np.clip(
              r_fallback / (r_fallback + fallback_mean_abs), 1e-9, 1.0 - 1e-9
          )
          sampled_counts = nbinom.rvs(
              n=r_fallback, p=p_fallback, size=num_dummy_samples
          )
        else:  # fallback_mean_abs > 0 and var_weekly_admissions_abs > 0
          if var_weekly_admissions_abs > fallback_mean_abs:
            r_fallback = max(
                0.5,
                fallback_mean_abs**2
                / (var_weekly_admissions_abs - fallback_mean_abs),
            )
          else:  # var_weekly_admissions_abs <= fallback_mean_abs (or very close to it) -> Poisson-like, but still maintain some spread
            r_fallback = 1.0  # Use r=1.0 instead of a very large number like 10000.0 to allow for some variability

          p_fallback = np.clip(
              r_fallback / (r_fallback + fallback_mean_abs), 1e-9, 1.0 - 1e-9
          )
          sampled_counts = nbinom.rvs(
              n=r_fallback, p=p_fallback, size=num_dummy_samples
          )
        # --- MODIFICATION END ---

        dummy_samples_abs[:, week_idx] = np.maximum(0, sampled_counts)

      for week_idx, test_idx in enumerate(loc_test_x_indices):
        # Calculate quantiles for this specific forecast point from dummy_samples_abs
        weekly_quantiles_float = np.percentile(
            dummy_samples_abs[:, week_idx], [q * 100 for q in QUANTILES]
        )
        # Ensure monotonicity and non-negativity before rounding
        weekly_quantiles_sorted = (
            np.maximum(0, np.sort(weekly_quantiles_float)).round().astype(int)
        )
        test_y_hat_quantiles.loc[
            test_idx, [f'quantile_{q}' for q in QUANTILES]
        ] = weekly_quantiles_sorted

      continue

    else:  # SIRH model was successfully fitted
      # --- Stochastic Simulations from SIR[H]2 model (using MCMC parameter distributions) ---
      sirh_simulations_abs = sirh_forecaster.predict_stochastic(
          forecast_horizon_weeks=forecast_horizon_weeks,
          num_simulations=NUM_SIMULATIONS,
      )

      # Generate final quantiles based on SIRH simulations spread
      predictions_per_quantile_abs = {}
      for q_val in QUANTILES:
        predictions_per_quantile_abs[f'quantile_{q_val}'] = np.percentile(
            sirh_simulations_abs, q_val * 100, axis=0
        )

      # Fill into DataFrame
      loc_predictions_df = pd.DataFrame(
          index=loc_test_x_indices, columns=[f'quantile_{q}' for q in QUANTILES]
      )
      for q_col, preds_abs in predictions_per_quantile_abs.items():
        loc_predictions_df[q_col] = preds_abs

      # --- Enforce monotonicity and non-negativity (sorted before rounding) ---
      for idx in loc_predictions_df.index:
        sorted_quantiles = np.sort(loc_predictions_df.loc[idx].values)
        loc_predictions_df.loc[idx, :] = (
            np.maximum(0, sorted_quantiles).round().astype(int)
        )

      test_y_hat_quantiles.loc[loc_test_x_indices] = loc_predictions_df

  return test_y_hat_quantiles


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
    cdc_submission = format_for_cdc(test_forecasts, 'wk inc flu hosp')
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
