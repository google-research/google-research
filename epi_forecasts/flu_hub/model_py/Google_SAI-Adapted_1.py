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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name

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
MODEL_NAME = 'Google_SAI-Adapted_1'
TARGET_STR = ''

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import warnings
import numpy as np
import pandas as pd
from scipy.stats import gamma, nbinom
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

TARGET_STR = 'Total Influenza Admissions'

# --- Configuration Constants ---

N_SIMULATIONS = (
    1000  # Number of Monte Carlo simulations for quantile generation
)

# --- Fixed Model Parameters and Distributions (heuristic for this benchmark model) ---

# Generation Interval (GI) - parameters for Gamma distribution
# Mean around 3.5 days, Std dev around 1.0 day for flu
GI_MEAN = 3.5
GI_STD = 1.0
GI_SHAPE = (GI_MEAN / GI_STD) ** 2
GI_SCALE = GI_STD**2 / GI_MEAN
GI_MAX_DAYS = 30  # Max days to consider for GI kernel

# Hospitalization Interval Delay (HID) - parameters for Gamma distribution
# Mean around 9.0 days, Std dev around 4.0 days for flu
HID_MEAN = 9.0
HID_STD = 4.0
HID_SHAPE = (HID_MEAN / HID_STD) ** 2
HID_SCALE = HID_STD**2 / HID_MEAN
HID_MAX_DAYS = 30  # Max days to consider for HID kernel

# SIGMA_RT: Standard deviation for log(Rt) random walk (fixed - part of the Bayesian process definition)
SIGMA_RT = 0.1  # Allows Rt to change slowly, ensure strictly positive

# Default IHR and NB_PHI to use if local inference is not possible (e.g., no data)
# Default slope for linear regression (equivalent to IHR for simple scaling)
DEFAULT_ILI_SLOPE = 0.001  # 1 hospitalization per 1000 ILI cases
DEFAULT_ILI_INTERCEPT = 1.0  # Default baseline hospitalizations
DEFAULT_NB_PHI = (
    5.0  # A common value for overdispersed count data, or when estimation fails
)

# Parameters for sampling IHR and NB_PHI (for simple Bayesian inference)
IHR_CV = 0.2  # Coefficient of Variation for IHR Gamma distribution (now applied to regression slope)
DEFAULT_NB_PHI_LOG_STD = 0.5  # Std dev on log scale for NB_PHI LogNormal distribution (robust fallback)

# Number of recent weeks to average for initial infections / trend estimation
N_WEEKS_FOR_INIT = 8  # Increased from 4 for more robust initial Rt estimation

# History length needed for initial conditions (convolutions and Rt estimation)
HISTORY_DAYS_FOR_INIT_WINDOW = (
    max(GI_MAX_DAYS, HID_MAX_DAYS) + 7 * N_WEEKS_FOR_INIT
)

# Epsilon for numerical stability, especially for log and division
EPSILON = 1e-9

# Minimum standard deviation for initial log(Rt) sampling to ensure initial uncertainty
MIN_INITIAL_LOG_RT_STD_DEV = (
    0.1  # A reasonable minimum for log-scale Rt uncertainty
)

# Minimum number of overlap points required to calculate location-specific IHR (regression).
MIN_OVERLAP_POINTS_FOR_LOC_IHR = 5

# Minimum number of positive observations in recent history for Rt estimation
MIN_POSITIVE_OBS_FOR_RT_EST = 3


def _create_discrete_gamma_pmf(shape, scale, max_days):
  """Creates a discrete probability mass function from a Gamma distribution."""
  if max_days <= 0:
    return np.array([])

  # Robustness: Ensure valid gamma parameters, fallback to uniform if invalid or non-finite
  if (
      not np.isfinite(shape)
      or shape <= EPSILON
      or not np.isfinite(scale)
      or scale <= EPSILON
  ):
    # Fallback to uniform distribution if parameters are invalid
    return np.ones(max_days) / max_days

  pmf = gamma.pdf(np.arange(max_days), a=shape, scale=scale)
  pmf_sum = np.sum(pmf)
  if pmf_sum < EPSILON or not np.isfinite(
      pmf_sum
  ):  # Avoid division by zero if all pdf values are tiny
    return np.ones(max_days) / max_days
  pmf = pmf / pmf_sum  # Normalize to sum to 1
  return pmf


def _nbinom_neg_log_likelihood(phi, data_mean, data_positive):
  """Negative log-likelihood for Negative Binomial distribution."""
  # Ensure phi > 0 and data_mean > 0 for valid log-likelihood calculation
  if (
      phi <= EPSILON
      or data_mean <= EPSILON
      or not np.isfinite(phi)
      or not np.isfinite(data_mean)
  ):
    return np.inf
  p = phi / (phi + data_mean)
  # Ensure p is within (0, 1) for nbinom.logpmf
  p = np.clip(p, EPSILON, 1.0 - EPSILON)

  try:
    log_likelihood = nbinom.logpmf(k=np.asarray(data_positive), n=phi, p=p)
    # Replace -inf with log(EPSILON) to prevent NaN sum and ensure finite objective.
    log_likelihood[np.isneginf(log_likelihood)] = np.log(EPSILON)
    return -np.sum(log_likelihood)
  except Exception:
    # Catch any other numerical exceptions during logpmf calculation
    return np.inf


def _robust_sample_gamma(
    mean, cv, default_value, min_std_ratio=0.001, clip_min=1e-9, clip_max=0.1
):
  """Robustly samples from a Gamma distribution."""
  mean = max(mean, EPSILON)  # Ensure mean is positive
  std = mean * cv
  std_robust = max(
      std, mean * min_std_ratio
  )  # Ensure minimum std to prevent zero variance

  # Calculate shape and scale
  shape = mean**2 / (std_robust**2 + EPSILON)
  scale = std_robust**2 / (mean + EPSILON)

  sampled_val = default_value  # Default in case of failure
  if (
      np.isfinite(shape)
      and shape > EPSILON
      and np.isfinite(scale)
      and scale > EPSILON
  ):
    try:
      sampled_val = np.random.gamma(shape=shape, scale=scale)
    except ValueError:  # Catch potential domain errors with np.random.gamma
      pass  # Use default_value
  return np.clip(sampled_val, clip_min, clip_max)


def _robust_sample_lognormal(
    mean_val, log_std, default_value, clip_min=1.0, clip_max=1000.0
):
  """Robustly samples from a Log-Normal distribution."""
  mean_val = max(mean_val, EPSILON)  # Ensure mean is positive for log
  log_mean = np.log(mean_val)
  log_std_robust = max(log_std, EPSILON)  # Ensure positive std

  sampled_val = default_value  # Default in case of failure
  if (
      np.isfinite(log_mean)
      and np.isfinite(log_std_robust)
      and log_std_robust > EPSILON
  ):
    try:
      sampled_val = np.random.lognormal(mean=log_mean, sigma=log_std_robust)
    except ValueError:  # Catch potential domain errors with np.random.lognormal
      pass  # Use default_value
  return np.clip(sampled_val, clip_min, clip_max)


# Pre-calculate PMFs using the robust function (assuming these are globally accessible constants)
GI_PMF = _create_discrete_gamma_pmf(GI_SHAPE, GI_SCALE, GI_MAX_DAYS)
HID_PMF = _create_discrete_gamma_pmf(HID_SHAPE, HID_SCALE, HID_MAX_DAYS)


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # IMPLEMENTATION PLAN.
  # This plan details how the function adheres to the Method Contract, particularly concerning the "Bayesian statistical framework" in a "simple renewal model" context.
  #
  # ## Core Principles Checklist:
  # 1.  **Model Latent Infections:** The code estimates daily latent infections using a renewal equation, where the time-varying reproduction number (Rt) is modeled as a latent random walk on the log scale, evolving weekly by adding Gaussian noise. Initial daily infection levels are derived through a smoothed, shifted deconvolution of reconstructed historical hospitalizations, with robust flooring to ensure positivity.
  # 2.  **Estimate Latent Hospitalizations:** Daily latent hospitalizations are calculated by convolving the daily latent infections with a fixed hospitalization interval distribution (discrete Gamma PMF) and multiplying by a sampled infection-to-hospitalization rate (IHR).
  # 3.  **Observation Model:** Modeled daily latent hospitalizations are aggregated to weekly totals. Observed weekly admissions are then linked via a Negative Binomial distribution, whose dispersion parameter (phi) is inferred and sampled for each simulation, ensuring parameters are within valid ranges.
  # 4.  **Parameter Inference (Simulated Bayesian Framework):** This implementation robustly estimates point values for IHR, NB_PHI, and initial `log(Rt)` using methods like sums-ratio, MLE, or linear regression. These point estimates then rigorously parameterize sampling distributions (Gamma for IHR, Log-Normal for `nb_phi`, Normal for `log(Rt)`), from which parameters are drawn for each Monte Carlo simulation. Each sampling step includes checks for valid parameters and clipping of sampled values to ensure numerical stability and prevent unrealistic parameter magnitudes, thereby simulating Bayesian uncertainty propagation within the constraints of a simple model.
  #
  # ## Step-by-Step Logic:
  # 1.  **Data Preprocessing and Initialization:**
  #     *   Convert `target_end_date` and `reference_date` columns in `train_x`, `train_y`, and `test_x` to datetime objects.
  #     *   Combine `train_x` and `train_y` into `train_df`.
  #     *   Determine `train_last_obs_date` and `max_forecast_date`.
  # 2.  **ILINet Augmentation & Global Parameter Estimation (Revised):**
  #     *   Load and preprocess `ilinet_state`, ensuring `ilitotal` column is robustly handled (filling NaNs with 0 and converting to float). Merge with `locations` to add FIPS codes.
  #     *   Add `week_end_date` to `ilinet_augmented` to align with `target_end_date` (Saturday).
  #     *   **Revised Global IHR/Scaling:** Perform a linear regression globally: `TARGET_STR` ~ `ilitotal` on the overlapping data. Robustly handle cases where regression is impossible or degenerate (e.g., no variance in `ilitotal`) by falling back to `DEFAULT_ILI_SLOPE` and `DEFAULT_ILI_INTERCEPT`.
  #     *   Estimate a `global_nb_phi_estimate` using MLE on positive target values, with a robust method-of-moments fallback if MLE fails or is unstable, ensuring it's strictly positive and within a reasonable range (1.0 to 1000.0).
  #     *   Estimate `global_nb_phi_log_std_estimate` from the standard deviation of log-transformed positive target values, used as a robust fallback for sampling `nb_phi`, with clipping to prevent extreme values.
  #     *   Initialize an empty dictionary `location_params` to store location-specific estimates.
  # 3.  **Location-wise Parameter & Synthetic History Creation (Revised):**
  #     *   Iterate through each unique location in the augmented ILINet data:
  #         *   **Revised Location IHR/Scaling:** If enough overlap points (`MIN_OVERLAP_POINTS_FOR_LOC_IHR`) and sufficient variance in `ilitotal` exist, perform a local linear regression: `TARGET_STR` ~ `ilitotal`. Otherwise, fall back to global slope and intercept. Store `loc_ili_regression_slope` and `loc_ili_regression_intercept`. Clip slope and intercept to be non-negative and finite.
  #         *   Calculate `loc_nb_phi_estimate` using location-specific `TARGET_STR` data (MLE with method-of-moments fallback), ensuring numerical stability and appropriate range.
  #         *   Calculate `loc_nb_phi_log_std_estimate` from location-specific log-transformed positive target values, falling back to the global estimate, with robust clipping.
  #         *   Store these estimates (`ili_slope`, `ili_intercept`, `nb_phi_est`, `nb_phi_log_std_est`) in `location_params`.
  #         *   Generate `synthetic_loc_history` by applying the location-specific (or global fallback) linear regression model to historical `ilitotal` data (for dates where actual target data is missing). Ensure synthetic values are non-negative.
  #     *   Combine `train_df` with all generated `synthetic_ilinet_history`, prioritizing actual observations, to form `full_target_history_df`.
  # 4.  **Location-wise Forecasting Loop (Revised `I_daily_history` Initialization):** Iterate through each unique location in `test_x`:
  #     *   **Retrieve Parameters:** Get `loc_ili_regression_slope`, `loc_ili_regression_intercept`, `loc_nb_phi_estimate`, and `loc_nb_phi_log_std_estimate` from `location_params`.
  #     *   **Handle Empty/Zero History:** If a location has insufficient recent positive target data, predict zero for all quantiles and continue.
  #     *   **Revised Initialize Daily Latent Infections (`I_daily_history`):**
  #         *   Define `daily_sim_start_date` (day after `train_last_obs_date`).
  #         *   Reconstruct `daily_hospitalization_history` by distributing weekly `TARGET_STR` values uniformly across days, applying robust smoothing (forward/backward fill, median fallback for NaNs) and enforcing strict positivity (`np.maximum(EPSILON)`). *Ensure `initial_fill_value_for_H` is robustly calculated to prevent `NaN` or `inf` values if `location_history_weekly` is empty or contains only zeros.*
  #         *   Derive `I_daily_history` by a simplified deconvolution: shift `daily_hospitalization_history` by `HID_MEAN` days, divide by `loc_ili_regression_slope` (or a robust default `> EPSILON` if slope is problematic) to represent the effective IHR for this reconstruction, apply a 7-day rolling mean for smoothing, and enforce a minimum floor (`1.0`) to prevent zero infections. *Ensure the division by `effective_ihr_for_init_deconv` is always valid by checking its positivity and clamping to a reasonable range.*
  #     *   **Infer and Sample Initial Log(Rt):** Aggregate `I_daily_history` into weekly sums, fit a linear regression to log-transformed sums to estimate `initial_log_Rt_point_estimate` and `initial_log_Rt_std_dev_for_sampling`, ensuring robustness and a minimum standard deviation. *Explicitly ensure `n_reg > 2` and `S_xx > EPSILON` for the standard error calculation, falling back to `MIN_INITIAL_LOG_RT_STD_DEV` if conditions are not met or calculation is degenerate.*
  #     *   **Monte Carlo Simulation (N_SIMULATIONS iterations - Robust Sampling):**
  #         *   For each simulation:
  #             *   **Sample Initial `current_log_rt`:** Draw from a Normal distribution, clamping the value.
  #             *   **Robustly Sample IHR and NB_PHI:** Draw `ihr_sim` from a Gamma distribution parameterized by `loc_ili_regression_slope` (as mean) and its CV, *after robustly checking for finite and positive Gamma parameters and falling back to the mean if invalid or the shape/scale are degenerate*. Draw `nb_phi_sim` from a Log-Normal distribution parameterized by `loc_nb_phi_estimate` and its log-std, *after robustly checking for finite and positive Log-Normal parameters and falling back to the mean if invalid or the mean/sigma are degenerate*. Sampled values are clipped.
  #             *   Initialize a dynamic buffer `I_daily_sim_buffer` with the last `REQUIRED_CONV_HISTORY_DAYS` values from `I_daily_history`.
  #             *   Loop day-by-day: Update `current_log_rt` weekly by random walk. Calculate `new_infections` using the renewal equation. Calculate `new_hospitalizations_daily_val` using convolution with `HID_PMF` and `ihr_sim`.
  #             *   **Weekly Aggregation and Sampling:** For each forecast `target_end_date`, sum simulated daily hospitalizations (`weekly_H_mean_sim`). Sample `sampled_weekly_H` from a Negative Binomial distribution using `nb_phi_sim` and `weekly_H_mean_sim`, ensuring numerical stability.
  #             *   Append `sampled_weekly_H` to `location_forecasts_sims`.
  # 5.  **Quantile Calculation (Revised Monotonicity):** For each `target_end_date`, compute quantiles from the `N_SIMULATIONS` sampled values. Ensure non-negativity and enforce monotonicity using `np.maximum.accumulate` *before* rounding to integer. Round to integer.
  # 6.  **Output Formatting:** Merge calculated quantiles with `test_x_loc`, fill NaNs with 0, set index, and apply a final check for monotonicity and non-negativity.

  # Suppress all warnings locally within the function, assuming non-critical for competition
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    # 1.a: Convert dates to datetime objects
    train_x_copy = train_x.copy()
    test_x_copy = test_x.copy()
    train_y_copy = train_y.copy()

    train_x_copy['target_end_date'] = pd.to_datetime(
        train_x_copy['target_end_date']
    )
    test_x_copy['target_end_date'] = pd.to_datetime(
        test_x_copy['target_end_date']
    )
    test_x_copy['reference_date'] = pd.to_datetime(
        test_x_copy['reference_date']
    )

    # 1.b: Combine train_x and train_y
    train_df = train_x_copy.copy()
    train_df[TARGET_STR] = train_y_copy.values

    # Determine the earliest date for daily simulation (day after last training week)
    train_last_obs_date = (
        train_df['target_end_date'].max()
        if not train_df.empty
        else pd.Timestamp('1900-01-01')
    )  # Robust default

    max_forecast_date = test_x_copy['target_end_date'].max()

    # --- 2. ILINet Augmentation & Global Parameter Estimation ---
    ilinet_augmented = globals().get('ilinet_state').copy()
    locations_df = globals().get('locations').copy()

    ilinet_augmented['week_start'] = pd.to_datetime(
        ilinet_augmented['week_start']
    )
    ilinet_augmented['week_end_date'] = ilinet_augmented[
        'week_start'
    ] + pd.Timedelta(days=6)

    ilinet_augmented = ilinet_augmented.rename(
        columns={'region': 'location_name'}
    )
    ilinet_augmented = ilinet_augmented.merge(
        locations_df[['location_name', 'location']],
        on='location_name',
        how='left',
    )
    ilinet_augmented = ilinet_augmented.dropna(
        subset=['location']
    )  # Drop rows if location cannot be mapped
    ilinet_augmented['location'] = ilinet_augmented['location'].astype(int)

    # Use 'ilitotal' for augmentation, ensuring it's float and NaNs are zero-filled
    ilinet_augmented['ilitotal'] = (
        ilinet_augmented['ilitotal'].fillna(0).astype(float)
    )

    # 2.b: Global Parameter Estimation for robust defaults (IHR as regression slope/intercept)
    global_overlap_data = pd.merge(
        train_df[['target_end_date', 'location', TARGET_STR]],
        ilinet_augmented[['week_end_date', 'location', 'ilitotal']],
        left_on=['target_end_date', 'location'],
        right_on=['week_end_date', 'location'],
        how='inner',
    ).dropna(subset=[TARGET_STR, 'ilitotal'])

    global_ili_regression_slope = DEFAULT_ILI_SLOPE
    global_ili_regression_intercept = DEFAULT_ILI_INTERCEPT

    # Perform global linear regression TARGET_STR ~ ilitotal
    if (
        len(global_overlap_data) >= MIN_OVERLAP_POINTS_FOR_LOC_IHR
        and global_overlap_data['ilitotal'].var() > EPSILON
    ):
      try:
        model = LinearRegression()
        # Reshape 'ilitotal' for scikit-learn
        X_global = global_overlap_data['ilitotal'].values.reshape(-1, 1)
        y_global = global_overlap_data[TARGET_STR].values
        model.fit(X_global, y_global)
        if (
            np.isfinite(model.coef_[0]) and model.coef_[0] >= 0
        ):  # Slope should be non-negative
          global_ili_regression_slope = model.coef_[0]
        if (
            np.isfinite(model.intercept_) and model.intercept_ >= 0
        ):  # Intercept should be non-negative
          global_ili_regression_intercept = model.intercept_
      except Exception:
        pass  # Fallback to defaults

    # Clip values to reasonable ranges
    global_ili_regression_slope = np.clip(
        global_ili_regression_slope, 1e-9, 0.1
    )
    global_ili_regression_intercept = np.clip(
        global_ili_regression_intercept, 0.0, 100.0
    )

    global_nb_phi_estimate = DEFAULT_NB_PHI
    positive_target_values_global = train_df[train_df[TARGET_STR] > 0][
        TARGET_STR
    ]
    if (
        not positive_target_values_global.empty
        and len(positive_target_values_global) >= 5
    ):  # Require enough data for MLE
      mu_H_global = positive_target_values_global.mean()
      if mu_H_global > EPSILON and np.isfinite(mu_H_global):
        try:
          x0_phi = max(1.0, np.sqrt(mu_H_global), EPSILON)
          res = minimize(
              _nbinom_neg_log_likelihood,
              x0=[x0_phi],
              args=(mu_H_global, positive_target_values_global.values),
              bounds=[(EPSILON, 1000.0)],
              method='L-BFGS-B',
          )
          if (
              res.success
              and np.isfinite(res.x[0])
              and res.x[0] > EPSILON
              and res.x[0] <= 1000.0
          ):
            global_nb_phi_estimate = np.clip(res.x[0], 1.0, 1000.0)
        except Exception:
          pass
    if (
        global_nb_phi_estimate == DEFAULT_NB_PHI
        and not positive_target_values_global.empty
        and len(positive_target_values_global) > 1
    ):
      mu_H_global = positive_target_values_global.mean()
      var_H_global = positive_target_values_global.var()
      if (
          np.isfinite(mu_H_global)
          and np.isfinite(var_H_global)
          and var_H_global > mu_H_global + EPSILON
      ):
        potential_phi = mu_H_global**2 / (var_H_global - mu_H_global)
        if np.isfinite(potential_phi) and potential_phi > EPSILON:
          global_nb_phi_estimate = np.clip(potential_phi, 1.0, 1000.0)
      else:
        global_nb_phi_estimate = (
            1000.0  # Default to high phi for near-Poisson if variance is low
        )

    global_nb_phi_log_std_estimate = DEFAULT_NB_PHI_LOG_STD
    if (
        not positive_target_values_global.empty
        and len(positive_target_values_global) >= 5
    ):
      log_positive_targets = np.log(
          positive_target_values_global.values + EPSILON
      )
      std_log_targets = np.std(log_positive_targets)
      if std_log_targets > EPSILON and np.isfinite(std_log_targets):
        global_nb_phi_log_std_estimate = np.clip(std_log_targets, EPSILON, 2.0)

    location_params = {}

    # 3. Location-wise Parameter & Synthetic History Creation
    synthetic_ilinet_history_list = []
    all_train_target_dates = set(train_df['target_end_date'].unique())

    for loc_fips in ilinet_augmented['location'].unique():
      ilinet_loc_data = ilinet_augmented[
          ilinet_augmented['location'] == loc_fips
      ]
      train_loc_data = train_df[train_df['location'] == loc_fips]

      # 3.a: Infer location-specific ILI regression parameters
      loc_ili_regression_slope = global_ili_regression_slope
      loc_ili_regression_intercept = global_ili_regression_intercept

      loc_overlap_data = pd.merge(
          train_loc_data[['target_end_date', TARGET_STR]],
          ilinet_loc_data[['week_end_date', 'ilitotal']],
          left_on='target_end_date',
          right_on='week_end_date',
          how='inner',
      ).dropna(subset=[TARGET_STR, 'ilitotal'])

      # Perform local linear regression if enough data and variance
      if (
          len(loc_overlap_data) >= MIN_OVERLAP_POINTS_FOR_LOC_IHR
          and loc_overlap_data['ilitotal'].var() > EPSILON
      ):
        try:
          model = LinearRegression()
          X_loc = loc_overlap_data['ilitotal'].values.reshape(-1, 1)
          y_loc = loc_overlap_data[TARGET_STR].values
          model.fit(X_loc, y_loc)
          if np.isfinite(model.coef_[0]) and model.coef_[0] >= 0:
            loc_ili_regression_slope = model.coef_[0]
          if np.isfinite(model.intercept_) and model.intercept_ >= 0:
            loc_ili_regression_intercept = model.intercept_
        except Exception:
          pass  # Fallback to global defaults

      # Clip for stability
      loc_ili_regression_slope = np.clip(loc_ili_regression_slope, 1e-9, 0.1)
      loc_ili_regression_intercept = np.clip(
          loc_ili_regression_intercept, 0.0, 100.0
      )

      # 3.b: Infer location-specific NB_PHI estimate (Negative Binomial dispersion parameter)
      loc_nb_phi_estimate = global_nb_phi_estimate  # Fallback
      if not train_loc_data.empty:
        positive_target_loc_values = train_loc_data[
            train_loc_data[TARGET_STR] > 0
        ][TARGET_STR]
        if (
            not positive_target_loc_values.empty
            and len(positive_target_loc_values) >= 5
        ):
          mu_H_loc = positive_target_loc_values.mean()
          if mu_H_loc > EPSILON and np.isfinite(mu_H_loc):
            try:
              x0_phi = max(1.0, np.sqrt(mu_H_loc), EPSILON)
              res = minimize(
                  _nbinom_neg_log_likelihood,
                  x0=[x0_phi],
                  args=(mu_H_loc, positive_target_loc_values.values),
                  bounds=[(EPSILON, 1000.0)],
                  method='L-BFGS-B',
              )
              if (
                  res.success
                  and np.isfinite(res.x[0])
                  and res.x[0] > EPSILON
                  and res.x[0] <= 1000.0
              ):
                loc_nb_phi_estimate = np.clip(res.x[0], 1.0, 1000.0)
            except Exception:
              pass
        if (
            loc_nb_phi_estimate == global_nb_phi_estimate
            and not positive_target_loc_values.empty
            and len(positive_target_loc_values) > 1
        ):
          mu_H_loc = positive_target_loc_values.mean()
          var_H_loc = positive_target_loc_values.var()
          if (
              np.isfinite(mu_H_loc)
              and np.isfinite(var_H_loc)
              and var_H_loc > mu_H_loc + EPSILON
          ):
            potential_phi = mu_H_loc**2 / (var_H_loc - mu_H_loc)
            if np.isfinite(potential_phi) and potential_phi > EPSILON:
              loc_nb_phi_estimate = np.clip(potential_phi, 1.0, 1000.0)
          else:
            loc_nb_phi_estimate = 1000.0  # Default to high phi for near-Poisson if variance is low

      loc_nb_phi_log_std_estimate = (
          global_nb_phi_log_std_estimate  # Fallback to global
      )
      if (
          not positive_target_loc_values.empty
          and len(positive_target_loc_values) >= 5
      ):
        log_positive_targets = np.log(
            positive_target_loc_values.values + EPSILON
        )
        std_log_targets = np.std(log_positive_targets)
        if std_log_targets > EPSILON and np.isfinite(std_log_targets):
          loc_nb_phi_log_std_estimate = np.clip(std_log_targets, EPSILON, 2.0)

      location_params[loc_fips] = {
          'ili_slope': loc_ili_regression_slope,
          'ili_intercept': loc_ili_regression_intercept,
          'nb_phi_est': loc_nb_phi_estimate,
          'nb_phi_log_std_est': loc_nb_phi_log_std_estimate,
      }

      # 3.c: Create synthetic history for this location for ALL dates where TARGET_STR is missing
      ilinet_dates_with_no_target = ilinet_loc_data[
          ~ilinet_loc_data['week_end_date'].isin(all_train_target_dates)
      ].copy()
      if not ilinet_dates_with_no_target.empty:
        synthetic_loc_history = ilinet_dates_with_no_target.copy()
        # Apply the learned linear regression model to generate synthetic hospitalizations
        synthetic_loc_history[TARGET_STR] = (
            synthetic_loc_history['ilitotal'] * loc_ili_regression_slope
            + loc_ili_regression_intercept
        )
        synthetic_loc_history[TARGET_STR] = np.maximum(
            0, synthetic_loc_history[TARGET_STR]
        )  # Ensure non-negative
        synthetic_loc_history = synthetic_loc_history[
            ['week_end_date', 'location', 'location_name', TARGET_STR]
        ].rename(columns={'week_end_date': 'target_end_date'})
        synthetic_loc_history = synthetic_loc_history.merge(
            locations_df[['location', 'population']], on='location', how='left'
        )
        synthetic_ilinet_history_list.append(synthetic_loc_history)

    synthetic_ilinet_history = (
        pd.concat(synthetic_ilinet_history_list, ignore_index=True)
        if synthetic_ilinet_history_list
        else pd.DataFrame()
    )

    # 3.d: Combine original train_df and synthetic history, prioritizing actual data
    full_target_history_df = pd.concat(
        [train_df, synthetic_ilinet_history], ignore_index=True
    )
    full_target_history_df = full_target_history_df.drop_duplicates(
        subset=['target_end_date', 'location'], keep='first'
    )
    full_target_history_df = full_target_history_df.sort_values(
        by=['location', 'target_end_date']
    )
    full_target_history_df = full_target_history_df.reset_index(drop=True)

    all_location_predictions = []

    # 4. Location-wise Forecasting Loop
    for location_fips, test_x_loc in test_x_copy.groupby('location'):

      location_history_weekly = full_target_history_df[
          (full_target_history_df['location'] == location_fips)
          & (full_target_history_df['target_end_date'] <= train_last_obs_date)
      ].copy()

      # 4.a: Retrieve Parameter Estimates for this location
      loc_ili_regression_slope = location_params.get(location_fips, {}).get(
          'ili_slope', global_ili_regression_slope
      )
      loc_ili_regression_intercept = location_params.get(location_fips, {}).get(
          'ili_intercept', global_ili_regression_intercept
      )
      loc_nb_phi_estimate = location_params.get(location_fips, {}).get(
          'nb_phi_est', global_nb_phi_estimate
      )
      loc_nb_phi_log_std_estimate = location_params.get(location_fips, {}).get(
          'nb_phi_log_std_est', global_nb_phi_log_std_estimate
      )

      # 4.b: Handle Empty/Zero History: If no actual target data or all zeros in recent history, predict zeros
      recent_train_history_for_check = location_history_weekly[
          location_history_weekly['target_end_date']
          > (train_last_obs_date - pd.Timedelta(weeks=N_WEEKS_FOR_INIT))
      ].copy()

      num_positive_recent_obs = (
          recent_train_history_for_check[TARGET_STR] > EPSILON
      ).sum()
      has_sufficient_positive_history = (
          num_positive_recent_obs >= MIN_POSITIVE_OBS_FOR_RT_EST
      )

      if not has_sufficient_positive_history:
        loc_preds_df = test_x_loc.copy()
        for q in QUANTILES:
          loc_preds_df[f'quantile_{q}'] = 0
        all_location_predictions.append(loc_preds_df)
        continue

      # --- 4.c: Initialization of Daily Latent Infections (`I_daily_history`) ---
      daily_sim_start_date = train_last_obs_date + pd.Timedelta(days=1)

      daily_history_dates = pd.date_range(
          end=train_last_obs_date,
          periods=HISTORY_DAYS_FOR_INIT_WINDOW,
          freq='D',
      )

      # Robustly calculate an initial fill value
      median_val = location_history_weekly[TARGET_STR].median()
      initial_fill_value_for_H = (
          median_val / 7.0
          if np.isfinite(median_val) and median_val >= 0
          else EPSILON
      )
      initial_fill_value_for_H = max(
          EPSILON, initial_fill_value_for_H
      )  # Ensure strictly positive

      # Initialize daily_H_reconstructed_array to full length with fill value
      daily_H_reconstructed_array = np.full(
          HISTORY_DAYS_FOR_INIT_WINDOW, initial_fill_value_for_H
      )

      # Populate with weekly data, distributing uniformly over 7 days
      for _, row in location_history_weekly.iterrows():
        week_end = row['target_end_date']
        week_start = week_end - pd.Timedelta(days=6)

        # Find the index corresponding to the week_start_clipped in daily_history_dates
        # Ensure date falls within the range of daily_history_dates
        if (
            week_start <= daily_history_dates[-1]
            and week_end >= daily_history_dates[0]
        ):
          start_idx_in_arr = max(0, (week_start - daily_history_dates[0]).days)
          end_idx_in_arr = min(
              HISTORY_DAYS_FOR_INIT_WINDOW - 1,
              (week_end - daily_history_dates[0]).days,
          )

          if start_idx_in_arr <= end_idx_in_arr:
            daily_H_reconstructed_array[
                start_idx_in_arr : end_idx_in_arr + 1
            ] = (row[TARGET_STR] / 7.0)

      # Convert to DataFrame for robust ffill/bfill, then back to array.
      # Mark initial fills as NaN to allow ffill/bfill to prioritize actual data.
      temp_df = pd.DataFrame(
          daily_H_reconstructed_array,
          index=daily_history_dates,
          columns=['H_daily'],
      )
      temp_df['H_daily'] = temp_df['H_daily'].replace(
          initial_fill_value_for_H, np.nan
      )  # Mark initial fills as NaN for proper ffill/bfill
      temp_df['H_daily'] = (
          temp_df['H_daily']
          .fillna(method='ffill')
          .fillna(method='bfill')
          .fillna(value=initial_fill_value_for_H)
      )
      daily_H_reconstructed_array = np.maximum(
          EPSILON, temp_df['H_daily'].values
      )

      # Derive I_daily_history (simplified deconvolution)
      effective_ihr_for_init_deconv = np.clip(
          loc_ili_regression_slope, 1e-9, 0.1
      )  # Clamped for stability

      h_lookahead_days = int(round(HID_MEAN))

      # Extend the hospitalization history with the last known value for the required lookahead
      H_ext = np.concatenate([
          daily_H_reconstructed_array,
          np.full(h_lookahead_days, daily_H_reconstructed_array[-1]),
      ])

      # Calculate I_daily_history by shifting H_ext back
      I_daily_history = (
          H_ext[
              h_lookahead_days : h_lookahead_days + HISTORY_DAYS_FOR_INIT_WINDOW
          ]
          / effective_ihr_for_init_deconv
      )
      I_daily_history = np.convolve(
          I_daily_history, np.ones(7) / 7.0, mode='same'
      )  # 7-day rolling mean
      I_daily_history = np.maximum(
          1.0, I_daily_history
      )  # Minimum 1 infection for stability

      # 4.d: Improved Inference and Sampling of Initial Log(Rt)
      initial_log_Rt_point_estimate = np.log(1.0)
      initial_log_Rt_std_dev_for_sampling = (
          MIN_INITIAL_LOG_RT_STD_DEV  # Default to minimum uncertainty
      )

      weekly_infection_sums = []
      # Ensure enough history for N_WEEKS_FOR_INIT
      if len(I_daily_history) >= N_WEEKS_FOR_INIT * 7:
        for i in range(N_WEEKS_FOR_INIT):
          start_idx = len(I_daily_history) - (N_WEEKS_FOR_INIT - i) * 7
          end_idx = start_idx + 7
          weekly_infection_sums.append(
              np.sum(I_daily_history[start_idx:end_idx])
          )

      weekly_infection_sums = np.array(weekly_infection_sums)
      weekly_infection_sums_for_log = np.maximum(1.0, weekly_infection_sums)

      n_reg = len(weekly_infection_sums_for_log)
      if n_reg >= 2 and np.std(weekly_infection_sums_for_log) > EPSILON:
        log_I_values = np.log(weekly_infection_sums_for_log)
        week_indices = np.arange(n_reg).reshape(-1, 1)

        if np.std(log_I_values) > EPSILON and np.all(np.isfinite(log_I_values)):
          model = LinearRegression()
          model.fit(week_indices, log_I_values)

          weekly_log_growth_rate = model.coef_[0]

          se_slope_log_rt_weekly = 0.0
          # Calculate standard error only if enough data and variability for meaningful estimation
          if n_reg > 2:
            y_hat = model.predict(week_indices)
            residuals = log_I_values - y_hat
            sum_sq_residuals = np.sum(residuals**2)
            S_xx = np.sum((week_indices - np.mean(week_indices)) ** 2)

            if (
                sum_sq_residuals > EPSILON
                and S_xx > EPSILON
                and np.isfinite(sum_sq_residuals)
                and np.isfinite(S_xx)
            ):
              mse_residuals = sum_sq_residuals / (n_reg - 2)
              se_slope_log_rt_weekly = np.sqrt(mse_residuals / S_xx)
            else:
              se_slope_log_rt_weekly = MIN_INITIAL_LOG_RT_STD_DEV  # Fallback to a minimum uncertainty
          else:
            se_slope_log_rt_weekly = (
                MIN_INITIAL_LOG_RT_STD_DEV  # Fallback for very few points
            )

          potential_initial_log_Rt = weekly_log_growth_rate * (GI_MEAN / 7.0)

          if np.isfinite(potential_initial_log_Rt):
            initial_log_Rt_point_estimate = np.clip(
                potential_initial_log_Rt, np.log(0.1), np.log(5.0)
            )
            initial_log_Rt_std_dev_for_sampling = se_slope_log_rt_weekly * (
                GI_MEAN / 7.0
            )
            initial_log_Rt_std_dev_for_sampling = np.clip(
                initial_log_Rt_std_dev_for_sampling,
                MIN_INITIAL_LOG_RT_STD_DEV,
                2.0,
            )
          else:  # Fallback if potential_initial_log_Rt is not finite
            initial_log_Rt_point_estimate = np.log(1.0)
            initial_log_Rt_std_dev_for_sampling = (
                MIN_INITIAL_LOG_RT_STD_DEV  # Ensure minimum uncertainty
            )

      initial_log_Rt_std_dev_for_sampling = max(
          initial_log_Rt_std_dev_for_sampling, MIN_INITIAL_LOG_RT_STD_DEV
      )

      total_forecast_days = (max_forecast_date - daily_sim_start_date).days + 1

      location_forecasts_sims = {
          date: [] for date in test_x_loc['target_end_date'].unique()
      }

      REQUIRED_CONV_HISTORY_DAYS = max(GI_MAX_DAYS, HID_MAX_DAYS)

      # 4.e: Monte Carlo Simulation Loop
      for _ in range(N_SIMULATIONS):
        sampled_initial_log_rt_std = max(
            initial_log_Rt_std_dev_for_sampling, EPSILON
        )
        current_log_rt = np.random.normal(
            initial_log_Rt_point_estimate, sampled_initial_log_rt_std
        )
        current_log_rt = np.clip(current_log_rt, np.log(0.1), np.log(5.0))

        # Ensure I_daily_history is long enough before slicing for convolution buffer
        if len(I_daily_history) < REQUIRED_CONV_HISTORY_DAYS:
          # Pad with minimum infection values if history is too short for convolution buffer
          padded_history = np.full(
              REQUIRED_CONV_HISTORY_DAYS - len(I_daily_history), 1.0
          )
          I_daily_sim_buffer = np.concatenate([padded_history, I_daily_history])
        else:
          I_daily_sim_buffer = np.copy(
              I_daily_history[-REQUIRED_CONV_HISTORY_DAYS:]
          )

        daily_hospitalizations_forecast_sim = np.zeros(total_forecast_days)

        # --- Robust Bayesian Parameter Sampling for IHR (slope) and NB_PHI ---
        ihr_sim = _robust_sample_gamma(
            mean=loc_ili_regression_slope,
            cv=IHR_CV,
            default_value=loc_ili_regression_slope,
            clip_min=1e-9,
            clip_max=0.1,
        )

        nb_phi_sim = _robust_sample_lognormal(
            mean_val=loc_nb_phi_estimate,
            log_std=loc_nb_phi_log_std_estimate,
            default_value=loc_nb_phi_estimate,
            clip_min=1.0,
            clip_max=1000.0,
        )

        for d_idx in range(total_forecast_days):
          current_date_in_sim = daily_sim_start_date + pd.Timedelta(days=d_idx)

          if (
              d_idx > 0
              and (current_date_in_sim - daily_sim_start_date).days % 7 == 0
          ):
            current_log_rt += np.random.normal(0, max(SIGMA_RT, EPSILON))
            current_log_rt = np.clip(current_log_rt, np.log(0.1), np.log(5.0))

          infections_for_gi_window = I_daily_sim_buffer[::-1][:GI_MAX_DAYS]
          new_infections_contributing_sum = np.dot(
              infections_for_gi_window, GI_PMF[: len(infections_for_gi_window)]
          )

          new_infections = (
              np.exp(current_log_rt) * new_infections_contributing_sum
          )
          new_infections = np.clip(new_infections, 0.0, 1_000_000.0)

          I_daily_sim_buffer = np.roll(I_daily_sim_buffer, -1)
          I_daily_sim_buffer[-1] = new_infections

          infections_for_hosp_window = I_daily_sim_buffer[::-1][:HID_MAX_DAYS]
          new_hospitalizations_contributing_sum = np.dot(
              infections_for_hosp_window,
              HID_PMF[: len(infections_for_hosp_window)],
          )

          new_hospitalizations_daily_val = (
              ihr_sim * new_hospitalizations_contributing_sum
              + loc_ili_regression_intercept / 7.0
          )
          new_hospitalizations_daily_val = np.clip(
              new_hospitalizations_daily_val, 0.0, 100_000.0
          )
          daily_hospitalizations_forecast_sim[d_idx] = (
              new_hospitalizations_daily_val
          )

        # Weekly Aggregation and Sampling
        for week_end_date in sorted(location_forecasts_sims.keys()):
          week_start_date = week_end_date - pd.Timedelta(days=6)

          start_idx = (week_start_date - daily_sim_start_date).days
          end_idx = (week_end_date - daily_sim_start_date).days + 1

          start_idx = np.clip(start_idx, 0, total_forecast_days)
          end_idx = np.clip(end_idx, 0, total_forecast_days)

          weekly_H_mean_sim = np.sum(
              daily_hospitalizations_forecast_sim[start_idx:end_idx]
          )

          if weekly_H_mean_sim > EPSILON and nb_phi_sim > EPSILON:
            # Ensure p_val is stable for nbinom.rvs
            p_val = nb_phi_sim / (nb_phi_sim + weekly_H_mean_sim)
            p_val = np.clip(p_val, EPSILON, 1.0 - EPSILON)
            sampled_weekly_H = nbinom.rvs(n=nb_phi_sim, p=p_val, size=1)[0]
          else:
            sampled_weekly_H = 0

          location_forecasts_sims[week_end_date].append(sampled_weekly_H)

      # 5. Compute quantiles for this location
      loc_predictions = []
      for week_end_date in sorted(location_forecasts_sims.keys()):
        sim_values = np.array(location_forecasts_sims[week_end_date])
        sim_values[sim_values < 0] = (
            0  # Ensure non-negativity of simulated values first
        )

        quantiles_for_date = np.percentile(
            sim_values, [q * 100 for q in QUANTILES]
        )

        # Enforce strict monotonicity AND non-negativity before rounding
        quantiles_for_date = np.maximum(
            0, quantiles_for_date
        )  # Ensure non-negativity
        quantiles_for_date = np.maximum.accumulate(
            quantiles_for_date
        )  # Then enforce monotonicity

        quantiles_for_date = np.round(quantiles_for_date).astype(int)

        loc_predictions.append({
            'target_end_date': week_end_date,
            **{
                f'quantile_{q}': q_val
                for q, q_val in zip(QUANTILES, quantiles_for_date)
            },
        })

      loc_predictions_df = pd.DataFrame(loc_predictions)

      loc_predictions_df = test_x_loc.merge(
          loc_predictions_df,
          on='target_end_date',
          how='left',
          suffixes=('_test_x', None),
      )
      for q in QUANTILES:
        loc_predictions_df[f'quantile_{q}'] = (
            loc_predictions_df[f'quantile_{q}'].fillna(0).astype(int)
        )

      all_location_predictions.append(loc_predictions_df)

    # 6. Output Formatting
    if not all_location_predictions:
      empty_df = pd.DataFrame(
          columns=[f'quantile_{q}' for q in QUANTILES], index=test_x_copy.index
      )
      return empty_df

    final_predictions_df = pd.concat(
        all_location_predictions, ignore_index=True
    )

    output_df = test_x_copy.merge(
        final_predictions_df,
        on=['reference_date', 'horizon', 'location', 'target_end_date'],
        how='left',
        suffixes=('_orig', None),
    )
    output_df = output_df.set_index(test_x_copy.index)

    output_cols = [f'quantile_{q}' for q in QUANTILES]
    output_df = output_df[output_cols]

    output_df = output_df.fillna(0).astype(int)

    # Final check for monotonicity and non-negativity across all quantiles in the output
    # This redundant final check is removed as it's now handled per-row earlier
    # in the quantile calculation.

    return output_df


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

  dataset['Total Influenza Admissions'] = (
      pd.to_numeric(dataset['Total Influenza Admissions'], errors='coerce')
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
      dataset.groupby('target_end_date')['Total Influenza Admissions']
      .sum()
      .reset_index()
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
