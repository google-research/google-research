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
MODEL_NAME = 'Google_SAI-Adapted_15'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
import warnings
import scipy.stats
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Ensure numpy arithmetic errors raise exceptions for debugging (global setting)
# The preamble sets `np.seterr(over='raise')`, we will respect that and not override.

# QUANTILES and locations are globals provided by the environment.


def get_epiweek_features(date_series):
  """Extracts epiweek and seasonal sin/cos features from a date Series."""
  if date_series.empty:
    return pd.DataFrame(
        columns=['epiweek', 'sin_epiweek', 'cos_epiweek'],
        index=date_series.index,
    )

  # Ensure date_series is datetime for consistent processing
  date_series = pd.to_datetime(date_series)
  epiweek = date_series.dt.isocalendar().week.astype(int)

  # Handle cases where week 53 might exist, normalizing to 52 weeks for consistent seasonality
  # This is a common practice to ensure consistent seasonal patterns regardless of leap weeks.
  epiweek_normalized = epiweek.apply(lambda w: 52 if w == 53 else w)

  sin_epiweek = np.sin(2 * np.pi * epiweek_normalized / 52)
  cos_epiweek = np.cos(2 * np.pi * epiweek_normalized / 52)
  return pd.DataFrame(
      {
          'epiweek': (
              epiweek
          ),  # Keep original epiweek for potential debugging/context
          'sin_epiweek': sin_epiweek,
          'cos_epiweek': cos_epiweek,
      },
      index=date_series.index,
  )


def calculate_lags(
    df, target_col, lags
):
  """Calculates lagged features for a given target column."""
  df_with_lags = df.copy()  # Work on a copy to prevent SettingWithCopyWarning
  for lag in lags:
    # Using numeric_only=False to allow shifting any column type, then filter to target_col.
    # However, shift on groupby already ensures it operates on the group.
    # Ensure 'location' is the groupby key.
    df_with_lags[f'lag_{lag}_{target_col}'] = df_with_lags.groupby('location')[
        target_col
    ].shift(lag)
  return df_with_lags


def enforce_monotonicity_and_nonnegativity(
    predictions_df, quantiles
):
  """Ensures quantiles are monotonically increasing and non-negative for each row."""
  # Ensure all predictions are non-negative
  predictions_df = predictions_df.clip(lower=0)

  # Enforce monotonicity
  # Apply np.maximum.accumulate row-wise on quantile columns
  quantile_cols = [f'quantile_{q}' for q in quantiles]
  # Ensure output is a DataFrame
  predictions_df[quantile_cols] = predictions_df[quantile_cols].apply(
      lambda row: np.maximum.accumulate(row.values),
      axis=1,
      result_type='broadcast',
  )
  return predictions_df


def _estimate_weighted_lag1_correlation(
    series, weights
):
  """Estimates the lag-1 Pearson correlation of normal pseudo-observations from a time series,

  using given weights. This provides a stable, weighted estimate of temporal
  dependence.
  The series is Gaussianized first using its weighted empirical CDF, then
  lagged.
  """
  min_samples_for_corr = (
      5  # Require at least 5 points for a meaningful correlation
  )
  epsilon_rank = 1e-7  # Stricter clipping for improved robustness in ppf
  epsilon_var_check = (
      1e-9  # Added epsilon for variance check to prevent division by near-zero
  )

  # Create a temporary DataFrame to align series and weights by index, then drop NaNs from the series.
  temp_df = pd.DataFrame(
      {'value': series, 'weight': weights}, index=series.index
  )
  temp_df.dropna(subset=['value'], inplace=True)
  temp_df.sort_index(
      inplace=True
  )  # Ensure sorted by index (time) for correct lagging

  if len(temp_df) < min_samples_for_corr or np.sum(temp_df['weight']) < 1e-9:
    return 0.0

  valid_series = temp_df['value']
  valid_weights = temp_df['weight']

  # --- Gaussianization using Weighted Empirical CDF ---
  # Sort by value to create the empirical CDF
  sorted_indices = np.argsort(valid_series.values)
  sorted_values = valid_series.values[sorted_indices]
  sorted_weights = valid_weights.values[sorted_indices]

  if np.sum(sorted_weights) == 0:
    return 0.0  # No effective weights for Gaussianization

  # Calculate cumulative weights and normalize to get empirical probabilities (ranks)
  cumulative_weights = np.cumsum(sorted_weights)
  total_weight = cumulative_weights[-1]

  # Handle cases where total_weight is zero or very small to prevent division by zero / inf
  if total_weight < 1e-9:
    return 0.0

  uniform_pseudo_observations = cumulative_weights / total_weight

  # Clip ranks to avoid -inf/inf for ppf if ranks are extremely close to 0 or 1
  uniform_pseudo_observations = np.clip(
      uniform_pseudo_observations, epsilon_rank, 1 - epsilon_rank
  )

  # Transform to standard normal scale using inverse CDF (quantile function)
  pseudo_normal_observations = scipy.stats.norm.ppf(uniform_pseudo_observations)

  # --- Create lagged pseudo-normal pairs from the full Gaussianized series ---
  # Re-order pseudo_normal_observations back to time order using original index mapping
  pseudo_normal_time_ordered = pd.Series(
      pseudo_normal_observations, index=valid_series.index[sorted_indices]
  ).sort_index()

  pseudo_normal_t = pseudo_normal_time_ordered.iloc[:-1].values
  pseudo_normal_tplus1 = pseudo_normal_time_ordered.iloc[1:].values
  weights_for_pairs = valid_weights.iloc[
      :-1
  ].values  # Weights corresponding to pseudo_normal_t (series_t)

  if (
      len(pseudo_normal_t) < (min_samples_for_corr - 1)
      or np.sum(weights_for_pairs) < 1e-9
  ):
    return 0.0

  # --- Robustness improvement: Check for finite pseudo-normal values ---
  if not (
      np.all(np.isfinite(pseudo_normal_t))
      and np.all(np.isfinite(pseudo_normal_tplus1))
  ):
    return 0.0

  # Calculate weighted mean
  with warnings.catch_warnings():
    warnings.simplefilter(
        'ignore', RuntimeWarning
    )  # Suppress warnings if sum of weights is zero (already handled)
    mean_t = np.average(pseudo_normal_t, weights=weights_for_pairs)
    mean_tplus1 = np.average(pseudo_normal_tplus1, weights=weights_for_pairs)

  if not (np.isfinite(mean_t) and np.isfinite(mean_tplus1)):
    return 0.0

  # Calculate weighted covariance and variances
  var_x = np.average((pseudo_normal_t - mean_t) ** 2, weights=weights_for_pairs)
  var_y = np.average(
      (pseudo_normal_tplus1 - mean_tplus1) ** 2, weights=weights_for_pairs
  )

  if var_x < epsilon_var_check or var_y < epsilon_var_check:
    return 0.0

  cov_xy = np.average(
      (pseudo_normal_t - mean_t) * (pseudo_normal_tplus1 - mean_tplus1),
      weights=weights_for_pairs,
  )

  denominator = np.sqrt(var_x * var_y)
  if denominator < epsilon_var_check:
    return 0.0

  rho = cov_xy / denominator
  rho = np.clip(rho, -0.99, 0.99)
  return rho


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Implements a semi-parametric forecasting method using kernel-based conditional

  densities and an isotropic normal copula for joint dependence,
  adhering to the method contract.
  """
  # Adhere to "quiet as possible" constraint.

  # --- 0. IMPLEMENTATION PLAN ---
  # ## Core Principles Checklist
  # 1. Estimate Weekly Conditional Densities: My code uses a multivariate Gaussian KernelDensity estimator for the *feature kernel* (seasonal + lagged log1p incidence). It uses a *scalar* bandwidth chosen by cross-validation, which is a practical simplification of a "fully parameterized bandwidth matrix" for `sklearn.KernelDensity`. For the *marginal conditional densities* of future incidence values, it uses separate univariate `KernelDensity` models for each horizon, dynamically estimating their bandwidths. Robust fallbacks to weighted empirical quantiles are implemented for insufficient data, constant data, or failed KDEs.
  # 2. Model Joint Dependence with Copulas: My code constructs an isotropic normal copula with an AR(1) correlation structure. The lag-1 correlation (`local_rho`) is estimated from weighted historical normal pseudo-observations using a robust helper function that first Gaussianizes the full historical series using its weighted empirical CDF, then extracts lagged pairs. The correlation matrix is explicitly checked for and adjusted to ensure positive definiteness by adjusting eigenvalues if needed, before sampling `N_SIMULATIONS` trajectories from the multivariate normal distribution. These samples are then transformed through the inverse CDFs (empirical or KDE-derived) of the marginal distributions to generate `log1p` incidence trajectories. A fallback to independent sampling occurs if copula construction or sampling fails due to numerical instability or insufficient simulations/horizons.
  # 3. Derive Target Predictions: My code directly estimates the required quantiles for incidence at each horizon `h` by applying `np.quantile` on the `N_SIMULATIONS` `log1p` trajectories generated from the joint distribution. Predictions are then inverse-transformed (`expm1`), clipped to non-negative values, rounded to integers, and finally ensured to be monotonically increasing across quantiles.

  # ## Step-by-Step Logic
  # 1. Data Preprocessing and Augmentation: Process `train_x` and `train_y`, apply `log1p` transformation, extract epiweek features. Process `ilinet_state` data, merge with `locations`, filter and add `target_end_date`. Apply a time-slicing logic to `ilinet_state` to only use data historically available for the current fold's `latest_train_date`, up to `ILINET_GLOBAL_AVAILABILITY_END`. Calculate location-specific and global scaling factors from the dynamic overlap period between `dataset.csv` target data and `ilinet_state` data (from `earliest_train_date` of current fold). Apply these factors to pre-overlap `ilinet_state` data to create `synthetic_admissions`, which are rounded to int and non-negative. Combine `synthetic_admissions` with actual `current_train_data` into a unified `augmented_historical_data`. Compute lagged `log1p_Total Influenza Admissions` features for this combined dataset, filling NaNs from lags with 0.
  # 2. Pre-calculate Bandwidths for Feature Kernels: For each location, use `GridSearchCV` with `TimeSeriesSplit` on `KernelDensity` to find the optimal bandwidth for the combined seasonal and lagged features by maximizing log-likelihood. This includes robust checks for insufficient data points or constant features (adding jitter if necessary). The bandwidth grid is made denser for improved search, and robust fallbacks are in place for insufficient data or if estimation fails.
  # 3. Prediction Loop for `test_x`:
  #    a. Determine Current Prediction Context: Extract seasonal features for the `reference_date`. Fetch lagged `log1p_Total Influenza Admissions` values for `reference_date - lag` from the `augmented_historical_data`, using 0.0 as default for missing lags.
  #    b. Kernel-Weighted Historical Analogs: Calculate Gaussian kernel weights based on similarity between the current context and historical feature vectors, using the pre-calculated bandwidths. Normalize weights, with fallback to uniform weights if all calculated weights are near zero. Estimate weighted lag-1 correlation (`local_rho`) using a dedicated helper function.
  #    c. Estimate Weekly Conditional Densities (KDEs): For each forecast horizon, collect historical `log1p` values and their corresponding kernel weights. Fit a `KernelDensity` model with dynamic bandwidth estimation and sample weighting. Generate samples from the KDE to form empirical inverse CDFs, with robust fallbacks to weighted empirical quantiles if KDE fails, has insufficient data, or constant data.
  #    d. Model Joint Dependence with Isotropic Normal Copula: Construct a robust AR(1)-like correlation matrix using `local_rho`, explicitly ensuring its positive definiteness by adjusting eigenvalues if needed. Sample `N_SIMULATIONS` trajectories from the multivariate normal distribution (copula) and transform them using the marginal inverse CDFs to get `log1p` incidence trajectories. Fallback to independent sampling if copula fails due to numerical instability or insufficient simulations/horizons.
  #    e. Derive Quantile Predictions: Calculate quantiles from the simulated `log1p` trajectories for each horizon. Inverse transform, clip to non-negative, and round to integer values.
  # 4. Enforce Monotonicity and Non-negativity: Apply `enforce_monotonicity_and_nonnegativity` to the final predictions and convert to integer type.
  # --- END IMPLEMENTATION PLAN ---

  # Initialize a global random number generator for reproducibility across all functions using randomness
  RANDOM_STATE = 42
  rng = np.random.default_rng(
      RANDOM_STATE
  )  # Moved RNG initialization inside the function for self-containment

  # --- 0. Configuration ---
  TARGET_COL = 'Total Influenza Admissions'
  LOG1P_TARGET_COL = 'log1p_Total Influenza Admissions'
  LAGGED_FEATURES = [1, 2, 3]  # Lags to use for current state
  LOG1P_LAGGED_FEATURES = [
      f'lag_{lag}_{LOG1P_TARGET_COL}' for lag in LAGGED_FEATURES
  ]
  N_SIMULATIONS = 1000  # Number of trajectories to sample for robust quantiles
  N_SAMPLES_FROM_MARGINAL_KDE = 5000  # Samples to draw from marginal KDEs to form their empirical inverse CDFs

  # Global quantiles (QUANTILES) and locations (locations) are assumed available from notebook context
  global QUANTILES, locations, ilinet_state  # Explicitly reference globals

  # Convert test_x dates to pd.Timestamp upfront
  test_x_copy = (
      test_x.copy()
  )  # Work with a copy of test_x to avoid modifying original input
  test_x_copy['reference_date'] = pd.to_datetime(test_x_copy['reference_date'])
  test_x_copy['target_end_date'] = pd.to_datetime(
      test_x_copy['target_end_date']
  )

  # Determine the latest date available in the current training data for time-slicing
  latest_train_date = pd.to_datetime(train_x['target_end_date'].max())
  earliest_train_date = pd.to_datetime(train_x['target_end_date'].min())

  # --- 1. Data Preprocessing and Augmentation (Time-Aware) ---

  # 1a. Combine train_x and train_y
  current_train_data = train_x.copy()
  current_train_data[TARGET_COL] = train_y.copy()
  current_train_data['target_end_date'] = pd.to_datetime(
      current_train_data['target_end_date']
  )
  current_train_data[LOG1P_TARGET_COL] = np.log1p(
      current_train_data[TARGET_COL].clip(lower=0)
  )  # Ensure non-negative before log
  current_train_data = current_train_data.join(
      get_epiweek_features(current_train_data['target_end_date'])
  )
  current_train_data.sort_values(['location', 'target_end_date'], inplace=True)

  # 1b. Process `ilinet_state` and filter based on current `latest_train_date`
  ilinet_state_proc = ilinet_state.copy()
  ilinet_state_proc['week_start'] = pd.to_datetime(
      ilinet_state_proc['week_start']
  )  # Ensure Timestamp consistency
  ilinet_state_proc = ilinet_state_proc[
      ilinet_state_proc['region_type'] == 'States'
  ].copy()
  ilinet_state_proc = ilinet_state_proc.rename(
      columns={'region': 'location_name'}
  )
  ilinet_state_proc = pd.merge(
      ilinet_state_proc,
      locations[['location', 'location_name', 'population']],
      on='location_name',
      how='left',
  )
  ilinet_state_proc.dropna(
      subset=['location'], inplace=True
  )  # Drop rows where location merge failed
  ilinet_state_proc['location'] = ilinet_state_proc['location'].astype(int)
  ilinet_state_proc['target_end_date'] = ilinet_state_proc[
      'week_start'
  ]  # Consistent date type
  ilinet_state_proc['ilitotal'] = ilinet_state_proc['ilitotal'].fillna(0)

  # Crucial time-slicing for ilinet data: only use data historically available for the current fold
  ILINET_GLOBAL_AVAILABILITY_END = pd.to_datetime(
      '2022-10-14'
  )  # As per problem description
  available_ilinet_for_fold = ilinet_state_proc[
      ilinet_state_proc['target_end_date']
      <= min(latest_train_date, ILINET_GLOBAL_AVAILABILITY_END)
  ].copy()

  # 1c. Dynamic Overlap Period for Scaling Factors
  effective_ratio_overlap_start = earliest_train_date  # Start from when actual admissions data begins in current train_x
  effective_ratio_overlap_end = min(
      latest_train_date, ILINET_GLOBAL_AVAILABILITY_END
  )

  overlap_train = current_train_data[
      (current_train_data['target_end_date'] >= effective_ratio_overlap_start)
      & (current_train_data['target_end_date'] <= effective_ratio_overlap_end)
  ].copy()

  overlap_ilinet = available_ilinet_for_fold[
      (
          available_ilinet_for_fold['target_end_date']
          >= effective_ratio_overlap_start
      )
      & (
          available_ilinet_for_fold['target_end_date']
          <= effective_ratio_overlap_end
      )
  ].copy()

  overlap_merged = pd.merge(
      overlap_train,
      overlap_ilinet,
      on=['location', 'target_end_date'],
      suffixes=('_admissions', '_ilitotal'),
      how='inner',
  )

  # 1d. Calculate Scaling Factors (with robust fallback)
  location_scaling_factors = {}

  global_median_ratio = 1.0  # Initialize robust default
  valid_overlap_merged_for_ratio_global = overlap_merged[
      overlap_merged['ilitotal'] > 0
  ].copy()
  if not valid_overlap_merged_for_ratio_global.empty:
    # Calculate raw_ratio, ignoring division by zero (will produce np.inf or np.nan, handled next)
    with warnings.catch_warnings():
      warnings.simplefilter(
          'ignore', RuntimeWarning
      )  # Suppress division by zero warning
      valid_overlap_merged_for_ratio_global['raw_ratio'] = (
          valid_overlap_merged_for_ratio_global[TARGET_COL + '_admissions']
          / valid_overlap_merged_for_ratio_global['ilitotal']
      )

    # Filter out inf and NaN values from raw_ratio before taking the median
    finite_ratios_global = valid_overlap_merged_for_ratio_global['raw_ratio'][
        np.isfinite(valid_overlap_merged_for_ratio_global['raw_ratio'])
    ]
    if not finite_ratios_global.empty:
      temp_global_median = finite_ratios_global.median()
      if np.isfinite(temp_global_median) and temp_global_median > 0:
        global_median_ratio = temp_global_median
    global_median_ratio = np.clip(
        global_median_ratio, 0.01, 100.0
    )  # Clip global factor for stability

  default_scaling_factor = (
      global_median_ratio  # Use the robust global median as default
  )

  for loc_fips in locations['location'].unique():
    loc_median_ratio = np.nan
    loc_overlap_merged_for_ratio = overlap_merged[
        (overlap_merged['location'] == loc_fips)
        & (overlap_merged['ilitotal'] > 0)
    ].copy()

    if not loc_overlap_merged_for_ratio.empty:
      with warnings.catch_warnings():
        warnings.simplefilter(
            'ignore', RuntimeWarning
        )  # Suppress division by zero warning
        loc_overlap_merged_for_ratio['raw_ratio'] = (
            loc_overlap_merged_for_ratio[TARGET_COL + '_admissions']
            / loc_overlap_merged_for_ratio['ilitotal']
        )

      # Filter out inf and NaN values from raw_ratio before taking the median
      finite_ratios_loc = loc_overlap_merged_for_ratio['raw_ratio'][
          np.isfinite(loc_overlap_merged_for_ratio['raw_ratio'])
      ]
      if not finite_ratios_loc.empty:
        loc_median_ratio = finite_ratios_loc.median()

    if not np.isfinite(loc_median_ratio) or loc_median_ratio <= 0:
      location_scaling_factors[loc_fips] = default_scaling_factor
    else:
      location_scaling_factors[loc_fips] = np.clip(
          loc_median_ratio, 0.01, 100.0
      )

  # 1e. Apply transformation to pre-overlap ILINet data to create `synthetic_admissions`
  pre_overlap_ilinet = available_ilinet_for_fold[
      available_ilinet_for_fold['target_end_date'] < earliest_train_date
  ].copy()

  pre_overlap_ilinet['synthetic_admissions'] = pre_overlap_ilinet.apply(
      lambda row: row['ilitotal']
      * location_scaling_factors.get(row['location'], default_scaling_factor)
      if row['ilitotal'] > 0
      else 0,
      axis=1,
  )
  pre_overlap_ilinet['synthetic_admissions'] = (
      pre_overlap_ilinet['synthetic_admissions'].round().astype(int)
  )
  pre_overlap_ilinet.loc[
      pre_overlap_ilinet['synthetic_admissions'] < 0, 'synthetic_admissions'
  ] = 0

  # 1f. Create unified `augmented_historical_data` for the current fold
  augmented_data_ilinet = pre_overlap_ilinet[
      ['location', 'target_end_date', 'synthetic_admissions', 'population']
  ].rename(columns={'synthetic_admissions': TARGET_COL})
  augmented_data_ilinet[LOG1P_TARGET_COL] = np.log1p(
      augmented_data_ilinet[TARGET_COL].clip(lower=0)
  )

  # Combine ILINet synthetic data with actual data from current_train_data
  augmented_historical_data = pd.concat(
      [
          augmented_data_ilinet,
          current_train_data[[
              'location',
              'target_end_date',
              TARGET_COL,
              LOG1P_TARGET_COL,
              'population',
          ]],
      ],
      ignore_index=True,
  )

  # Add epiweek features for augmented historical data
  augmented_historical_data = augmented_historical_data.join(
      get_epiweek_features(augmented_historical_data['target_end_date'])
  )
  augmented_historical_data.sort_values(
      ['location', 'target_end_date'], inplace=True
  )

  # 1g. Feature Engineering (Lags) for Augmented Historical Data
  augmented_historical_data = calculate_lags(
      augmented_historical_data, LOG1P_TARGET_COL, LAGGED_FEATURES
  )
  # Fill NaNs from lags with 0 (no admissions). Log1p(0) is 0.
  augmented_historical_data.fillna(0, inplace=True)

  # --- 2. Pre-calculate Bandwidths for all locations (for feature kernel) ---
  # Adhering to "Select bandwidth parameters for the kernels by maximizing a cross-validated log-likelihood score"
  location_best_bw_multipliers = {}
  KERNEL_FEATURES = ['sin_epiweek', 'cos_epiweek'] + LOG1P_LAGGED_FEATURES

  MIN_BW_FEATURE = 0.005  # Increased from 0.0005 for improved stability
  MAX_BW_FEATURE = 3.0
  DEFAULT_ROBUST_BW_FEATURE = 0.1  # Default if estimation fails

  # Define minimum data points required for meaningful cross-validation
  MIN_DATA_POINTS_FOR_CV = max(
      10, 2 * len(KERNEL_FEATURES) + 5
  )  # Increased for robustness

  for loc in locations[
      'location'
  ].unique():  # Iterate over all required CDC locations
    historical_loc_data = augmented_historical_data[
        augmented_historical_data['location'] == loc
    ].copy()
    historical_loc_data.sort_values('target_end_date', inplace=True)

    n_data_points = historical_loc_data.shape[0]
    X_train_combined = historical_loc_data[KERNEL_FEATURES].values

    best_bw_for_kde = (
        DEFAULT_ROBUST_BW_FEATURE  # Initialize with default robust bandwidth
    )

    # Robustness check: Ensure features have sufficient variance and data before attempting KDE or GridSearchCV
    if n_data_points < MIN_DATA_POINTS_FOR_CV or np.linalg.matrix_rank(
        X_train_combined
    ) < len(KERNEL_FEATURES):
      # Not enough data or insufficient feature variance/linear independence for GridSearchCV/KDE
      location_best_bw_multipliers[loc] = DEFAULT_ROBUST_BW_FEATURE
      continue  # Skip GridSearchCV for this location

    X_train_combined_for_kde = X_train_combined
    # Add jitter if feature std is very low (near constant) to prevent singular covariance during KDE fit
    # Check overall std deviation to determine if jitter is needed
    if (
        np.max(np.std(X_train_combined, axis=0)) < 1e-6
    ):  # Using a stricter threshold for 'near-constant'
      jitter = rng.normal(
          0, 1e-7, X_train_combined.shape
      )  # Use rng for consistency, smaller jitter
      X_train_combined_for_kde = X_train_combined + jitter

    try:
      # Attempt to estimate a base bandwidth using scipy's KDE factor heuristic
      # Note: scipy.stats.gaussian_kde expects data as (ndim, n_points)
      # For multivariate input, kde.factor is typically N^(-1/(d+4)). To get a scalar bandwidth
      # for sklearn.KernelDensity, a common heuristic is factor * mean(std_devs).
      # Suppress potential RuntimeWarnings from std or other numpy ops within
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        std_devs = np.std(X_train_combined_for_kde, axis=0)
        if (
            np.max(std_devs) > 1e-9
        ):  # Only try heuristic if there's actual variance
          # CORRECTED: Transpose X_train_combined_for_kde to (ndim, n_points) for scipy.stats.gaussian_kde
          kde_for_bw_heuristic = scipy.stats.gaussian_kde(
              X_train_combined_for_kde.T
          )
          estimated_base_bw = kde_for_bw_heuristic.factor * np.mean(std_devs)
          if np.isfinite(estimated_base_bw) and estimated_base_bw > 0:
            best_bw_for_kde = np.clip(
                estimated_base_bw, MIN_BW_FEATURE, MAX_BW_FEATURE
            )
        else:  # Data is effectively constant, heuristic won't help
          pass
    except (np.linalg.LinAlgError, ValueError) as e:
      pass  # scipy.stats.gaussian_kde heuristic failed. Falling back to default.

    # --- IMPROVED: Expanded and logarithmically spaced bandwidth grid ---
    # Increased num from 40 to 75 for a denser grid search to find a better optimal bandwidth.
    bandwidth_grid = np.logspace(
        np.log10(MIN_BW_FEATURE), np.log10(MAX_BW_FEATURE), num=75
    ).tolist()
    # Add the default robust bandwidth and the heuristic best_bw_for_kde to ensure they are considered
    bandwidth_grid.append(DEFAULT_ROBUST_BW_FEATURE)
    if best_bw_for_kde not in bandwidth_grid:
      bandwidth_grid.append(best_bw_for_kde)
    bandwidth_grid = sorted(
        list(set(np.round(bandwidth_grid, 5)))
    )  # Round and ensure uniqueness

    if not bandwidth_grid:  # Ensure at least one point in the grid
      bandwidth_grid = [DEFAULT_ROBUST_BW_FEATURE]

    # --- IMPROVED: Robust TimeSeriesSplit parameters for GridSearchCV ---
    # Determine a robust test_size for TimeSeriesSplit: at least 1, max 15 (increased from 10)
    # and proportional to data available relative to feature complexity.
    features_plus_buffer = len(KERNEL_FEATURES) + 5
    dynamic_ts_test_size = max(
        1, min(15, n_data_points // features_plus_buffer)
    )  # Increased max from 10 to 15

    # min_train_size should be at least (num_features + 1) for KDE to avoid singular matrices.
    min_train_size_for_ts_split = len(KERNEL_FEATURES) + 1

    # Calculate possible splits, ensuring enough data for initial train and a test set
    possible_splits = 0
    if n_data_points >= min_train_size_for_ts_split + dynamic_ts_test_size:
      possible_splits = (
          n_data_points - min_train_size_for_ts_split
      ) // dynamic_ts_test_size

    # Cap max splits at 8 (increased from 5), min at 1. If no possible splits, it will be 0, leading to the warning below.
    n_splits_cv = min(8, max(1, possible_splits))  # Increased max from 5 to 8

    # --- IMPROVEMENT: Skip GridSearchCV if not enough splits for meaningful CV ---
    if (
        n_splits_cv < 2
    ):  # If after calculation, there are still no valid splits for proper cross-validation
      location_best_bw_multipliers[loc] = (
          best_bw_for_kde
          if np.isfinite(best_bw_for_kde) and best_bw_for_kde > 0
          else DEFAULT_ROBUST_BW_FEATURE
      )
      continue  # Skip GridSearchCV for this location

    try:
      grid = GridSearchCV(
          KernelDensity(kernel='gaussian', rtol=1e-5, atol=1e-5),
          {'bandwidth': bandwidth_grid},
          cv=TimeSeriesSplit(
              n_splits=n_splits_cv, test_size=dynamic_ts_test_size
          ),  # Use dynamic_ts_test_size
          scoring=None,  # Use estimator's default score method (log-likelihood)
          n_jobs=-1,  # Use all available cores
          verbose=0,
      )
      # Catch warnings from GridSearchCV fit
      with warnings.catch_warnings():
        warnings.simplefilter(
            'ignore', UserWarning
        )  # Ignore sklearn's common UserWarnings
        warnings.simplefilter(
            'ignore', RuntimeWarning
        )  # Ignore numpy runtime warnings
        grid.fit(
            X_train_combined_for_kde
        )  # Fit on potentially jittered data for stability
      best_bw_for_kde = grid.best_params_['bandwidth']
    except (
        Exception
    ) as e:  # Catch broader exceptions for robustness in GridSearchCV
      pass  # GridSearchCV for bandwidth failed. Falling back to initial best estimate.

    # Store the final bandwidth for this location, ensuring it's clipped
    location_best_bw_multipliers[loc] = np.clip(
        best_bw_for_kde, MIN_BW_FEATURE, MAX_BW_FEATURE
    )

  # --- 3. Prediction Loop for test_x ---
  test_y_hat_quantiles = pd.DataFrame(
      index=test_x_copy.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  # Group test_x by (location, reference_date) to process all horizons for a forecast together
  unique_forecast_contexts = (
      test_x_copy[['location', 'reference_date']]
      .drop_duplicates()
      .itertuples(index=False)
  )

  for loc, ref_date in unique_forecast_contexts:
    # 3a. Determine the Current Prediction Context (features for the reference_date)
    current_epi_features_df = get_epiweek_features(
        pd.Series([ref_date], index=[0])
    )
    current_sin_epiweek = current_epi_features_df.iloc[0]['sin_epiweek']
    current_cos_epiweek = current_epi_features_df.iloc[0]['cos_epiweek']

    # IMPROVEMENT: Fetch lagged values from augmented_historical_data
    current_log1p_lag_values = []
    relevant_lag_source_data = augmented_historical_data[
        (augmented_historical_data['location'] == loc)
        & (
            augmented_historical_data['target_end_date'] < ref_date
        )  # Strictly before ref_date
    ].set_index(
        'target_end_date'
    )  # Index by date for easier lookup

    for lag_amount in LAGGED_FEATURES:
      lag_date = ref_date - pd.Timedelta(weeks=lag_amount)
      # Use .get() on the series, or .loc[idx, col] with check for robustness
      lag_value = 0.0
      if lag_date in relevant_lag_source_data.index:
        lag_value = relevant_lag_source_data.loc[lag_date, LOG1P_TARGET_COL]
      current_log1p_lag_values.append(lag_value)

    # Feature vectors for current context
    current_features_combined = np.array(
        [current_sin_epiweek, current_cos_epiweek] + current_log1p_lag_values
    )

    # 3b. Kernel-Weighted Historical Analogs
    # Filter `augmented_historical_data` to only include data prior to the current reference_date
    historical_loc_data_for_kernels = augmented_historical_data[
        (augmented_historical_data['location'] == loc)
        & (augmented_historical_data['target_end_date'] < ref_date)
    ].copy()
    historical_loc_data_for_kernels.sort_values(
        'target_end_date', inplace=True
    )  # Ensure sorted for lag correlation

    # Filter out rows with any NaN in KERNEL_FEATURES before calculating weights
    historical_loc_data_for_kernels.dropna(subset=KERNEL_FEATURES, inplace=True)

    # This series is used to lookup target values at horizon_val, so it should be from the full augmented data, not just for kernels
    historical_loc_data_indexed = augmented_historical_data[
        augmented_historical_data['location'] == loc
    ].set_index('target_end_date')[LOG1P_TARGET_COL]

    # Handle cases where historical data is insufficient for kernel estimation
    min_kernel_data_points = (
        len(KERNEL_FEATURES) + 1
    )  # At least one more point than features for covariance
    if (
        historical_loc_data_for_kernels.empty
        or historical_loc_data_for_kernels.shape[0] < min_kernel_data_points
    ):
      forecast_rows = test_x_copy[
          (test_x_copy['location'] == loc)
          & (test_x_copy['reference_date'] == ref_date)
      ]
      # Fill with 0s if no valid historical data
      test_y_hat_quantiles.loc[
          forecast_rows.index, [f'quantile_{q}' for q in QUANTILES]
      ] = 0
      continue

    historical_features_combined = historical_loc_data_for_kernels[
        KERNEL_FEATURES
    ].values

    # Use the optimized bandwidth for this location (guaranteed to be clipped or default)
    optimized_bw_scalar = location_best_bw_multipliers.get(
        loc, DEFAULT_ROBUST_BW_FEATURE
    )  # Fallback if somehow not present

    # Calculate Gaussian kernel weights for combined features
    diff_combined = historical_features_combined - current_features_combined
    # Using optimized_bw_scalar as the std dev of the kernel, so variance is optimized_bw_scalar^2
    # Use a small epsilon to avoid division by zero if optimized_bw_scalar is extremely close to zero.
    kernel_weights = np.exp(
        -0.5
        * np.sum((diff_combined / (optimized_bw_scalar + 1e-9)) ** 2, axis=1)
    )

    # Fallback to uniform weights if all calculated weights are effectively zero
    sum_kernel_weights = np.sum(kernel_weights)
    if sum_kernel_weights == 0 or np.all(
        kernel_weights < 1e-10
    ):  # Check for extremely small sum as well
      normalized_weights = np.ones_like(kernel_weights) / len(kernel_weights)
    else:
      normalized_weights = kernel_weights / sum_kernel_weights

    # --- Use Dynamically Calculated Lag-1 Correlation for Copula ---
    series_for_dynamic_rho = historical_loc_data_for_kernels[LOG1P_TARGET_COL]
    # Pass series and weights explicitly aligned
    local_rho = _estimate_weighted_lag1_correlation(
        series_for_dynamic_rho,
        pd.Series(normalized_weights, index=series_for_dynamic_rho.index),
    )

    forecast_horizons_for_ref_date = sorted(
        test_x_copy[
            (test_x_copy['location'] == loc)
            & (test_x_copy['reference_date'] == ref_date)
        ]['horizon'].unique()
    )

    if (
        not forecast_horizons_for_ref_date
    ):  # No horizons to forecast for this context
      forecast_rows = test_x_copy[
          (test_x_copy['location'] == loc)
          & (test_x_copy['reference_date'] == ref_date)
      ]
      test_y_hat_quantiles.loc[
          forecast_rows.index, [f'quantile_{q}' for q in QUANTILES]
      ] = 0
      continue

    # --- 3c. Estimate Weekly Conditional Densities using KDE ---
    marginal_inv_ecdfs = []
    MIN_BW_MARGINAL = 0.005  # Increased from 0.001 for improved stability
    MAX_BW_MARGINAL = 1.0
    DEFAULT_ROBUST_BW_MARGINAL = 0.1  # This is for log1p space

    for h_idx in range(len(forecast_horizons_for_ref_date)):
      # horizon_val = forecast_horizons_for_ref_date[h_idx] # Not directly used for data collection here

      Y_h_for_kde_list = []
      weights_for_kde_list = []

      # Iterate through historical analogs used to calculate kernel weights
      # and find their corresponding target values at `horizon_val` relative to analog_ref_date
      # The analog_ref_date is the 'current' date for the historical analog.
      # target_date_h is the 'future' date relative to the analog_ref_date.
      for i, analog_ref_date in enumerate(
          historical_loc_data_for_kernels['target_end_date']
      ):
        # The horizon_val here is the specific horizon from forecast_horizons_for_ref_date
        # for which we are building the marginal distribution.
        target_date_h = analog_ref_date + pd.Timedelta(
            weeks=forecast_horizons_for_ref_date[h_idx]
        )

        # Get the actual historical target value for this date and location
        target_val = historical_loc_data_indexed.get(target_date_h)
        if target_val is not None and not np.isnan(target_val):
          Y_h_for_kde_list.append(target_val)
          weights_for_kde_list.append(
              normalized_weights[i]
          )  # Use the pre-calculated weights

      Y_h_for_kde = np.array(Y_h_for_kde_list)
      weights_for_kde = np.array(weights_for_kde_list)

      # Filter out points with effectively zero weight to avoid numerical issues and improve KDE stability
      valid_mask_kde_target = weights_for_kde > 1e-9
      Y_h_for_kde = Y_h_for_kde[valid_mask_kde_target]
      weights_for_kde = weights_for_kde[valid_mask_kde_target]

      # Dynamically estimate bandwidth for marginal KDE, with robustness checks
      dynamic_bw_marginal = DEFAULT_ROBUST_BW_MARGINAL

      # --- Robustness improvement for marginal KDE bandwidth estimation and fallback ---
      # Check for sufficient data points and variance for meaningful KDE
      # If not enough data, or data is constant, or weights sum to near zero, fallback to empirical.
      if len(Y_h_for_kde) > 1 and np.sum(weights_for_kde) > 1e-9:
        std_y_h = np.std(Y_h_for_kde)
        if (
            std_y_h < 1e-9
        ):  # Effectively constant data for marginal, use min bandwidth
          dynamic_bw_marginal = MIN_BW_MARGINAL
        else:
          try:
            # Use scipy.stats.gaussian_kde to estimate an adaptive bandwidth, passing weights
            # For 1D data, a common heuristic is factor * std(data)
            with warnings.catch_warnings():
              warnings.simplefilter(
                  'ignore', RuntimeWarning
              )  # Ignore potential warnings from np.std on small data
              kde_bw_estimator = scipy.stats.gaussian_kde(
                  Y_h_for_kde.reshape(1, -1), weights=weights_for_kde
              )
            estimated_bw = (
                kde_bw_estimator.factor * std_y_h
            )  # Use calculated std_y_h

            # --- Robustness improvement: Ensure estimated_bw is finite and positive ---
            if np.isfinite(estimated_bw) and estimated_bw > 0:
              dynamic_bw_marginal = np.clip(
                  estimated_bw, MIN_BW_MARGINAL, MAX_BW_MARGINAL
              )
            else:  # Fallback if heuristic gave non-finite or non-positive result
              dynamic_bw_marginal = np.clip(
                  DEFAULT_ROBUST_BW_MARGINAL, MIN_BW_MARGINAL, MAX_BW_MARGINAL
              )

          except (np.linalg.LinAlgError, ValueError):
            # Heuristic marginal bandwidth estimation failed, fall back to default clipped value
            dynamic_bw_marginal = np.clip(
                DEFAULT_ROBUST_BW_MARGINAL, MIN_BW_MARGINAL, MAX_BW_MARGINAL
            )

        try:
          # Fit KernelDensity with sample_weight to reflect weighted observations
          kde_marginal = KernelDensity(
              kernel='gaussian',
              bandwidth=dynamic_bw_marginal,
              rtol=1e-5,
              atol=1e-5,
          )
          # Suppress KDE warnings during fit
          with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            kde_marginal.fit(
                Y_h_for_kde.reshape(-1, 1), sample_weight=weights_for_kde
            )
          # Generate samples from the KDE to form a smooth empirical inverse CDF
          # Use the initialized rng for sampling from KDE for consistency (though KDE's random_state is int)
          kde_samples = np.sort(
              kde_marginal.sample(
                  N_SAMPLES_FROM_MARGINAL_KDE, random_state=RANDOM_STATE
              )
          )
          marginal_inv_ecdfs.append(
              lambda q, samples=kde_samples: np.quantile(samples, q)
          )
        except Exception as e:
          # Fallback: if KDE fails (e.g., LinAlgError), use weighted empirical quantiles of the raw data.
          if len(Y_h_for_kde) > 0 and np.sum(weights_for_kde) > 0:
            # Create a weighted ECDF and sample from it.
            sorted_indices = np.argsort(Y_h_for_kde)
            sorted_y = Y_h_for_kde[sorted_indices]
            sorted_weights = weights_for_kde[sorted_indices]
            cumulative_weights = np.cumsum(sorted_weights)
            total_weight = cumulative_weights[-1]
            normalized_cum_weights = cumulative_weights / total_weight

            # Use interpolation to find quantiles from weighted ECDF
            marginal_inv_ecdfs.append(
                lambda q, y_vals=sorted_y, norm_cum_w=normalized_cum_weights: np.interp(
                    q, norm_cum_w, y_vals, left=y_vals[0], right=y_vals[-1]
                )
            )
          else:  # If no data or weights for this horizon, predict 0.0
            marginal_inv_ecdfs.append(lambda q: 0.0)
      elif (
          len(Y_h_for_kde) > 0 and np.sum(weights_for_kde) > 0
      ):  # Insufficient for KDE, but some data exists. Use weighted empirical.
        sorted_indices = np.argsort(Y_h_for_kde)
        sorted_y = Y_h_for_kde[sorted_indices]
        sorted_weights = weights_for_kde[sorted_indices]
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        normalized_cum_weights = cumulative_weights / total_weight
        marginal_inv_ecdfs.append(
            lambda q, y_vals=sorted_y, norm_cum_w=normalized_cum_weights: np.interp(
                q, norm_cum_w, y_vals, left=y_vals[0], right=y_vals[-1]
            )
        )
      else:  # If no data for this horizon, predict 0.0
        marginal_inv_ecdfs.append(lambda q: 0.0)

    # --- 3d. Model Joint Dependence with Isotropic Normal Copula (using local_rho) ---
    num_horizons = len(forecast_horizons_for_ref_date)

    # --- Robustness improvement: Ensure enough horizons for meaningful copula or revert to independent sampling ---
    if (
        num_horizons < 2
        or N_SIMULATIONS < 2
        or np.isclose(local_rho, 0.0, atol=1e-5)
    ):
      # Fallback to independence if not enough horizons/simulations, or rho is effectively zero
      simulated_forecast_trajectories_log1p = np.zeros(
          (N_SIMULATIONS, num_horizons)
      )
      for h_idx in range(num_horizons):
        # Ensure random samples for quantile are distinct for each simulation, using rng
        simulated_forecast_trajectories_log1p[:, h_idx] = marginal_inv_ecdfs[
            h_idx
        ](rng.random(N_SIMULATIONS))
    else:
      # Construct AR(1)-like correlation matrix for the copula
      R_isotropic = np.array([
          [local_rho ** (abs(i - j)) for j in range(num_horizons)]
          for i in range(num_horizons)
      ])
      np.fill_diagonal(R_isotropic, 1.0)  # Ensure diagonal is exactly 1

      try:
        # --- Improvement: Robust Positive Definiteness for Correlation Matrix ---
        # Ensure matrix is perfectly symmetric
        R_stable = (R_isotropic + R_isotropic.T) / 2

        # Check if the matrix is positive semi-definite (eigenvalues >= 0)
        # If not strictly positive definite, adjust negative eigenvalues to a small positive value.
        eigenvalues = np.linalg.eigvalsh(R_stable)
        min_eig = np.min(eigenvalues)
        epsilon_pd = 1e-7  # Threshold for numerical positive definiteness
        if min_eig < epsilon_pd:
          # Add just enough to the diagonal to make it positive definite, ensuring only non-negative additions
          R_stable += np.eye(num_horizons) * max(0, epsilon_pd - min_eig)
          # Correlation matrix was not positive definite. Adjusted.

        # Use the initialized rng for multivariate_normal
        simulated_normal_samples = rng.multivariate_normal(
            np.zeros(num_horizons),
            R_stable,
            size=N_SIMULATIONS,
            check_valid=(  # Warn if matrix is not positive definite, but attempt to use.
                'warn'
            ),
        )
        simulated_uniform_samples = scipy.stats.norm.cdf(
            simulated_normal_samples
        )

        simulated_forecast_trajectories_log1p = np.zeros_like(
            simulated_uniform_samples
        )
        for h_idx in range(num_horizons):
          simulated_forecast_trajectories_log1p[:, h_idx] = marginal_inv_ecdfs[
              h_idx
          ](simulated_uniform_samples[:, h_idx])

      except np.linalg.LinAlgError:
        # R_isotropic not positive definite even with adjustment. Falling back to independence.
        simulated_forecast_trajectories_log1p = np.zeros(
            (N_SIMULATIONS, num_horizons)
        )
        for h_idx in range(num_horizons):
          simulated_forecast_trajectories_log1p[:, h_idx] = marginal_inv_ecdfs[
              h_idx
          ](rng.random(N_SIMULATIONS))
      except Exception as e:
        # Error in multivariate normal sampling. Falling back to independence.
        simulated_forecast_trajectories_log1p = np.zeros(
            (N_SIMULATIONS, num_horizons)
        )
        for h_idx in range(num_horizons):
          simulated_forecast_trajectories_log1p[:, h_idx] = marginal_inv_ecdfs[
              h_idx
          ](rng.random(N_SIMULATIONS))

    # --- 3e. Derive Quantile Predictions ---
    for i, horizon_val in enumerate(forecast_horizons_for_ref_date):
      log1p_admissions_at_horizon = simulated_forecast_trajectories_log1p[:, i]

      # --- Robustness improvement: Ensure there are valid samples before calculating quantiles ---
      if len(log1p_admissions_at_horizon) == 0 or not np.any(
          np.isfinite(log1p_admissions_at_horizon)
      ):
        quantile_predictions_log1p = np.zeros(
            len(QUANTILES)
        )  # Explicitly initialize as zeros
      else:
        # Ensure quantiles are calculated robustly even if data is constant
        with warnings.catch_warnings():
          warnings.simplefilter(
              'ignore', RuntimeWarning
          )  # Ignore warnings if all elements are NaN or inf for quantile
          quantile_predictions_log1p = np.quantile(
              log1p_admissions_at_horizon, QUANTILES
          )

      # Ensure predictions are non-negative before inverse transform and rounding
      quantile_predictions = (
          np.expm1(quantile_predictions_log1p).clip(min=0).round().astype(int)
      )

      current_forecast_row_index = test_x_copy[
          (test_x_copy['location'] == loc)
          & (test_x_copy['reference_date'] == ref_date)
          & (test_x_copy['horizon'] == horizon_val)
      ].index

      test_y_hat_quantiles.loc[
          current_forecast_row_index, [f'quantile_{q}' for q in QUANTILES]
      ] = quantile_predictions

  # 3f. Enforce Monotonicity and Non-negativity
  test_y_hat_quantiles = enforce_monotonicity_and_nonnegativity(
      test_y_hat_quantiles, QUANTILES
  )

  for q_col in [f'quantile_{q}' for q in QUANTILES]:
    test_y_hat_quantiles[q_col] = test_y_hat_quantiles[q_col].astype(int)

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
