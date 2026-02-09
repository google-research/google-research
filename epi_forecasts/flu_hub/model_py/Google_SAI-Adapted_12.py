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
MODEL_NAME = 'Google_SAI-Adapted_12'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
import warnings
import datetime
from typing import Any, Callable, Dict, List, Set, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.linear_model import LinearRegression

# Global constants (QUANTILES, TARGET_STR, locations, ilinet_state, REQUIRED_CDC_LOCATIONS)
# are assumed to be available from the notebook's global scope as per instructions.

# Change global numpy error handling for robustness during intermediate calculations
np.seterr(over='warn', divide='warn', under='ignore', invalid='warn')


class FluSeasonForecaster:
  """A class to encapsulate the flu season forecasting model using Gaussian Process Regression

  with heteroskedasticity and latent season severity variables.
  Adheres strictly to the Method Contract provided in the problem description.
  """

  # Small epsilon for numerical stability, consistent across the class.
  _EPSILON = 1e-6  # Slightly smaller epsilon for greater numerical precision in some calculations

  def __init__(
      self,
      locations_df,
      ilinet_df,
      required_cdc_locations,
      quantiles,
  ):
    self.locations_df = locations_df
    self.ilinet_df = ilinet_df
    self.required_cdc_locations = required_cdc_locations
    self.quantiles = quantiles
    self.location_models: Dict[int, GaussianProcessRegressor] = (
        {}
    )  # Stores a single GPR model per location
    # Stores location-specific severity thresholds (transformed scale) used for x4 classification
    self.location_severity_thresholds: Dict[int, Dict[str, float]] = {}
    # Stores location-specific transformation parameters (slope, intercept) for ILINet data
    self.location_transformation_params: Dict[int, Dict[str, Any]] = {}
    # Stores the estimated eta_sq values for each severity category for each location.
    # These are the additional, category-specific variance components (beyond GPR's WhiteKernel).
    self.location_etas: Dict[int, Dict[int, float]] = {}
    # Default thresholds on the transformed scale, representing raw counts like 24 and 100.
    self.default_transformed_mild_threshold = self.f(
        24.0
    )  # Corresponds to <25 raw admissions
    self.default_transformed_severe_threshold = self.f(100.0)

  @staticmethod
  def f(x):
    """Transforms incidence counts using f(x) = sqrt(x + 1) - 1.

    Ensures input is non-negative before sqrt.
    """
    return np.sqrt(np.maximum(0, x) + 1) - 1

  @staticmethod
  def f_inv(y):
    """Inverse transforms the model output back to incidence counts.

    f_inv(y) = (y + 1)**2 - 1. Ensures output is non-negative.
    """
    return np.maximum(0, (y + 1) ** 2 - 1)

  @staticmethod
  def get_epiweek(date):
    """Get ISO week number (1-52 or 53)."""
    return date.isocalendar()[1]

  @staticmethod
  def get_season_year(date):
    """Determines the flu season's starting calendar year for a given date.

    Flu season typically starts in epiweek 40.
    """
    if FluSeasonForecaster.get_epiweek(date) >= 40:
      return date.year
    else:
      return date.year - 1

  @staticmethod
  def _get_season_str(date):
    """Returns season string in 'YYYY/YY' format (e.g., '1997/98')."""
    start_year = FluSeasonForecaster.get_season_year(date)
    end_year_short = (start_year + 1) % 100  # Last two digits of end year
    return (  # Use 02d for two digits, e.g. 1997/98
        f'{start_year}/{end_year_short:02d}'
    )

  @staticmethod
  def _get_season_week_num(epiweek):
    """Calculates the week number within a flu season (1-52).

    Week 40 of a calendar year is defined as season week 1.
    """
    season_week = (
        (epiweek - 40 + 52) % 52 + 1 if epiweek >= 40 else (epiweek + 12)
    )
    return int(np.clip(season_week, 1, 52))  # Ensure range 1-52

  def _prepare_historical_data(
      self, train_x, train_y
  ):
    """Prepares and augments historical data.

    This includes combining real 'Total Influenza Admissions' with 'ilitotal'
    data, learning a per-location linear transformation from transformed ILINet
    to transformed Admissions during an overlap period, and generating synthetic
    Admissions history from ILINet. Prioritizes real data over synthetic for
    overlapping dates.
    """
    # 1. Prepare core dataset (real admissions)
    train_df = train_x.copy()
    train_df[TARGET_STR] = train_y

    # Convert date columns to datetime.date objects for consistency
    train_df['target_end_date'] = pd.to_datetime(
        train_df['target_end_date']
    ).dt.date
    train_df['location_name'] = train_df['location_name'].astype(str)
    train_df['location'] = train_df['location'].astype(int)

    # Merge population from self.locations_df to ensure consistent and accurate population figures.
    train_df = pd.merge(
        train_df.drop(columns='population', errors='ignore'),
        self.locations_df[['location', 'population']],
        on='location',
        how='left',
    )
    # Mark real data for prioritization later
    train_df['is_real_data'] = True
    # Apply f() transformation to raw admissions as per Method Contract Step 1.
    train_df['transformed_raw_admissions'] = train_df[TARGET_STR].apply(self.f)

    # 2. Prepare ILINet historical data (for synthetic admissions)
    ilinet_processed = self.ilinet_df.copy()
    ilinet_processed['week_start'] = pd.to_datetime(
        ilinet_processed['week_start']
    ).dt.date
    # Target end date for ILINet data, consistent with the main dataset definition (Saturday of the epiweek)
    ilinet_processed['target_end_date'] = ilinet_processed[
        'week_start'
    ] + pd.Timedelta(days=6)
    ilinet_processed['region'] = ilinet_processed['region'].astype(str)

    # --- IMPORTANT: Filter ILINet data to be only from before 2022-10-15 ---
    # This adheres to the problem description's data availability constraint.
    ilinet_processed = ilinet_processed[
        ilinet_processed['target_end_date'] < datetime.date(2022, 10, 15)
    ]

    # Map ILINet regions (location names) to FIPS codes and merge population from self.locations_df.
    region_to_fips = self.locations_df.set_index('location_name')[
        'location'
    ].to_dict()
    ilinet_processed['location'] = ilinet_processed['region'].map(
        region_to_fips
    )
    # Drop ILINet entries for regions that do not map to a required FIPS code
    ilinet_processed = ilinet_processed.dropna(subset=['location'])
    ilinet_processed['location'] = ilinet_processed['location'].astype(int)

    # Merge population data for ILINet entries as well.
    ilinet_processed = pd.merge(
        ilinet_processed,
        self.locations_df[['location', 'population']],
        on='location',
        how='left',
    )

    # Filter ILINet data for the set of required CDC locations.
    ilinet_processed = ilinet_processed[
        ilinet_processed['location'].isin(self.required_cdc_locations)
    ]

    # Fill potential NaNs in 'ilitotal' with 0 (as it's a count variable) and ensure float type for f().
    ilinet_processed['ilitotal'] = (
        ilinet_processed['ilitotal'].fillna(0).astype(float)
    )
    # Apply f() transformation to raw ILI data. This will be the input for the mapping regression.
    ilinet_processed['transformed_raw_ilitotal'] = ilinet_processed[
        'ilitotal'
    ].apply(self.f)

    # 3. Learn a PER-LOCATION and GLOBAL robust transformation.
    # This implements Method Contract Section 4, "Learn a Transformation", with improved fallbacks.
    self.location_transformation_params = {}
    min_overlap_points_for_linear_scaling = (
        15  # Increased for more robust linear regression (y=mx+c)
    )
    min_overlap_points_for_ratio_scaling = (
        5  # For simpler median ratio scaling (y=mx)
    )

    # Use raw count thresholds for robustness, then transform for regression.
    raw_admissions_threshold_linear = 10.0  # For robust linear regression
    raw_ilitotal_threshold_linear = 50.0

    raw_admissions_threshold_ratio = 2.0  # For median ratio scaling
    raw_ilitotal_threshold_ratio = 10.0

    # Clamping bounds for learned slope and intercept, adjusted for greater flexibility.
    SLOPE_CLAMP_BOUNDS = (
        self._EPSILON,
        50.0,
    )  # Lowered minimum slope and increased max slope slightly
    INTERCEPT_CLAMP_BOUNDS = (-50.0, 50.0)  # Wider intercept bounds

    # Lists to collect all overlap data for global fallback models.
    all_robust_overlap_x_linear = (
        []
    )  # For Global Robust Linear Regression (y=mx+c)
    all_robust_overlap_y_linear = []
    all_any_signal_overlap_x = (
        []
    )  # For Global Robust Median Ratio Scaling (y=mx)
    all_any_signal_overlap_y = []

    # Determine unique locations from ILINet data that are also in REQUIRED_CDC_LOCATIONS
    unique_ilinet_locations = ilinet_processed['location'].unique()
    locations_to_process = [
        loc
        for loc in self.required_cdc_locations
        if loc in unique_ilinet_locations
    ]

    for loc_id in locations_to_process:
      loc_train_df_current = train_df[
          (train_df['location'] == loc_id) & (train_df[TARGET_STR].notna())
      ].copy()
      loc_ilinet_df_current = ilinet_processed[
          ilinet_processed['location'] == loc_id
      ].copy()

      # Identify overlap period
      overlap_df = pd.merge(
          loc_train_df_current,
          loc_ilinet_df_current[
              ['target_end_date', 'transformed_raw_ilitotal', 'ilitotal']
          ],  # Include raw ilitotal for filtering
          on='target_end_date',
          how='inner',
          suffixes=('_admissions', '_ilitotal'),
      )

      if overlap_df.empty:
        continue

      # Filter for robust data points for per-location linear regression (Tier 1)
      # Filter on raw counts, then use transformed for regression.
      robust_overlap_df_linear = overlap_df[
          (overlap_df['ilitotal'] > raw_ilitotal_threshold_linear)
          & (overlap_df[TARGET_STR] > raw_admissions_threshold_linear)
      ].copy()

      # Filter for data points with any signal for per-location ratio scaling (used for global Tier 3 fallback)
      # Filter on raw counts, then use transformed for regression.
      robust_overlap_df_ratio = overlap_df[
          (overlap_df['ilitotal'] > raw_ilitotal_threshold_ratio)
          & (overlap_df[TARGET_STR] > raw_admissions_threshold_ratio)
      ].copy()

      # Collect global data for fallbacks
      if not robust_overlap_df_linear.empty:
        all_robust_overlap_x_linear.extend(
            robust_overlap_df_linear['transformed_raw_ilitotal'].tolist()
        )
        all_robust_overlap_y_linear.extend(
            robust_overlap_df_linear['transformed_raw_admissions'].tolist()
        )
      if not robust_overlap_df_ratio.empty:
        all_any_signal_overlap_x.extend(
            robust_overlap_df_ratio['transformed_raw_ilitotal'].tolist()
        )
        all_any_signal_overlap_y.extend(
            robust_overlap_df_ratio['transformed_raw_admissions'].tolist()
        )

      # --- Per-Location Transformation Learning (Tier 1 & 2) ---
      # Tier 1: Per-location Robust Linear Regression
      if (
          len(robust_overlap_df_linear) >= min_overlap_points_for_linear_scaling
          and robust_overlap_df_linear['transformed_raw_ilitotal'].nunique() > 1
      ):
        X_overlap = np.asarray(
            robust_overlap_df_linear[['transformed_raw_ilitotal']]
        )
        y_overlap = np.asarray(
            robust_overlap_df_linear['transformed_raw_admissions']
        )
        try:
          reg = LinearRegression()
          reg.fit(X_overlap, y_overlap)
          # Clamping slope and intercept with slightly wider bounds for flexibility
          slope = np.clip(reg.coef_[0], *SLOPE_CLAMP_BOUNDS)
          intercept = np.clip(reg.intercept_, *INTERCEPT_CLAMP_BOUNDS)
          self.location_transformation_params[loc_id] = {
              'slope': slope,
              'intercept': intercept,
          }
        except Exception:
          pass  # Keep as None, proceed to next fallback

      # Tier 2: Per-location Median Ratio Scaling (if Tier 1 failed)
      if (
          self.location_transformation_params.get(loc_id) is None
          and len(robust_overlap_df_ratio)
          >= min_overlap_points_for_ratio_scaling
      ):
        median_ilitotal_transformed = np.median(
            robust_overlap_df_ratio['transformed_raw_ilitotal']
        )
        median_admissions_transformed = np.median(
            robust_overlap_df_ratio['transformed_raw_admissions']
        )
        if (
            median_ilitotal_transformed > self._EPSILON
        ):  # Avoid division by zero
          slope = np.clip(
              median_admissions_transformed / median_ilitotal_transformed,
              *SLOPE_CLAMP_BOUNDS,
          )
          self.location_transformation_params[loc_id] = {
              'slope': slope,
              'intercept': 0.0,
          }

    # --- Calculate GLOBAL fallback transformation parameters (Hierarchical Tiers 3, 4, 5) ---
    global_fallback_params = None  # Initialize with None

    # Tier 3: Global Robust Linear Regression (y=mx+c)
    if (
        len(all_robust_overlap_x_linear)
        >= min_overlap_points_for_linear_scaling
        and np.asarray(all_robust_overlap_x_linear).nunique() > 1
    ):
      try:
        global_reg_robust = LinearRegression()
        global_reg_robust.fit(
            np.asarray(all_robust_overlap_x_linear).reshape(-1, 1),
            np.asarray(all_robust_overlap_y_linear),
        )
        global_fallback_params = {
            'slope': np.clip(global_reg_robust.coef_[0], *SLOPE_CLAMP_BOUNDS),
            'intercept': np.clip(
                global_reg_robust.intercept_, *INTERCEPT_CLAMP_BOUNDS
            ),
        }
      except Exception:
        pass

    # Tier 4: Global Robust Median Ratio Scaling (y=mx)
    if (
        global_fallback_params is None
        and len(all_any_signal_overlap_x)
        >= min_overlap_points_for_ratio_scaling
    ):
      median_ili_global = np.median(all_any_signal_overlap_x)
      median_admissions_global = np.median(all_any_signal_overlap_y)
      if median_ili_global > self._EPSILON:
        global_fallback_params = {
            'slope': np.clip(
                median_admissions_global / median_ili_global,
                *SLOPE_CLAMP_BOUNDS,
            ),
            'intercept': 0.0,
        }

    # Tier 5: Absolute last resort default - More informed than 1.0, 0.0
    if global_fallback_params is None:
      global_fallback_params = {
          'slope': 2.0,
          'intercept': 0.0,
      }  # More aggressive/realistic default
      warnings.warn(
          'No robust ILINet transformations could be learned (neither'
          ' per-location nor any global fit). Falling back to slope=2.0,'
          ' intercept=0.0 for all locations without a specific fit. This is'
          ' highly suboptimal but better than 1.0, 0.0.',
          RuntimeWarning,
      )
    elif not any(
        self.location_transformation_params.get(loc_id)
        for loc_id in locations_to_process
    ):
      # Only warn if NO per-location models were found, but a global one was, indicating per-location sparsity.
      warnings.warn(
          'No per-location ILINet transformations could be learned. Falling'
          ' back to a global transformation'
          f" (slope={global_fallback_params['slope']:.2f},"
          f" intercept={global_fallback_params['intercept']:.2f}) for all"
          ' locations.',
          RuntimeWarning,
      )

    # Assign derived or best global fallback parameters to any location that did not get a specific fit.
    for loc_id in locations_to_process:
      if self.location_transformation_params.get(loc_id) is None:
        self.location_transformation_params[loc_id] = global_fallback_params

    # 4. Generate synthetic admissions (raw counts) using the learned per-location (or global fallback) transformations.
    # This implements Method Contract Section 4, Step 3.
    synthetic_admissions_data = []
    for loc_id in locations_to_process:
      loc_ilinet_df = ilinet_processed[
          ilinet_processed['location'] == loc_id
      ].copy()

      if not loc_ilinet_df.empty:
        transform_params = self.location_transformation_params.get(loc_id)

        # Apply the learned slope and intercept to the transformed_raw_ilitotal
        scaled_values = (
            loc_ilinet_df['transformed_raw_ilitotal']
            * transform_params['slope']
        ) + transform_params['intercept']
        # Clamp values at f(0.0) to avoid negative raw counts on the transformed scale
        transformed_synthetic_values_f_scale = np.maximum(
            scaled_values, self.f(0.0)
        )

        # Inverse transform the f-scaled synthetic values back to raw counts.
        loc_ilinet_df['synthetic_admissions'] = self.f_inv(
            transformed_synthetic_values_f_scale
        )
      else:
        # If no ILINet data for a location, no synthetic data can be generated.
        continue

      loc_name = self.locations_df[self.locations_df['location'] == loc_id][
          'location_name'
      ].iloc[0]
      ilinet_cols_to_keep = [
          'target_end_date',
          'location',
          'population',
          'synthetic_admissions',
      ]
      synthetic_df_for_loc = loc_ilinet_df[ilinet_cols_to_keep].rename(
          columns={'synthetic_admissions': TARGET_STR}
      )
      synthetic_df_for_loc['location_name'] = loc_name
      synthetic_df_for_loc['is_real_data'] = False  # Mark as synthetic data
      synthetic_df_for_loc['transformed_raw_admissions'] = synthetic_df_for_loc[
          TARGET_STR
      ].apply(self.f)
      synthetic_admissions_data.append(synthetic_df_for_loc)

    # 5. Combine real and synthetic data, prioritizing real data
    if synthetic_admissions_data:
      synthetic_df_all = pd.concat(synthetic_admissions_data, ignore_index=True)
      historical_data = pd.concat(
          [train_df, synthetic_df_all], ignore_index=True
      )
    else:
      historical_data = train_df  # Only real data available if no synthetic data was generated

    # Prioritize real data over synthetic for overlapping dates
    # Sort by location, then date, then by is_real_data (True first)
    historical_data = historical_data.sort_values(
        by=['location', 'target_end_date', 'is_real_data'],
        ascending=[True, True, False],
    ).drop_duplicates(subset=['target_end_date', 'location'], keep='first')

    historical_data = historical_data.drop(
        columns=['is_real_data']
    )  # Remove the helper column
    historical_data = historical_data.sort_values(
        by=['location', 'target_end_date']
    ).reset_index(drop=True)

    # 6. Add season and epiweek info based on target_end_date.
    historical_data['epiweek'] = historical_data['target_end_date'].apply(
        self.get_epiweek
    )
    historical_data['season'] = historical_data['target_end_date'].apply(
        self._get_season_str
    )

    # Calculate week within season (1-52) for x1 feature, using the centralized helper.
    historical_data['season_week_num'] = historical_data['epiweek'].apply(
        self._get_season_week_num
    )

    # Rename the unified transformed target column
    historical_data = historical_data.rename(
        columns={'transformed_raw_admissions': 'transformed_target'}
    )

    return historical_data

  def _engineer_features(self, historical_data):
    """Calculates the four predictor variables (x1, x2, x3, x4) based on Method Contract Step 2.

    Derives x4 severity thresholds from transformed-scale peaks per location.
    """
    data_with_features = historical_data.copy()

    # 'transformed_target' is already present from _prepare_historical_data

    # Initialize x1, x2, x3, x4 columns
    # x1: Week number within the season (1 to 52) - Method Contract Step 2.a
    data_with_features['x1'] = data_with_features[
        'season_week_num'
    ]  # Already calculated in _prepare_historical_data
    # x2: Sinusoidal function of x1 for cyclic behavior - Method Contract Step 2.b
    data_with_features['x2'] = np.sin(2 * np.pi * data_with_features['x1'] / 52)
    # x3: Transformed incidence value of the last week of previous season - Method Contract Step 2.c
    data_with_features['x3'] = (
        np.nan
    )  # Initialize with NaN, to be filled per season
    # x4: Categorical season severity (-1: mild, 0: moderate, 1: severe) - Method Contract Step 2.d
    data_with_features['x4'] = (
        0  # Default to moderate severity, will be updated
    )

    # Calculate PER-LOCATION severity thresholds based on transformed-scale peaks.
    # This is an adaptive approach generalizing the example thresholds provided in Method Contract Step 2.d.
    self.location_severity_thresholds = {}
    # Increased to 5 seasons for more robust percentile calculation.
    min_seasons_for_percentile_thresholds = 5

    all_global_transformed_scale_season_peaks = (
        []
    )  # For global fallback thresholds

    for loc_id in data_with_features['location'].unique():
      loc_data_sorted = data_with_features[
          data_with_features['location'] == loc_id
      ].sort_values('target_end_date')
      seasons = loc_data_sorted['season'].unique()

      loc_transformed_scale_season_peaks = []
      for season_str in seasons:
        current_season_data_subset = loc_data_sorted[
            loc_data_sorted['season'] == season_str
        ]
        if not current_season_data_subset.empty:
          peak = current_season_data_subset['transformed_target'].max()
          loc_transformed_scale_season_peaks.append(peak)

      # Add to global list for fallback
      all_global_transformed_scale_season_peaks.extend(
          loc_transformed_scale_season_peaks
      )

      # Tier 1: Per-location percentile thresholds
      if (
          len(loc_transformed_scale_season_peaks)
          >= min_seasons_for_percentile_thresholds
      ):
        loc_peaks_np_transformed = np.array(loc_transformed_scale_season_peaks)
        mild_threshold = np.percentile(loc_peaks_np_transformed, 33)
        severe_threshold = np.percentile(loc_peaks_np_transformed, 66)

        # Adjusted clamping of derived thresholds to more epidemiologically plausible ranges.
        # Mild threshold can be clamped between f(0) and f(40) (raw 0 to ~5.4)
        # Severe threshold can be clamped between f(20) and f(400) (raw ~3.3 to ~19)
        mild_threshold = np.clip(mild_threshold, self.f(0.0), self.f(40.0))
        severe_threshold = np.clip(
            severe_threshold, self.f(20.0), self.f(400.0)
        )
        # Ensure severe threshold is always strictly greater than mild threshold
        if mild_threshold >= severe_threshold:
          severe_threshold = (
              mild_threshold + self._EPSILON
          )  # Use EPSILON for strict separation

        self.location_severity_thresholds[loc_id] = {
            'mild': mild_threshold,
            'severe': severe_threshold,
        }
      # Else, leave self.location_severity_thresholds[loc_id] as None to be filled by global fallback below.

    # --- Tier 2: Global percentile thresholds fallback ---
    global_mild_threshold = self.default_transformed_mild_threshold
    global_severe_threshold = self.default_transformed_severe_threshold

    if (
        len(all_global_transformed_scale_season_peaks)
        >= min_seasons_for_percentile_thresholds
    ):
      global_peaks_np_transformed = np.array(
          all_global_transformed_scale_season_peaks
      )
      global_mild_threshold = np.percentile(global_peaks_np_transformed, 33)
      global_severe_threshold = np.percentile(global_peaks_np_transformed, 66)

      # Apply same clamping as for per-location thresholds
      global_mild_threshold = np.clip(
          global_mild_threshold, self.f(0.0), self.f(40.0)
      )
      global_severe_threshold = np.clip(
          global_severe_threshold, self.f(20.0), self.f(400.0)
      )
      if global_mild_threshold >= global_severe_threshold:
        global_severe_threshold = (
            global_mild_threshold + self._EPSILON
        )  # Use EPSILON for strict separation
    else:
      warnings.warn(
          'Insufficient total seasons across all locations to derive global'
          ' percentile thresholds. Using default absolute thresholds for global'
          ' fallback.',
          RuntimeWarning,
      )

    # Assign derived or best global fallback parameters to any location that did not get a specific fit.
    for loc_id in data_with_features['location'].unique():
      if self.location_severity_thresholds.get(loc_id) is None:
        # Tier 2 fallback: use global thresholds
        self.location_severity_thresholds[loc_id] = {
            'mild': global_mild_threshold,
            'severe': global_severe_threshold,
        }
        # Tier 3 implicit: if global thresholds were still defaults, then these are the defaults.
        warnings.warn(
            f'Loc {loc_id}: Insufficient seasons to derive percentile'
            ' thresholds. Using global fallback thresholds (Mild:'
            f' {self.f_inv(global_mild_threshold):.1f}, Severe:'
            f' {self.f_inv(global_severe_threshold):.1f} raw).',
            RuntimeWarning,
        )

    # Calculate x3 and assign x4 for each season and location.
    for loc_id in data_with_features['location'].unique():
      loc_data_idx = data_with_features['location'] == loc_id
      loc_data_sorted_by_date = (
          data_with_features.loc[loc_data_idx]
          .sort_values('target_end_date')
          .copy()
      )

      # Retrieve location-specific thresholds.
      current_loc_thresholds = self.location_severity_thresholds.get(
          loc_id
      )  # Guaranteed to exist by now

      for season_str in loc_data_sorted_by_date['season'].unique():
        season_idx = loc_data_sorted_by_date['season'] == season_str
        current_season_data_subset_all_weeks = loc_data_sorted_by_date.loc[
            season_idx
        ].copy()

        if current_season_data_subset_all_weeks.empty:
          continue  # Skip if no data for this season for this location

        # Step 2.c: Determine x3 for the current season (constant for all weeks within a season).
        # Default if no prior data can be found (transformed value for 0 raw admissions).
        x3_val_for_season = self.f(0.0)

        # Find the last observed transformed target from the *previous* season.
        # Strictly use data from a season that *completely precedes* the current season.
        current_season_start_date = current_season_data_subset_all_weeks[
            'target_end_date'
        ].min()
        prev_season_data = (
            loc_data_sorted_by_date.loc[(
                loc_data_sorted_by_date['target_end_date']
                < current_season_start_date
            )]
            .sort_values('target_end_date', ascending=False)
            .head(1)
        )

        if not prev_season_data.empty:
          x3_val_for_season = prev_season_data['transformed_target'].iloc[0]
        else:
          # If no previous season data, use the transformed value of the first week (epiweek 40)
          # of the current season IF that week is present in the historical data.
          # This ensures x3 is based on data *at or before* the season start.
          first_week_current_season_data = loc_data_sorted_by_date.loc[
              (loc_data_sorted_by_date['season'] == season_str)
              & (loc_data_sorted_by_date['season_week_num'] == 1)
          ]
          if not first_week_current_season_data.empty:
            x3_val_for_season = first_week_current_season_data[
                'transformed_target'
            ].iloc[0]

        # Assign this calculated x3 value to all weeks within the current season.
        data_with_features.loc[loc_data_idx & season_idx, 'x3'] = (
            x3_val_for_season
        )

        # Step 2.d: Assign severity categories (x4) based on PER-LOCATION TRANSFORMED thresholds.
        current_season_peak_transformed = current_season_data_subset_all_weeks[
            'transformed_target'
        ].max()

        if current_season_peak_transformed < current_loc_thresholds['mild']:
          data_with_features.loc[loc_data_idx & season_idx, 'x4'] = -1  # Mild
        elif current_season_peak_transformed > current_loc_thresholds['severe']:
          data_with_features.loc[loc_data_idx & season_idx, 'x4'] = 1  # Severe
        else:
          data_with_features.loc[loc_data_idx & season_idx, 'x4'] = (
              0  # Moderate
          )

    return data_with_features

  def _calculate_log_likelihood(
      self,
      x4_latent_val,  # The single scalar value (continuous latent variable) to optimize for x4
      gpr_model,
      observed_x_features,  # x1, x2, x3 for observed points in current season
      observed_y_transformed,  # transformed_target for observed points
      loc_id,
  ):
    """Calculates the negative sum of log-likelihoods for observed data given a

    GPR model and a specific x4_latent_val, incorporating heteroskedasticity.
    This function is designed to be the objective for `minimize_scalar` (Method
    Contract Step 5.c.ii).
    """
    if (
        gpr_model is None
        or observed_x_features.empty
        or observed_y_transformed.size == 0
    ):
      return np.inf  # Return infinity if no model or no observed data

    # Create X for GPR prediction by adding the constant x4_latent_val as the x4 feature
    X_for_predict = observed_x_features.copy()
    X_for_predict['x4'] = x4_latent_val  # Assign the continuous latent value

    # Determine the severity category based on the rounded x4_latent_val.
    # This is used to select the correct category-specific noise component (eta_sq).
    x4_category_for_eta = int(np.clip(round(x4_latent_val), -1, 1))

    # Retrieve the estimated category-specific additional noise (variance component, eta_sq)
    # as estimated during the `fit` phase (Method Contract Step 4).
    # Fallback to a small positive value if category is not in self.location_etas (this should be rare with global fallback).
    eta_category_sq = self.location_etas.get(loc_id, {}).get(
        x4_category_for_eta, self._EPSILON
    )

    try:
      # Predict mean and standard deviation from the GPR for the observed points.
      mean_pred, std_pred = gpr_model.predict(
          X_for_predict.values, return_std=True
      )

      # Add the category-specific noise (eta_category_sq) to the GPR's predicted variance.
      # This implements the heteroskedasticity: Λ's diagonal elements depend on x4.
      effective_std = np.sqrt(std_pred**2 + eta_category_sq)

      # Ensure standard deviation is positive for logpdf calculation
      effective_std_safe = np.maximum(
          effective_std, self._EPSILON
      )  # Add small epsilon for numerical stability

      # Calculate sum of log-likelihoods for all observed points
      log_likelihoods = norm.logpdf(
          observed_y_transformed, loc=mean_pred, scale=effective_std_safe
      )

      return -np.sum(
          log_likelihoods
      )  # Return negative sum for minimization (Method Contract Step 5.c.ii)

    except Exception as e:
      # warnings.warn(f"Likelihood calculation failed for loc {loc_id}, x4_latent_val {x4_latent_val}: {e}", RuntimeWarning)
      return np.inf  # Return infinity to penalize problematic x4_latent values

  def fit(self, historical_data_with_features):
    """Fits a single Gaussian Process Regressor model per location using x1, x2, x3, x4 as inputs.

    Then estimates severity-dependent nugget terms (eta_sq) from residuals,
    implementing heteroskedasticity. This covers Method Contract Steps 3 and 4.
    """
    self.location_models = {}
    # Stores {'loc_id': {-1: eta_sq_mild, 0: eta_sq_moderate, 1: eta_sq_severe}}
    # Initialized here, actual global values filled later
    self.location_etas = {}

    # Define the Gaussian Process Kernel as per Method Contract Step 3.
    # ConstantKernel scales the RBF (C matrix). RBF captures correlation based on features (x1, x2, x3, x4).
    # WhiteKernel adds independent base noise (nugget effect) (Λ_base matrix) for the GPR itself.
    # length_scale includes a 4th dimension for x4, which is treated as a continuous input.
    base_kernel = ConstantKernel(
        constant_value_bounds=(0.1, 10.0)
    ) * RBF(  # Adjusted ConstantKernel bounds
        # Adjusted length_scale_bounds to provide more focused search space for hyperparameters.
        length_scale=[
            10.0,
            0.5,
            5.0,
            0.5,
        ],  # Initial value for x4 length_scale 0.5
        # Bounds for length_scale: (x1: seasonal pattern, x2: cyclical, x3: prev season, x4: severity latent)
        # Increased upper bound for x3 length_scale_bounds from 10.0 to 30.0 for more flexibility.
        # Tightened x4 length_scale_bounds from (0.1, 2.0) to (0.1, 1.0)
        length_scale_bounds=[(5.0, 50.0), (0.1, 5.0), (0.1, 30.0), (0.1, 1.0)],
    ) + WhiteKernel(
        noise_level_bounds=(0.001, 0.5)
    )  # Adjusted WhiteKernel noise bounds for normalized y

    # Temporary storage for global eta_sq calculation
    global_eta_sq_by_category = {-1: [], 0: [], 1: []}

    for loc_id in historical_data_with_features['location'].unique():
      loc_data = historical_data_with_features[
          historical_data_with_features['location'] == loc_id
      ].copy()
      # Drop rows with NaN in features or target before fitting to ensure GPR stability.
      loc_data = loc_data.dropna(
          subset=['x1', 'x2', 'x3', 'x4', 'transformed_target']
      )

      # GPR requires at least 5 samples for a 4-dimensional input space (x1, x2, x3, x4) for robust fitting.
      if not loc_data.empty and len(loc_data) >= 5:
        X_train = loc_data[['x1', 'x2', 'x3', 'x4']].values
        y_train = loc_data['transformed_target'].values

        gpr = GaussianProcessRegressor(
            kernel=base_kernel,
            n_restarts_optimizer=20,  # Increased restarts to 20 for better kernel hyperparameter optimization
            random_state=42,
            normalize_y=True,  # Normalize y to potentially improve fitting stability and performance
        )

        try:
          with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore'
            )  # Suppress convergence warnings from GPR optimization
            gpr.fit(X_train, y_train)
          self.location_models[loc_id] = gpr

          # Estimate category-specific nugget terms (eta_sq) from residuals.
          # This implements Method Contract Step 4: three separate nugget parameters.
          y_pred_train, std_pred_train = gpr.predict(X_train, return_std=True)
          residuals = y_train - y_pred_train

          self.location_etas[loc_id] = {}
          for severity_category in [-1, 0, 1]:
            category_mask = (loc_data['x4'] == severity_category).values
            # Tier 1: Per-location category variance
            if np.any(category_mask) and len(residuals[category_mask]) > 1:
              category_noise_sq = np.var(residuals[category_mask])
              # Ensure non-negative and add a small floor for numerical stability
              self.location_etas[loc_id][severity_category] = max(
                  self._EPSILON, category_noise_sq
              )
              global_eta_sq_by_category[severity_category].append(
                  self.location_etas[loc_id][severity_category]
              )
            else:
              # Mark for global fallback if not enough data for local estimate.
              # Set to None for now, to be filled by global median later.
              self.location_etas[loc_id][severity_category] = None

        except Exception as e:
          warnings.warn(
              f'GPR fit failed for loc {loc_id}: {e}. Skipping model for this'
              ' location.',
              RuntimeWarning,
          )
          self.location_models[loc_id] = None  # Mark model as unavailable
      else:
        warnings.warn(
            f'Loc {loc_id}: Insufficient data ({len(loc_data)} samples) to'
            ' train GPR. Skipping model.',
            RuntimeWarning,
        )
        self.location_models[loc_id] = None  # Not enough data to train a model

    # --- Tier 2: Global category average (median) fallback for eta_sq ---
    global_eta_medians = {}
    for category, eta_sq_list in global_eta_sq_by_category.items():
      if eta_sq_list:
        global_eta_medians[category] = np.median(eta_sq_list)
      else:
        global_eta_medians[category] = (
            self._EPSILON
        )  # Fallback to epsilon if no global data for category

    # Fill in missing per-location eta_sq values with global medians
    for loc_id in historical_data_with_features['location'].unique():
      if loc_id not in self.location_etas:
        self.location_etas[loc_id] = {}  # Initialize if not already present
      for severity_category in [-1, 0, 1]:
        if self.location_etas[loc_id].get(severity_category) is None:
          # Use global median as fallback
          self.location_etas[loc_id][severity_category] = (
              global_eta_medians.get(severity_category, self._EPSILON)
          )
          warnings.warn(
              f'Loc {loc_id}: Insufficient data to estimate eta_sq for severity'
              f' {severity_category}. Using global median fallback:'
              f' {self.location_etas[loc_id][severity_category]:.4f}.',
              RuntimeWarning,
          )

  def predict_quantiles(
      self,
      test_x,
      historical_data_with_features,
      num_monte_carlo_samples = 5000,  # Increased Monte Carlo samples from 2000 to 5000
  ):
    """Generates quantile predictions for the time points specified in test_x,

    using Monte Carlo sampling conditioned on severity hypotheses with latent
    variable optimization.
    This covers Method Contract Steps 5 and 6.
    """
    quantiles = self.quantiles

    output_df = pd.DataFrame(
        index=test_x.index, columns=[f'quantile_{q}' for q in quantiles]
    )
    output_df[:] = (
        0  # Initialize with zeros to handle skipped forecasts gracefully
    )

    # Ensure test_x date columns are in datetime.date format for feature generation consistency.
    test_x_processed = test_x.copy()
    test_x_processed['target_end_date'] = pd.to_datetime(
        test_x_processed['target_end_date']
    ).dt.date
    test_x_processed['reference_date'] = pd.to_datetime(
        test_x_processed['reference_date']
    ).dt.date

    # Process forecasts location by location, for each unique reference date.
    for (ref_date, loc_id), group in test_x_processed.groupby(
        ['reference_date', 'location']
    ):
      gpr_model = self.location_models.get(loc_id)

      # If no model is available for this location, return zeros for all its forecasts.
      if gpr_model is None:
        output_df.loc[group.index, [f'quantile_{q}' for q in quantiles]] = 0
        continue

      # --- Determine x3_for_forecast_season based on Method Contract Step 2.c logic ---
      x3_for_forecast_season = self.f(
          0.0
      )  # Default transformed value for 0 raw admissions

      loc_historical_data_sorted = historical_data_with_features[
          historical_data_with_features['location'] == loc_id
      ].sort_values('target_end_date')

      current_ref_season_str = self._get_season_str(ref_date)
      # Find the start date of the current reference season from historical data
      current_ref_season_data = loc_historical_data_sorted[
          loc_historical_data_sorted['season'] == current_ref_season_str
      ]
      current_ref_season_start_date = (
          current_ref_season_data['target_end_date'].min()
          if not current_ref_season_data.empty
          else ref_date
      )  # Fallback to ref_date if no data found

      # 1. Try to get x3 from the last week of the *previous* season, strictly before current season start.
      prev_season_data = (
          loc_historical_data_sorted[
              loc_historical_data_sorted['target_end_date']
              < current_ref_season_start_date
          ]
          .sort_values('target_end_date', ascending=False)
          .head(1)
      )

      if not prev_season_data.empty:
        x3_for_forecast_season = prev_season_data['transformed_target'].iloc[0]
      else:
        # 2. If no previous season data, use the first week of the current season (epiweek 40)
        # but only if it's observed data before or at the ref_date.
        first_week_current_season_obs = (
            loc_historical_data_sorted[
                (loc_historical_data_sorted['season'] == current_ref_season_str)
                & (
                    loc_historical_data_sorted['season_week_num'] == 1
                )  # Corresponds to epiweek 40
                & (
                    loc_historical_data_sorted['target_end_date'] <= ref_date
                )  # Only use data up to/before ref_date
            ]
            .sort_values('target_end_date')
            .head(1)
        )

        if not first_week_current_season_obs.empty:
          x3_for_forecast_season = first_week_current_season_obs[
              'transformed_target'
          ].iloc[0]

      # --- Get observed data for current season (up to ref_date - 1 week) to weight severity hypotheses ---
      observed_mask = (
          (loc_historical_data_sorted['location'] == loc_id)
          & (loc_historical_data_sorted['season'] == current_ref_season_str)
          & (loc_historical_data_sorted['target_end_date'] < ref_date)
      )

      y_obs_current_season_transformed = loc_historical_data_sorted.loc[
          observed_mask, 'transformed_target'
      ].values

      # Construct X_obs for current season for x1, x2, x3. x4 will be added during optimization.
      x_obs_current_season_features_df = loc_historical_data_sorted.loc[
          observed_mask, ['x1', 'x2']
      ].copy()
      x_obs_current_season_features_df['x3'] = x3_for_forecast_season

      # --- Severity Hypothesis Weighting & Latent Variable Optimization (Method Contract Steps 5.c.i & 5.c.ii) ---
      log_likelihoods_for_weighting = []
      optimized_x4_latents = []

      # Refined bounds for x4_latent optimization (e.g., -1.5 to 1.5 to stay closer to -1, 0, 1)
      x4_latent_bounds = (-1.5, 1.5)

      # Consider three initial hypotheses for the latent severity variable
      for severity_initial_val in [-1.0, 0.0, 1.0]:  # Method Contract Step 5.a
        if (
            len(y_obs_current_season_transformed) > 0
            and not x_obs_current_season_features_df.empty
        ):
          # Objective function for minimize_scalar: negative log-likelihood
          objective_fn = lambda x4_latent: self._calculate_log_likelihood(
              x4_latent,
              gpr_model,
              x_obs_current_season_features_df,
              y_obs_current_season_transformed,
              loc_id,
          )

          # Optimize the continuous latent severity variable (Method Contract Step 5.c.ii)
          with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore'
            )  # Suppress convergence warnings from optimizer
            res = minimize_scalar(
                objective_fn,
                bounds=x4_latent_bounds,
                method='bounded',
                options={'xatol': 1e-3, 'maxiter': 50},
            )

          # Store optimized x4_latent and its maximum log-likelihood (negated objective function value)
          optimized_x4_latents.append(res.x)
          log_likelihoods_for_weighting.append(-res.fun)
        else:
          # If no observed data for current season, assume initial value is optimal and log-likelihood is 0 (equal weight).
          # This ensures equal initial weights for hypotheses if no data is available to distinguish them.
          optimized_x4_latents.append(severity_initial_val)
          log_likelihoods_for_weighting.append(
              0.0
          )  # Represents neutral evidence for log-likelihoods

      # Calculate weights based on log-likelihoods using logsumexp for numerical stability (Method Contract Step 5.c.i).
      if len(log_likelihoods_for_weighting) > 0 and not np.all(
          np.isneginf(log_likelihoods_for_weighting)
      ):
        weights = np.exp(
            log_likelihoods_for_weighting
            - logsumexp(log_likelihoods_for_weighting)
        )
      else:
        weights = (
            np.ones(3) / 3.0
        )  # Fallback to equal weights if all log-likelihoods are -inf or empty

      # --- Generate Future Features for Test Horizons ---
      future_features_df = group.copy()
      future_features_df['epiweek'] = future_features_df[
          'target_end_date'
      ].apply(self.get_epiweek)
      # Use the centralized _get_season_week_num helper
      future_features_df['x1'] = future_features_df['epiweek'].apply(
          self._get_season_week_num
      )
      future_features_df['x2'] = np.sin(
          2 * np.pi * future_features_df['x1'] / 52
      )
      future_features_df['x3'] = (
          x3_for_forecast_season  # Use the determined season-wide x3 for the forecast season.
      )

      # --- Monte Carlo Sampling and Quantile Calculation (Method Contract Step 5.d & 5.e) ---
      all_samples_transformed = []

      # If all models are unavailable or weights sum to 0, return zeros for this location's forecasts.
      if np.sum(weights) == 0:
        output_df.loc[group.index, [f'quantile_{q}' for q in quantiles]] = 0
        continue

      for i, severity_hypothesis_initial_val in enumerate(
          [-1, 0, 1]
      ):  # Loop through hypotheses
        opt_x4_latent = optimized_x4_latents[i]
        weight = weights[i]

        num_samples_for_s = round(num_monte_carlo_samples * weight)
        # Ensure at least one sample is drawn if the weight is positive, to contribute to diversity
        if (
            num_samples_for_s == 0 and weight > self._EPSILON
        ):  # Use a small epsilon for "positive" weight
          num_samples_for_s = 1

        if num_samples_for_s == 0:  # Skip if no samples to draw
          continue

        # Construct X_test_features with the optimized x4_latent for this hypothesis
        x_test_features_for_hyp = future_features_df[['x1', 'x2', 'x3']].copy()
        x_test_features_for_hyp['x4'] = (
            opt_x4_latent  # Add the optimized continuous latent variable
        )
        x_test_features_for_hyp_values = x_test_features_for_hyp.values

        # Get the category-specific additional noise (eta_category_sq) based on the rounded latent x4.
        # Fallback to a small positive value if category is not in self.location_etas.
        x4_cat_for_eta = int(np.clip(round(opt_x4_latent), -1, 1))
        eta_category_sq = self.location_etas.get(loc_id, {}).get(
            x4_cat_for_eta, self._EPSILON
        )

        try:
          # Predict mean and covariance matrix from the GPR model.
          # cov_pred already includes the GPR's inherent WhiteKernel noise (Λ_base).
          mean_pred, cov_pred = gpr_model.predict(
              x_test_features_for_hyp_values, return_cov=True
          )

          # Add the category-specific additional noise (eta_category_sq) to the diagonal of the covariance matrix.
          # This completes the Λ matrix (Λ_base + Λ_category_specific * I).
          cov_final = cov_pred + np.eye(cov_pred.shape[0]) * eta_category_sq

          # Explicitly check for positive definiteness using Cholesky decomposition.
          # This will raise LinAlgError if cov_final is not positive definite, leading to fallback.
          _ = np.linalg.cholesky(cov_final)
          samples = np.random.multivariate_normal(
              mean_pred, cov_final, size=num_samples_for_s
          )
          all_samples_transformed.append(samples)

        except np.linalg.LinAlgError as e:
          # Fallback to independent sampling if multivariate_normal fails (e.g., non-positive definite covariance)
          warnings.warn(
              f'Multivariate MC sampling failed for loc {loc_id}, latent_x4'
              f' {opt_x4_latent}: {e} (Cholesky failed). Falling back to'
              ' independent sampling.',
              RuntimeWarning,
          )
          mean_pred_single, std_pred_single = gpr_model.predict(
              x_test_features_for_hyp_values, return_std=True
          )

          # Adjust std_pred_single with category-specific additional noise for independent sampling
          effective_std_single = np.sqrt(std_pred_single**2 + eta_category_sq)

          samples = norm.rvs(
              loc=mean_pred_single,
              scale=np.maximum(effective_std_single, self._EPSILON),
              size=(num_samples_for_s, len(mean_pred_single)),
          )
          all_samples_transformed.append(samples)
        except Exception as e:
          warnings.warn(
              f'General Monte Carlo sampling failed for loc {loc_id}, latent_x4'
              f' {opt_x4_latent}: {e}. Skipping samples for this model.',
              RuntimeWarning,
          )

      if not all_samples_transformed:
        # If no samples were generated for any hypothesis, default to zero (already initialized output_df with 0).
        continue

      all_samples_transformed = np.concatenate(all_samples_transformed, axis=0)

      # Step 6: Inverse transform samples and calculate quantiles.
      all_samples_original_scale = self.f_inv(all_samples_transformed)

      # Calculate quantiles for each forecast horizon.
      for j, (_, row) in enumerate(group.iterrows()):
        forecast_samples_for_horizon = all_samples_original_scale[:, j]
        forecast_samples_for_horizon = np.maximum(
            0, forecast_samples_for_horizon
        )  # Ensure non-negative counts.

        # Calculate quantiles from the aggregated Monte Carlo samples.
        quantile_preds = np.percentile(
            forecast_samples_for_horizon, [q * 100 for q in quantiles]
        )
        output_df.loc[row.name, [f'quantile_{q}' for q in quantiles]] = (
            quantile_preds
        )

    # Crucial Constraint: Ensure monotonicity for predicted quantiles.
    # This sorts the quantiles for each forecast to satisfy the problem requirement.
    for idx in output_df.index:
      row_values = output_df.loc[idx].values.astype(float)
      # Ensure row_values are not all NaNs or non-numeric before sorting.
      if pd.isna(row_values).all():
        output_df.loc[idx] = 0.0  # Default to 0 if all NaNs
      else:
        # Fill NaNs with 0 before sorting if some values are present, to prevent NaNs propagating.
        # Then sort to ensure monotonicity.
        output_df.loc[idx] = np.sort(np.nan_to_num(row_values, nan=0.0))

    # Round to nearest integer before casting to int, as per sample submission format.
    # Ensure values are non-negative after rounding.
    return output_df.round().astype(int).clip(lower=0)


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Main function to make predictions for test_x using the required method

  by modelling train_x to train_y. Returns quantile predictions.
  This orchestrates Method Contract Steps 1-6.
  """

  # Initialize output DataFrame with zeros as a fallback
  output_df = pd.DataFrame(
      index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )
  output_df[:] = 0

  # Early exit if no training data
  np_train_y_values = train_y.values
  if train_x.empty or train_y.empty or np.all(np.isnan(np_train_y_values)):
    warnings.warn(
        'Empty or all-NaN training data provided. Returning all zeros.',
        RuntimeWarning,
    )
    return output_df

  # REQUIRED_CDC_LOCATIONS and QUANTILES are assumed to be available from the notebook's global scope.
  forecaster = FluSeasonForecaster(
      locations_df=locations,
      ilinet_df=ilinet_state,  # Using ilinet_state as the relevant historical ILINet data
      required_cdc_locations=REQUIRED_CDC_LOCATIONS,
      quantiles=QUANTILES,
  )

  # Step 1: Data Preparation and Augmentation with historical ILINet data.
  historical_data = forecaster._prepare_historical_data(train_x, train_y)

  # Step 2: Feature Engineering (x1, x2, x3, x4).
  historical_data_with_features = forecaster._engineer_features(historical_data)

  # Check for empty data or insufficient samples after feature engineering
  # for a 4D GPR fitting (needs at least 5 valid samples).
  valid_data_for_gpr_check = historical_data_with_features.dropna(
      subset=['x1', 'x2', 'x3', 'x4', 'transformed_target']
  )
  if valid_data_for_gpr_check.empty or len(valid_data_for_gpr_check) < 5:
    warnings.warn(
        'Insufficient historical data for model training after feature'
        ' engineering. Returning zeros.',
        RuntimeWarning,
    )
    return output_df  # Return the zero-initialized output_df

  # Step 3 & 4: Model Training (Single GPR per location, with x4 as input, and heteroskedasticity).
  forecaster.fit(historical_data_with_features)

  # Check if any GPR models were successfully trained.
  if not any(
      model is not None for model in forecaster.location_models.values()
  ):
    warnings.warn(
        'No GPR models were successfully trained for any location. Returning'
        ' zeros.',
        RuntimeWarning,
    )
    return output_df  # Return the zero-initialized output_df

  # Step 5 & 6: Forecasting New Seasons and Inverse Transformation.
  predictions_df = forecaster.predict_quantiles(
      test_x, historical_data_with_features
  )

  return predictions_df


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
