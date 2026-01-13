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
MODEL_NAME = 'Google_SAI-Adapted_6'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import warnings
from collections import deque


# Global dataframes `locations`, `ilinet_state`, `dataset` are available.


# Helper class for simple scaling when linear regression mapping is not possible
class SimpleScalingMapper:

  def __init__(self, scaling_factor=1.0, intercept=0.0):
    self.scaling_factor = scaling_factor
    self.intercept = intercept

  def predict(self, X):
    return (X * self.scaling_factor) + self.intercept


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # IMPLEMENTATION PLAN.
  # ## Core Principles Checklist:
  # 1. AR(6) model with exogenous regressors for holiday effects, applied after a fourth root data transform.
  #    - My code will apply a (value + 1)**0.25 transformation to the target variable before modeling, create an exogenous binary Christmas covariate, and include 6 lags of the transformed target as autoregressive terms in a linear model.
  # 2. AR and exogenous regressor coefficients are shared across all locations.
  #    - My code will fit a single Ordinary Least Squares (OLS) model using combined data from all locations (augmented with synthetic historical data), thus pooling the coefficients for AR terms and the Christmas covariate. Additionally, location-specific intercepts will be incorporated as fixed effects.
  # 3. A separate variance parameter is estimated for each location.
  #    - After fitting the pooled OLS model, my code will calculate residuals, group them by location, and estimate a unique error variance (sigma^2) for each location from these grouped residuals.

  # ## Step-by-Step Logic:
  # 1. Initialize constants: `AR_LAG_MAX` (6), `TRANSFORM_OFFSET` (1), `ILINET_OFFSET` (1), `MIN_MAPPER_SLOPE` (for robust scaling), `MAX_PROPORTIONAL_SCALING_FACTOR`, `min_transformed_val_for_zero`, and `ILINET_GLOBAL_MAX_DATE` (2022-10-15).
  # 2. Derive `train_end_date` from `train_x`.
  # 3. Prepare `admissions_base_history` from the global `dataset`, converting `target_end_date` to `datetime` and filtering by `REQUIRED_CDC_LOCATIONS`. Ensure explicit copying after filtering to avoid SettingWithCopyWarning.
  # 4. Prepare `ili_base_history` from the global `ilinet_state`, converting `week_start` to `datetime`, renaming `region` to `location_name`, merging with `locations` to get FIPS codes and population, and filtering by `REQUIRED_CDC_LOCATIONS`. Calculate `scaled_ili_count` (population-scaled `unweighted_ili`). Crucially, drop rows with NaN values in `scaled_ili_count` from `ili_base_history` at this stage. Filter `ili_base_history` to `target_end_date <= ILINET_GLOBAL_MAX_DATE`. Ensure explicit copying after filtering/merging/cleaning.
  # 5. Learn the ILINet to Admissions transformation once using fixed historical overlap (with robust fallbacks):
  #    a. Define `fixed_mapper_training_start_date` as the earliest `target_end_date` from admissions data.
  #    b. Define `fixed_mapper_training_end_date` as `ILINET_GLOBAL_MAX_DATE`. This is the crucial change to ensure a stable, maximal mapping period.
  #    c. Filter `admissions_for_mapper_training` and `ili_for_mapper_training` within this dynamically defined window.
  #    d. Perform an **inner merge** to create `overlap_df_for_mapper`. Drop `NaN` values from relevant columns (`TARGET_STR`, `scaled_ili_count`).
  #    e. Calculate `transformed_admissions = (overlap_df_for_mapper[TARGET_STR] + TRANSFORM_OFFSET)**0.25` and `transformed_ili = (overlap_df_for_mapper['scaled_ili_count'] + ILINET_OFFSET)**0.25`.
  #    f. Initialize robust fallback parameters (`mapper_scaling_factor_fallback`, `mapper_intercept_fallback`) ensuring they are always finite.
  #    g. **Robust Mapper Determination (IMPROVED LOGIC):**
  #       i. If `overlap_df_for_mapper` is empty, or `transformed_admissions` is all NaN, or `transformed_admissions.nunique() < 2`, or `transformed_ili.nunique() < 2`:
  #          Issue a warning indicating insufficient data for robust mapper learning. Default `SimpleScalingMapper` is used with a 0.0 slope and the robust `mapper_intercept_fallback_base`.
  #       ii. Else (sufficient variance in both for LR):
  #           1. Attempt to fit a `LinearRegression` model (`temp_mapper`).
  #           2. If successful and `temp_mapper.coef_[0]` is non-negative and `np.isfinite(temp_mapper.intercept_)`:
  #              Set `ili_to_admissions_mapper` to `temp_mapper`.
  #           3. Else (LR failed or yielded a negative slope or non-finite intercept):
  #              Issue a warning about Linear Regression failure, falling back to a `SimpleScalingMapper` using a **proportional scaling factor** derived from the means of transformed admissions and ILI, capped to a reasonable range. If means are problematic, then fallback to `0.0` slope and `mapper_intercept_fallback_base`.
  #    h. **Final Mapper Initialization:** `ili_to_admissions_mapper` is guaranteed to be set, either by LR or a `SimpleScalingMapper` with robust parameters.
  # 6. Prepare `current_admissions_df`: Combine `train_x` and `train_y`, convert `target_end_date` to `datetime`, and filter by `REQUIRED_CDC_LOCATIONS`. Ensure explicit copying.
  # 7. Filter `ili_base_history` to `ili_history_for_fold` to include only data where `target_end_date` is less than or equal to `train_end_date`. This ensures no future ILI data is used for augmentation in the current fold.
  # 8. Combine `current_admissions_df` with `ili_history_for_fold` for augmentation using an **outer merge**, sort. Ensure explicit copying.
  # 9. Generate synthetic `Total Influenza Admissions` data:
  #    a. Identify rows where `Total Influenza Admissions` is null but `scaled_ili_count` is not null.
  #    b. Apply the robust `ili_to_admissions_mapper` to transformed `scaled_ili_count` in these rows.
  #    c. Invert the transformation and fill the `Total Influenza Admissions` column, ensuring non-negativity.
  # 10. Finalize the `train_df` for the AR model: **Copy `full_merged_history_for_augmentation` to `train_df`**. Drop rows with `NaN` in `Total Influenza Admissions` (ensure explicit copying), sort (ensure explicit copying), add `week_of_year` and `is_christmas_week` features to `train_df` and `test_x_processed`. Apply `transformed_target = (train_df[TARGET_STR] + TRANSFORM_OFFSET)**0.25` to `train_df`.
  # 11. Generate lagged features: For each location, create `AR_LAG_MAX` lagged columns of `transformed_target`. Drop rows with `NaN` values resulting from lag creation (ensure explicit copying).
  # 12. Fit the pooled linear regression model: Define features (`X_train`) and target (`y_train_transformed`). **IMPROVEMENT: Add one-hot encoded location fixed effects to `X_train`**. Fit `sklearn.linear_model.LinearRegression`.
  # 13. Estimate location-specific error variances: Calculate residuals, group by `location`, and compute the sample variance (`ddof=1`), handling single-point cases and replacing zero or NaN variances with a small positive number.
  # 14. Pre-calculate AR forecast error variance multipliers: Extract AR coefficients, calculate `psi_weights`. Define `max_h_needed`. Calculate `cumulative_psi_squared_sum` based on AR theory (sum of squared psi_weights up to h-1).
  # 15. Generate multi-step ahead forecasts for `test_x`:
  #     a. Calculate `global_avg_transformed_target = train_df_model['transformed_target'].mean()`.
  #     b. Initialize storage for `forecast_results` and `current_lag_values_per_location`.
  #     c. Iterate through unique locations and their forecast horizons in `test_x`.
  #     d. For each forecast step: retrieve latest lags from `train_df_model` (the dataframe used for fitting) for robust initialization, predict mean transformed forecast, clamp, retrieve location-specific standard deviation, scale it by horizon-dependent multiplier, calculate quantiles using `norm.ppf`, clamp transformed quantiles, update `current_lag_values`, store results.
  #        i. When initializing lags, if `loc_train_model_data_for_lags` is too short, pad with `loc_train_model_data_for_lags['transformed_target'].mean()`, or `global_avg_transformed_target` if that's NaN, or `min_transformed_val_for_zero` if both are NaN. (Streamlined to default to `min_transformed_val_for_zero` if no valid average).
  #        ii. **IMPROVEMENT: Ensure the correct location-specific fixed effect is included in the feature vector for prediction.**
  # 16. Invert the data transformations to return the quantile forecasts to the original scale of hospitalization counts. This involves raising to the power of 4 and reversing the centering and scaling. Ensure no negative predictions and monotonicity. Perform clipping on *transformed* values before raising to power 4 to prevent intermediate floating-point overflow to `inf`. Then, sort and clip again after inversion, before converting to integer.
  # 17. Format the quantile forecasts into the standard CDC FluSight Hub submission file.

  AR_LAG_MAX = 6
  TRANSFORM_OFFSET = 1  # Used for (y + offset)**0.25 transformation for Total Influenza Admissions
  ILINET_OFFSET = 1  # Using 1 for population-scaled ILI data for consistency and handling zeros.
  MIN_MAPPER_SLOPE = (
      1e-3  # Minimum slope to ensure some positive relationship if LR fails
  )
  MAX_PROPORTIONAL_SCALING_FACTOR = (
      5.0  # Max factor for transformed values to prevent extreme scaling
  )
  min_transformed_val_for_zero = (
      0 + TRANSFORM_OFFSET
  ) ** 0.25  # Transformed value corresponding to 0 original admissions
  ILINET_GLOBAL_MAX_DATE = pd.to_datetime(
      '2022-10-15'
  )  # Global cutoff for ILINet data availability

  # Derive train_end_date from the actual input train_x
  train_end_date = pd.to_datetime(train_x['target_end_date']).max()

  # REQUIRED_CDC_LOCATIONS is derived from the globally filtered `locations` DataFrame.
  REQUIRED_CDC_LOCATIONS = locations['location'].unique()

  # 3. Prepare full admissions history from the global 'dataset' variable
  # `dataset` is a global variable. Explicitly copy for safety and consistency.
  admissions_base_history = dataset.copy()
  admissions_base_history['target_end_date'] = pd.to_datetime(
      admissions_base_history['target_end_date']
  )
  admissions_base_history = admissions_base_history[
      admissions_base_history['location'].isin(REQUIRED_CDC_LOCATIONS)
  ].copy()

  # 4. Prepare full ILI history from the global 'ilinet_state' variable
  # `ilinet_state` is a global variable. Explicitly copy.
  ili_base_history = ilinet_state.copy()
  ili_base_history['target_end_date'] = pd.to_datetime(
      ili_base_history['week_start']
  )
  ili_base_history = ili_base_history.rename(
      columns={'region': 'location_name'}
  )

  # Merge with global `locations` DataFrame to get FIPS codes AND population
  ili_base_history = pd.merge(
      ili_base_history,
      locations[['location', 'location_name', 'population']],
      on='location_name',
      how='inner',
  ).copy()  # Explicitly copy after merge
  ili_base_history = ili_base_history[
      ili_base_history['location'].isin(REQUIRED_CDC_LOCATIONS)
  ].copy()

  # --- CRITICAL IMPROVEMENT: Global filter for ILINet data availability ---
  ili_base_history = ili_base_history[
      ili_base_history['target_end_date'] <= ILINET_GLOBAL_MAX_DATE
  ].copy()

  # Scale unweighted_ili by population to make it count-like, rename for clarity
  ili_base_history.loc[:, 'scaled_ili_count'] = (
      ili_base_history['unweighted_ili'] / 100
  ) * ili_base_history['population']

  # IMPROVEMENT: Drop NaNs from `scaled_ili_count` from `ili_base_history` *early*
  # This ensures that only valid ILI data points are considered for mapper training
  ili_base_history = ili_base_history.dropna(subset=['scaled_ili_count']).copy()
  # Select relevant columns after dropping NaNs
  ili_base_history = ili_base_history[
      ['target_end_date', 'location', 'scaled_ili_count']
  ].copy()

  # 5. Learn the ILINet to Admissions transformation ONCE using a fixed historical overlap:
  # This change ensures the mapper is trained on a consistent and maximal historical overlap.
  fixed_mapper_training_start_date = admissions_base_history[
      'target_end_date'
  ].min()
  fixed_mapper_training_end_date = ILINET_GLOBAL_MAX_DATE

  transformed_zero_ili = (0 + ILINET_OFFSET) ** 0.25
  ili_to_admissions_mapper = (
      None  # Will be set to LinearRegression or SimpleScalingMapper
  )

  # Filter admissions data for mapper training using the fixed overlap window
  admissions_for_mapper_training = admissions_base_history[
      (
          admissions_base_history['target_end_date']
          >= fixed_mapper_training_start_date
      )
      & (
          admissions_base_history['target_end_date']
          <= fixed_mapper_training_end_date
      )
  ].copy()

  # Filter ILI history for mapper training using the fixed overlap window
  ili_for_mapper_training = ili_base_history[
      (ili_base_history['target_end_date'] >= fixed_mapper_training_start_date)
      & (ili_base_history['target_end_date'] <= fixed_mapper_training_end_date)
  ].copy()

  # Perform an inner merge to find the overlap for mapper training
  overlap_df_for_mapper = pd.merge(
      admissions_for_mapper_training[
          ['target_end_date', 'location', TARGET_STR]
      ],
      ili_for_mapper_training[
          ['target_end_date', 'location', 'scaled_ili_count']
      ],
      on=['target_end_date', 'location'],
      how='inner',
  ).copy()
  # Drop NaNs from the target (admissions) column (scaled_ili_count already cleaned in step 4)
  overlap_df_for_mapper = overlap_df_for_mapper.dropna(
      subset=[TARGET_STR]
  ).copy()

  # Calculate transformed values for overlap_df_for_mapper for easier use
  if not overlap_df_for_mapper.empty:
    overlap_df_for_mapper.loc[:, 'transformed_admissions'] = (
        overlap_df_for_mapper[TARGET_STR] + TRANSFORM_OFFSET
    ) ** 0.25
    overlap_df_for_mapper.loc[:, 'transformed_ili'] = (
        overlap_df_for_mapper['scaled_ili_count'] + ILINET_OFFSET
    ) ** 0.25

    transformed_admissions = overlap_df_for_mapper['transformed_admissions']
    transformed_ili = overlap_df_for_mapper['transformed_ili']
  else:
    transformed_admissions = pd.Series(
        dtype=float
    )  # Empty series if no overlap data
    transformed_ili = pd.Series(dtype=float)

  # Initialize robust fallback parameters, always ensuring they are finite and reasonable
  # If transformed_admissions has valid non-NaN values, use its mean for a better intercept fallback
  mapper_intercept_fallback_base = min_transformed_val_for_zero
  if transformed_admissions.notnull().any():
    mean_admissions_val_for_fallback = transformed_admissions.mean()
    if pd.notna(mean_admissions_val_for_fallback):
      mapper_intercept_fallback_base = mean_admissions_val_for_fallback
  # mapper_scaling_factor_fallback = MIN_MAPPER_SLOPE # Not directly used if LR fails, but kept as a general min
  # mapper_intercept_fallback = mapper_intercept_fallback_base - mapper_scaling_factor_fallback * transformed_zero_ili # Not directly used if LR fails

  # Robust Mapper Determination (IMPROVED LOGIC):
  # Consolidate checks for insufficient or constant data that prevent a meaningful Linear Regression
  if (
      overlap_df_for_mapper.empty
      or transformed_admissions.isnull().all()
      or transformed_admissions.nunique() < 2
      or transformed_ili.nunique() < 2
  ):

    warnings.warn(
        'Insufficient data or constant values in overlap for mapper training'
        f' (fixed window ending {fixed_mapper_training_end_date}). Using'
        f' constant mapping: slope {0.0:.4f}, intercept'
        f' {mapper_intercept_fallback_base:.4f}.'
    )
    ili_to_admissions_mapper = SimpleScalingMapper(
        scaling_factor=0.0,  # If target constant or ILI constant, ILI provides no explanatory power
        intercept=mapper_intercept_fallback_base,  # Use the best available mean as intercept
    )
  else:  # Both transformed_admissions and transformed_ili have sufficient variance for Linear Regression
    try:
      temp_mapper = LinearRegression(fit_intercept=True)
      temp_mapper.fit(transformed_ili.to_frame(), transformed_admissions)

      # Validate the fitted LinearRegression: ensure non-negative slope and finite intercept
      if temp_mapper.coef_[0] >= 0 and np.isfinite(temp_mapper.intercept_):
        ili_to_admissions_mapper = temp_mapper
      else:
        warnings.warn(
            'LinearRegression yielded a non-robust slope'
            f' ({temp_mapper.coef_[0]:.4f}) or intercept'
            f' ({temp_mapper.intercept_:.4f}) for mapper training (fixed window'
            f' ending {fixed_mapper_training_end_date}). Falling back to'
            ' proportional scaling.'
        )
        # Fallback to proportional scaling if LR fails validation
        mean_transformed_admissions_overlap = transformed_admissions.mean()
        mean_transformed_ili_overlap = transformed_ili.mean()

        final_scaling_factor = 0.0  # Default if means are problematic
        final_intercept = (
            mapper_intercept_fallback_base  # Default to robust intercept
        )

        if (
            pd.notna(mean_transformed_admissions_overlap)
            and pd.notna(mean_transformed_ili_overlap)
            and mean_transformed_ili_overlap > 0
        ):
          ratio_scaling_factor = (
              mean_transformed_admissions_overlap / mean_transformed_ili_overlap
          )
          final_scaling_factor = max(
              MIN_MAPPER_SLOPE,
              min(ratio_scaling_factor, MAX_PROPORTIONAL_SCALING_FACTOR),
          )
          final_intercept = (
              mean_transformed_admissions_overlap
              - final_scaling_factor * mean_transformed_ili_overlap
          )

          if not np.isfinite(final_intercept):  # Defensive check
            final_intercept = mapper_intercept_fallback_base
        else:  # Means are problematic for ratio, fallback to constant admissions with robust intercept
          warnings.warn(
              f'Means for proportional scaling were problematic. Falling back'
              f' to constant admissions for intercept.'
          )
          # final_scaling_factor remains 0.0, final_intercept remains mapper_intercept_fallback_base

        ili_to_admissions_mapper = SimpleScalingMapper(
            scaling_factor=final_scaling_factor, intercept=final_intercept
        )

    except ValueError as e:  # Catch fit errors like singular matrix
      warnings.warn(
          'LinearRegression fit failed due to data issues for mapper training'
          f' (fixed window ending {fixed_mapper_training_end_date}): {e}.'
          ' Falling back to proportional scaling.'
      )
      mean_transformed_admissions_overlap = transformed_admissions.mean()
      mean_transformed_ili_overlap = transformed_ili.mean()

      final_scaling_factor = 0.0
      final_intercept = mapper_intercept_fallback_base

      if (
          pd.notna(mean_transformed_admissions_overlap)
          and pd.notna(mean_transformed_ili_overlap)
          and mean_transformed_ili_overlap > 0
      ):
        ratio_scaling_factor = (
            mean_transformed_admissions_overlap / mean_transformed_ili_overlap
        )
        final_scaling_factor = max(
            MIN_MAPPER_SLOPE,
            min(ratio_scaling_factor, MAX_PROPORTIONAL_SCALING_FACTOR),
        )
        final_intercept = (
            mean_transformed_admissions_overlap
            - final_scaling_factor * mean_transformed_ili_overlap
        )
        if not np.isfinite(final_intercept):
          final_intercept = mapper_intercept_fallback_base
      else:
        warnings.warn(
            f'Means for proportional scaling were problematic. Falling back to'
            f' constant admissions for intercept.'
        )

      ili_to_admissions_mapper = SimpleScalingMapper(
          scaling_factor=final_scaling_factor, intercept=final_intercept
      )

  # 6. Prepare the `current_admissions_df` from `train_x` and `train_y`:
  current_admissions_df = train_x.copy()  # Keep .copy() for function input
  current_admissions_df[TARGET_STR] = train_y
  current_admissions_df['target_end_date'] = pd.to_datetime(
      current_admissions_df['target_end_date']
  )
  current_admissions_df = current_admissions_df[
      current_admissions_df['location'].isin(REQUIRED_CDC_LOCATIONS)
  ].copy()

  # 7. Filter `ili_base_history` to `ili_history_for_fold`
  # This is for augmenting the current fold's training data.
  # It correctly uses ILI data up to the current fold's train_end_date, respecting the global ILINet max date.
  ili_history_for_fold = ili_base_history[
      ili_base_history['target_end_date'] <= train_end_date
  ].copy()

  # 8. Combine `current_admissions_df` with `ili_history_for_fold` for augmentation:
  full_merged_history_for_augmentation = pd.merge(
      current_admissions_df[['target_end_date', 'location', TARGET_STR]],
      ili_history_for_fold,
      on=['target_end_date', 'location'],
      how='outer',
  ).copy()
  full_merged_history_for_augmentation = (
      full_merged_history_for_augmentation.sort_values(
          by=['location', 'target_end_date']
      )
      .reset_index(drop=True)
      .copy()
  )

  # 9. Generate synthetic `Total Influenza Admissions` data:
  # Identify rows where `Total Influenza Admissions` is null but `scaled_ili_count` is not null.
  synthetic_data_mask = (
      full_merged_history_for_augmentation[TARGET_STR].isnull()
      & full_merged_history_for_augmentation['scaled_ili_count'].notnull()
  )

  if synthetic_data_mask.any():
    synthetic_rows = full_merged_history_for_augmentation.loc[
        synthetic_data_mask
    ].copy()

    synthetic_rows.loc[:, 'transformed_ili'] = (
        synthetic_rows['scaled_ili_count'] + ILINET_OFFSET
    ) ** 0.25

    # Apply the learned mapper
    synthetic_transformed_admissions = ili_to_admissions_mapper.predict(
        synthetic_rows[['transformed_ili']]
    )
    # Invert transformation and ensure non-negative integers
    synthetic_admissions = np.maximum(
        0, (synthetic_transformed_admissions**4) - TRANSFORM_OFFSET
    ).astype(int)

    full_merged_history_for_augmentation.loc[
        synthetic_data_mask, TARGET_STR
    ] = synthetic_admissions

  # 10. Finalize the `train_df` for the AR model:
  # Copy `full_merged_history_for_augmentation` to `train_df` for explicit independence.
  train_df = full_merged_history_for_augmentation.copy()

  # Drop rows where Total Influenza Admissions is still null (e.g., no ILI data either)
  train_df = train_df.dropna(subset=[TARGET_STR]).copy()

  if train_df.empty:
    warnings.warn(
        'All training data (actual + synthetic) was dropped or filtered.'
        ' Returning empty predictions.'
    )
    output_df = pd.DataFrame(
        index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    return output_df.fillna(0).astype(int)

  train_df = train_df.sort_values(by=['location', 'target_end_date']).copy()

  # Using isocalendar().week for week_of_year for epiweeks
  train_df.loc[:, 'week_of_year'] = (
      train_df['target_end_date'].dt.isocalendar().week.astype(int)
  )

  # Ensure test_x dates are datetime for consistency
  test_x_processed = test_x.copy()  # Keep .copy() for function input
  test_x_processed.loc[:, 'target_end_date'] = pd.to_datetime(
      test_x_processed['target_end_date']
  )
  test_x_processed.loc[:, 'week_of_year'] = (
      test_x_processed['target_end_date'].dt.isocalendar().week.astype(int)
  )

  # Create the Christmas holiday covariate:
  # Weeks 51, 52, 53, 1, 2 are considered 'Christmas' period.
  christmas_weeks = [51, 52, 53, 1, 2]
  train_df.loc[:, 'is_christmas_week'] = (
      train_df['week_of_year'].isin(christmas_weeks).astype(int)
  )
  test_x_processed.loc[:, 'is_christmas_week'] = (
      test_x_processed['week_of_year'].isin(christmas_weeks).astype(int)
  )

  # Transform the target variable (on the augmented data):
  train_df.loc[:, 'transformed_target'] = (
      train_df[TARGET_STR] + TRANSFORM_OFFSET
  ) ** 0.25

  # 11. Generate lagged features for the transformed target:
  lag_cols = [f'lag_{i}' for i in range(1, AR_LAG_MAX + 1)]
  for lag in range(1, AR_LAG_MAX + 1):
    train_df.loc[:, f'lag_{lag}'] = train_df.groupby('location')[
        'transformed_target'
    ].shift(lag)

  # Drop rows with NaN values resulting from lag creation (these are early time points for each location)
  train_df_model = train_df.dropna(
      subset=['transformed_target'] + lag_cols
  ).copy()

  if train_df_model.empty:
    warnings.warn(
        'Training data is too short (even with augmentation) to form AR lags.'
        ' Returning empty predictions.'
    )
    output_df = pd.DataFrame(
        index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    return output_df.fillna(0).astype(int)

  # 12. Fit the pooled linear regression model:
  base_features = lag_cols + ['is_christmas_week']

  # IMPROVEMENT: Add location-specific fixed effects
  # Create one-hot encoded location dummies for training data
  # Include all locations present in both train_df_model and test_x_processed to ensure consistent dummy columns
  all_locations_encountered = pd.Series(
      train_df_model['location'].unique().tolist()
      + test_x_processed['location'].unique().tolist()
  ).unique()
  all_locations_encountered.sort()  # Ensure consistent ordering of dummy columns

  # Create dummy variables for the training data
  location_dummies_train = pd.get_dummies(
      train_df_model['location'], prefix='loc', drop_first=True
  ).reindex(
      columns=[
          f'loc_{loc}'
          for loc in all_locations_encountered
          if loc != all_locations_encountered[0]
      ],
      fill_value=0,
  )  # Ensure all possible dummy cols are there, filled with 0

  # Align the index of location_dummies_train with train_df_model
  location_dummies_train = location_dummies_train.reindex(
      train_df_model.index, fill_value=0
  )

  # Combine all features for training
  X_train = pd.concat(
      [train_df_model[base_features], location_dummies_train], axis=1
  )

  # Store the final list of features including dummies for prediction
  final_features_for_model = X_train.columns.tolist()

  y_train_transformed = train_df_model['transformed_target']

  model = LinearRegression(
      fit_intercept=True
  )  # Retain intercept, drop_first=True handles collinearity
  model.fit(X_train, y_train_transformed)

  # 13. Estimate location-specific error variances:
  train_df_model.loc[:, 'predicted_transformed'] = model.predict(X_train)
  train_df_model.loc[:, 'residuals'] = (
      y_train_transformed - train_df_model['predicted_transformed']
  )

  # Calculate sample variance (ddof=1). If len(x) <= 1, np.var(ddof=1) returns NaN.
  location_variances = train_df_model.groupby('location')['residuals'].apply(
      lambda x: np.var(x, ddof=1)
  )

  # Replace 0 or NaN variance with a small number to avoid division by zero or overly narrow intervals
  location_variances = location_variances.replace(0, 1e-6).fillna(1e-6)

  # 14. Pre-calculate AR forecast error variance multipliers:
  phis = model.coef_[:AR_LAG_MAX]  # AR coefficients from the pooled model

  # max_h_needed is max steps ahead from train_end_date for horizon 3 (+2 for steps adjustment)
  max_h_needed = test_x_processed['horizon'].max() + 2

  psi_weights = [1.0]  # psi_0 = 1

  # cumulative_psi_squared_sum[k] will store sum_{j=0}^{k} psi_j^2.
  # For an h-step forecast, we need sum_{j=0}^{h-1} psi_j^2, which will be cumulative_psi_squared_sum[h-1].
  cumulative_psi_squared_sum = [
      psi_weights[0] ** 2
  ]  # Stores psi_0^2 at index 0
  current_psi_sq_sum_accumulator = psi_weights[0] ** 2

  # Calculate psi_h and cumulative squared psi_weights for h-step forecasts
  for h_step_idx in range(
      1, max_h_needed + 1
  ):  # h_step_idx represents the index for psi (e.g., psi_1, psi_2...)
    current_psi = 0.0
    for k in range(1, AR_LAG_MAX + 1):
      if h_step_idx - k >= 0:
        # phis are 0-indexed (phi_1 is phis[0]), psi_weights are 0-indexed (psi_0 is psi_weights[0])
        current_psi += phis[k - 1] * psi_weights[h_step_idx - k]
    psi_weights.append(current_psi)

    current_psi_sq_sum_accumulator += current_psi**2
    cumulative_psi_squared_sum.append(current_psi_sq_sum_accumulator)

  # 15. Generate multi-step ahead forecasts for `test_x`:
  forecast_results = []

  current_lag_values_per_location = {}

  test_x_sorted = test_x_processed.sort_values(
      by=['location', 'target_end_date']
  ).copy()
  unique_test_locations = test_x_sorted['location'].unique()

  # Calculate global average transformed target for padding if location-specific average is not available.
  global_avg_transformed_target = train_df_model['transformed_target'].mean()
  if pd.isna(global_avg_transformed_target):
    # If train_df_model itself is empty or only NaNs, global_avg_transformed_target will be NaN.
    # Fallback to the transformed value for 0 admissions.
    global_avg_transformed_target = min_transformed_val_for_zero

  for loc_id in unique_test_locations:
    # Get the latest AR_LAG_MAX transformed target values from the `train_df_model` (which has valid lags)
    loc_train_model_data_for_lags = train_df_model[
        (train_df_model['location'] == loc_id)
    ].sort_values('target_end_date')

    # We need the last AR_LAG_MAX *actual* transformed target values to seed the forecast.
    last_actual_transformed_values = deque(
        loc_train_model_data_for_lags['transformed_target']
        .tail(AR_LAG_MAX)
        .tolist()
    )

    # Determine padding value for short series (streamlined logic):
    padding_value = min_transformed_val_for_zero  # Default fallback if no valid averages exist
    loc_train_avg_transformed = loc_train_model_data_for_lags[
        'transformed_target'
    ].mean()

    if pd.notna(loc_train_avg_transformed):
      padding_value = loc_train_avg_transformed
    elif pd.notna(global_avg_transformed_target):
      padding_value = global_avg_transformed_target
    # Else, padding_value remains min_transformed_val_for_zero

    # Pad with the determined padding_value if insufficient history in train_df_model
    if len(last_actual_transformed_values) < AR_LAG_MAX:
      warnings.warn(
          f'Location {loc_id} has less than {AR_LAG_MAX} valid historical'
          ' points in `train_df_model` to form initial AR lags. Padding with'
          f' {padding_value:.2f}.'
      )
      while len(last_actual_transformed_values) < AR_LAG_MAX:
        last_actual_transformed_values.appendleft(padding_value)

    current_lag_values_per_location[loc_id] = last_actual_transformed_values

    loc_test_data = test_x_sorted[
        test_x_sorted['location'] == loc_id
    ].sort_values('target_end_date')

    # Prepare dummy variables for the current location for prediction
    # Get all unique locations encountered in train/test to ensure consistent dummy columns

    # Create a full dummy vector initialized to 0, matching the columns from model training
    current_loc_fixed_effects_vector_dict = {
        col: 0 for col in final_features_for_model if col.startswith('loc_')
    }

    # If loc_id is not the one dropped (i.e., not the first in all_locations_encountered), set its dummy to 1
    if loc_id != all_locations_encountered[0]:
      dummy_col_name = f'loc_{loc_id}'
      if dummy_col_name in current_loc_fixed_effects_vector_dict:
        current_loc_fixed_effects_vector_dict[dummy_col_name] = 1

    current_loc_fixed_effects_vector = list(
        current_loc_fixed_effects_vector_dict.values()
    )

    for _, row in loc_test_data.iterrows():
      # Apply Christmas effect coefficient
      # The coefficient for 'is_christmas_week' is at index AR_LAG_MAX
      christmas_effect_val = model.coef_[AR_LAG_MAX] * row['is_christmas_week']

      # Construct feature vector for prediction
      # Lags are 1 to AR_LAG_MAX, so we need values in reverse chronological order
      lag_features = list(current_lag_values_per_location[loc_id])[::-1]

      # Combine lag features, Christmas effect, and location fixed effects
      feature_values_for_prediction = (
          lag_features
          + [christmas_effect_val]
          + current_loc_fixed_effects_vector
      )

      # Create a DataFrame for prediction, ensuring columns match `final_features_for_model`
      X_forecast_data = pd.DataFrame(
          [feature_values_for_prediction], columns=final_features_for_model
      )

      # Predict mean transformed value
      mean_transformed_forecast = model.predict(X_forecast_data)[0]

      # Ensure transformed forecast is at least the value corresponding to 0 original admissions
      mean_transformed_forecast = np.maximum(
          min_transformed_val_for_zero, mean_transformed_forecast
      )

      # Get location-specific standard deviation
      # Default to a small sigma if location not in training, though should be rare with augmentation
      base_sigma_l = np.sqrt(location_variances.get(loc_id, 1e-6))

      # Calculate horizon-dependent scaling to the standard deviation using AR theory
      # steps_after_train_end = 1 for horizon -1, 2 for horizon 0, 3 for horizon 1, etc.
      steps_after_train_end = row['horizon'] + 2

      # CRITICAL CORRECTION: Indexing for cumulative_psi_squared_sum
      # cumulative_psi_squared_sum[k] stores sum_{j=0}^{k} psi_j^2.
      # For an h-step forecast (where h = steps_after_train_end), we need sum_{j=0}^{h-1} psi_j^2.
      # This corresponds to index h-1.
      scale_factor_var = cumulative_psi_squared_sum[steps_after_train_end - 1]

      sigma_l_scaled = base_sigma_l * np.sqrt(scale_factor_var)

      # Calculate quantiles on transformed scale
      quantiles_transformed = norm.ppf(
          QUANTILES, loc=mean_transformed_forecast, scale=sigma_l_scaled
      )

      # Ensure transformed quantiles are at least the value corresponding to 0 original admissions
      quantiles_transformed = np.maximum(
          min_transformed_val_for_zero, quantiles_transformed
      )

      # Update lag values for next prediction step for this location
      # Use the clamped mean_transformed_forecast for lags
      current_lag_values_per_location[loc_id].append(mean_transformed_forecast)
      current_lag_values_per_location[loc_id].popleft()  # Remove the oldest lag

      # Store results for this row
      row_forecast = pd.Series(
          quantiles_transformed,
          index=[f'quantile_{q}' for q in QUANTILES],
          name=row.name,
      )
      forecast_results.append(row_forecast)

  # Combine all forecasts
  forecast_quantiles_transformed_df = pd.DataFrame(
      forecast_results, index=[r.name for r in forecast_results]
  )

  # 16. Invert the transformation for quantiles and ensure monotonicity:
  final_quantiles_df = pd.DataFrame(
      index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  # Calculate the maximum transformed value that, when inverted, will fit into int64 without overflow.
  # We add TRANSFORM_OFFSET to np.iinfo(np.int64).max because we subtract it after **4
  max_transformed_val_for_int64 = (
      np.iinfo(np.int64).max + TRANSFORM_OFFSET
  ) ** 0.25

  # First, clip the *transformed* quantiles before inversion (power of 4)
  # This is crucial to prevent intermediate float overflow to 'inf' before inversion.
  clipped_transformed_quantiles_df = pd.DataFrame(
      index=test_x.index,
      columns=[f'quantile_{q}' for q in QUANTILES],
      dtype=float,
  )
  for q_col in final_quantiles_df.columns:
    clipped_transformed = np.clip(
        forecast_quantiles_transformed_df[q_col],
        min_transformed_val_for_zero,
        max_transformed_val_for_int64,
    )
    clipped_transformed_quantiles_df[q_col] = clipped_transformed

  # Now, invert the (already clipped) transformed values
  inverted_quantiles_float_df = (
      clipped_transformed_quantiles_df**4
  ) - TRANSFORM_OFFSET

  # Then, for each row, sort the float values to ensure monotonicity
  # And finally, convert to non-negative integer using rounding, clipping *again* for final safety.
  for idx in final_quantiles_df.index:
    sorted_float_quantiles = np.sort(
        inverted_quantiles_float_df.loc[idx].values
    )

    # Final clip to ensure non-negativity and stay within int64 range after sorting, then round and cast.
    # np.round can produce inf for very large floats, so clip should precede it for absolute safety.
    # The clip before rounding is redundant given the earlier clip, but safe.
    final_quantiles_df.loc[idx] = np.round(
        np.clip(sorted_float_quantiles, 0, np.iinfo(np.int64).max)
    ).astype(int)

  return final_quantiles_df


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
