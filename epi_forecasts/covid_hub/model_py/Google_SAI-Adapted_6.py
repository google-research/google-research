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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-short-docstring-punctuation

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
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Describe the core principles of the required method:
# 1. Pooled AR(6) Model with Population-Normalized Fourth-Root Transformation: Apply a variance-stabilizing fourth-root transformation to the target variable after normalizing by population (setting normalized target to 0 for zero-population locations). Then, estimate global AR(6) coefficients (including an intercept, linear time trend, and day-of-year seasonal components) by pooling data from all locations using regularized Ordinary Least Squares (Ridge regression).
# 2. Local Error Standard Deviation: Calculate residuals for each specific location using the global coefficients, then compute location-specific standard deviations of these residuals, applying an absolute floor and falling back to a robust global standard deviation if local data is insufficient or too small.
# 3. Recursive Monte Carlo Forecasting: Generate multiple future trajectories by iteratively applying the global AR model equation forward in time, adding Gaussian noise sampled from the location-specific error distributions, and enforcing non-negativity at each transformed step.
# 4. Inverse Transformation & Quantile Aggregation: Apply the inverse transformation (raise to the power of 4), denormalize by multiplying back the population factor (predicting 0 for zero-population locations), enforce non-negativity on raw counts, and compute the required quantiles for each forecast point, ensuring monotonicity *before* final rounding.


# Constants for the model (TARGET_STR and QUANTILES are assumed globally available)
AR_ORDER = 6
N_SIMULATIONS = 1000
POPULATION_NORMALIZATION_FACTOR = 100_000
RANDOM_SEED = 42  # For reproducibility of simulations
DAYS_IN_YEAR = 366  # For seasonal features (accounts for leap years)
RIDGE_ALPHA = 1.0  # Regularization strength for Ridge regression
DEFAULT_GLOBAL_STD_INITIAL_FALLBACK = 0.1  # Fallback std if no data or std is zero after transform, based on initial target series
MIN_NOISE_STD_ABSOLUTE_FLOOR = (
    1e-6  # A very small absolute floor for noise std (to prevent zero noise)
)


# Helper function to apply the standard deviation floors
def _apply_std_floors(std_val):
  """Applies an absolute floor to a standard deviation value."""
  if pd.isna(std_val):
    return DEFAULT_GLOBAL_STD_INITIAL_FALLBACK
  return max(MIN_NOISE_STD_ABSOLUTE_FLOOR, std_val)


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using a pooled AR(6) model with global coefficients

  and local variance, incorporating linear trend and seasonality.

  Args:
      train_x: DataFrame containing historical feature data.
      train_y: Series containing historical target values ('Total COVID-19
        Admissions').
      test_x: DataFrame containing feature data for prediction horizons.

  Returns:
      pd.DataFrame: A DataFrame with quantile predictions for each row in
      test_x.
  """
  np.random.seed(RANDOM_SEED)  # Set seed for reproducible simulations

  # 0. Initial setup and data consolidation
  # Safely combine train_x and train_y, assuming their indices align.
  full_train_df = pd.concat([train_x, train_y], axis=1)

  # Ensure target values are non-negative before any transformations
  full_train_df[TARGET_STR] = np.maximum(0, full_train_df[TARGET_STR])

  # Ensure target_end_date column is datetime object for robustness.
  full_train_df['target_end_date'] = pd.to_datetime(
      full_train_df['target_end_date']
  )

  # 0.1. Robustness: Drop rows where the target is NaN
  full_train_df.dropna(subset=[TARGET_STR], inplace=True)

  # If no valid training data remains after dropping NaNs,
  # the model cannot be trained or provide meaningful forecasts.
  if full_train_df.empty:
    test_y_hat_quantiles = pd.DataFrame(index=test_x.index)
    for q in QUANTILES:
      test_y_hat_quantiles[f'quantile_{q}'] = 0
    return test_y_hat_quantiles

  full_train_df = full_train_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # Calculate a global time index for the training data, ensuring min_train_date is a Timestamp
  min_train_date = pd.to_datetime(full_train_df['target_end_date'].min())
  full_train_df['time_idx'] = (
      full_train_df['target_end_date'] - min_train_date
  ).dt.days // 7

  # 1. Population-normalized Fourth-root Transformation
  full_train_df['population_safe'] = full_train_df['population'].replace(0, 1)

  # Explicitly handle zero-population locations: normalized target should be 0.
  full_train_df['normalized_target'] = np.where(
      full_train_df['population'] == 0,
      0,
      (full_train_df[TARGET_STR] / full_train_df['population_safe'])
      * POPULATION_NORMALIZATION_FACTOR,
  )
  full_train_df['transformed_normalized_target'] = np.power(
      full_train_df['normalized_target'], 1 / 4
  )

  # Add seasonal features for training data using dayofyear for smoother periodicity
  full_train_df['day_of_year'] = full_train_df['target_end_date'].dt.dayofyear
  full_train_df['sin_dayofyear'] = np.sin(
      2 * np.pi * full_train_df['day_of_year'] / DAYS_IN_YEAR
  )
  full_train_df['cos_dayofyear'] = np.cos(
      2 * np.pi * full_train_df['day_of_year'] / DAYS_IN_YEAR
  )

  # Robustness: Calculate initial global mean and std for transformed target as fallbacks
  base_transformed_mean = full_train_df['transformed_normalized_target'].mean()
  if pd.isna(base_transformed_mean):
    base_transformed_mean = (
        0.0  # Default to 0 if all transformed targets are NaN
    )

  # 2. Create lagged features for AR(6)
  for i in range(1, AR_ORDER + 1):
    full_train_df[f'lag_{i}'] = full_train_df.groupby('location')[
        'transformed_normalized_target'
    ].shift(i)

  # Define feature columns programmatically
  lag_feature_names = [f'lag_{i}' for i in range(1, AR_ORDER + 1)]
  trend_feature_names = ['time_idx']
  seasonal_feature_names = ['sin_dayofyear', 'cos_dayofyear']
  feature_cols = (
      lag_feature_names + trend_feature_names + seasonal_feature_names
  )

  # Drop rows with NaN in lags (first AR_ORDER entries for each location) or features/target
  model_train_df = full_train_df.dropna(
      subset=feature_cols + ['transformed_normalized_target']
  ).copy()

  # Dynamically calculate MIN_SAMPLES_FOR_OLS based on the actual number of features
  min_samples_for_ols_dynamic = len(feature_cols) + 1  # +1 for intercept

  # 3. Global Estimation: Ridge Regression for AR coefficients, intercept, linear time trend, and seasonal components
  # Initialize coefficients for fallback scenario (when model cannot be trained)
  ar_coeffs = np.zeros(AR_ORDER)
  trend_coeff = 0.0
  sin_dayofyear_coeff = 0.0
  cos_dayofyear_coeff = 0.0
  ar_int = base_transformed_mean  # Default intercept to global mean if no model is trained

  global_ar_model = None  # Default to no model trained
  location_error_std = pd.Series(
      dtype=float
  )  # Default empty, will trigger global fallback std

  residuals_for_global_std_calc = pd.Series(dtype=float)

  if len(model_train_df) >= min_samples_for_ols_dynamic:
    X_train_features = model_train_df[feature_cols]
    y_train_transformed_normalized = model_train_df[
        'transformed_normalized_target'
    ]

    global_ar_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    global_ar_model.fit(X_train_features, y_train_transformed_normalized)

    # Extract coefficients from trained model based on feature_cols order
    coeffs_dict = dict(zip(X_train_features.columns, global_ar_model.coef_))
    ar_coeffs = np.array(
        [coeffs_dict.get(f'lag_{i}', 0.0) for i in range(1, AR_ORDER + 1)]
    )
    trend_coeff = coeffs_dict.get('time_idx', 0.0)
    sin_dayofyear_coeff = coeffs_dict.get('sin_dayofyear', 0.0)
    cos_dayofyear_coeff = coeffs_dict.get('cos_dayofyear', 0.0)
    ar_int = global_ar_model.intercept_

    # 4. Local Error Variance Estimation
    model_train_df['predicted_transformed_normalized'] = (
        global_ar_model.predict(X_train_features)
    )
    model_train_df['residuals'] = (
        model_train_df['transformed_normalized_target']
        - model_train_df['predicted_transformed_normalized']
    )
    residuals_for_global_std_calc = model_train_df[
        'residuals'
    ]  # Use residuals from trained model
    location_error_std = model_train_df.groupby('location')['residuals'].std()
  else:
    # Model not trained, fallback coefficients are already initialized.
    # Calculate residuals using the global mean (ar_int) as the "prediction" for global std.
    if not full_train_df.empty:
      residuals_for_global_std_calc = (
          full_train_df['transformed_normalized_target'] - ar_int
      )
    # location_error_std remains empty (as initialized), ensuring all locations use global_std_for_noise_fallback

  # Determine the global standard deviation for noise, applying floors
  global_std_for_noise = _apply_std_floors(
      residuals_for_global_std_calc.std()
      if not residuals_for_global_std_calc.empty
      else DEFAULT_GLOBAL_STD_INITIAL_FALLBACK
  )

  # Combine all coefficients into one array for unified prediction calculation
  # Order must match feature_cols: lags, time_idx, sin_dayofyear, cos_dayofyear
  all_coeffs = np.concatenate(
      (ar_coeffs, [trend_coeff], [sin_dayofyear_coeff], [cos_dayofyear_coeff])
  )

  # 5. Recursive Forecasting
  test_y_hat_quantiles = pd.DataFrame(index=test_x.index)
  all_quantiles_output = np.empty((len(test_x), len(QUANTILES)), dtype=int)

  unique_locations_in_test = test_x['location'].unique()
  max_forecast_horizon = test_x['horizon'].max()

  test_x_ref_date = pd.to_datetime(
      test_x['reference_date'].iloc[0]
  )  # Ensure datetime type
  current_time_idx_for_forecast_start = (
      test_x_ref_date - min_train_date
  ).days // 7

  # Prepare a series of final location standard deviations (applying floors)
  final_location_stds = pd.Series(
      global_std_for_noise, index=unique_locations_in_test
  )
  if not location_error_std.empty:
    # Update with local stds where available, filling any missing with the global fallback
    location_error_std_aligned = location_error_std.reindex(
        unique_locations_in_test
    ).fillna(global_std_for_noise)
    final_location_stds.update(location_error_std_aligned)

  # Apply the standard deviation floors to all final location standard deviations
  final_location_stds = final_location_stds.apply(_apply_std_floors)

  location_initial_lags = {}
  for loc in unique_locations_in_test:
    loc_history = full_train_df[full_train_df['location'] == loc].sort_values(
        'target_end_date', ascending=False
    )

    available_lags = (
        loc_history['transformed_normalized_target'].head(AR_ORDER).values
    )

    # Create a full array filled with 'ar_int' if actual lags are missing for consistency with model intercept
    padded_lags = np.full(AR_ORDER, ar_int, dtype=float)
    # Overwrite the beginning with available lags (if any)
    padded_lags[: len(available_lags)] = available_lags
    location_initial_lags[loc] = padded_lags

  all_simulated_paths_by_location = {}

  for loc in unique_locations_in_test:
    # Retrieve the pre-calculated and floored standard deviation for this location
    loc_std_dev = final_location_stds.loc[loc]

    current_lags_for_simulations = np.tile(
        location_initial_lags[loc], (N_SIMULATIONS, 1)
    )

    simulated_values_for_location_all_horizons = np.zeros(
        (N_SIMULATIONS, max_forecast_horizon + 1)
    )

    for h in range(max_forecast_horizon + 1):
      current_forecast_time_idx = current_time_idx_for_forecast_start + h
      current_forecast_date = test_x_ref_date + pd.Timedelta(weeks=h)
      current_day_of_year = current_forecast_date.dayofyear
      current_sin_dayofyear = np.sin(
          2 * np.pi * current_day_of_year / DAYS_IN_YEAR
      )
      current_cos_dayofyear = np.cos(
          2 * np.pi * current_day_of_year / DAYS_IN_YEAR
      )

      # Construct the full feature matrix for N_SIMULATIONS paths at this horizon
      # Order must match feature_cols: lags, time_idx, sin_dayofyear, cos_dayofyear
      X_pred_features_combined = np.hstack((
          current_lags_for_simulations,  # AR lags (N_SIMULATIONS, AR_ORDER)
          np.full(
              (N_SIMULATIONS, 1), current_forecast_time_idx
          ),  # Time index (N_SIMULATIONS, 1)
          np.full(
              (N_SIMULATIONS, 1), current_sin_dayofyear
          ),  # Sin seasonal (N_SIMULATIONS, 1)
          np.full(
              (N_SIMULATIONS, 1), current_cos_dayofyear
          ),  # Cos seasonal (N_SIMULATIONS, 1)
      ))

      # Calculate predicted value using the combined coefficients and intercept
      predicted_next_transformed = ar_int + np.dot(
          X_pred_features_combined, all_coeffs
      )

      noise = np.random.normal(0, loc_std_dev, N_SIMULATIONS)
      next_transformed_with_noise = predicted_next_transformed + noise

      # Enforce non-negativity in the transformed space
      next_transformed_with_noise = np.maximum(0, next_transformed_with_noise)

      simulated_values_for_location_all_horizons[:, h] = (
          next_transformed_with_noise
      )

      # Update lags for the next step in simulation
      current_lags_for_simulations = np.hstack((
          next_transformed_with_noise[:, np.newaxis],
          current_lags_for_simulations[:, :-1],
      ))

    all_simulated_paths_by_location[loc] = (
        simulated_values_for_location_all_horizons
    )

  # 6. Inverse Transformation, Denormalization, and Quantile Aggregation
  for row_idx_numeric, (row_idx_original, row_data) in enumerate(
      test_x.iterrows()
  ):
    loc = row_data['location']
    horizon = row_data['horizon']
    loc_population = row_data['population']

    # Explicitly handle zero-population locations: predicted admissions should be 0.
    if loc_population == 0:
      simulated_final_values = np.zeros(N_SIMULATIONS, dtype=int)
    else:
      simulated_transformed_normalized_values = all_simulated_paths_by_location[
          loc
      ][:, horizon]

      # Apply inverse transformation, ensuring non-negativity
      simulated_normalized_values = np.maximum(
          0, np.power(simulated_transformed_normalized_values, 4)
      )

      # Denormalize to get raw admissions
      simulated_final_values = (
          simulated_normalized_values * loc_population
      ) / POPULATION_NORMALIZATION_FACTOR

      # Ensure final admissions are non-negative integers
      simulated_final_values = np.round(
          np.maximum(0, simulated_final_values)
      ).astype(int)

    # Calculate quantiles (as floats first)
    calculated_quantiles = np.quantile(simulated_final_values, QUANTILES)

    # Enforce monotonicity on floating-point quantiles, then round to integer
    calculated_quantiles = np.maximum.accumulate(calculated_quantiles)
    calculated_quantiles_int = np.round(calculated_quantiles).astype(int)

    all_quantiles_output[row_idx_numeric, :] = calculated_quantiles_int

  test_y_hat_quantiles[[f'quantile_{q}' for q in QUANTILES]] = (
      all_quantiles_output
  )

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
