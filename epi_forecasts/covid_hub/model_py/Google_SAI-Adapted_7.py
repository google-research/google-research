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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,anomalous-backslash-in-string,reimported,g-multiple-import,g-doc-return-or-yield,g-doc-args,broad-exception-caught

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
MODEL_NAME = 'Google_SAI-Adapted_7'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles of the method:
# 1. Feature Engineering & Target Transformation: Prepare input features (scaled time, seasonal components) and apply log1p transformation to the population-normalized target variable (`Total COVID-19 Admissions per 100k`) to model count data appropriately.
# 2. Kernel Specification & Hyperparameter Optimization: Define a composite Gaussian Process covariance kernel (e.g., ConstantKernel * Matern + WhiteKernel) to model trends, seasonality, and noise, and optimize its hyperparameters (e.g., length-scales, amplitude, noise-level) by maximizing the marginal likelihood.
# 3. Posterior Prediction: Conditioning the Gaussian Process on the transformed training data to derive the posterior predictive distribution, then drawing samples from this distribution for the forecast horizon.
# 4. Inverse Transformation & Quantile Derivation: Applying inverse transformations to return samples to the original count scale (ensuring non-negativity), and then computing the required quantiles, enforcing monotonicity.

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

# from scipy.stats import norm # Removed as not directly used in the improved fallback logic.
import warnings
from sklearn.preprocessing import StandardScaler

# TARGET_STR and QUANTILES are assumed to be available in the global scope
# as defined in the preamble of the notebook.

# MODIFICATION: Moved warnings filter to the top of the cell for efficiency and consistency with the provided cell.
# This ensures they are configured only once when the cell is executed.
warnings.filterwarnings(
    'ignore',
    message=(
        'The optimal value found for dimension \d of the parameter length_scale'
        ' is close to the specified lower bound'
    ),
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message=(
        'The optimal value found for dimension \d of the parameter length_scale'
        ' is close to the specified upper bound'
    ),
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message=(
        'A not a number (NaN) or infinity initial parameter was passed to the'
        ' optimizer, this typically occurs when the kernel is not returning a'
        ' finite value and may indicate a numerical problem.'
    ),
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message=(
        'The number of training samples \(.*\) should be greater than 1 to'
        ' estimate the standard deviation properly. Results might be'
        ' inaccurate.'
    ),
    category=UserWarning,
)


# MODIFICATION: Moved _add_time_features outside the main function for better code organization.
def _add_time_features(
    df, min_ref_date
):
  """Adds time-based features to a DataFrame for GP modeling."""
  df_processed = df.copy()
  df_processed['time_feature'] = (
      df_processed['target_end_date'] - min_ref_date
  ).dt.days
  df_processed['week_of_year'] = (
      df_processed['target_end_date'].dt.isocalendar().week.astype(int)
  )
  # 52.1775 is the average number of weeks in a year (365.25 / 7)
  df_processed['sin_week'] = np.sin(
      2 * np.pi * df_processed['week_of_year'] / 52.1775
  )
  df_processed['cos_week'] = np.cos(
      2 * np.pi * df_processed['week_of_year'] / 52.1775
  )
  return df_processed


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Return quantiles.

  Do not do any cross-validation in here.
  """
  # Ensure target_end_date is datetime type for all operations.
  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])

  # Determine the minimum date across the entire training set to normalize time features.
  # This ensures 'time_feature' is consistently anchored for all locations within a given fold.
  min_overall_date = train_x['target_end_date'].min()

  # Pre-process features for GP input.
  # MODIFICATION: Using the _add_time_features helper function defined globally in this cell.
  train_x_processed = _add_time_features(train_x, min_overall_date)
  test_x_processed = _add_time_features(test_x, min_overall_date)

  # Combine train_x and train_y for easier per-location filtering.
  train_df = train_x_processed.copy()
  train_df[TARGET_STR] = train_y

  all_location_predictions = []

  # Define features to be used in the GP (already outside the loop, good practice).
  GP_FEATURES = ['time_feature', 'sin_week', 'cos_week']

  # Initialize and fit StandardScaler ONCE on the entire training data.
  # This ensures consistent scaling across all locations.
  scaler = StandardScaler()
  # MODIFICATION: Added more robust initialization for scaler in case of empty or missing feature data.
  if not train_df.empty and all(
      feature in train_df.columns for feature in GP_FEATURES
  ):
    scaler.fit(train_df[GP_FEATURES])
  else:
    warnings.warn(
        'Training data is empty or missing GP features. StandardScaler will be'
        ' initialized without fitting.',
        UserWarning,
    )
    # Create a scaler that does nothing if it can't be fitted
    scaler.mean_ = np.zeros(len(GP_FEATURES))
    scaler.scale_ = np.ones(len(GP_FEATURES))

  # Number of samples to draw from the posterior predictive distribution
  N_SAMPLES = 1000

  # Define the Gaussian Process Kernel ONCE outside the location loop for efficiency.
  # MODIFICATION: Adjusted initial length_scale for 'time_feature' to be closer to 1 year for better trend capture.
  base_kernel = ConstantKernel(1.0, (1e-3, 1e4)) * Matern(
      length_scale=[
          365.25,
          1.0,
          1.0,
      ],  # MODIFIED: Initial value for time_feature length_scale set to ~1 year
      length_scale_bounds=[
          (1.0, 365.25 * 3),
          (0.1, 2.0),
          (0.1, 2.0),
      ],  # Adjusted bounds for time_feature
      nu=1.5,
  ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e2))

  # Iterate over each unique location in the test set to train and predict models.
  for location_code in test_x_processed['location'].unique():
    train_loc_df = train_df[train_df['location'] == location_code]
    test_loc_df = test_x_processed[
        test_x_processed['location'] == location_code
    ]

    # Raw target values for current location's training data
    y_train_loc = train_loc_df[TARGET_STR].values

    # Ensure population values are robust (not zero or NaN) before scaling.
    population_train_for_scaling = (
        train_loc_df['population'].replace(0, 1).fillna(1).values
    )
    population_test_reshaped = (
        test_loc_df['population'].replace(0, 1).fillna(1).values.reshape(-1, 1)
    )

    # Calculate population-normalized and log1p-transformed target for GP (and fallback)
    # Ensure target values are non-negative before log transformation.
    y_train_loc_scaled = (
        np.maximum(0, y_train_loc) / population_train_for_scaling * 100000
    )
    y_train_loc_transformed = np.log1p(y_train_loc_scaled)

    # MODIFICATION: Implement robust fallback logic based on training data availability.
    # Case 1: No positive training data for the location, predict all zeros.
    if not np.any(y_train_loc > 0):
      print(
          f'Warning: No positive training data for location {location_code}.'
          ' Predicting all zeros.'
      )
      quantile_predictions = np.zeros(
          (len(test_loc_df), len(QUANTILES)), dtype=int
      )
    else:
      try:
        # Prepare input features for the GP.
        X_train_loc_raw = train_loc_df[GP_FEATURES].copy()
        X_test_loc_raw = test_loc_df[GP_FEATURES].copy()

        # Apply the GLOBALLY FITTED scaler to transform per-location data.
        X_train_loc = scaler.transform(X_train_loc_raw)
        X_test_loc = scaler.transform(X_test_loc_raw)

        # Initialize and fit Gaussian Process Regressor with the pre-defined kernel.
        gp = GaussianProcessRegressor(
            kernel=base_kernel,
            alpha=1e-4,
            n_restarts_optimizer=10,
            random_state=42,
            normalize_y=True,
        )

        # Fit the GP model to the transformed and scaled training data.
        gp.fit(X_train_loc, y_train_loc_transformed)

        # Draw N_SAMPLES function samples from the posterior predictive distribution.
        y_samples_log = gp.sample_y(
            X_test_loc, n_samples=N_SAMPLES, random_state=42
        )

        # Inverse transform each sample: expm1, de-normalize by population, round, clamp non-negative.
        y_samples_scaled = np.expm1(y_samples_log)
        y_samples_count = (
            (y_samples_scaled / 100000 * population_test_reshaped)
            .round()
            .astype(int)
        )
        y_samples_count = np.maximum(0, y_samples_count)

        # Compute quantiles from the ensemble of transformed samples.
        quantile_predictions = np.percentile(
            y_samples_count, [q * 100 for q in QUANTILES], axis=1
        ).T

        # Enforce monotonicity for quantiles in each row.
        for i in range(quantile_predictions.shape[0]):
          quantile_predictions[i, :] = np.maximum.accumulate(
              quantile_predictions[i, :]
          )

      except Exception as e:
        # Case 2: GP model failed (e.g., optimization error), but positive data exists.
        # Use transformed space probabilistic fallback, consistent with GP.
        print(
            f'Warning: GP model failed for location {location_code} with error:'
            f' {e}. Using transformed space probabilistic fallback.'
        )

        # Calculate mean and std from *transformed* training data for fallback.
        # Ensure a minimum std for probabilistic spread, even if training data has little variance.
        fallback_mean_transformed = np.mean(y_train_loc_transformed)
        fallback_std_transformed = (
            np.std(y_train_loc_transformed)
            if len(y_train_loc_transformed) > 1
            else 0.5
        )
        fallback_std_transformed = max(
            0.1, fallback_std_transformed
        )  # Enforce minimum spread in log-space

        # Generate samples from a normal distribution in the transformed space.
        samples_transformed = np.random.normal(
            fallback_mean_transformed,
            fallback_std_transformed,
            size=(len(test_loc_df), N_SAMPLES),
        )

        # Inverse transform these samples consistently with the GP path.
        samples_scaled = np.expm1(samples_transformed)
        y_samples_count = (
            (samples_scaled / 100000 * population_test_reshaped)
            .round()
            .astype(int)
        )
        y_samples_count = np.maximum(
            0, y_samples_count
        )  # Ensure non-negativity

        # Compute quantiles from these samples.
        quantile_predictions = np.percentile(
            y_samples_count, [q * 100 for q in QUANTILES], axis=1
        ).T

        # Ensure non-negativity and apply monotonicity for this row.
        for i in range(quantile_predictions.shape[0]):
          quantile_predictions[i, :] = np.maximum.accumulate(
              np.maximum(0, quantile_predictions[i, :])
          )

    # Create a DataFrame for the current location's predictions.
    loc_predictions_df = pd.DataFrame(
        quantile_predictions,
        index=test_loc_df.index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )
    all_location_predictions.append(loc_predictions_df)

  # Concatenate all individual location predictions into a single DataFrame.
  final_predictions_df = pd.concat(all_location_predictions)

  # Reindex to match the exact order of the original test_x DataFrame.
  final_predictions_df = final_predictions_df.reindex(test_x.index)

  # Final type and value enforcement.
  for col in final_predictions_df.columns:
    final_predictions_df[col] = final_predictions_df[col].astype(int)
    final_predictions_df[col] = np.maximum(
        0, final_predictions_df[col]
    )  # Ensure no negative predictions

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
