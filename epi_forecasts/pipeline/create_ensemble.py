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

r"""Creates and saves an ensembled forecast from multiple model submissions.

This script reads forecast submission CSV files for multiple models, combines
them, calculates an ensemble mean for each quantile, enforces monotonicity using
Isotonic Regression, and saves the final ensembled forecast to a new CSV file.

Submissions are expected at
`<model_output_dir>/<model_name>/<forecast_date>-<model_name>.csv`, which is the
layout of the `*_hub/model_output` directories in this repository.

By default, the ensemble members, the model output directory and the ensemble
name are read from `infection_configs.json`.

Example usage for flu (run from the `epi_forecasts` directory):
python pipeline/create_ensemble.py \
    --forecast_date=2026-01-10 \
    --infection=flu

Example usage for covid, with an explicit list of ensemble members:
python pipeline/create_ensemble.py \
    --forecast_date=2026-01-10 \
    --infection=covid \
    --model_names=Google_SAI-Adapted_1,Google_SAI-Hybrid_1,Google_SAI-Novel_1

Example usage for rsv, writing the ensemble to a custom directory:
python pipeline/create_ensemble.py \
    --forecast_date=2026-01-10 \
    --infection=rsv \
    --output_dir=/tmp/rsv_ensemble
"""

from collections.abc import Sequence
import json
import os
from typing import Any

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sklearn import isotonic


IsotonicRegression = isotonic.IsotonicRegression


# Define the quantiles used in the forecasts
QUANTILES = [
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99,
]

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
# Root of the `epi_forecasts` directory. Relative paths in the config are
# resolved against it.
_REPO_DIR = os.path.dirname(_PIPELINE_DIR)

_INFECTION = flags.DEFINE_string(
    'infection',
    None,
    'Infection name to use for the dataset. E.g. "flu", "covid" or "rsv".',
    required=True,
)

# Define command-line flags
_FORECAST_DATE = flags.DEFINE_string(
    'forecast_date',
    None,
    'The date of the forecast to load and ensemble (e.g., "2025-11-22").',
    required=True,
)
_MODEL_NAMES = flags.DEFINE_list(
    'model_names',
    None,
    'A comma-separated list of model names to include in the ensemble. '
    'Defaults to `model_names_for_ensemble` in the infection config.',
    required=False,
)
_CONFIG_PATH = flags.DEFINE_string(
    'config_path',
    os.path.join(_PIPELINE_DIR, 'infection_configs.json'),
    'Path to the shared infection configuration JSON.',
)
_MODEL_OUTPUT_DIR = flags.DEFINE_string(
    'model_output_dir',
    None,
    'Directory containing one sub-directory of submissions per model. '
    'Defaults to `model_output_dir` in the infection config.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Directory where the ensemble CSV is saved. Defaults to '
    '`<model_output_dir>/<ensemble_name>`.',
)


def _load_infection_config(config_path, infection):
  """Loads the configuration of one infection from the shared config JSON."""
  print(f'Loading infection configs from: {config_path}')
  with open(config_path, 'r') as f:
    config_data = json.load(f)
  if infection not in config_data:
    raise ValueError(
        f'Unknown infection "{infection}". Available: {sorted(config_data)}'
    )
  print(f'Successfully loaded config from {config_path}')
  return config_data[infection]


def cdc_submission_to_internal(cdc_submission_df):
  """Converts a CDC-format submission DataFrame to an internal wide format.

  This function pivots the DataFrame from a long format (where each quantile
  is a row) to a wide format where each quantile has its own column.

  Args:
    cdc_submission_df: A pandas DataFrame in the CDC submission format.

  Returns:
    A pandas DataFrame pivoted to a wide format with 'quantile_' columns.
  """
  df_pivot: pd.DataFrame = cdc_submission_df.pivot(
      index=[
          'reference_date',
          'target',
          'horizon',
          'location',
          'target_end_date',
      ],
      columns='output_type_id',
      values='value',
  )
  df_pivot = df_pivot.add_prefix('quantile_')
  df_pivot.columns.name = None
  return df_pivot.reset_index()


def create_ensemble_from_files(
    forecast_date,
    model_names,
    base_path,
):
  """Finds, loads, and ensembles multiple model submissions."""
  all_submissions = []
  print('--- Searching for submission files ---')

  for model_name in model_names:
    # Construct the expected file path
    file_path = f'{base_path}/{model_name}/{forecast_date}-{model_name}.csv'

    if not os.path.exists(file_path):
      print(
          f"[WARNING] Submission not found for model '{model_name}' at:"
          f' {file_path}'
      )
      continue

    try:
      df = pd.read_csv(file_path)
      df = cdc_submission_to_internal(df)
      print(f"Successfully loaded submission for model '{model_name}'")
      all_submissions.append(df)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as e:
      print(f"[ERROR] Failed to read file for model '{model_name}': {e}")

  if not all_submissions:
    print('\nNo valid submission files were found. Cannot create an ensemble.')
    return pd.DataFrame()

  # Concatenate all loaded dataframes into one
  combined_df = pd.concat(all_submissions, ignore_index=True)
  print(f'\nSuccessfully combined {len(all_submissions)} submission files.')

  # --- Ensembling Logic ---
  print('Calculating ensemble predictions...')
  grouping_cols = ['reference_date', 'horizon', 'location', 'target_end_date']
  quantile_cols = [
      col for col in combined_df.columns if col.startswith('quantile_')
  ]

  ensemble_df = (
      combined_df.groupby(grouping_cols)[quantile_cols].mean().reset_index()
  )
  ensemble_df['model'] = 'ensemble'

  print('Ensemble created successfully.')
  return ensemble_df


def ensure_monotonicity(df_group):
  df_group = df_group.sort_values(by='output_type_id')
  y_ = df_group['value'].values
  quantiles_numeric = df_group['output_type_id'].values
  ir = IsotonicRegression(y_min=0, increasing=True, out_of_bounds='clip')
  df_group['value'] = ir.fit_transform(quantiles_numeric, y_)
  return df_group


def save_ensemble(
    df, output_dir, filename, forecast_date
):
  """Saves the ensembled DataFrame as a CSV.

  Args:
      df: The pandas DataFrame to save.
      output_dir: The directory where the file will be saved.
      filename: The base filename. Can contain '{forecast_date}' as a
        placeholder.
      forecast_date: The date string used to format the filename.
  """
  resolved_filename = filename.format(forecast_date=forecast_date)
  file_path = os.path.join(output_dir, resolved_filename)

  # Ensure the directory exists.
  os.makedirs(output_dir, exist_ok=True)
  df.to_csv(file_path, index=False)
  print(f'\nSuccessfully saved ensembled forecast to: {file_path}')


def format_for_cdc(
    submission,
    target_col = 'wk inc flu hosp',
):
  """Formats a forecast DataFrame into the CDC's required format.

  The CDC requires a DataFrame with separate rows for each quantile. the
  following columns:
  - reference_date: The date of the forecast.
  - horizon: The horizon of the forecast.
  - location: The location of the forecast.
  - target_end_date: The date of the target.
  - output_type: The type of the output.
  - output_type_id: The id of the output type.
  - value: The value of the output.

  Args:
      submission: DataFrame with forecast data.
      target_col: The name of the target column.

  Returns:
      DataFrame: CDC-formatted forecast DataFrame.
  """
  id_vars = [
      'reference_date',
      'horizon',
      'location',
      'target_end_date',
  ]
  value_vars = [f'quantile_{q}' for q in QUANTILES]

  cdc_df = pd.melt(
      submission,
      id_vars=id_vars,
      value_vars=value_vars,
      var_name='output_type_id',
      value_name='value',
  )

  cdc_df = cdc_df.rename(columns={'abbreviation': 'location'})
  cdc_df['target'] = target_col
  cdc_df['output_type'] = 'quantile'
  cdc_df['output_type_id'] = cdc_df['output_type_id'].str.replace(
      'quantile_', ''
  )
  cdc_df['output_type_id'] = cdc_df['output_type_id'].astype(float).round(3)
  cdc_df['location'] = cdc_df['location'].astype(str).str.zfill(2)
  cdc_df['value'] = pd.to_numeric(cdc_df['value']).clip(lower=0)

  # Each group is processed explicitly (rather than with `groupby().apply()`)
  # so that the grouping columns are kept in the result with every pandas
  # version: pandas >= 3 drops them from the frames passed to `apply`.
  cdc_monotonic_df = pd.concat(
      [
          ensure_monotonicity(df_group)
          for _, df_group in cdc_df.groupby(
              ['reference_date', 'target', 'horizon', 'target_end_date',
               'location']
          )
      ]
  ).reset_index(drop=True)
  diff_df = pd.merge(
      cdc_df,
      cdc_monotonic_df,
      on=[
          'reference_date',
          'target',
          'horizon',
          'target_end_date',
          'location',
          'output_type',
          'output_type_id',
      ],
      how='inner',
      suffixes=('_cdc', '_monotonic'),
  )
  diff_df['value_diff'] = np.abs(
      diff_df['value_monotonic'] - diff_df['value_cdc']
  )
  print(
      'difference between original forecasts and monotonic version:'
      f' {diff_df.value_diff.sum()}'
  )

  return cdc_monotonic_df[[
      'reference_date',
      'target',
      'horizon',
      'target_end_date',
      'location',
      'output_type',
      'output_type_id',
      'value',
  ]]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  forecast_date = _FORECAST_DATE.value
  infection_config = _load_infection_config(
      _CONFIG_PATH.value, _INFECTION.value
  )
  model_names = _MODEL_NAMES.value or infection_config[
      'model_names_for_ensemble'
  ]
  if not model_names:
    print('Error: No model names provided. Please use the --model_names flag.')
    return

  model_output_dir = _MODEL_OUTPUT_DIR.value or os.path.join(
      _REPO_DIR, infection_config['model_output_dir']
  )
  ensemble_name = infection_config['ensemble_name']
  output_dir = _OUTPUT_DIR.value or os.path.join(
      model_output_dir, ensemble_name
  )

  print(f'--- Creating Ensemble for {forecast_date} ---')
  ensemble_forecast = create_ensemble_from_files(
      forecast_date=forecast_date,
      model_names=model_names,
      base_path=model_output_dir,
  )

  if not ensemble_forecast.empty:
    cdc_formatted_ensemble = format_for_cdc(
        ensemble_forecast,
        infection_config['cdc_metric_name'],
    )
    cdc_formatted_ensemble['location'] = cdc_formatted_ensemble[
        'location'
    ].astype(str)

    print('\n--- CDC Formatted Ensemble Head ---')
    print(cdc_formatted_ensemble.head())
    save_ensemble(
        cdc_formatted_ensemble,
        output_dir,
        f'{forecast_date}-{ensemble_name}.csv',
        forecast_date,
    )
  else:
    print('\nNo ensemble created. Skipping save.')


if __name__ == '__main__':
  app.run(main)
