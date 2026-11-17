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

r"""Processes CDC RSV, Flu, and COVID-19 hospital data and NSSP data.

This binary loads, cleans, interpolates, and merges weekly respiratory data
from CDC sources (NHSN and NSSP) for specified US jurisdictions.

The NHSN input is the "Weekly Hospital Respiratory Data (HRD) Metrics by
Jurisdiction, National Healthcare Safety Network (NHSN) (Preliminary)" CSV
export from https://data.cdc.gov. The NSSP input can be downloaded with
`download_nssp_parquet.sh`. The locations file is the
`auxiliary-data/locations.csv` file of the CDC forecast hub repositories.

Example usage (run from the `epi_forecasts` directory):
python pipeline/data_prep/preprocess_input_rsv.py \
  --nhsn_input_path=data/Weekly_Hospital_Respiratory_Data_Preliminary_20260107.csv \
  --nssp_input_path=data/latest_20260107.parquet \
  --locations_path=data/locations.csv \
  --output_csv_path=data/rsv/dataset_with_nssp_20260107.csv
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import pandas as pd

TARGET_FIPS_CODES = [
    '01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '15',
    '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
    '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53',
    '54', '55', '56', '72'
]

COL_ABBREVIATION = 'abbreviation'
COL_GEO_AGGREGATION = 'Geographic aggregation'
COL_LOCATION_CODE = 'location'
COL_LOCATION_NAME = 'location_name'
COL_POPULATION = 'population'
COL_WEEK_END_DATE = 'Week Ending Date'
COL_TARGET_DATE = 'target_end_date'

METRIC_COVID = 'Total COVID-19 Admissions'
METRIC_FLU = 'Total Influenza Admissions'
METRIC_RSV = 'Total RSV Admissions'

# Flags for input and output paths
_NHSN_INPUT_PATH = flags.DEFINE_string(
    'nhsn_input_path',
    None,
    'Path to the NHSN input CSV file.',
    required=True,
)
_NSSP_INPUT_PATH = flags.DEFINE_string(
    'nssp_input_path',
    None,
    'Path to the NSSP input Parquet file.',
    required=True,
)
_LOCATIONS_PATH = flags.DEFINE_string(
    'locations_path',
    None,
    'Path to the locations metadata CSV file.',
    required=True,
)
_OUTPUT_CSV_PATH = flags.DEFINE_string(
    'output_csv_path',
    None,
    'Path to save the combined processed CSV output.',
    required=True,
)


def clean_numeric_column(df, column_name):
  """Cleans a specific column by removing commas and converting to numeric."""
  df = df.copy()
  df[column_name] = (
      df[column_name].astype(str).str.replace(',', '', regex=False)
  )
  df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
  return df


def interpolate_missing_weeks(
    df,
    target_col,
    date_col = COL_WEEK_END_DATE,
    group_col = COL_GEO_AGGREGATION
):
  """Fills gaps in the weekly data using linear interpolation."""
  df[date_col] = pd.to_datetime(df[date_col])

  min_date = df[date_col].min()
  max_date = df[date_col].max()
  all_locations = df[group_col].unique()

  all_dates = pd.date_range(start=min_date, end=max_date, freq='W-SAT')
  index_combinations = pd.MultiIndex.from_product(
      [all_dates, all_locations],
      names=[date_col, group_col]
  )

  full_grid_df = pd.DataFrame(index=index_combinations).reset_index()
  df_merged = pd.merge(
      full_grid_df,
      df,
      on=[group_col, date_col],
      how='left'
  )
  df_merged = df_merged.sort_values(by=[group_col, date_col])
  df_merged[target_col] = df_merged.groupby(group_col)[target_col].transform(
      lambda x: x.interpolate(method='linear', limit_direction='both')
  )
  return df_merged


def load_nssp(path, locations_df):
  """Loads and processes NSSP data, merging with locations."""
  nssp = pd.read_parquet(path)

  nssp = nssp[nssp['county'] == 'All']

  processed_df = pd.merge(
      nssp,
      locations_df,
      left_on='geography',
      right_on=COL_LOCATION_NAME,
      how='inner',
  )
  processed_df[COL_TARGET_DATE] = processed_df['week_end']
  processed_df.drop(columns=['county', 'week_end'], inplace=True)

  missing_locations = set(locations_df[COL_LOCATION_NAME].unique()) - set(
      processed_df['geography'].unique()
  )
  if missing_locations:
    print(
        f"These locations don't have NSSP data available: {missing_locations}"
    )
  return processed_df


def load_locations_metadata(path):
  """Loads location metadata and filters for specific FIPS codes."""
  locations = pd.read_csv(path)
  locations = locations[locations[COL_LOCATION_CODE].isin(TARGET_FIPS_CODES)]
  return locations


def process_nhsn(
    input_path,
    locations_df,
    metric_name
):
  """Loads, cleans, interpolates, and merges NHSN data for a specific metric."""
  raw_df = pd.read_csv(input_path, low_memory=False)

  df = raw_df[[COL_WEEK_END_DATE, COL_GEO_AGGREGATION, metric_name]].copy()
  df = clean_numeric_column(df, metric_name)
  df = interpolate_missing_weeks(df, target_col=metric_name)

  if df[metric_name].isnull().sum() > 0:
    print(f'Warning: Null values remain in {metric_name} after interpolation.')

  df = df.rename(columns={COL_WEEK_END_DATE: COL_TARGET_DATE})

  processed_df = pd.merge(
      df,
      locations_df,
      left_on=COL_GEO_AGGREGATION,
      right_on=COL_ABBREVIATION,
      how='inner'
  )

  final_df = processed_df[[
      COL_TARGET_DATE,
      COL_LOCATION_CODE,
      COL_LOCATION_NAME,
      COL_POPULATION,
      metric_name,
  ]].copy()

  final_df[metric_name] = final_df[metric_name].astype(int)
  final_df[COL_TARGET_DATE] = pd.to_datetime(final_df[COL_TARGET_DATE]).dt.date

  return final_df.sort_values(
      by=[COL_LOCATION_CODE, COL_TARGET_DATE], ascending=[True, False]
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print(f'Loading locations from: {_LOCATIONS_PATH.value}')
  locations_df = load_locations_metadata(_LOCATIONS_PATH.value)

  print('Processing NHSN Data...')
  df_covid = process_nhsn(_NHSN_INPUT_PATH.value, locations_df, METRIC_COVID)
  df_flu = process_nhsn(_NHSN_INPUT_PATH.value, locations_df, METRIC_FLU)
  df_rsv = process_nhsn(_NHSN_INPUT_PATH.value, locations_df, METRIC_RSV)

  print('Processing NSSP Data...')
  df_nssp = load_nssp(_NSSP_INPUT_PATH.value, locations_df)

  print('\nMerging datasets...')
  merge_keys = [
      COL_TARGET_DATE,
      COL_LOCATION_CODE,
      COL_LOCATION_NAME,
      COL_POPULATION,
  ]

  df_combined = pd.merge(df_rsv, df_flu, on=merge_keys, how='inner')
  df_combined = pd.merge(df_combined, df_covid, on=merge_keys, how='inner')

  # NSSP uses 'geography' which was mapped to COL_LOCATION_NAME in load_nssp
  df_combined = pd.merge(df_combined, df_nssp, on=merge_keys, how='outer')

  print('Combined Dataset Info:')
  df_combined.info()
  print(f'Combined shape: {df_combined.shape}')

  output_path = _OUTPUT_CSV_PATH.value
  print(f'\nSaving combined dataset to {output_path}...')
  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  df_combined.to_csv(output_path, index=False)
  print(f'Successfully saved to {output_path}')


if __name__ == '__main__':
  app.run(main)
