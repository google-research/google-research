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

r"""Preprocesses the NHSN input data for the flu and covid datasets.

The NHSN input is the "Weekly Hospital Respiratory Data (HRD) Metrics by
Jurisdiction, National Healthcare Safety Network (NHSN) (Preliminary)" CSV
export from https://data.cdc.gov. The locations file is the
`auxiliary-data/locations.csv` file of the CDC forecast hub repositories.

The output is written to `<output_path>/dataset_<suffix>.csv`, where `<suffix>`
is the part of the NHSN file name after the last underscore (typically the
download date, e.g. `20250924`).

Example usage for flu (run from the `epi_forecasts` directory):
python pipeline/data_prep/preprocess_input.py \
    --infection=flu \
    --nhsn_input=data/Weekly_Hospital_Respiratory_Data_Preliminary_20250924.csv \
    --locations_path=data/locations.csv \
    --output_path=data/flu/

Example usage for covid:
python pipeline/data_prep/preprocess_input.py \
    --infection=covid \
    --nhsn_input=data/Weekly_Hospital_Respiratory_Data_Preliminary_20250924.csv \
    --locations_path=data/locations.csv \
    --output_path=data/covid/
"""

from collections.abc import Sequence
import json
import os
from typing import Any

from absl import app
from absl import flags
import pandas as pd

_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_INFECTION = flags.DEFINE_string(
    'infection',
    None,
    'Infection name to use for the dataset. E.g. "flu" or "covid".',
    required=True,
)

_NHSN_INPUT = flags.DEFINE_string(
    'nhsn_input',
    None,
    'Path to the NHSN input csv file.',
    required=True,
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Path to the directory where the output csv file is written.',
    required=True,
)
_LOCATIONS_PATH = flags.DEFINE_string(
    'locations_path',
    None,
    'Path to the locations csv file.',
    required=True,
)
_CONFIG_PATH = flags.DEFINE_string(
    'config_path',
    os.path.join(_PIPELINE_DIR, 'infection_configs.json'),
    'Path to the shared infection configuration JSON.',
)
REQUIRED_CDC_LOCATIONS = [
    '01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '15',
    '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
    '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53',
    '54', '55', '56', '72'
]
ABBREVIATION_COL = 'abbreviation'
GEO_AGGREGATION_COL = 'Geographic aggregation'
LOCATION_CODE_COL = 'location'
LOCATION_NAME_COL = 'location_name'
POPULATION_COL = 'population'
WEEK_ENDING_DATE_COL = 'Week Ending Date'


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


def interpolate_for_missing_values(
    df, target_col
):
  """Interpolates missing values in the target column."""

  df[WEEK_ENDING_DATE_COL] = pd.to_datetime(df[WEEK_ENDING_DATE_COL])

  all_geo_agg = df[GEO_AGGREGATION_COL].unique()
  min_date = df[WEEK_ENDING_DATE_COL].min()
  max_date = df[WEEK_ENDING_DATE_COL].max()

  # Create full date range of Saturdays
  all_dates = pd.date_range(start=min_date, end=max_date, freq='W-SAT')
  all_combinations = pd.MultiIndex.from_product(
      [all_dates, all_geo_agg],
      names=[WEEK_ENDING_DATE_COL, GEO_AGGREGATION_COL],
  )
  full_df = pd.DataFrame(index=all_combinations).reset_index()
  df_no_missing = pd.merge(
      full_df,
      df,
      on=[GEO_AGGREGATION_COL, WEEK_ENDING_DATE_COL],
      how='left',
  )

  # Sort for correct interpolation
  df_no_missing = df_no_missing.sort_values(
      by=[GEO_AGGREGATION_COL, WEEK_ENDING_DATE_COL]
  )

  # Interpolate
  df_no_missing[target_col] = df_no_missing.groupby(GEO_AGGREGATION_COL)[
      target_col
  ].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))

  return df_no_missing


def preprocess_data(
    input_df,
    locations_df,
    infection_config,
):
  """Preprocesses the data, ensuring numeric conversion before interpolation."""

  # Ensure column is a string to handle potential mixed types, replace commas
  clean_df = input_df.copy()
  clean_df[infection_config['target_column']] = (
      clean_df[infection_config['target_column']]
      .astype(str)
      .str.replace(',', '', regex=False)
  )
  # Coerce any non-numeric values to NaN, preparing for interpolation
  clean_df[infection_config['target_column']] = pd.to_numeric(
      clean_df[infection_config['target_column']], errors='coerce'
  )

  interpolated_df = interpolate_for_missing_values(
      clean_df, infection_config['target_column']
  )

  assert (
      interpolated_df[infection_config['target_column']].isnull().sum() == 0
  ), 'Error: Null values remain in the target column after interpolation.'

  interpolated_df = interpolated_df.rename(
      columns={WEEK_ENDING_DATE_COL: 'target_end_date'}
  )

  # Merge with locations data
  processed_df = pd.merge(
      interpolated_df,
      locations_df,
      left_on=GEO_AGGREGATION_COL,
      right_on=ABBREVIATION_COL,
      how='inner',
  )
  processed_df = processed_df[[
      'target_end_date',
      LOCATION_NAME_COL,
      infection_config['target_column'],
  ]].copy()
  processed_df[infection_config['target_column']] = processed_df[
      infection_config['target_column']
  ].astype(int)
  processed_df.sort_values(by='target_end_date', ascending=False, inplace=True)
  processed_df['target_end_date'] = pd.to_datetime(
      processed_df['target_end_date']
  ).dt.date

  return processed_df


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  locations = pd.read_csv(_LOCATIONS_PATH.value)
  locations_df = locations[
      locations[LOCATION_CODE_COL].isin(REQUIRED_CDC_LOCATIONS)
  ]
  infection_config = _load_infection_config(
      _CONFIG_PATH.value, _INFECTION.value
  )

  df = pd.read_csv(_NHSN_INPUT.value)[[
      'Week Ending Date',
      'Geographic aggregation',
      infection_config['target_column'],
  ]]

  df = preprocess_data(df, locations_df, infection_config)
  df = pd.merge(
      df,
      locations[[LOCATION_CODE_COL, LOCATION_NAME_COL, POPULATION_COL]],
      on=LOCATION_NAME_COL,
      how='inner',
  )
  df[infection_config['target_column']] = pd.to_numeric(
      df[infection_config['target_column']], errors='coerce'
  ).astype('Int64')

  # Note: duplicated (location, target_end_date) rows have been observed before
  # due to leap years. To check for them:
  # df[df.duplicated(subset=[LOCATION_NAME_COL, 'target_end_date'], keep=False)]

  # The suffix is the part of the NHSN file name after the last underscore,
  # e.g. "20250924" for "..._Preliminary__20250924.csv".
  input_suffix = _NHSN_INPUT.value.split('_')[-1].split('.')[0]
  os.makedirs(_OUTPUT_PATH.value, exist_ok=True)
  output_file = os.path.join(_OUTPUT_PATH.value, f'dataset_{input_suffix}.csv')
  df.to_csv(output_file, index=False)
  print(f'Successfully wrote to {output_file}')


if __name__ == '__main__':
  app.run(main)
