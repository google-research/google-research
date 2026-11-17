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

r"""Collect and merge individual weekly forecasts into a single wide-format CSV.

For every model listed under `model_names_for_ensemble` in
`infection_configs.json`, this script reads all weekly submission files found in
`<model_output_dir>/<model_name>/` and writes them to one CSV with a column per
quantile.

Example usage (run from the `epi_forecasts` directory):
python pipeline/merge_forecasts.py \
    --infection=flu \
    --output_path=/tmp/flu_merged_forecasts.csv
"""

from collections.abc import Sequence
import json
import os
import re

from absl import app
from absl import flags
import numpy as np
import pandas as pd

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
# Root of the `epi_forecasts` directory. Relative paths in the config are
# resolved against it.
_REPO_DIR = os.path.dirname(_PIPELINE_DIR)

_INFECTION = flags.DEFINE_string(
    'infection',
    None,
    'Infection type: "flu", "covid", or "rsv".',
    required=True,
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Destination path for the merged CSV file.',
    required=True,
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


def standardize_forecast(df, model_name):
  """Standardizes column names, types, and pivots long formats to wide format."""
  # Ensure columns are lowercase
  df.columns = [c.lower() for c in df.columns]

  # Coerce basic column types
  df['location'] = df['location'].astype(int)
  df['horizon'] = df['horizon'].astype(int)
  df['reference_date'] = pd.to_datetime(df['reference_date'])
  df['target_end_date'] = pd.to_datetime(df['target_end_date'])

  quantiles = [
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
  quantile_cols = [f'quantile_{q}' for q in quantiles]

  if 'output_type' in df.columns and 'value' in df.columns:
    # Long format: filter for quantiles and pivot
    df_q = df[df['output_type'] == 'quantile'].copy()
    # Format output_type_id as float to match quantiles list representation
    df_q['quantile_name'] = (
        df_q['output_type_id'].astype(float).map(lambda x: f'quantile_{x}')
    )

    # Pivot to wide format
    pivoted = df_q.pivot(
        index=['reference_date', 'horizon', 'location', 'target_end_date'],
        columns='quantile_name',
        values='value',
    ).reset_index()

    # Ensure all expected quantile columns exist (fill missing with NaN)
    for col in quantile_cols:
      if col not in pivoted.columns:
        pivoted[col] = np.nan

    df_wide = pivoted[
        ['reference_date', 'horizon', 'location', 'target_end_date']
        + quantile_cols
    ].copy()
  else:
    # Wide format: select required columns
    # Ensure all expected quantile columns exist (fill missing with NaN)
    for col in quantile_cols:
      if col not in df.columns:
        df[col] = np.nan
    df_wide = df[
        ['reference_date', 'horizon', 'location', 'target_end_date']
        + quantile_cols
    ].copy()

  df_wide['model_name'] = model_name
  return df_wide


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  infection = _INFECTION.value
  output_path = _OUTPUT_PATH.value

  print(f'Loading configuration for: {infection}')
  with open(_CONFIG_PATH.value, 'r') as f:
    all_configs = json.load(f)
  if infection not in all_configs:
    raise ValueError(
        f'Unknown infection "{infection}". Available: {sorted(all_configs)}'
    )
  config_data = all_configs[infection]

  model_names = config_data['model_names_for_ensemble']
  base_dir = _MODEL_OUTPUT_DIR.value or os.path.join(
      _REPO_DIR, config_data['model_output_dir']
  )

  print(f'Models to merge: {model_names}')
  print(f'Base directory: {base_dir}')

  all_dfs = []
  for model_name in model_names:
    model_dir = os.path.join(base_dir, model_name)
    if not os.path.isdir(model_dir):
      print(f'[WARNING] Directory not found for model: {model_name}')
      continue

    print(f'Scanning forecasts for model: {model_name}')
    files = os.listdir(model_dir)
    pattern = re.compile(
        rf'^\d{{4}}-\d{{2}}-\d{{2}}-{re.escape(model_name)}\.csv$'
    )
    matching_files = [f for f in files if pattern.match(f)]
    print(f'  Found {len(matching_files)} forecast files.')

    for filename in sorted(matching_files):
      file_path = os.path.join(model_dir, filename)
      df = pd.read_csv(file_path)

      std_df = standardize_forecast(df, model_name)
      all_dfs.append(std_df)

  if not all_dfs:
    print('No forecast data loaded. Exiting.')
    return

  merged_df = pd.concat(all_dfs, ignore_index=True)
  print(f'Merged dataframe shape: {merged_df.shape}')

  # Ensure the directory exists
  out_dir = os.path.dirname(output_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  print(f'Saving merged dataset to: {output_path}')
  merged_df.to_csv(output_path, index=False)
  print('Done successfully.')


if __name__ == '__main__':
  app.run(main)
