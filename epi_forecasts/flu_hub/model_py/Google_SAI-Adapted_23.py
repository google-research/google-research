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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,missing-function-docstring

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
MODEL_NAME = 'Google_SAI-Adapted_23'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles of this Method:
# 1. Strategy 2 - Adaptive Synthetic History: Use historical ILI data as a seasonal
#    prior, but allow the scaling factor to be updated by recent admissions data.
# 2. Hierarchical Negative Binomial State-Space Model: Model US states nested within
#    HHS regions using non-centered parameterizations for intercepts and AR(1) terms.
# 3. Enhanced Dynamic Features: Capture epidemic growth using log-rate momentum,
#    3rd-order Fourier seasonality, and specific holiday-effect indicators.
# 4. Fast Monte Carlo Trajectory Sampling: Generate quantile forecasts by
#    simulating forward paths from the joint posterior of the latent log-rate process.

import pymc as pm
import numpy as np
import pandas as pd
import logging
import warnings

# Speed up PyMC and suppress unnecessary logs
logging.getLogger('pymc').setLevel(logging.ERROR)
logging.getLogger('pytensor').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # 1. Geography & Mapping
  hhs_map = {
      'Connecticut': 1,
      'Maine': 1,
      'Massachusetts': 1,
      'New Hampshire': 1,
      'Rhode Island': 1,
      'Vermont': 1,
      'New Jersey': 2,
      'New York': 2,
      'Puerto Rico': 2,
      'Delaware': 3,
      'District of Columbia': 3,
      'Maryland': 3,
      'Pennsylvania': 3,
      'Virginia': 3,
      'West Virginia': 3,
      'Alabama': 4,
      'Florida': 4,
      'Georgia': 4,
      'Kentucky': 4,
      'Mississippi': 4,
      'North Carolina': 4,
      'South Carolina': 4,
      'Tennessee': 4,
      'Illinois': 5,
      'Indiana': 5,
      'Michigan': 5,
      'Minnesota': 5,
      'Ohio': 5,
      'Wisconsin': 5,
      'Arkansas': 6,
      'Louisiana': 6,
      'New Mexico': 6,
      'Oklahoma': 6,
      'Texas': 6,
      'Iowa': 7,
      'Kansas': 7,
      'Missouri': 7,
      'Nebraska': 7,
      'Colorado': 8,
      'Montana': 8,
      'North Dakota': 8,
      'South Dakota': 8,
      'Utah': 8,
      'Wyoming': 8,
      'Arizona': 9,
      'California': 9,
      'Hawaii': 9,
      'Nevada': 9,
      'Alaska': 10,
      'Idaho': 10,
      'Oregon': 10,
      'Washington': 10,
  }

  # 2. Process Data & Create Synthetic Baseline
  df_train = train_x.copy()
  df_train[TARGET_STR] = train_y.values
  df_train['target_end_date'] = pd.to_datetime(df_train['target_end_date'])

  ili = ilinet_state.copy()
  ili['target_end_date'] = pd.to_datetime(ili['week_start'])
  ili['week'] = ili['target_end_date'].dt.isocalendar().week.astype(int)

  # Calculate state-specific scaling (Admissions per 100k vs ILI %)
  overlap = df_train.merge(
      ili[['region', 'target_end_date', 'unweighted_ili']],
      left_on=['location_name', 'target_end_date'],
      right_on=['region', 'target_end_date'],
  )
  overlap['ratio'] = (overlap[TARGET_STR] / (overlap['population'] / 1e5)) / (
      overlap['unweighted_ili'] + 0.1
  )
  state_ratios = overlap.groupby('location_name')['ratio'].median().to_dict()

  # Historical Climatology (Strategy 2)
  ili_clim = (
      ili.groupby(['region', 'week'])['unweighted_ili'].median().reset_index()
  )
  ili_clim['unweighted_ili'] = ili_clim.groupby('region')[
      'unweighted_ili'
  ].transform(lambda x: x.rolling(3, center=True, min_periods=1).mean())

  def add_features(input_df):
    d = input_df.copy()
    d['target_end_date'] = pd.to_datetime(d['target_end_date'])
    d['week'] = d['target_end_date'].dt.isocalendar().week.astype(int)
    d['hhs_idx'] = d['location_name'].map(hhs_map).fillna(11).astype(int) - 1
    # Fourier Seasonality (3rd Order)
    for i in range(1, 4):
      d[f'cos_{i}'] = np.cos(2 * i * np.pi * d['week'] / 52.18)
      d[f'sin_{i}'] = np.sin(2 * i * np.pi * d['week'] / 52.18)
    d['is_holiday'] = d['week'].isin([52, 1]).astype(float)
    # Baseline from ILI
    d = d.merge(
        ili_clim,
        left_on=['location_name', 'week'],
        right_on=['region', 'week'],
        how='left',
    )
    ratio_med = overlap['ratio'].median()
    d['ratio'] = d['location_name'].map(state_ratios).fillna(ratio_med)
    d['synth_baseline'] = np.log(
        (d['unweighted_ili'] * d['ratio'] * (d['population'] / 1e5) + 0.1)
        / d['population']
    )
    return d.fillna(0)

  # Filter to recent history (last 75 weeks) for model relevance
  train_proc = add_features(
      df_train[
          df_train['target_end_date']
          > (df_train['target_end_date'].max() - pd.Timedelta(weeks=75))
      ]
  )
  locs = sorted(train_proc['location'].unique())
  loc_map = {l: i for i, l in enumerate(locs)}
  hhs_ids = (
      train_proc.groupby('location')['hhs_idx'].first().reindex(locs).values
  )

  train_proc['loc_idx'] = train_proc['location'].map(loc_map)
  train_proc['log_rate'] = np.log(
      (train_proc[TARGET_STR] + 0.5) / train_proc['population']
  )
  train_proc = train_proc.sort_values(['location', 'target_end_date'])
  train_proc['lag_log_rate'] = train_proc.groupby('location')['log_rate'].shift(
      1
  )
  train_proc['momentum'] = train_proc.groupby('location')['log_rate'].shift(
      1
  ) - train_proc.groupby('location')['log_rate'].shift(2)
  m_data = train_proc.dropna(subset=['lag_log_rate', 'momentum'])

  # 3. PyMC Hierarchical Model
  n_locs, n_hhs = len(locs), 11
  with pm.Model() as model:
    # Hierarchical Intercepts
    mu_a = pm.Normal('mu_a', -10.0, 1.5)
    s_hhs = pm.HalfNormal('s_hhs', 0.8)
    s_loc = pm.HalfNormal('s_loc', 0.8)
    z_hhs = pm.Normal('z_hhs', 0, 1, shape=n_hhs)
    z_loc = pm.Normal('z_loc', 0, 1, shape=n_locs)
    alpha = pm.Deterministic(
        'alpha', mu_a + z_hhs[hhs_ids] * s_hhs + z_loc * s_loc
    )

    # Adaptive AR(1) logic (non-centered)
    mu_b = pm.Normal('mu_b', 1.8, 0.4)
    s_b = pm.HalfNormal('s_b', 0.4)
    z_b = pm.Normal('z_b', 0, 1, shape=n_locs)
    beta = pm.Deterministic('beta', pm.math.sigmoid(mu_b + z_b * s_b) * 1.1)

    # Fixed effects
    b_mom = pm.Normal('b_mom', 0.15, 0.1)
    b_base = pm.Normal('b_base', 0.4, 0.2)
    b_hol = pm.Normal('b_hol', -0.2, 0.1)
    b_f = pm.Normal('b_f', 0, 0.4, shape=6)  # 3 Fourier pairs

    # Process and Observation Noise
    sig_p = pm.HalfNormal('sig_p', 0.25)
    phi_mu = pm.Exponential('phi_mu', 0.1)
    phi_s = pm.HalfNormal('phi_s', 0.4)
    z_phi = pm.Normal('z_phi', 0, 1, shape=n_locs)
    phi = pm.Deterministic(
        'phi', pm.math.exp(pm.math.log(phi_mu) + z_phi * phi_s)
    )

    l_idx = m_data['loc_idx'].values
    mu_log = (
        alpha[l_idx]
        + beta[l_idx] * m_data['lag_log_rate'].values
        + b_mom * m_data['momentum'].values
        + b_base * m_data['synth_baseline'].values
        + b_hol * m_data['is_holiday'].values
        + b_f[0] * m_data['cos_1'].values
        + b_f[1] * m_data['sin_1'].values
        + b_f[2] * m_data['cos_2'].values
        + b_f[3] * m_data['sin_2'].values
        + b_f[4] * m_data['cos_3'].values
        + b_f[5] * m_data['sin_3'].values
    )

    pm.NegativeBinomial(
        'obs',
        mu=pm.math.exp(np.log(m_data['population'].values) + mu_log),
        alpha=phi[l_idx],
        observed=m_data[TARGET_STR].values,
    )

    # Efficient sampling for rolling windows
    trace = pm.sample(
        500,
        tune=500,
        chains=1,
        target_accept=0.8,
        progressbar=False,
        random_seed=42,
    )

  # 4. Vectorized Projection
  post = trace.posterior
  n_s = 500
  test_proc = add_features(test_x)
  test_proc['loc_idx'] = test_proc['location'].map(loc_map)

  last_pts = train_proc.groupby('location').tail(2)
  r_prev = np.tile(
      last_pts.groupby('location').first().reindex(locs)['log_rate'].values,
      (n_s, 1),
  )
  r_curr = np.tile(
      last_pts.groupby('location').last().reindex(locs)['log_rate'].values,
      (n_s, 1),
  )

  # Extract posterior means/samples
  p_a, p_b = post['alpha'].values[0], post['beta'].values[0]
  p_bm, p_bb, p_bh = (
      post['b_mom'].values[0][:, None],
      post['b_base'].values[0][:, None],
      post['b_hol'].values[0][:, None],
  )
  p_bf = post['b_f'].values[0]
  p_phi, p_sig = post['phi'].values[0], post['sig_p'].values[0][:, None]

  forecasts = {}
  for h in sorted(test_proc['horizon'].unique()):
    h_df = test_proc[test_proc['horizon'] == h].sort_values('loc_idx')
    idx = h_df['loc_idx'].values

    mom = r_curr[:, idx] - r_prev[:, idx]
    mu_h = (
        p_a[:, idx]
        + p_b[:, idx] * r_curr[:, idx]
        + p_bm * mom
        + p_bb * h_df['synth_baseline'].values
        + p_bh * h_df['is_holiday'].values
        + p_bf[:, 0:1] * h_df['cos_1'].values
        + p_bf[:, 1:2] * h_df['sin_1'].values
        + p_bf[:, 2:3] * h_df['cos_2'].values
        + p_bf[:, 3:4] * h_df['sin_2'].values
        + p_bf[:, 4:5] * h_df['cos_3'].values
        + p_bf[:, 5:6] * h_df['sin_3'].values
    )

    r_new = mu_h + np.random.normal(0, p_sig, (n_s, len(idx)))
    r_prev[:, idx], r_curr[:, idx] = r_curr[:, idx], r_new

    # Negative Binomial Simulation
    mu_obs = np.exp(np.log(h_df['population'].values) + r_new)
    p_nb = p_phi[:, idx] / (p_phi[:, idx] + mu_obs)
    y_sim = np.random.negative_binomial(p_phi[:, idx], p_nb)

    q_vals = np.percentile(y_sim, [q * 100 for q in QUANTILES], axis=0)
    for i, row_idx in enumerate(h_df.index):
      forecasts[row_idx] = q_vals[:, i]

  # 5. Output Formatting
  out = pd.DataFrame.from_dict(
      forecasts, orient='index', columns=[f'quantile_{q}' for q in QUANTILES]
  )
  out = out.reindex(test_x.index).fillna(0)
  # Ensure strict monotonicity and non-negativity
  final = np.sort(np.maximum(0, out.values), axis=1)
  return pd.DataFrame(
      final.astype(int), index=test_x.index, columns=out.columns
  )


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
