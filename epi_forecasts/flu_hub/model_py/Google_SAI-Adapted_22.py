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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,missing-function-docstring,bare-except

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
MODEL_NAME = 'Google_SAI-Adapted_22'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles of the Method:
# 1. Branching Process Model: Hospitalizations are treated as a proxy for a latent infection chain where E[Z_{t+1}] = R_t * Z_t.
# 2. Symbolic Moment Propagation: Uses SymPy to derive the recursive mean and variance of a branching process (Var_{t+1} = m^2 * Var_t + Mean_t * Var_offspring).
# 3. Bayesian Inference via PyMC: Estimates location-specific R0 and current growth trends by combining recent hospital data with historical ILI seasonality.
# 4. Mechanistic Immunity: Incorporates a seasonally-reset susceptibility constraint (S/N) where S updates based on infections (hospitalizations / IHR).

import numpy as np
import pandas as pd
import pymc as pm
import sympy as sp
from scipy.stats import nbinom


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # --- 1. Symbolic Derivation of Moments ---
  # We derive the mean (M) and variance (V) of Z_{t+1} given Z_t ~ (M_t, V_t)
  # and offspring distribution ~ (m, var_off)
  M_t, V_t, m, g2 = sp.symbols('M_t V_t m g2')
  M_next = M_t * m
  # Variance of branching process: V_{t+1} = m^2 * V_t + M_t * Var_offspring
  # Factorial moment g2 = E[X(X-1)] = Var_off + m^2 - m
  V_next = (V_t + M_t**2 - M_t) * (m**2) + M_t * g2 + M_next - (M_next**2)

  f_mean = sp.lambdify((M_t, m), M_next, 'numpy')
  f_var = sp.lambdify((M_t, V_t, m, g2), V_next, 'numpy')

  # --- 2. Data Preparation & ILI Seasonality ---
  locations = test_x['location'].unique()
  loc_names = (
      test_x.groupby('location')['location_name']
      .first()
      .reindex(locations)
      .values
  )
  num_locs = len(locations)

  train_df = train_x.copy()
  train_df[TARGET_STR] = train_y.values
  train_df['target_end_date'] = pd.to_datetime(train_df['target_end_date'])
  train_df['week'] = train_df['target_end_date'].dt.isocalendar().week

  # Calculate seasonal growth priors from historical ILI
  try:
    ili = globals()['ilinet_state'].copy()
    ili['target_end_date'] = pd.to_datetime(ili['week_start']) + pd.Timedelta(
        days=6
    )
    ili['week'] = ili['target_end_date'].dt.isocalendar().week
    # Median growth factor per week across all history
    ili_pivot = ili.pivot_table(
        index='week',
        columns='region',
        values='unweighted_ili',
        aggfunc='median',
    )
    seasonal_growth = (
        (ili_pivot.pct_change().fillna(0) + 1.0)
        .rolling(3, center=True, min_periods=1)
        .mean()
    )
  except:
    seasonal_growth = pd.DataFrame(1.0, index=range(1, 54), columns=loc_names)

  # --- 3. Bayesian Inference for Local Parameters ---
  history_len = 6
  pivot_y = train_df.pivot(
      index='target_end_date', columns='location', values=TARGET_STR
  ).reindex(columns=locations, fill_value=0)
  y_obs = pivot_y.values[-history_len:]

  # Recent growth profile
  curr_weeks = pivot_y.index[-history_len:].isocalendar().week.values
  growth_prior = np.ones((history_len - 1, num_locs))
  for t in range(history_len - 1):
    wk = curr_weeks[t + 1]
    for i, name in enumerate(loc_names):
      if name in seasonal_growth.columns:
        growth_prior[t, i] = seasonal_growth.loc[wk, name]

  with pm.Model() as model:
    # Hierarchical reproduction number: Base * Seasonal * Local_Trend
    r_base = pm.Lognormal('r_base', mu=0.0, sigma=0.2, shape=num_locs)
    # local_k allows the model to adjust for this specific year's growth speed
    local_k = pm.Normal('local_k', mu=1.0, sigma=0.1, shape=num_locs)
    phi = pm.Exponential('phi', 0.5, shape=num_locs)  # Dispersion

    # Expected value propagation
    mu = pm.math.maximum(0.1, y_obs[:-1, :] * (r_base * local_k * growth_prior))
    pm.NegativeBinomial('obs', mu=mu, alpha=phi, observed=y_obs[1:, :])

    map_est = pm.find_MAP(progressbar=False)

  # --- 4. Mechanistic Projection ---
  # Define Infection-to-Hospitalization Ratio (IHR)
  IHR = 0.005  # 0.5%

  # Calculate susceptibility (S/N) reset at start of season (week 30)
  # Get cumulative hospitalizations in the current season (since last week 30)
  ref_date = test_x['reference_date'].min()
  ref_date = pd.Timestamp(ref_date)  ## Manually added

  season_start = (
      ref_date - pd.Timedelta(weeks=52)
      if ref_date.week < 30
      else ref_date.replace(month=7, day=20)
  )  # Approx week 30
  curr_season_data = train_df[train_df['target_end_date'] >= season_start]
  cum_hosp = (
      curr_season_data.groupby('location')[TARGET_STR]
      .sum()
      .reindex(locations, fill_value=0)
      .values
  )

  pop_map = (
      test_x.groupby('location')['population'].first().reindex(locations).values
  )
  s_curr = np.clip(pop_map - (cum_hosp / IHR), 0.2 * pop_map, pop_map)

  # Initial state for projection
  mu_curr = np.maximum(0.5, y_obs[-1, :])
  phi_vals = map_est['phi']
  # Starting variance includes observation uncertainty
  var_curr = mu_curr + (mu_curr**2 / phi_vals)

  res_df = pd.DataFrame(
      index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  for h in range(4):
    target_wk = (curr_weeks[-1] + h) % 52 + 1
    m_seas = np.array([
        seasonal_growth.loc[target_wk, n]
        if n in seasonal_growth.columns
        else 1.0
        for n in loc_names
    ])

    # Effective R = Base * Seasonal * Trend * (S/N)
    m_eff = map_est['r_base'] * map_est['local_k'] * m_seas * (s_curr / pop_map)
    m_eff = np.maximum(0.01, m_eff)

    # Offspring variance (Negative Binomial)
    # g2 = E[X(X-1)] = m^2 * (1 + 1/phi)
    g2_eff = (m_eff**2) * (1.0 + 1.0 / phi_vals)

    # Propagate Moments
    mu_next = f_mean(mu_curr, m_eff)
    var_next = f_var(mu_curr, var_curr, m_eff, g2_eff)

    # Variance Inflation: Add 10% to account for parameter uncertainty ignored by MAP
    var_next = var_next * 1.1 + (mu_next * 0.1)

    # Robust conversion to NB parameters
    mu_next = np.maximum(0.1, mu_next)
    var_next = np.maximum(mu_next + 1e-5, var_next)
    nb_p = np.clip(mu_next / var_next, 1e-8, 1 - 1e-8)
    nb_n = (mu_next**2) / (var_next - mu_next)

    for i, loc in enumerate(locations):
      q_vals = nbinom.ppf(QUANTILES, nb_n[i], nb_p[i])
      mask = (test_x['horizon'] == h) & (test_x['location'] == loc)
      if mask.any():
        res_df.loc[mask, :] = q_vals

    # Update latent states
    mu_curr, var_curr = mu_next, var_next
    s_curr = np.maximum(0.1 * pop_map, s_curr - (mu_curr / IHR))

  # Final Monotonicity & Formatting
  res_df = res_df.apply(pd.to_numeric).fillna(0)
  final_vals = np.sort(np.maximum(0, res_df.values), axis=1)
  res_df.iloc[:, :] = final_vals
  return res_df.astype(int)


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
