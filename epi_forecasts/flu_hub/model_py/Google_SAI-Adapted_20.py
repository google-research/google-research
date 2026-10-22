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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-multiple-import,g-doc-return-or-yield,g-doc-args,bare-except,global-variable-not-assigned

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
MODEL_NAME = 'Google_SAI-Adapted_20'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Enhanced Mechanistic Branching Process using PGF Moment Propagation.

  Incorporates scaled historical ILINet data for robust seasonal priors.
  """
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

  # Preparation
  train_df = train_x.copy()
  train_df['target'] = train_y.values
  train_df['target_end_date'] = pd.to_datetime(train_df['target_end_date'])

  # 1. Improved Seasonality Mapping (Learn Transformation from ILINet)
  # We use global variables ilinet_state and ilinet provided in the environment
  seasonal_profiles = {}
  try:
    global ilinet_state, ilinet
    # Calculate a mapping factor: Admissions / ILI
    # Overlap period is late 2020 to late 2022
    ili_ref = ilinet_state.copy()
    ili_ref['week_start'] = pd.to_datetime(ili_ref['week_start'])

    for loc_name in train_df['location_name'].unique():
      loc_ili = ili_ref[ili_ref['region'] == loc_name]
      # Simple average profile by week
      profile = loc_ili.groupby('week')['unweighted_ili'].mean()
      # Smooth profile
      profile = (
          profile.reindex(range(1, 54))
          .interpolate()
          .rolling(5, center=True, min_periods=1)
          .mean()
      )
      seasonal_profiles[loc_name] = profile.to_dict()
  except:
    pass

  # Generation Interval: Discrete Gamma (Mean=3.6, SD=1.6)
  # Weights for weeks t-1, t-2, t-3, t-4
  w_gen = np.array([0.62, 0.26, 0.08, 0.04])

  results = []
  for loc in test_x['location'].unique():
    loc_train = train_df[train_df['location'] == loc].sort_values(
        'target_end_date'
    )
    loc_test = test_x[test_x['location'] == loc].sort_values('horizon')
    if loc_train.empty:
      continue

    loc_name = loc_train['location_name'].iloc[-1]
    pop = loc_train['population'].iloc[-1]
    raw_hist = loc_train['target'].values

    # Determine seasonal timing
    last_date = loc_train['target_end_date'].max()
    last_week = last_date.isocalendar()[1]

    def get_seas(w):
      w_idx = int(((w - 1) % 52) + 1)
      # Fallback to 1.0 if loc_name not in profiles
      return max(0.01, seasonal_profiles.get(loc_name, {}).get(w_idx, 1.0))

    # 2. Mechanistic Inference
    # Infections per Admission (IPA) proxy:
    # assume total seasonal infections ~20% of pop, scaling by historical volume
    hist_peak = np.percentile(raw_hist, 95) if len(raw_hist) > 0 else 10
    ipa = np.clip((0.20 * pop) / (hist_peak * 25 + 1), 40, 600)

    # Renewal term (lambda_t): weighted sum of past cases
    if len(raw_hist) >= 4:
      lambda_t = np.dot(w_gen, raw_hist[-4:][::-1])
    else:
      lambda_t = raw_hist[-1] if len(raw_hist) > 0 else 1.0
    lambda_t = max(0.1, lambda_t)

    # Bayesian Rt estimation
    # Prior: Seasonal growth trend
    s_curr, s_next = get_seas(last_week), get_seas(last_week + 1)
    r_prior = np.clip(s_next / s_curr, 0.5, 2.5)

    # Momentum: recent growth rate (smoothed)
    if len(raw_hist) >= 3:
      recent_mu = np.mean(raw_hist[-2:]) + 1.0
      prev_mu = np.mean(raw_hist[-4:-2]) + 1.0
      r_momentum = np.clip(recent_mu / prev_mu, 0.5, 2.5)
    else:
      r_momentum = 1.0

    # Combine Prior and Momentum
    r_mu = 0.4 * r_prior + 0.6 * r_momentum
    kappa = 10.0  # Prior strength

    # Gamma-Poisson Posterior
    alpha_post = r_mu * kappa + raw_hist[-1]
    beta_post = kappa + lambda_t
    r_inferred = alpha_post / beta_post
    r_var = alpha_post / (beta_post**2)  # Uncertainty in R parameter

    # 3. PGF Moment Propagation
    # phi is the dispersion parameter (lower = higher variance)
    phi = np.clip(1.2 + (raw_hist[-1] / 50.0), 1.0, 15.0)

    # SLIR Susceptibility
    cum_ads_season = np.sum(
        raw_hist[-12:]
    )  # Proxy for current season infections
    s_t = np.exp(-(cum_ads_season * ipa) / pop)

    moments = {}
    curr_mean = raw_hist[-1]
    # Initial variance includes observation noise
    curr_var = curr_mean + (curr_mean**2 / phi) + 2.0
    # History buffer for renewal equation during projection
    mu_buffer = (
        list(raw_hist[-4:]) if len(raw_hist) >= 4 else [raw_hist[-1]] * 4
    )

    for h in [-1, 0, 1, 2, 3]:
      if h == -1:
        moments[h] = (raw_hist[-1], max(1.0, raw_hist[-1] * 0.1))
        continue

      # Mechanical drivers
      seas_factor = get_seas(last_week + h + 1) / get_seas(last_week + h)
      r_eff = r_inferred * s_t * seas_factor

      # Mean propagation (Renewal Equation)
      rt_sum = (
          w_gen[0] * mu_buffer[-1]
          + w_gen[1] * mu_buffer[-2]
          + w_gen[2] * mu_buffer[-3]
          + w_gen[3] * mu_buffer[-4]
      )
      mu_next = max(0.01, r_eff * rt_sum)

      # PGF Variance propagation:
      # Var(Z_t+1) = E[Z_t]*Var(Offspring) + E[Offspring]^2 * Var(Z_t)
      # Offspring variance (Negative Binomial)
      v_offspring = r_eff + (r_eff**2 / phi)

      # Add parameter uncertainty (r_var) and horizon-based drift
      param_noise = (rt_sum**2) * r_var * (1.2**h)
      var_next = (rt_sum * v_offspring) + (curr_var * (r_eff**2)) + param_noise
      var_next += mu_next * 0.1 + 0.5  # Add small process noise

      moments[h] = (mu_next, var_next)

      # Update state for next step
      mu_buffer.append(mu_next)
      curr_var = var_next
      # Deplete susceptibles
      s_t *= np.exp(-(mu_next * ipa) / pop)

    # 4. Distributional Mapping
    for _, row in loc_test.iterrows():
      h = row['horizon']
      mu, var = moments.get(h, (0.01, 1.0))

      # Map mean/variance to Negative Binomial
      if var > mu + 1e-5:
        p = np.clip(mu / var, 1e-9, 1.0 - 1e-9)
        n = np.clip((mu**2) / (var - mu), 0.01, 1e7)
        q_vals = nbinom.ppf(quantiles, n, p)
      else:
        q_vals = poisson.ppf(quantiles, mu)

      res_row = {'index': row.name}
      for i, q in enumerate(quantiles):
        res_row[f'quantile_{q}'] = int(max(0, np.round(q_vals[i])))
      results.append(res_row)

  if not results:
    return pd.DataFrame(
        index=test_x.index, columns=[f'quantile_{q}' for q in quantiles]
    ).fillna(0)

  out_df = pd.DataFrame(results).set_index('index')
  out_df = out_df.reindex(test_x.index).fillna(0)

  # Ensure Monotonicity
  cols = [f'quantile_{q}' for q in quantiles]
  for i in range(len(cols) - 1):
    out_df[cols[i + 1]] = np.maximum(out_df[cols[i + 1]], out_df[cols[i]])

  return out_df.astype(int)


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
