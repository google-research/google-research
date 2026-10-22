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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,missing-function-docstring,broad-exception-caught,bare-except,cell-var-from-loop

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
MODEL_NAME = 'Google_SAI-Adapted_25'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles:
# 1. Mechanistic SIR Model: Discrete-time daily dynamics (S, I, R) to model influenza spread.
# 2. Splined Effective Reproduction Number (Rt): Time-varying Rt modeled via knots to allow flexible epidemic curves.
# 3. Bayesian Hierarchical Priors: Priors for timing, scale (rho), and Rt derived from historical ILINet and overlap training data.
# 4. Ensemble MCMC Sampling: Goodman & Weare sampler (emcee) used to characterize uncertainty via the posterior.

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import gammaln
from scipy.optimize import minimize
import warnings

try:
  import emcee
except ImportError:
  emcee = None


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  QUANTILES = [
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
  q_cols = [f'quantile_{q}' for q in QUANTILES]

  # --- 1. Hierarchical Prior Extraction ---
  # Analyze ILI history and learn the ILI-to-Admissions transformation
  state_h_priors = {}
  try:
    ili_df = globals().get('ilinet_state', pd.DataFrame()).copy()
    if not ili_df.empty:
      ili_df['target_end_date'] = pd.to_datetime(
          ili_df['week_start']
      ) + pd.Timedelta(days=6)
      # Find peak weeks per state
      peaks = ili_df.groupby(['region', 'season'])['unweighted_ili'].idxmax()
      peak_weeks = (
          ili_df.loc[peaks]
          .groupby('region')['season_week']
          .agg(['median', 'std'])
          .fillna(5)
      )

      # Estimate scaling rho: Admissions ~ rho * (unweighted_ili/100) * population
      # We look at the overlap period (roughly 2020-2022)
      merged = pd.merge(
          train_x,
          train_y.to_frame(name='y'),
          on=train_x.index.name if train_x.index.name else None,
          left_index=True,
          right_index=True,
      )
      merged['target_end_date'] = pd.to_datetime(merged['target_end_date'])
      overlap = pd.merge(
          merged,
          ili_df[['region', 'target_end_date', 'unweighted_ili']],
          left_on=['location_name', 'target_end_date'],
          right_on=['region', 'target_end_date'],
      )

      def calc_rho(df):
        num = df['y'] * 100
        den = df['unweighted_ili'] * df['population']
        return np.median(num / den.replace(0, np.nan).dropna())

      rho_map = overlap.groupby('location_name').apply(calc_rho).to_dict()

      for loc in peak_weeks.index:
        state_h_priors[loc] = {
            'peak_wk': peak_weeks.loc[loc, 'median'],
            'rho': rho_map.get(loc, 0.0001),  # Default tiny ascertainment
        }
  except Exception:
    pass

  # --- 2. Mechanistic Model Definition ---
  def run_sir(params, pop, n_weeks, knot_weeks):
    # params: [log_Rk1, log_Rk2, log_Rk3, log_I0, log_rho, log_phi, logit_s0]
    rt_knots = np.exp(np.clip(params[0:3], -2, 2))
    I0 = np.exp(params[3])
    rho = np.exp(params[4])
    phi = np.exp(params[5])
    s0_frac = 1.0 / (1.0 + np.exp(-params[6]))
    gamma = 1.0 / 3.5  # Recovery rate

    interp_f = interp1d(
        knot_weeks, rt_knots, kind='linear', fill_value='extrapolate'
    )
    rt_daily = interp_f(np.arange(n_weeks * 7) / 7.0)

    S, I = pop * s0_frac, I0
    weekly_h = np.zeros(n_weeks)
    for w in range(n_weeks):
      new_inf_w = 0.0
      for d in range(7):
        beta = rt_daily[min(w * 7 + d, len(rt_daily) - 1)] * gamma
        new_cases = S * (1.0 - np.exp(-beta * I / pop))
        S = max(0, S - new_cases)
        I = max(0, I + new_cases - (gamma * I))
        new_inf_w += new_cases
      weekly_h[w] = new_inf_w * rho
    return np.maximum(weekly_h, 1e-5), phi

  def log_prob(params, y_obs, pop, n_weeks, knot_weeks, prior_vals):
    # Priors based on hierarchical stats
    lp = -0.5 * np.sum(((params[0:3] - 0.1) / 0.4) ** 2)  # Rt knots
    lp -= 0.5 * ((params[3] - 2.0) / 2.5) ** 2  # log_I0
    lp -= 0.5 * ((params[4] - np.log(prior_vals['rho'])) / 0.8) ** 2  # log_rho
    lp -= 0.5 * ((params[5] - 2.5) / 1.5) ** 2  # log_phi
    lp -= 0.5 * ((params[6] - 2.2) / 1.0) ** 2  # logit_s0
    if not np.isfinite(lp):
      return -1e30

    mu_w, phi = run_sir(params, pop, n_weeks, knot_weeks)
    mu = mu_w[: len(y_obs)]
    # Negative Binomial Likelihood
    ll = (
        gammaln(y_obs + phi)
        - gammaln(y_obs + 1)
        - gammaln(phi)
        + phi * np.log(phi / (phi + mu))
        + y_obs * np.log(mu / (phi + mu))
    ).sum()
    return lp + ll if np.isfinite(ll) else -1e30

  # --- 3. Prediction Loop ---
  ref_date = pd.to_datetime(test_x['reference_date'].iloc[0])
  season_year = ref_date.year if ref_date.month >= 8 else ref_date.year - 1
  season_start = pd.to_datetime(f'{season_year}-08-01')

  df_train = train_x.copy()
  df_train['y'] = train_y.values
  df_train['target_end_date'] = pd.to_datetime(df_train['target_end_date'])

  results = []
  for loc in test_x['location'].unique():
    loc_test = test_x[test_x['location'] == loc].sort_values('target_end_date')
    loc_train = df_train[
        (df_train['location'] == loc)
        & (df_train.target_end_date >= season_start)
    ].sort_values('target_end_date')

    pop = loc_test['population'].iloc[0]
    y_obs = loc_train['y'].values.astype(float)
    loc_name = loc_test['location_name'].iloc[0]
    priors = state_h_priors.get(loc_name, {'peak_wk': 22, 'rho': 0.0001})

    # n_sim_weeks = int((loc_test.target_end_date.max() - season_start).days // 7) + 1
    n_sim_weeks = (
        int(
            (pd.Timestamp(loc_test.target_end_date.max()) - season_start).days
            // 7
        )
        + 1
    )  # manually added
    knot_weeks = np.array([0, priors['peak_wk'], n_sim_weeks + 2])

    # Initial guess for optimization
    init_p = np.array([0.1, 0.3, -0.1, 1.0, np.log(priors['rho']), 2.5, 2.2])

    samples = None
    if emcee is not None and len(y_obs) > 2:
      try:
        res = minimize(
            lambda p: -log_prob(p, y_obs, pop, n_sim_weeks, knot_weeks, priors),
            init_p,
            method='Nelder-Mead',
            options={'maxiter': 400},
        )
        ndim, nwalkers = 7, 20
        pos = res.x + 1e-3 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob,
            args=(y_obs, pop, n_sim_weeks, knot_weeks, priors),
        )
        sampler.run_mcmc(pos, 350, progress=False)
        samples = sampler.get_chain(discard=200, flat=True)
      except:
        pass

    if samples is not None:
      draws = samples[np.random.choice(len(samples), 400)]
      trajectories = []
      for p in draws:
        mu_w, phi = run_sir(p, pop, n_sim_weeks, knot_weeks)
        # Sample a coherent trajectory
        trajectories.append(
            np.random.negative_binomial(phi, phi / (phi + mu_w))
        )
      pred_matrix = np.array(trajectories)
    else:
      # High-quality fallback using historical shape
      weeks = np.arange(n_sim_weeks)
      shape = np.exp(-0.5 * ((weeks - priors['peak_wk']) / 6.0) ** 2)
      scale = (
          (y_obs[-1] / max(0.01, shape[len(y_obs) - 1]))
          if len(y_obs) > 0
          else (pop * priors['rho'] * 0.01)
      )
      mu_f = shape * scale
      pred_matrix = np.array([
          np.random.negative_binomial(
              3.0, 3.0 / (3.0 + mu_f * np.random.uniform(0.7, 1.3))
          )
          for _ in range(400)
      ])

    loc_forecasts = []
    for dt in loc_test.target_end_date:
      w_idx = min(
          int((pd.to_datetime(dt) - season_start).days // 7), n_sim_weeks - 1
      )
      loc_forecasts.append(np.quantile(pred_matrix[:, w_idx], QUANTILES))

    res_df = pd.DataFrame(loc_forecasts, columns=q_cols, index=loc_test.index)
    results.append(res_df)

  final_df = pd.concat(results).sort_index()
  # Ensure non-negative, monotonic, and rounded
  arr = np.maximum(0, final_df.values)
  arr.sort(axis=1)
  return (
      pd.DataFrame(arr, columns=q_cols, index=final_df.index)
      .round()
      .astype(int)
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
