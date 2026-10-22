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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-multiple-import,missing-function-docstring,cell-var-from-loop,unnecessary-lambda

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
MODEL_NAME = 'Google_SAI-Adapted_18'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm
from scipy.special import expit, logit
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

  train_df = train_x.copy()
  train_df['target'] = train_y.values
  train_df['target_end_date'] = pd.to_datetime(train_df['target_end_date'])
  test_x_df = test_x.copy()
  test_x_df['target_end_date'] = pd.to_datetime(test_x_df['target_end_date'])

  # Access historical ILI data from global scope for hierarchical priors
  ili_state = globals().get('ilinet_state', pd.DataFrame())

  def get_hierarchical_params(loc_name, population):
    # Default priors based on national averages
    rho_prior = (logit(0.018), 0.5)
    rt_knot_prior = (np.log(1.2), 0.3)

    if not ili_state.empty:
      loc_ili = ili_state[ili_state['region'] == loc_name].copy()
      if not loc_ili.empty:
        loc_ili['target_end_date'] = pd.to_datetime(
            loc_ili['week_start']
        ) + pd.Timedelta(days=6)
        overlap = pd.merge(
            loc_ili,
            train_df[train_df['location_name'] == loc_name],
            on='target_end_date',
        )
        if len(overlap) > 8:
          # Robust estimate of mapping between ILI% and Admissions
          # Admissions / Pop = rho * (ILI / 100)
          ratios = (overlap['target'] / population) / (
              overlap['unweighted_ili'].clip(0.01) / 100.0
          )
          rho_val = np.clip(np.nanmedian(ratios), 0.005, 0.1)
          rho_prior = (logit(rho_val), 0.35)
    return rho_prior, rt_knot_prior

  results = []
  for loc in test_x_df['location'].unique():
    loc_train_all = train_df[train_df['location'] == loc].sort_values(
        'target_end_date'
    )
    ref_date = test_x_df['target_end_date'].min()
    sim_start = pd.Timestamp(
        year=ref_date.year if ref_date.month >= 7 else ref_date.year - 1,
        month=7,
        day=1,
    )

    loc_train = loc_train_all[loc_train_all['target_end_date'] >= sim_start]
    if len(loc_train) < 5:
      loc_train = loc_train_all.tail(15)

    loc_test = test_x_df[test_x_df['location'] == loc].sort_values(
        'target_end_date'
    )
    pop = loc_train['population'].iloc[0]
    y_obs = loc_train['target'].values.astype(float)

    all_dates = sorted(
        pd.to_datetime(
            pd.concat(
                [loc_train['target_end_date'], loc_test['target_end_date']]
            ).unique()
        )
    )
    date_map = {d: i for i, d in enumerate(all_dates)}
    train_idx = [
        date_map[pd.Timestamp(d)] for d in loc_train['target_end_date']
    ]
    test_idx = [date_map[pd.Timestamp(d)] for d in loc_test['target_end_date']]
    total_weeks = len(all_dates)

    rho_p, rt_p = get_hierarchical_params(
        loc_train['location_name'].iloc[0], pop
    )

    def sir_model(p, length):
      # p: [logit_S0, log_I0, log_gamma, logit_rho, log_phi, rt_knots x 7]
      S = expit(p[0]) * pop
      I = np.exp(p[1])
      gamma = np.exp(p[2])
      rho = expit(p[3])
      phi = np.exp(p[4])
      rt_knots = np.exp(p[5:])

      # Use non-uniform knots: focus more on mid-season
      knot_x = np.linspace(0, length - 1, len(rt_knots))
      rt_t = np.interp(np.arange(length), knot_x, rt_knots)

      h_out = np.zeros(length)
      dt = 0.25
      seeding = (
          0.01  # Small constant background seeding to prevent total die-out
      )

      for t in range(length):
        beta = rt_t[t] * gamma
        infections_this_week = 0
        for _ in range(int(1 / dt)):
          # SIR step with stability corrections
          lambda_t = (beta * I / pop) + (seeding / pop)
          new_inf = S * (1.0 - np.exp(-lambda_t * dt))
          new_inf = np.clip(new_inf, 0, S)

          S -= new_inf
          I = I + new_inf - (gamma * dt) * I
          I = max(1e-8, I)
          infections_this_week += new_inf
        h_out[t] = infections_this_week * rho
      return np.maximum(h_out, 1e-4), phi

    def ln_prior(p):
      # Hierarchical Hyperparameter Priors
      lp = norm.logpdf(p[0], 1.8, 0.4)  # Initial Susceptibility (~85%)
      lp += norm.logpdf(p[1], np.log(max(0.1, y_obs[0] * 2)), 2.0)  # I0
      lp += norm.logpdf(p[2], np.log(0.45), 0.1)  # Gamma
      lp += norm.logpdf(
          p[3], rho_p[0], rho_p[1]
      )  # Ascertainment (Hierarchical)
      lp += norm.logpdf(p[4], 2.0, 1.0)  # Phi dispersion
      lp += np.sum(norm.logpdf(p[5:], rt_p[0], rt_p[1]))  # Rt knots
      return lp if np.isfinite(lp) else -1e30

    def ln_likelihood(p):
      mu_full, phi = sir_model(p, total_weeks)
      mu = mu_full[train_idx]
      # NB likelihood: n=phi, p=phi/(phi+mu)
      prob = phi / (phi + mu)
      ll = np.sum(nbinom.logpmf(y_obs, phi, prob))
      return ll if np.isfinite(ll) else -1e30

    def ln_prob(p):
      lp = ln_prior(p)
      if not np.isfinite(lp):
        return -1e30
      ll = ln_likelihood(p)
      return lp + ll

    # 1. MAP Estimation for MCMC initialization
    ndim = 12
    p0_start = np.array(
        [1.8, np.log(max(1.0, y_obs[0])), np.log(0.45), rho_p[0], 2.0]
        + [rt_p[0]] * 7
    )
    res = minimize(
        lambda p: -ln_prob(p),
        p0_start,
        method='L-BFGS-B',
        bounds=[(0, 5), (-5, 15), (-1.5, -0.2), (-10, -1), (-1, 8)]
        + [(-1.5, 2.0)] * 7,
    )

    # 2. Ensemble MCMC Sampling
    if emcee and res.success:
      n_walkers = 32
      sampler = emcee.EnsembleSampler(n_walkers, ndim, ln_prob)
      # Start in a tight ball around MAP
      pos = res.x + 1e-3 * np.random.randn(n_walkers, ndim)
      sampler.run_mcmc(pos, 350, progress=False)
      flat_samples = sampler.get_chain(discard=200, flat=True)
    else:
      # Robust fallback: Perturbed MAP
      flat_samples = res.x + 0.08 * np.random.randn(400, ndim)

    # 3. Generate Trajectories & Quantiles
    n_sims = min(len(flat_samples), 400)
    chosen_samples = flat_samples[
        np.random.choice(len(flat_samples), n_sims, replace=False)
    ]

    preds = np.zeros((n_sims, len(test_idx)))
    for i in range(n_sims):
      mu_full, phi = sir_model(chosen_samples[i], total_weeks)
      mu_test = mu_full[test_idx]
      preds[i, :] = nbinom.rvs(phi, phi / (phi + mu_test))

    q_table = np.percentile(preds, [q * 100 for q in QUANTILES], axis=0).T
    loc_res = pd.DataFrame(
        q_table,
        index=loc_test.index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )

    # Enforce constraints: Monotonicity and Integers
    loc_res = loc_res.clip(lower=0).apply(
        lambda x: np.sort(x), axis=1, result_type='broadcast'
    )
    results.append(loc_res.round().astype(int))

  final_df = pd.concat(results).sort_index()
  return final_df


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
