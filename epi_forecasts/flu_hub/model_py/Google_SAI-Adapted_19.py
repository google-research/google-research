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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-multiple-import,missing-function-docstring,unused-argument

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
MODEL_NAME = 'Google_SAI-Adapted_19'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
from scipy.special import expit, logit, gammaln


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # --- Config ---
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
  LAMBDA = 120.0  # Concentration
  N_ITER, BURN_IN = 1000, 500
  T_HIST = 30  # Increased history
  SCALE_VAL = 6000  # Population scale denominator

  locs = sorted(test_x['location'].unique())
  n_locs = len(locs)

  # --- Prepare Data ---
  df_all = pd.concat([train_x, train_y.rename('target')], axis=1)
  df_all['target_end_date'] = pd.to_datetime(df_all['target_end_date'])
  max_date = df_all['target_end_date'].max()
  ref_dates = pd.date_range(end=max_date, periods=T_HIST, freq='W-SAT')

  y_obs = np.zeros((n_locs, T_HIST))
  pops = np.zeros(n_locs)
  for i, loc in enumerate(locs):
    ldf = df_all[df_all['location'] == loc].sort_values('target_end_date')
    pops[i] = ldf['population'].iloc[-1]
    vals = ldf[ldf['target_end_date'].isin(ref_dates)]['target'].values
    padded = np.zeros(T_HIST)
    padded[-len(vals) :] = vals
    # Scale to admissions per (pop/SCALE_VAL)
    y_obs[i] = np.clip(padded / (pops[i] / SCALE_VAL), 1e-5, 0.999)

  # Pre-calc logs for Beta likelihood
  log_y = np.log(y_obs)
  log_1_y = np.log(1 - y_obs)

  def log_beta_pdf_vec(y_val, pi_val, ly, l1y):
    a = np.clip(pi_val * LAMBDA, 1e-3, 1e6)
    b = np.clip((1 - pi_val) * LAMBDA, 1e-3, 1e6)
    return (
        (a - 1) * ly + (b - 1) * l1y + gammaln(a + b) - gammaln(a) - gammaln(b)
    )

  def solve_sir_rk4(beta, rho, I0, steps):
    gamma = beta * rho
    I_path = np.zeros((len(beta), steps))
    S, I = np.full(len(beta), 0.9), I0.copy()
    I_path[:, 0] = I
    for t in range(1, steps):
      k1s = -beta * S * I
      k1i = beta * S * I - gamma * I
      k2s = -beta * (S + 0.5 * k1s) * (I + 0.5 * k1i)
      k2i = beta * (S + 0.5 * k1s) * (I + 0.5 * k1i) - gamma * (I + 0.5 * k1i)
      k3s = -beta * (S + 0.5 * k2s) * (I + 0.5 * k2i)
      k3i = beta * (S + 0.5 * k2s) * (I + 0.5 * k2i) - gamma * (I + 0.5 * k2i)
      k4s = -beta * (S + k3s) * (I + k3i)
      k4i = beta * (S + k3s) * (I + k3i) - gamma * (I + k3i)
      S = np.clip(S + (k1s + 2 * k2s + 2 * k3s + k4s) / 6.0, 0, 1)
      I = np.clip(I + (k1i + 2 * k2i + 2 * k3i + k4i) / 6.0, 1e-9, 1)
      I_path[:, t] = I
    return I_path

  # --- Initialization ---
  beta = np.random.uniform(0.8, 1.2, n_locs)
  rho = np.random.uniform(0.7, 0.9, n_locs)
  I0 = np.full(n_locs, 1e-4)
  mu = np.zeros(T_HIST)
  delta = np.zeros((n_locs, T_HIST))
  alpha = np.full(n_locs, 0.8)
  tau_delta = np.full(n_locs, 100.0)
  tau_mu = 200.0

  # Fixed hyper-priors
  mu_alpha_logit, tau_alpha = logit(0.9), 10.0
  a_delta, b_delta = 2.0, 0.01

  samples = []

  # --- MCMC Loop ---
  for it in range(N_ITER):
    # 1. Update SIR Params
    p_beta = np.clip(beta + np.random.normal(0, 0.04, n_locs), 0.2, 2.0)
    p_rho = np.clip(rho + np.random.normal(0, 0.02, n_locs), 0.4, 0.99)
    I_path = solve_sir_rk4(beta, rho, I0, T_HIST)
    p_I_path = solve_sir_rk4(p_beta, p_rho, I0, T_HIST)

    lI, p_lI = logit(I_path), logit(p_I_path)

    for j in range(n_locs):
      # delta_T constraint enforcement
      curr_delta = delta[j].copy()
      curr_delta[-1] = -lI[j, -1]
      prop_delta = delta[j].copy()
      prop_delta[-1] = -p_lI[j, -1]

      curr_pi = expit(np.clip(lI[j] + mu + curr_delta, -12, 12))
      prop_pi = expit(np.clip(p_lI[j] + mu + prop_delta, -12, 12))

      log_acc = np.sum(
          log_beta_pdf_vec(y_obs[j], prop_pi, log_y[j], log_1_y[j])
      ) - np.sum(log_beta_pdf_vec(y_obs[j], curr_pi, log_y[j], log_1_y[j]))
      # SIR priors
      log_acc += -0.5 * (
          (p_beta[j] - 1.0) ** 2 / 0.25**2 - (beta[j] - 1.0) ** 2 / 0.25**2
      )

      if log_acc > np.log(np.random.rand()):
        beta[j], rho[j], delta[j] = p_beta[j], p_rho[j], prop_delta
        lI[j] = p_lI[j]
      else:
        delta[j] = curr_delta

    # 2. Update Common Discrepancy mu (Reverse RW)
    for t in range(T_HIST):
      p_mu_val = mu[t] + np.random.normal(0, 0.05)
      # Prior mu_t | mu_{t+1} and mu_{t-1} | mu_t
      m_next = mu[t + 1] if t < T_HIST - 1 else 0
      m_prev = mu[t - 1] if t > 0 else mu[t + 1] if t < T_HIST - 1 else 0

      curr_pi = expit(np.clip(lI[:, t] + mu[t] + delta[:, t], -12, 12))
      prop_pi = expit(np.clip(lI[:, t] + p_mu_val + delta[:, t], -12, 12))

      log_acc = np.sum(
          log_beta_pdf_vec(y_obs[:, t], prop_pi, log_y[:, t], log_1_y[:, t])
      ) - np.sum(
          log_beta_pdf_vec(y_obs[:, t], curr_pi, log_y[:, t], log_1_y[:, t])
      )
      log_acc -= (
          0.5 * tau_mu * ((p_mu_val - m_next) ** 2 - (mu[t] - m_next) ** 2)
      )
      if t > 0:
        log_acc -= (
            0.5 * tau_mu * ((m_prev - p_mu_val) ** 2 - (m_prev - mu[t]) ** 2)
        )

      if log_acc > np.log(np.random.rand()):
        mu[t] = p_mu_val

    # 3. Update Specific Discrepancy delta (Reverse AR RW) - Skip delta_T
    for t in range(T_HIST - 1):
      p_d = delta[:, t] + np.random.normal(0, 0.08, n_locs)
      d_next = delta[:, t + 1]
      d_prev = delta[:, t - 1] if t > 0 else delta[:, t + 1]  # simplified edge

      curr_pi = expit(np.clip(lI[:, t] + mu[t] + delta[:, t], -12, 12))
      prop_pi = expit(np.clip(lI[:, t] + mu[t] + p_d, -12, 12))

      log_acc = log_beta_pdf_vec(
          y_obs[:, t], prop_pi, log_y[:, t], log_1_y[:, t]
      ) - log_beta_pdf_vec(y_obs[:, t], curr_pi, log_y[:, t], log_1_y[:, t])
      log_acc -= (
          0.5
          * tau_delta
          * ((p_d - alpha * d_next) ** 2 - (delta[:, t] - alpha * d_next) ** 2)
      )

      acc_mask = log_acc > np.log(np.random.rand(n_locs))
      delta[acc_mask, t] = p_d[acc_mask]

    # 4. Hierarchical Updates
    if it % 2 == 0:
      # Alpha update
      p_al_logit = logit(alpha) + np.random.normal(0, 0.1, n_locs)
      p_al_logit = np.clip(p_al_logit, logit(0.02), logit(0.98))
      p_al = expit(p_al_logit)
      for j in range(n_locs):
        curr_err = np.sum((delta[j, :-1] - alpha[j] * delta[j, 1:]) ** 2)
        prop_err = np.sum((delta[j, :-1] - p_al[j] * delta[j, 1:]) ** 2)
        log_acc = -0.5 * tau_delta[j] * (
            prop_err - curr_err
        ) - 0.5 * tau_alpha * (
            (p_al_logit[j] - mu_alpha_logit) ** 2
            - (logit(alpha[j]) - mu_alpha_logit) ** 2
        )
        if log_acc > np.log(np.random.rand()):
          alpha[j] = p_al[j]
      # Tau update (Gibbs)
      for j in range(n_locs):
        err_sq = np.sum((delta[j, :-1] - alpha[j] * delta[j, 1:]) ** 2)
        tau_delta[j] = np.random.gamma(
            a_delta + (T_HIST - 1) / 2, 1.0 / (b_delta + 0.5 * err_sq)
        )

    if it >= BURN_IN and it % 2 == 0:
      samples.append({
          'b': beta.copy(),
          'r': rho.copy(),
          'mu': mu[-1],
          'd': delta[:, -1],
          'a': alpha.copy(),
          'td': tau_delta.copy(),
          'tm': tau_mu,
      })

  # --- Prediction ---
  final_rows = []
  H_MAX = 4
  for j, loc in enumerate(locs):
    loc_test = test_x[test_x['location'] == loc]
    pop_scale = pops[j] / SCALE_VAL

    trajs = np.zeros((len(samples), H_MAX + 2))
    for i, s in enumerate(samples):
      # SIR Forward
      If = solve_sir_rk4(
          np.array([s['b'][j]]),
          np.array([s['r'][j]]),
          np.array([expit(lI[j, -1])]),
          H_MAX + 2,
      )[0]
      c_mu, c_d = s['mu'], s['d'][j]
      for h in range(H_MAX + 2):
        if h > 0:
          c_mu += np.random.normal(0, 1.0 / np.sqrt(s['tm']))
          c_d = s['a'][j] * c_d + np.random.normal(0, 1.0 / np.sqrt(s['td'][j]))
        pi = expit(np.clip(logit(If[h]) + c_mu + c_d, -12, 12))
        trajs[i, h] = np.random.beta(pi * LAMBDA, (1 - pi) * LAMBDA) * pop_scale

    for _, row in loc_test.iterrows():
      h = int(row['horizon'])
      idx = h + 1  # horizon -1 is index 0
      vals = trajs[:, min(max(0, idx), H_MAX + 1)]
      q_vals = np.sort(np.quantile(vals, QUANTILES))
      final_rows.append(
          pd.Series(
              q_vals, index=[f'quantile_{q}' for q in QUANTILES], name=row.name
          )
      )

  return pd.DataFrame(final_rows).reindex(test_x.index).fillna(0).astype(int)


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
