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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,g-bad-todo,reimported,g-doc-return-or-yield,g-doc-args,broad-exception-caught

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
MODEL_NAME = 'Google_SAI-Adapted_21'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# TODO: Improve the time series forecasting model.

# [rewrite_cell]
# --- Core Principles of the Method ---
# 1. Branching Process Dynamics: The epidemic is modeled as a discrete-time branching process where new cases Y_t arise from Y_{t-1} via a Negative Binomial offspring distribution.
# 2. Analytic PGF Solution: We use SymPy to analytically prove that the sum of independent Negative Binomial random variables (offspring) follows a Negative Binomial distribution, justifying the likelihood Y_t ~ NB(Y_{t-1}*R, Y_{t-1}*k).
# 3. Mechanistic Immunity: The effective reproduction number R_t is dynamically scaled by the susceptible fraction S_t. To handle multi-year data, immunity depletion resets at the start of each season (drift).
# 4. Bayesian Inference: A hierarchical PyMC model infers the time-varying reproduction number (decomposed into historical seasonality and a random walk deviation) and static parameters using the full available history.

import logging
import warnings
import numpy as np
import pandas as pd
import sympy as sp
import pymc as pm
import pytensor.tensor as pt

# Suppress PyMC output for cleaner logs
logger = logging.getLogger('pymc')
logger.setLevel(logging.ERROR)


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Fits a vectorized Hierarchical Bayesian branching process model and generates quantile forecasts.

  Uses SymPy for analytic validation of the likelihood and PyMC for inference.
  """

  # -------------------------------------------------------------------------
  # 1. Analytic Solution with SymPy
  # -------------------------------------------------------------------------
  try:
    R_sym, k_sym, Y_prev_sym = sp.symbols(
        'R k Y_prev', real=True, positive=True
    )
    z_var = sp.symbols('z')

    # NB parametrization: p = k / (k + R)
    p_sym = k_sym / (k_sym + R_sym)

    # PGF of single offspring Z: G_Z(z) = [p / (1 - (1-p)z)]^k
    pgf_single = (p_sym / (1 - (1 - p_sym) * z_var)) ** k_sym

    # PGF of sum of Y_prev copies: G_sum(z) = [G_Z(z)]^Y_prev
    pgf_sum = pgf_single**Y_prev_sym

    # Target distribution: NB(mean=Y_prev*R, shape=Y_prev*k)
    A_target = Y_prev_sym * k_sym
    p_target = p_sym
    pgf_target = (p_target / (1 - (1 - p_target) * z_var)) ** A_target

    if sp.simplify(pgf_sum - pgf_target) != 0:
      warnings.warn(
          'SymPy derivation failed: Sum of NB is not NB with hypothesized'
          ' parameters.'
      )

  except Exception as e:
    warnings.warn(f'SymPy validation skipped due to error: {e}')

  # -------------------------------------------------------------------------
  # 2. Data Preparation
  # -------------------------------------------------------------------------
  required_locations = sorted(test_x['location'].unique())
  loc_map = {loc: i for i, loc in enumerate(required_locations)}
  n_locs = len(required_locations)

  # Population Vector
  pop_df = (
      test_x[['location', 'population']]
      .drop_duplicates(subset=['location'])
      .set_index('location')
  )
  pop_vec = np.array(
      [pop_df.loc[loc, 'population'] for loc in required_locations]
  )
  median_pop = np.median(pop_vec)
  pop_scale = pop_vec / median_pop

  # Merge and Pivot Training Data
  df = train_x.copy()
  df[TARGET_STR] = train_y
  df = df[df['location'].isin(required_locations)]

  if df.empty:
    # Return zeros if no training data
    q_cols = [f'quantile_{q}' for q in globals().get('QUANTILES', [0.5])]
    return pd.DataFrame(0, index=test_x.index, columns=q_cols)

  # Use Full History: Determine grid from min date to max date
  min_date = df['target_end_date'].min()
  max_date = df['target_end_date'].max()

  # Ensure min_date aligns to Saturday
  weekday = min_date.weekday()
  if weekday != 5:
    min_date = min_date + pd.Timedelta(days=(5 - weekday))

  date_grid = pd.date_range(min_date, max_date, freq='W-SAT')
  T = len(date_grid)

  pivot_df = df.pivot(
      index='target_end_date', columns='location', values=TARGET_STR
  )
  pivot_df = pivot_df.reindex(index=date_grid)

  # Imputation: Fill gaps with interpolation, missing edges with 0
  pivot_df = pivot_df.interpolate(
      method='linear', limit_direction='both'
  ).fillna(0)
  Y_obs = np.maximum(pivot_df[required_locations].values, 0)  # Shape [T, N]

  # Pre-calculate Season-Resetting Cumulative Sums for Immunity
  # We assume immunity resets (or drift occurs) around July 1st (Week 27)
  season_ids = []
  for d in date_grid:
    y_val = d.year
    w_val = d.isocalendar().week
    # If week >= 27 (approx July), treat as next season ID
    s_id = y_val + 1 if w_val >= 27 else y_val
    season_ids.append(s_id)
  season_ids = np.array(season_ids)

  # Calculate cumulative sum for each location, resetting when season_id changes
  Y_cum_reset = np.zeros_like(Y_obs)
  for i in range(n_locs):
    temp_s = pd.Series(Y_obs[:, i], index=season_ids)
    Y_cum_reset[:, i] = temp_s.groupby(level=0).cumsum().values

  # -------------------------------------------------------------------------
  # 3. Historical Seasonality Prior
  # -------------------------------------------------------------------------
  seasonal_profile = np.zeros(54)
  try:
    hist_data = None
    if 'ilinet' in globals():
      hist_data = globals()['ilinet']
    elif 'ilinet_hhs' in globals():
      hist_data = globals()['ilinet_hhs']

    if hist_data is not None and not hist_data.empty:
      nat_ili = hist_data.groupby('week')['weighted_ili'].median().reset_index()
      # Sort order: Weeks 30-53 come before 1-29
      nat_ili['sort_order'] = nat_ili['week'].apply(
          lambda w: w + 53 if w < 30 else w
      )
      nat_ili = nat_ili.sort_values('sort_order')

      # Smooth and Calculate Growth
      nat_ili['smooth_ili'] = (
          nat_ili['weighted_ili']
          .rolling(window=3, center=True, min_periods=1)
          .mean()
      )
      growth = nat_ili['smooth_ili'].pct_change().fillna(0) + 1.0
      growth = growth.clip(0.8, 1.5)

      for _, row in nat_ili.iterrows():
        w = int(row['week'])
        if 0 <= w <= 53:
          seasonal_profile[w] = np.log(
              growth[row.name]
          )  # row.name matches index

      if seasonal_profile[53] == 0 and seasonal_profile[52] != 0:
        seasonal_profile[53] = seasonal_profile[52]
  except Exception:
    pass

  # -------------------------------------------------------------------------
  # 4. PyMC Inference (Full History)
  # -------------------------------------------------------------------------
  trace = None
  try:
    if T > 4:
      with pm.Model() as model:
        # -- Global Parameters --
        rho = pm.Beta('rho', alpha=5, beta=95)  # Reporting rate ~5%
        k = pm.Gamma('k', alpha=3, beta=0.5)  # Overdispersion

        # Seeding/Immigration
        eps_base = pm.Exponential('eps_base', 1.0)
        epsilon = eps_base * pop_scale[None, :]

        # -- Latent Dynamics (R_t) --
        sigma_rw = pm.HalfNormal('sigma_rw', 0.05)

        # RW over full history T
        rw_dev = pm.GaussianRandomWalk(
            'rw_dev',
            sigma=sigma_rw,
            init_dist=pm.Normal.dist(0, 0.1),
            shape=(T, n_locs),
        )

        # Map full history dates to seasonal profile
        week_indices = np.array([d.isocalendar().week for d in date_grid])
        safe_week_indices = np.clip(week_indices, 0, 53)
        prior_log_R = seasonal_profile[safe_week_indices][:, None]

        log_Rt = prior_log_R + rw_dev
        Rt_base = pm.math.exp(log_Rt)

        # -- Mechanistic Immunity (Resetting) --
        # Use pre-calculated resetting sums for observed period
        Y_cum_data = pm.Data('Y_cum', Y_cum_reset[:-1])

        depletion = (Y_cum_data / rho) / pop_vec[None, :]
        s_frac = pt.clip(1.0 - depletion, 0.05, 1.0)

        Rt_eff = Rt_base[:-1] * s_frac

        # -- Likelihood --
        Y_prev = pm.Data('Y_prev', Y_obs[:-1])
        Y_curr = pm.Data('Y_curr', Y_obs[1:])

        parent_pool = Y_prev + epsilon
        mu_t = parent_pool * Rt_eff
        alpha_t = parent_pool * k + 1e-6

        pm.NegativeBinomial('obs', mu=mu_t, alpha=alpha_t, observed=Y_curr)

        # -- Fit --
        approx = pm.fit(
            n=20000,
            method='advi',
            obj_optimizer=pm.adam(learning_rate=0.005),
            progressbar=False,
        )
        trace = approx.sample(draws=300)

  except Exception as e:
    print(f'PyMC modeling failed: {e}')
    trace = None

  # -------------------------------------------------------------------------
  # 5. Forecast Simulation
  # -------------------------------------------------------------------------
  horizons = sorted(test_x['horizon'].unique())
  max_h = int(max(horizons)) if horizons else 0
  sim_steps = max_h + 1
  n_samples = 300

  preds = np.zeros((n_samples, sim_steps, n_locs))

  if trace is not None:
    post = trace.posterior
    # Resample indices
    idxs = np.random.choice(300, n_samples, replace=True)

    # Extract last state of RW
    rw_last = post['rw_dev'].values[0][idxs, -1, :]  # [Draws, N]
    rho_s = post['rho'].values[0][idxs]
    k_s = post['k'].values[0][idxs]
    eps_base_s = post['eps_base'].values[0][idxs]
    eps_s = eps_base_s[:, None] * pop_scale[None, :]

    # Initialize simulation
    curr_Y = np.tile(Y_obs[-1], (n_samples, 1))
    # Initialize cumulative sum with last observed reset-cumulative
    curr_cum = np.tile(Y_cum_reset[-1], (n_samples, 1))

    current_season_id = season_ids[-1]

    for h in range(sim_steps):
      # Check for season change in forecast horizon
      fut_date = max_date + pd.Timedelta(weeks=h + 1)
      w = fut_date.isocalendar().week
      fut_season_id = fut_date.year + 1 if w >= 27 else fut_date.year

      # If season changes, reset cumulative count (simple approximation)
      # This logic assumes all samples follow the same calendar
      if fut_season_id != current_season_id:
        curr_cum = np.zeros_like(curr_cum)
        current_season_id = fut_season_id

      # Base Rt (Projected as martingale from last RW state)
      base_r_log = seasonal_profile[min(w, 53)]
      rt_proj = np.exp(base_r_log + rw_last)

      # Immunity
      depletion = (curr_cum / rho_s[:, None]) / pop_vec[None, :]
      s_frac = np.maximum(0.05, 1.0 - depletion)
      rt_eff = rt_proj * s_frac

      # Branching
      parent = curr_Y + eps_s
      mu = parent * rt_eff
      alpha = parent * k_s[:, None] + 1e-6

      # NB Draw
      p_nb = alpha / (alpha + mu + 1e-9)
      next_y = np.random.negative_binomial(alpha, p_nb)
      preds[:, h, :] = next_y

      # Update
      curr_Y = next_y
      curr_cum += next_y

  else:
    # Fallback
    for h in range(sim_steps):
      preds[:, h, :] = 0

  # -------------------------------------------------------------------------
  # 6. Formatting Output
  # -------------------------------------------------------------------------
  results = []
  q_levels = globals().get('QUANTILES', [0.01, 0.5, 0.99])
  q_cols = [f'quantile_{q}' for q in q_levels]

  for idx, row in test_x.iterrows():
    h = int(row['horizon'])
    loc_id = row['location']
    loc_idx = loc_map.get(loc_id)

    if loc_idx is not None and 0 <= h < sim_steps:
      samples = preds[:, h, loc_idx]
      q_vals = np.quantile(samples, q_levels)
    else:
      q_vals = np.zeros(len(q_levels))

    results.append(q_vals)

  # Ensure monotonicity
  results = np.sort(np.array(results), axis=1)
  return pd.DataFrame(results, columns=q_cols, index=test_x.index)


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
