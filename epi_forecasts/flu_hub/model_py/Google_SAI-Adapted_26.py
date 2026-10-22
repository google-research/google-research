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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-multiple-import,g-doc-return-or-yield,g-doc-args,broad-exception-caught,bare-except,missing-function-docstring,cell-var-from-loop

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
MODEL_NAME = 'Google_SAI-Adapted_26'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import nbinom, norm
from scipy.optimize import minimize
import warnings

# --- 1. Core Principles & Configuration ---
# Principle A: SIR model with unknown ascertainment, immunity, and splined Rt.
# Principle B: Across-season trends (from historical ILINet) used as priors for Rt.
# Principle C: Bayesian inference via custom Ensemble MCMC.
# Principle D: Scipy for splines/optimization.

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
KNOTS = 6  # Number of control points for Rt spline over the season
N_WALKERS = 32  # Number of MCMC walkers
N_STEPS = 300  # MCMC steps
BURN_IN = 150  # Burn-in period


# --- 2. Helper: Custom MCMC Sampler ---
def run_mcmc_ensemble(log_prob_fn, p0, n_steps, a=2.0):
  """Affine Invariant Ensemble Sampler (Goodman & Weare)."""
  n_walkers, ndim = p0.shape
  chain = np.zeros((n_steps, n_walkers, ndim))
  current_state = p0.copy()
  current_log_probs = np.full(n_walkers, -np.inf)

  # Init evaluation
  for i in range(n_walkers):
    current_log_probs[i] = log_prob_fn(current_state[i])

  for step in range(n_steps):
    for k in range(n_walkers):
      # Stretch move
      idxs = [x for x in range(n_walkers) if x != k]
      j = np.random.choice(idxs)
      xj, xk = current_state[j], current_state[k]

      u = np.random.rand()
      z = ((u * (a - 1) + 1) ** 2) / a
      y = xj + z * (xk - xj)

      lp_y = log_prob_fn(y)
      if np.isfinite(lp_y):
        log_ratio = (ndim - 1) * np.log(z) + lp_y - current_log_probs[k]
        if np.log(np.random.rand()) <= log_ratio:
          current_state[k] = y
          current_log_probs[k] = lp_y

    chain[step] = current_state.copy()
  return chain


# --- 3. Helper: Historical Priors ---
def get_historical_rt_prior(ilinet_df):
  """Derives a weekly Rt prior profile from 20 years of ILINet data.

  Approximates Rt via growth rate of ILI: Rt ~ 1 + d(log(ILI))/dt
  """
  try:
    # Use national data for robust shape
    df = ilinet_df[ilinet_df['region_type'] == 'National'].copy()
    # Aggregate by week
    weekly = df.groupby('week')['weighted_ili'].mean().sort_index()
    # Handle wraparound for smoothing (copy start to end)
    padded = np.concatenate(
        [weekly.values[-5:], weekly.values, weekly.values[:5]]
    )
    smoothed = np.convolve(padded, np.ones(5) / 5, mode='same')[5:-5]

    # Calculate log growth rate
    log_ili = np.log(np.maximum(smoothed, 1e-6))
    growth = np.diff(log_ili, prepend=log_ili[0])

    # Shift growth to align with "future" Rt impact (approx 1 week lag)
    # and scale to Rt (Rt=1 implies 0 growth).
    # Scaling factor: if generation time is ~1 week, Rt ~ 1 + growth
    rt_proxy = 1.0 + growth

    # Clip to reasonable physical bounds for flu
    rt_proxy = np.clip(rt_proxy, 0.5, 2.0)

    return dict(zip(weekly.index, rt_proxy))
  except Exception as e:
    # Fallback if ilinet is missing or malformed
    return {w: 1.0 for w in range(1, 54)}


# Pre-compute this once if possible, or inside the function
# We rely on 'ilinet' being in the global scope as per the challenge description
try:
  RT_HISTORY = get_historical_rt_prior(ilinet)
except:
  RT_HISTORY = {w: 1.0 for w in range(1, 54)}


# --- 4. Main Function ---
def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):

  # Merge for easier handling
  train = train_x.copy()
  train['y'] = train_y.values
  train['target_end_date'] = pd.to_datetime(train['target_end_date'])

  # Initialize output
  output = pd.DataFrame(
      index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  unique_locations = test_x['location'].unique()

  for loc in unique_locations:
    loc_mask = test_x['location'] == loc
    loc_test = test_x[loc_mask]
    ref_date = loc_test['reference_date'].iloc[0]

    # Define current season window (start Aug 1)
    season_year = ref_date.year if ref_date.month >= 8 else ref_date.year - 1
    season_start = pd.Timestamp(f'{season_year}-08-01')

    ref_date = pd.Timestamp(ref_date)  # manually added
    loc_train = train[
        (train['location'] == loc)
        & (train['target_end_date'] >= season_start)
        & (train['target_end_date'] <= ref_date)  # Include up to ref_date
    ].sort_values('target_end_date')

    # Fallback for insufficient data
    if len(loc_train) < 4:
      # Simple fallback: repeat last observation
      last_obs = loc_train['y'].iloc[-1] if not loc_train.empty else 0
      for col in output.columns:
        output.loc[loc_mask, col] = last_obs
      continue

    y_obs = loc_train['y'].values.astype(float)
    pop = loc_train['population'].iloc[0]

    # Time setup (in weeks)
    dates = loc_train['target_end_date']
    start_date = dates.min()
    t_obs = (dates - start_date).dt.days.values / 7.0

    # Forecast horizon
    horizons = loc_test['horizon'].values
    max_h = horizons.max()
    # Simulation duration: obs duration + max horizon + buffer
    t_sim_max = int(t_obs[-1] + max_h + 2)
    sim_steps = np.arange(t_sim_max + 1)

    # --- A. Spline Basis Setup ---
    # Knots spread evenly over the simulation period
    knot_locs = np.linspace(0, sim_steps[-1], KNOTS)

    # Precompute Basis Matrix B (T x K)
    # Rt(t) = B[t] @ weights
    B_matrix = np.zeros((len(sim_steps), KNOTS))
    eye = np.eye(KNOTS)
    for k in range(KNOTS):
      cs = CubicSpline(knot_locs, eye[k], bc_type='clamped')
      B_matrix[:, k] = cs(sim_steps)

    # --- B. Prior Construction ---
    # Map knot times to Week of Year to fetch historical prior
    knot_dates = [start_date + pd.Timedelta(weeks=k) for k in knot_locs]
    knot_weeks = [d.week for d in knot_dates]

    # Prior Mean for Rt Knots from History
    rt_prior_mean = np.array([RT_HISTORY.get(w, 1.0) for w in knot_weeks])

    # --- C. Model & Likelihood ---
    # Theta: [rt_knots (K), log_I0, logit_susecptibility, log_phi, log_alpha]
    # Fixed: Gamma = 1.0 (1 week infectious period -> Discrete generation step)

    def get_model_curves(theta):
      # 1. Unpack
      rt_coeffs = theta[:KNOTS]
      I0 = np.exp(theta[KNOTS])
      # Bounds check for overflow safety
      if I0 > pop:
        I0 = pop

      # Susceptible fraction (prior belief: ~70% susceptible start of season)
      susceptibility = 1.0 / (1.0 + np.exp(-theta[KNOTS + 1]))

      phi = np.exp(theta[KNOTS + 2])  # Dispersion
      alpha = np.exp(theta[KNOTS + 3])  # Reporting rate (Hosp / Infections)

      rt_curve = np.maximum(0, B_matrix @ rt_coeffs)

      # 2. Fast Discrete SIR (Generational)
      # S_{t+1} = S_t - I_{new}
      # I_{new} = (Rt * S_t / N) * I_t
      # I_{t+1} = I_{new} (assuming gamma=1, i.e., 1-week generation)

      S = pop * susceptibility
      I = I0

      infections = np.zeros(len(sim_steps))

      curr_S = S
      curr_I = I

      for t in range(len(sim_steps)):
        # Force of infection
        new_inf = rt_curve[t] * (curr_S / pop) * curr_I

        # Clipping
        if new_inf > curr_S:
          new_inf = curr_S

        infections[t] = new_inf

        # Update
        curr_S = curr_S - new_inf
        curr_I = new_inf  # Generation replacement

      return infections * alpha, phi

    def log_posterior(theta):
      # --- Priors ---
      lp = 0

      # 1. Rt Knots: Normal around historical mean
      rt = theta[:KNOTS]
      if np.any(rt < 0) or np.any(rt > 5):
        return -np.inf
      # Looser variance (0.25) allows adapting to this specific season
      lp += -0.5 * np.sum((rt - rt_prior_mean) ** 2 / 0.25)

      # 2. I0: LogNormal
      # Prior centered around observed start (approx)
      log_i0 = theta[KNOTS]
      obs_start = max(0.1, y_obs[0] / 0.05)  # Guess based on 5% reporting
      lp += -0.5 * ((log_i0 - np.log(obs_start)) / 2.0) ** 2

      # 3. Susceptibility: LogitNormal centered at 0.7 (logit ~ 0.85)
      # Weak prior to allow for different immunity levels
      logit_sus = theta[KNOTS + 1]
      lp += -0.5 * ((logit_sus - 0.85) / 1.0) ** 2

      # 4. Phi (Dispersion): LogNormal centered at 5
      log_phi = theta[KNOTS + 2]
      lp += -0.5 * ((log_phi - np.log(5)) / 1.0) ** 2

      # 5. Alpha (Reporting): LogNormal centered at 2% (approx flu hosp rate)
      log_alpha = theta[KNOTS + 3]
      lp += -0.5 * ((log_alpha - np.log(0.02)) / 0.5) ** 2

      if not np.isfinite(lp):
        return -np.inf

      # --- Likelihood ---
      expected_hosp, phi = get_model_curves(theta)

      # Map simulation time to observation indices
      # t_obs values are floats (0.0, 1.0, ...), convert to indices
      idx_map = np.round(t_obs).astype(int)
      # Clip safe
      idx_map = np.clip(idx_map, 0, len(expected_hosp) - 1)

      mu = expected_hosp[idx_map]
      mu = np.maximum(mu, 1e-6)  # Numerical stability

      # NegBinom LogLikelihood
      # n = phi, p = n / (n + mu)
      n = phi
      p = n / (n + mu)

      ll = nbinom.logpmf(y_obs, n, p).sum()

      return lp + ll

    # --- D. MAP Initialization ---
    # Start near the prior means
    p0 = np.zeros(KNOTS + 4)
    p0[:KNOTS] = rt_prior_mean
    p0[KNOTS] = np.log(max(1, y_obs[0] / 0.02))
    p0[KNOTS + 1] = 0.85
    p0[KNOTS + 2] = np.log(5.0)
    p0[KNOTS + 3] = np.log(0.02)

    # Bounded optimization
    bounds = [(0, 4)] * KNOTS + [(-5, 12), (-5, 5), (-5, 5), (-7, 0)]

    # Custom minimizer wrapper for log_posterior (negate for minimize)
    res = minimize(
        lambda x: -log_posterior(x), p0, bounds=bounds, method='L-BFGS-B'
    )
    best_theta = res.x

    # --- E. MCMC Sampling ---
    ndim = len(best_theta)
    # Initialize walkers in a tight ball around MAP
    pos = best_theta + 1e-3 * np.random.randn(N_WALKERS, ndim)

    chain = run_mcmc_ensemble(log_posterior, pos, N_STEPS)

    # --- F. Prediction Generation ---
    samples = chain[BURN_IN:].reshape(-1, ndim)
    # Thinning if too many samples
    if len(samples) > 500:
      idxs = np.random.choice(len(samples), 500, replace=False)
      samples = samples[idxs]

    pred_matrix = np.zeros((len(samples), len(sim_steps)))

    for i, theta in enumerate(samples):
      mu_curve, phi = get_model_curves(theta)
      # Generate predictive realizations (measurement noise)
      n = phi
      p = n / (n + mu_curve + 1e-9)
      pred_matrix[i, :] = nbinom.rvs(n, p)

    # --- G. Extract Quantiles for Horizons ---
    # Horizon target indices relative to sim_start
    # Last observation was at t_obs[-1]
    last_t = t_obs[-1]
    target_indices = np.round(last_t + horizons).astype(int)
    target_indices = np.clip(target_indices, 0, len(sim_steps) - 1)

    # Get quantiles at specific time steps
    q_values = np.quantile(pred_matrix[:, target_indices], QUANTILES, axis=0)
    # q_values shape: (n_quantiles, n_horizons)

    # Fill Output
    for h_idx, h in enumerate(horizons):
      mask = (test_x['location'] == loc) & (test_x['horizon'] == h)
      if mask.any():
        # Ensure monotonicity is naturally handled by np.quantile
        vals = q_values[:, h_idx]
        vals = np.maximum(0, vals).astype(int)
        output.loc[mask, output.columns] = vals

  return output.fillna(0).astype(int)


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
