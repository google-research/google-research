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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name

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
MODEL_NAME = 'Google_SAI-Adapted_2'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
import warnings

# Assume PyMC and its dependencies are available in the environment.
import pymc as pm
import pytensor.tensor as at
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression  # Added for refined ILINet scaling

np.seterr(
    over='raise'
)  # Ensure numerical errors are caught as specified globally.

# The global QUANTILES list is defined in the helper functions section and is accessible.

# IMPLEMENTATION PLAN.
# ## Core Principles Checklist
# 1. The PROF routines perform a deterministic fit of our compartmental SIR[H]2 model to daily hospitalization incidence profiles: My code implements the `SIRH2Model` class which uses `pm.ode.DifferentialEquation` to deterministically solve the SIR[H]2 ODE system for daily values during the MCMC fit. The likelihood function then compares the *weekly aggregated daily model incidence* with the *actual weekly observed data*. For simulations, the ODE is run efficiently only for the forecast horizon, starting from the last inferred state of the training period, ensuring parameter uncertainty is sampled.
# 2. The model includes a hospitalization compartment which is split into two sub-compartments: The `_sirh2_ode` function defines state variables `H1` and `H2` representing these two sub-compartments, with transitions `rho*I` to `H1`, `delta1*H1` to `H2`, and `delta2*H2` to `R`, precisely as described.
# 3. The transmission-rate coefficient (beta) is a flexible, time-dependent function composed of two or more arc-tangents: The `beta_t_closure` function within `fit_mcmc` computes `log_beta_t_unconstrained` from a base log-beta and two arc-tangent terms, then transforms it to `beta_t = at.exp(log_beta_t_unconstrained)` to ensure positivity and numerical stability, while its parameters (log_beta_base, slopes, k, t0) are inferred by MCMC.

# ## Step-by-Step Logic
# 1.  **PyMC Installation Handling**: As is.
# 2.  **Data Preprocessing and Augmentation (Revised Transformation using Robust Regression for Scaling Factor):**
#     a.  Convert `target_end_date` in `train_x` to `pd.Timestamp` for consistent date operations.
#     b.  Merge `train_x` and `train_y` into `train_df`.
#     c.  Preprocess `ilinet_state`: filter for 'States', map 'region' to 'location' (FIPS codes using `locations`), and create a consistent `target_end_date` as `pd.Timestamp`.
#     d.  Merge `ilinet_state_processed` with `locations` to explicitly add the `population` column, and drop rows with missing or non-positive populations for ILINet.
#     e.  **Revised Learn Location-Specific Transformation**:
#         i.  Identify the common date range where both `train_df` (actual target `Total Influenza Admissions`) and `ilinet_state_processed` (`unweighted_ili` and `population`) are available, limited to dates before '2022-10-15' as ILI data is only available up to this point.
#         ii. Filter both `train_df` and `ilinet_state_processed` to this overlapping range.
#         iii. Merge the filtered dataframes on `location` and `target_end_date` into `merged_overlap_data`.
#         iv. Calculate per-capita target and ILI rate. Filter `merged_overlap_data` to only include positive population, positive `unweighted_ili` and positive `Total Influenza Admissions` to calculate meaningful ratios for regression.
#         v.  **Calculate a `global_regression_factor`**: Fit a `LinearRegression(fit_intercept=False)` model to `target_per_capita` vs `unweighted_ili` on all valid `merged_overlap_data`. If the coefficient is negative, zero, or the data is insufficient, calculate a global median ratio from the valid `target_per_capita / ili_rate`. If that fails, use `1e-6`. Ensure the factor is always positive.
#         vi. Initialize an empty dictionary `location_mapping_functions` to store mapping lambda functions.
#         vii. For each unique `loc_id` in `ilinet_state_processed['location'].unique()`:
#             1.  Filter `merged_overlap_data` for the current `loc_id` to get `loc_overlap_data_valid`, ensuring `population > 0`, `unweighted_ili > 0`, and `target_per_capita > 0`.
#             2.  Attempt to fit a `LinearRegression(fit_intercept=False)` model for `target_per_capita` (Y) vs `unweighted_ili` (X).
#             3.  If the local regression is unreliable (e.g., fewer than 5 valid data points, or a non-positive coefficient), then calculate a local median ratio `target_per_capita / ili_rate`. If that also fails, `chosen_factor = global_regression_factor`.
#             4.  Otherwise, `chosen_factor = local_regression_coefficient`.
#             5.  Ensure `chosen_factor` is always positive; if zero or negative, default to a small positive constant (`1e-6`) to avoid `NaN` or `inf` during multiplication.
#             6.  Store a lambda function `lambda x, f=chosen_factor: x * f` in `location_mapping_functions[loc_id]`.
#     f.  **Apply Location-Specific Mapping to ILINet History (with Cutoff Enforcement)**:
#         i.  Iterate through each location in `ilinet_state_processed`.
#         ii. Retrieve the appropriate mapping lambda function from `location_mapping_functions`, explicitly defaulting to a lambda using `global_regression_factor` (or `1e-6`) if no specific mapping is found for that location.
#         iii. Apply the mapping to `unweighted_ili` to create `synthetic_admissions_per_capita`.
#         iv. Multiply `synthetic_admissions_per_capita` by the location's `population` to get `synthetic_admissions`, ensuring non-negativity and integer rounding, and setting to 0 if `population <= 0`.
#         v.  **Crucially, filter out synthetic data for dates ON OR AFTER '2022-10-15' (the ILI data cutoff) AND filter out dates already present in the original `train_df` for that location.**
#         vi. Format the synthetic data to match `train_df`'s structure.
#     g.  Concatenate the original `train_df` with the new formatted `synthetic_admissions` data into `augmented_train_df`.
#     h.  Sort `augmented_train_df` by `location` and `target_end_date`.
# 3.  **SIRH2Model Class Refinements (Tighter Priors, More Chains, Robust Initial `I0`)**:
#     a.  **`_sirh2_ode` function**: As is.
#     b.  **Transmission Rate (`beta_t`) Prior and Transformation**:
#         *   `log_beta_base`: `pm.Normal(..., sigma=0.3)` (from 0.5).
#         *   `slope1`, `slope2`: `pm.Normal(..., sigma=0.5)` (from 0.7).
#         *   `k1`, `k2`: `pm.HalfNormal(..., sigma=0.5)` (from 1.0).
#     c.  **Initial Conditions Priors (Refined `I0` Prior)**:
#         *   `I0`: `pm.LogNormal(..., sigma=0.7)` (from 1.0). Its `mu` will be calculated from `initial_admissions_estimate`, which is the median of the first 3 non-zero `Total Influenza Admissions` values from `loc_train_data` (or the first non-zero, or `loc_train_data.iloc[0]` if no non-zeros, or 0 if data is empty) divided by 7 days and prior mean of `rho` (simplified to 0.005 directly).
#         *   `H1_0`, `H2_0`: `pm.HalfNormal(..., sigma=0.7)` (from 1.0).
#     d.  **Epidemiological Parameter Priors**:
#         *   `gamma`: `pm.LogNormal(..., sigma=0.3)` (from 0.5).
#         *   `rho`: `pm.LogNormal(..., sigma=0.5)` (from 0.7).
#         *   `delta1`, `delta2`: `pm.LogNormal(..., sigma=0.5)` (from 0.7).
#     e.  **MCMC Sampling**: `pm.sample` is configured with `chains=4` (from 2). `draws=1500`, `tune=1500` (as is), `random_seed=42`, `cores=pm.cpu_count()`, `target_accept=0.85`, `max_treedepth=12`, and `progressbar=False`.
#     f.  **Likelihood Strategy**:
#         *   `alpha`: `pm.HalfNormal(..., sigma=1.0)` (from 1.5).
#     g.  Population/Robustness Checks, `simulate_forecasts`: As is.
# 4.  **Forecasting Loop (`fit_and_predict_fn`)**:
#     a.  Warnings: As is.
#     b.  Grouping: As is.
#     c.  `train_end_date` and `loc_train_data` filtering: As is.
#     d.  **Conditional Horizon Handling & MCMC Failure Fallback**:
#         i.  If `row['horizon'] == -1`: As is (retrieve actual observed value).
#         ii. Otherwise (for `horizon=0, 1, 2, 3`):
#             1.  If MCMC fitting is successful (`model.trace` is not None):
#                 *   Call `model.simulate_forecasts` for the `forecast_target_date`.
#                 *   Calculate `np.percentile`, `np.round`, `astype(int)`, and apply `np.maximum.accumulate` to ensure monotonicity and non-negativity.
#             2.  **MCMC Failed Fallback (IMPROVED):** If MCMC failed or was skipped (`should_fit_mcmc` is False or `model.trace` is None):
#                 *   Determine `last_known_admissions`: find the last non-zero `Total Influenza Admissions` in `loc_train_data` up to `train_end_date`. If no non-zero, use the last value, otherwise 0.
#                 *   If `last_known_admissions > 0`:
#                     *   Generate `n_simulations` samples from a `np.random.negative_binomial` distribution, with `mu = last_known_admissions` and a fixed `alpha_nb_fallback = 2.0`.
#                     *   Calculate `np.percentile` from these samples.
#                     *   Apply `np.round`, `astype(int)`, `np.maximum(0, ...)` (for non-negativity), and `np.maximum.accumulate` (for monotonicity).
#                 *   If `last_known_admissions == 0`: `quant_preds` will be all zeros.
#     e.  Predictions stored: As is.
# 5.  **Output Format**: As is.


# Helper for I0 prior
def get_initial_admissions_estimate(
    series, n_first_vals = 3
):
  """Estimates initial admissions by taking the median of the first `n_first_vals`

  non-zero observations, or the first observed value if fewer non-zeros,
  or 0 if all are zero/empty.
  """
  if series.empty:
    return 0.0

  non_zero_vals = series[series > 0]
  if not non_zero_vals.empty:
    return non_zero_vals.head(n_first_vals).median()
  else:
    return float(series.iloc[0])  # If all are zero, return the first zero.


class SIRH2Model:
  """Implements the SIR[H]2 model with MCMC inference for parameters and

  stochastic simulations for forecasting, as specified in the instructions.
  """

  def __init__(self, n_simulations = 2000):
    self.trace = None
    self.n_simulations = n_simulations
    self.daily_time_points_train = None
    self.observed_weekly_admissions = None
    self.N_pop_val = None
    self.train_start_date = None
    self.train_end_date = None
    self.train_full_duration_days = None

  @staticmethod
  def _sirh2_ode(t, y, N, beta_t_func, gamma, rho, delta1, delta2):
    """SIR[H]2 ODE system.

    y = [S, I, R, H1, H2] N = total population beta_t_func = function to get
    beta at time t gamma = recovery rate from I rho = hospitalization rate from
    I delta1 = rate from H1 to H2 delta2 = rate from H2 to R
    """
    S, I, R, H1, H2 = y

    # Ensure S, I, R, H1, H2 are non-negative for ODE stability
    S = at.maximum(0.0, S)
    I = at.maximum(0.0, I)
    R = at.maximum(0.0, R)
    H1 = at.maximum(0.0, H1)
    H2 = at.maximum(0.0, H2)

    beta_t = beta_t_func(t)

    d_S = -beta_t * S * I / N
    d_I = beta_t * S * I / N - (gamma + rho) * I
    d_R = gamma * I + delta2 * H2
    d_H1 = rho * I - delta1 * H1
    d_H2 = delta1 * H1 - delta2 * H2

    return [d_S, d_I, d_R, d_H1, d_H2]

  def fit_mcmc(
      self,
      train_data,
      target_col = 'Total Influenza Admissions',
  ):
    """Infers SIR[H]2 model parameters using MCMC."""
    # Ensure target_end_date is Timestamp for min/max and consistent operations
    train_data['target_end_date'] = pd.to_datetime(
        train_data['target_end_date']
    )
    train_data = train_data.sort_values('target_end_date').reset_index(
        drop=True
    )

    if (
        train_data.empty
        or train_data[target_col].sum() == 0
        or train_data[target_col].nunique() < 2
    ):
      warnings.warn(
          'Empty, zero-sum, or constant training data, skipping MCMC fit.'
      )
      self.trace = None
      return

    self.train_end_date = train_data[
        'target_end_date'
    ].max()  # Last Saturday of training data
    self.train_start_date = train_data['target_end_date'].min() - pd.Timedelta(
        days=6
    )  # Sunday of first training week

    # Prepare observed weekly data directly
    self.observed_weekly_admissions = train_data.set_index('target_end_date')[
        target_col
    ].sort_index()

    # Define daily time points for ODE solution, spanning the full training period
    self.train_full_duration_days = (
        self.train_end_date - self.train_start_date
    ).days + 1
    self.daily_time_points_train = np.arange(self.train_full_duration_days)

    # Ensure N_pop_val is a float
    self.N_pop_val = float(
        train_data['population'].iloc[0]
    )  # Assume population is constant for a location

    # If population is not positive, do not fit the model.
    if self.N_pop_val <= 0:
      warnings.warn(
          f'Population for location is non-positive ({self.N_pop_val}),'
          ' skipping MCMC fit.'
      )
      self.trace = None
      return

    try:
      with pm.Model() as model:
        # Priors for epidemiological parameters
        # gamma: recovery rate. Mean infectious period ~7 days (1/7 days)
        gamma = pm.LogNormal(
            'gamma', mu=np.log(1 / 7), sigma=0.3
        )  # Tighter sigma from 0.5 to 0.3
        # rho: hospitalization incidence rate. Mean 0.44% of infected get hospitalized
        rho = pm.LogNormal(
            'rho', mu=np.log(0.005), sigma=0.5
        )  # Tighter sigma from 0.7 to 0.5
        # delta1: rate from H1 to H2. Mean duration in H1 ~5 days (1/5 days)
        delta1 = pm.LogNormal(
            'delta1', mu=np.log(1 / 5), sigma=0.5
        )  # Tighter sigma from 0.7 to 0.5
        # delta2: rate from H2 to R. Mean duration in H2 ~7 days (1/7 days)
        delta2 = pm.LogNormal(
            'delta2', mu=np.log(1 / 7), sigma=0.5
        )  # Tighter sigma from 0.7 to 0.5

        # Priors for time-varying beta (2 arc-tangents)
        log_beta_base = pm.Normal(
            'log_beta_base', mu=np.log(0.8), sigma=0.3
        )  # Tighter sigma from 0.5 to 0.3

        slope1 = pm.Normal(
            'slope1', mu=0, sigma=0.5
        )  # Tighter sigma from 0.7 to 0.5
        k1 = pm.HalfNormal('k1', sigma=0.5)  # Tighter sigma from 1.0 to 0.5
        t_mid = self.daily_time_points_train.mean()
        t_span = (
            self.daily_time_points_train.max()
            - self.daily_time_points_train.min()
        )
        t01 = pm.Normal('t01', mu=t_mid, sigma=t_span / 6.0)

        slope2 = pm.Normal(
            'slope2', mu=0, sigma=0.5
        )  # Tighter sigma from 0.7 to 0.5
        k2 = pm.HalfNormal('k2', sigma=0.5)  # Tighter sigma from 1.0 to 0.5
        t02 = pm.Normal('t02', mu=t_mid, sigma=t_span / 6.0)

        def beta_t_closure(t_val):
          log_beta_t_unconstrained = (
              log_beta_base
              + slope1 * (0.5 + 0.5 * at.arctan(k1 * (t_val - t01)) / np.pi)
              + slope2 * (0.5 + 0.5 * at.arctan(k2 * (t_val - t02)) / np.pi)
          )
          return at.exp(log_beta_t_unconstrained)

        # Initial conditions
        # I0: Initial infected count. Using a more robust estimate of initial_admissions.
        # Consider initial admissions over the first few weeks of the training period for a robust estimate
        initial_admissions_estimate = get_initial_admissions_estimate(
            train_data.set_index('target_end_date')[target_col].loc[
                self.train_start_date
                + pd.Timedelta(days=6) : self.train_start_date
                + pd.Timedelta(weeks=3)
            ],
            n_first_vals=3,
        )

        # Ensure I0 prior mean is robustly calculated and floored at 1.0 to avoid log(0) and small initial infection numbers.
        # Simplified np.exp(np.log(0.005)) to 0.005
        I0_mu_val = np.log(max(1.0, initial_admissions_estimate / 7 / 0.005))
        I0 = pm.LogNormal(
            'I0', mu=I0_mu_val, sigma=0.7
        )  # Tighter sigma from 1.0 to 0.7

        H1_0 = pm.HalfNormal('H1_0', sigma=0.7)  # Tighter sigma from 1.0 to 0.7
        H2_0 = pm.HalfNormal('H2_0', sigma=0.7)  # Tighter sigma from 1.0 to 0.7

        R0 = pm.Constant('R0', 0.0)

        S0_unconstrained = self.N_pop_val - I0 - R0 - H1_0 - H2_0
        S0 = pm.Deterministic('S0', pm.math.maximum(0.0, S0_unconstrained))

        y0 = [S0, I0, R0, H1_0, H2_0]

        sirh2_ode_solution = pm.ode.DifferentialEquation(
            func=self._sirh2_ode,
            y0=y0,
            t=self.daily_time_points_train,
            args=(self.N_pop_val, beta_t_closure, gamma, rho, delta1, delta2),
        )

        sirh2_ode_solution_output = pm.Deterministic(
            'sirh2_ode_solution_output', sirh2_ode_solution
        )

        daily_hospital_incidence = pm.Deterministic(
            'daily_hospital_incidence', rho * sirh2_ode_solution_output[:, 1]
        )

        weekly_model_incidence_list = []
        for i, target_end_date_k in enumerate(
            self.observed_weekly_admissions.index
        ):
          week_start_date_k = target_end_date_k - pd.Timedelta(days=6)
          start_idx_in_ode = (week_start_date_k - self.train_start_date).days
          end_idx_in_ode = (target_end_date_k - self.train_start_date).days

          start_idx_in_ode = np.maximum(0, start_idx_in_ode)
          weekly_sum_k = at.sum(
              daily_hospital_incidence[start_idx_in_ode : end_idx_in_ode + 1]
          )
          weekly_model_incidence_list.append(weekly_sum_k)

        weekly_model_incidence = at.stack(weekly_model_incidence_list)

        mu_pred_weekly = pm.math.maximum(0.0, weekly_model_incidence)

        alpha = pm.HalfNormal(
            'alpha', sigma=1.0
        )  # Tighter sigma from 1.5 to 1.0

        pm.NegativeBinomial(
            'likelihood',
            mu=mu_pred_weekly,
            alpha=alpha,
            observed=self.observed_weekly_admissions.values,
        )

        self.trace = pm.sample(
            draws=1500,
            tune=1500,
            chains=4,
            random_seed=42,
            return_inferencedata=True,
            cores=pm.cpu_count(),
            target_accept=0.85,
            max_treedepth=12,
            progressbar=False,
        )  # Changed chains from 2 to 4, added progressbar=False
    except Exception as e:
      warnings.warn(
          'MCMC sampling failed for location (population:'
          f' {self.N_pop_val}): {e}'
      )
      self.trace = None

  def simulate_forecasts(
      self, forecast_target_date
  ):
    """Generates stochastic forecasts for a single target week ending on forecast_target_date.

    This optimized version starts simulations from the end of the training
    period.
    """
    if self.trace is None:
      return np.zeros(self.n_simulations, dtype=int)

    if self.N_pop_val <= 0:
      warnings.warn(
          'Population is non-positive during simulation, returning zeros.'
      )
      return np.zeros(self.n_simulations, dtype=int)

    all_simulated_weekly_admissions = []

    # Sample parameters from the trace outside the loop for efficiency
    num_samples_from_trace = min(
        self.n_simulations,
        len(self.trace.posterior.draw) * len(self.trace.posterior.chain),
    )

    flat_trace = self.trace.posterior.stack(sample=('chain', 'draw'))
    # Ensure 'replace=False' is used if num_samples_from_trace is less than the total available samples,
    # otherwise, use 'replace=True' to allow sampling with replacement.
    replace_sampling = num_samples_from_trace > len(flat_trace.sample)
    sampled_params_indices = np.random.choice(
        len(flat_trace.sample),  # Use the size of the stacked dimension
        size=num_samples_from_trace,
        replace=replace_sampling,  # Added 'replace' argument for robust sampling
    )

    forecast_duration_days = (forecast_target_date - self.train_end_date).days

    if forecast_duration_days < 0:
      warnings.warn(
          'Forecast target date is before the end of training data during'
          ' simulation (should be handled earlier), returning zeros.'
      )
      return np.zeros(self.n_simulations, dtype=int)

    forecast_relative_time_points = np.arange(0, forecast_duration_days + 1)

    for idx in sampled_params_indices:
      params = {
          var: flat_trace[var].values[idx] for var in flat_trace.data_vars
      }

      log_beta_base_s = params.get('log_beta_base')
      slope1_s = params.get('slope1')
      k1_s = params.get('k1')
      t01_s = params.get('t01')
      slope2_s = params.get('slope2')
      k2_s = params.get('k2')
      t02_s = params.get('t02')

      absolute_time_offset = self.train_full_duration_days - 1

      def beta_t_func_sim(t_val_relative_to_train_end):
        t_val_absolute = t_val_relative_to_train_end + absolute_time_offset
        log_beta_t_unconstrained_s = (
            log_beta_base_s
            + slope1_s
            * (0.5 + 0.5 * np.arctan(k1_s * (t_val_absolute - t01_s)) / np.pi)
            + slope2_s
            * (0.5 + 0.5 * np.arctan(k2_s * (t_val_absolute - t02_s)) / np.pi)
        )
        return np.exp(log_beta_t_unconstrained_s)

      # Ensure y0_forecast is robustly handled, potentially clamping values to non-negative
      y0_forecast = params['sirh2_ode_solution_output'][
          self.train_full_duration_days - 1, :
      ]
      y0_forecast = np.maximum(
          0, y0_forecast
      )  # Ensure initial state for forecast is non-negative

      sol_forecast = odeint(
          self._sirh2_ode,
          y0_forecast,
          forecast_relative_time_points,
          args=(
              float(self.N_pop_val),
              beta_t_func_sim,
              params['gamma'],
              params['rho'],
              params['delta1'],
              params['delta2'],
          ),
          tfirst=True,
      )

      sol_forecast = np.maximum(0, sol_forecast)

      daily_incidence_forecast = params['rho'] * sol_forecast[:, 1]

      # Sum the last 7 days for the target weekly incidence
      if len(daily_incidence_forecast) < 7:
        # This case implies forecast_duration_days < 6, which shouldn't happen for valid horizons (0,1,2,3)
        # but handle defensively. Sum all available days.
        daily_incidence_target_week = daily_incidence_forecast
      else:
        # For a week ending on forecast_target_date, sum the last 7 days of daily incidence
        daily_incidence_target_week = daily_incidence_forecast[-7:]

      stochastic_daily_incidence = np.random.poisson(
          daily_incidence_target_week
      )

      weekly_sum = np.sum(stochastic_daily_incidence)

      all_simulated_weekly_admissions.append(
          np.maximum(0, np.round(weekly_sum)).astype(int)
      )

    return np.array(all_simulated_weekly_admissions)


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Returns quantile predictions.
  """
  warnings.filterwarnings('ignore', category=FutureWarning)
  warnings.filterwarnings('ignore', category=UserWarning)
  warnings.filterwarnings('ignore', category=RuntimeWarning)

  train_x_copy = train_x.copy()
  train_x_copy['target_end_date'] = pd.to_datetime(
      train_x_copy['target_end_date']
  )
  train_df = train_x_copy.copy()
  train_df[train_y.name] = train_y
  train_df['target_end_date'] = pd.to_datetime(train_df['target_end_date'])

  global locations, ilinet_state

  ilinet_state_processed = ilinet_state[
      ilinet_state['region_type'] == 'States'
  ].copy()

  location_name_to_fips = locations.set_index('location_name')[
      'location'
  ].to_dict()
  ilinet_state_processed['location'] = ilinet_state_processed['region'].map(
      location_name_to_fips
  )
  ilinet_state_processed = ilinet_state_processed.dropna(
      subset=['location']
  ).copy()
  ilinet_state_processed['location'] = ilinet_state_processed[
      'location'
  ].astype(int)

  ilinet_state_processed['week_start'] = pd.to_datetime(
      ilinet_state_processed['week_start']
  )
  ilinet_state_processed['target_end_date'] = ilinet_state_processed[
      'week_start'
  ].apply(lambda d: d + pd.Timedelta(days=6))

  train_df = pd.merge(
      train_df, locations[['location', 'population']], on='location', how='left'
  )

  ilinet_state_processed = pd.merge(
      ilinet_state_processed,
      locations[['location', 'population']],
      on='location',
      how='left',
  )
  ilinet_state_processed['unweighted_ili'] = ilinet_state_processed[
      'unweighted_ili'
  ].fillna(0)

  ilinet_state_processed = ilinet_state_processed[
      ilinet_state_processed['population'] > 0
  ].copy()
  ilinet_state_processed = ilinet_state_processed.dropna(
      subset=['population']
  ).copy()

  # --- REVISED DATA AUGMENTATION: Location-Specific Regression Scaling Transformation with Robust Fallback ---
  ili_data_cutoff = pd.Timestamp('2022-10-15')

  overlap_train_df = train_df[
      train_df['target_end_date'] < ili_data_cutoff
  ].copy()
  overlap_ilinet_df = ilinet_state_processed[
      ilinet_state_processed['target_end_date'] < ili_data_cutoff
  ].copy()

  merged_overlap_data = pd.merge(
      overlap_train_df,
      overlap_ilinet_df[
          ['target_end_date', 'location', 'unweighted_ili', 'population']
      ],
      on=['target_end_date', 'location'],
      how='inner',
  )

  # Calculate per-capita metrics for regression
  merged_overlap_data['target_per_capita'] = (
      merged_overlap_data[train_y.name] / merged_overlap_data['population']
  )
  merged_overlap_data['ili_rate'] = merged_overlap_data[
      'unweighted_ili'
  ]  # ILI is already a rate/percentage

  # Filter for meaningful regression points: positive target, positive ili, positive population
  merged_overlap_data_for_reg = merged_overlap_data[
      (merged_overlap_data['population'] > 0)
      & (merged_overlap_data['ili_rate'] > 0)
      & (merged_overlap_data['target_per_capita'] > 0)
  ].copy()

  global_regression_factor = (
      1e-6  # Default small positive factor if global regression fails
  )
  if (
      not merged_overlap_data_for_reg.empty
      and len(merged_overlap_data_for_reg) >= 5
  ):  # Require at least 5 points for global regression
    X_global = merged_overlap_data_for_reg[['ili_rate']]
    y_global = merged_overlap_data_for_reg['target_per_capita']

    reg_global = LinearRegression(fit_intercept=False)
    reg_global.fit(X_global, y_global)

    if reg_global.coef_[0] > 0:
      global_regression_factor = reg_global.coef_[0]
    else:
      warnings.warn(
          'Global regression coefficient was non-positive, falling back to'
          ' median ratio.'
      )
      global_ratios = y_global / X_global.iloc[:, 0]
      global_ratios_finite = global_ratios[
          np.isfinite(global_ratios) & (global_ratios > 0)
      ]  # Only positive finite ratios
      global_regression_factor = (
          global_ratios_finite.median()
          if not global_ratios_finite.empty
          else 1e-6
      )
  global_regression_factor = np.maximum(
      global_regression_factor, 1e-6
  )  # Ensure global factor is always positive

  location_mapping_functions = {}

  for loc_id in ilinet_state_processed['location'].unique():
    loc_overlap_data_valid = merged_overlap_data_for_reg[
        merged_overlap_data_for_reg['location'] == loc_id
    ].copy()

    chosen_factor = float(global_regression_factor)  # Default to global factor

    if (
        not loc_overlap_data_valid.empty and len(loc_overlap_data_valid) >= 5
    ):  # Require at least 5 points for local regression
      X_loc = loc_overlap_data_valid[['ili_rate']]
      y_loc = loc_overlap_data_valid['target_per_capita']

      reg_loc = LinearRegression(fit_intercept=False)
      reg_loc.fit(X_loc, y_loc)

      if reg_loc.coef_[0] > 0:  # Ensure positive coefficient
        chosen_factor = reg_loc.coef_[0]
      else:
        warnings.warn(
            f'Local regression coefficient for location {loc_id} was'
            ' non-positive, using local median or global factor.'
        )
        local_ratios = y_loc / X_loc.iloc[:, 0]
        local_ratios_finite = local_ratios[
            np.isfinite(local_ratios) & (local_ratios > 0)
        ]  # Only positive finite ratios
        local_median_ratio = (
            local_ratios_finite.median()
            if not local_ratios_finite.empty
            else global_regression_factor
        )
        chosen_factor = local_median_ratio  # Use local median, or global factor if local median fails
    else:
      warnings.warn(
          f'Insufficient valid data for local regression for location {loc_id}'
          f' (count: {len(loc_overlap_data_valid)}), using global factor.'
      )

    chosen_factor = np.maximum(
        chosen_factor, 1e-6
    )  # Ensure factor is always positive
    location_mapping_functions[loc_id] = lambda x, f=chosen_factor: x * f

  synthetic_admissions_data = []
  for loc_id in ilinet_state_processed['location'].unique():
    loc_ilinet_df = ilinet_state_processed[
        ilinet_state_processed['location'] == loc_id
    ].copy()

    if loc_ilinet_df.empty:
      continue

    synthetic_data_for_loc = loc_ilinet_df.copy()

    # Filter synthetic data to be strictly before the ILI data cutoff
    synthetic_data_for_loc = synthetic_data_for_loc[
        synthetic_data_for_loc['target_end_date'] < ili_data_cutoff
    ].copy()

    if synthetic_data_for_loc.empty:
      continue

    # Retrieve the appropriate mapping function, defaulting to one using the global factor if not found
    mapping_object = location_mapping_functions.get(
        loc_id, lambda x, f=float(global_regression_factor): x * f
    )

    synthetic_admissions_per_capita = mapping_object(
        synthetic_data_for_loc['unweighted_ili']
    )
    synthetic_admissions_per_capita = np.maximum(
        0.0, synthetic_admissions_per_capita
    )  # Ensure non-negative per-capita

    # Calculate synthetic admissions, handling non-positive population
    synthetic_data_for_loc['synthetic_admissions'] = np.where(
        synthetic_data_for_loc['population'] > 0,
        np.round(
            synthetic_admissions_per_capita
            * synthetic_data_for_loc['population']
        ).astype(int),
        0,
    )
    synthetic_data_for_loc['synthetic_admissions'] = np.maximum(
        0, synthetic_data_for_loc['synthetic_admissions']
    )

    loc_train_df = train_df[train_df['location'] == loc_id].copy()
    existing_dates = loc_train_df['target_end_date'].unique()

    # Crucially filter out dates already present in actual train_df
    synthetic_data_for_loc = synthetic_data_for_loc[
        ~synthetic_data_for_loc['target_end_date'].isin(existing_dates)
    ].copy()

    synthetic_data_formatted = (
        synthetic_data_for_loc[[
            'target_end_date',
            'location',
            'synthetic_admissions',
            'population',
        ]]
        .rename(columns={'synthetic_admissions': train_y.name})
        .copy()
    )

    synthetic_admissions_data.append(synthetic_data_formatted)

  if synthetic_admissions_data:
    all_synthetic_data = pd.concat(synthetic_admissions_data, ignore_index=True)
    augmented_train_df = pd.concat(
        [train_df, all_synthetic_data], ignore_index=True
    )
  else:
    augmented_train_df = train_df.copy()

  augmented_train_df = augmented_train_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # --- End REVISED DATA AUGMENTATION ---

  # 2. Forecasting Loop
  all_predictions = []

  test_x = test_x.sort_values(
      by=['reference_date', 'horizon', 'location']
  ).reset_index(drop=True)

  for (ref_date, loc_id), group in test_x.groupby(
      ['reference_date', 'location']
  ):
    ref_date_ts = pd.to_datetime(ref_date)
    train_end_date = ref_date_ts - pd.Timedelta(weeks=1)

    loc_train_data = augmented_train_df[
        (augmented_train_df['location'] == loc_id)
        & (augmented_train_df['target_end_date'] <= train_end_date)
    ].copy()

    loc_train_data = loc_train_data.sort_values(
        by='target_end_date'
    ).reset_index(drop=True)

    should_fit_mcmc = not (
        loc_train_data.empty
        or loc_train_data[train_y.name].nunique() < 2
        or loc_train_data['population'].isnull().any()
        or loc_train_data['population'].iloc[0] <= 0
    )

    model = None
    if should_fit_mcmc:
      model = SIRH2Model(n_simulations=2000)
      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit_mcmc(loc_train_data, target_col=train_y.name)

      if model.trace is None:
        should_fit_mcmc = False

    for _, row in group.iterrows():
      current_target_end_date = pd.to_datetime(row['target_end_date'])
      row_predictions = {
          'reference_date': row['reference_date'],
          'horizon': row['horizon'],
          'location': row['location'],
          'target_end_date': row['target_end_date'],
      }

      if row['horizon'] == -1:
        # Use original train_df for observed historical values for horizon -1
        observed_value_series = train_df[
            (train_df['location'] == loc_id)
            & (train_df['target_end_date'] == current_target_end_date)
        ][train_y.name]

        observed_value = (
            observed_value_series.iloc[0]
            if not observed_value_series.empty
            else 0
        )

        # For horizon -1, all quantiles are the observed value. Ensure non-negative.
        observed_value = np.maximum(0, int(observed_value))
        for q_level in QUANTILES:
          row_predictions[f'quantile_{q_level}'] = observed_value
      else:
        if should_fit_mcmc and model is not None:
          simulations = model.simulate_forecasts(
              forecast_target_date=current_target_end_date
          )

          if len(simulations) > 0 and np.sum(simulations) > 0:
            quant_preds = np.round(
                np.percentile(simulations, [q * 100 for q in QUANTILES])
            ).astype(int)
            quant_preds = np.maximum(
                0, quant_preds
            )  # Ensure non-negative before monotonicity
            quant_preds = np.maximum.accumulate(
                quant_preds
            )  # Ensure monotonicity
          else:
            quant_preds = np.zeros(len(QUANTILES), dtype=int)
        else:
          # MCMC failed or skipped - IMPROVED fallback using Negative Binomial
          last_known_admissions_series = loc_train_data[
              (loc_train_data['target_end_date'] <= train_end_date)
              & (loc_train_data[train_y.name] >= 0)  # Ensure non-negative
          ][train_y.name]

          last_known_admissions = 0
          if not last_known_admissions_series.empty:
            # Prioritize last non-zero, otherwise the absolute last value
            last_non_zero_series = last_known_admissions_series[
                last_known_admissions_series > 0
            ]
            if not last_non_zero_series.empty:
              last_known_admissions = last_non_zero_series.iloc[-1]
            else:
              last_known_admissions = last_known_admissions_series.iloc[-1]

          if last_known_admissions > 0:
            # Generate samples from a Negative Binomial distribution
            # PyMC's NegativeBinomial(mu, alpha) has mean mu, variance mu + mu^2/alpha
            # numpy.random.negative_binomial(n, p) has mean n*(1-p)/p
            # So, n = alpha_nb_fallback, p = alpha_nb_fallback / (mu + alpha_nb_fallback)
            fallback_alpha_dispersion = 2.0  # A reasonable dispersion parameter
            mu_nb = last_known_admissions
            p_nb = fallback_alpha_dispersion / (
                mu_nb + fallback_alpha_dispersion
            )

            fallback_samples = np.random.negative_binomial(
                n=fallback_alpha_dispersion, p=p_nb, size=2000
            )

            # Calculate quantiles from these samples
            quant_preds_raw = np.percentile(
                fallback_samples, [q * 100 for q in QUANTILES]
            )

            # Ensure non-negative, rounded, and monotonic
            quant_preds = np.round(quant_preds_raw).astype(int)
            quant_preds = np.maximum(
                0, quant_preds
            )  # Ensure no negative values after rounding
            quant_preds = np.maximum.accumulate(
                quant_preds
            )  # Ensure monotonicity
          else:
            quant_preds = np.zeros(len(QUANTILES), dtype=int)

        for i, q_level in enumerate(QUANTILES):
          row_predictions[f'quantile_{q_level}'] = quant_preds[i]

      all_predictions.append(row_predictions)

  # 3. Final Output
  output_df = pd.DataFrame(all_predictions)

  test_x_indexed = test_x.set_index(['reference_date', 'horizon', 'location'])
  output_df_indexed = output_df.set_index(
      ['reference_date', 'horizon', 'location']
  )

  output_df = output_df_indexed.reindex(test_x_indexed.index).reset_index()

  quantile_cols = [f'quantile_{q}' for q in QUANTILES]
  output_df = output_df[
      ['reference_date', 'horizon', 'location', 'target_end_date']
      + quantile_cols
  ]

  output_df['target_end_date'] = pd.to_datetime(
      output_df['target_end_date']
  ).dt.date
  output_df['reference_date'] = pd.to_datetime(
      output_df['reference_date']
  ).dt.date

  return output_df[quantile_cols]


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

  dataset['Total Influenza Admissions'] = (
      pd.to_numeric(dataset['Total Influenza Admissions'], errors='coerce')
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
      dataset.groupby('target_end_date')['Total Influenza Admissions']
      .sum()
      .reset_index()
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
