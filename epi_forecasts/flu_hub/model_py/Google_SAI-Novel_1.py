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
MODEL_NAME = 'Google_SAI-Novel_1'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core principles from the prompt:
# 1. Multi-Layered Stacking Ensemble (Level-1) built from a portfolio of diverse, SOTA hybrid models (Level-0)
#    designed to solve the Proxy-Target Data Asymmetry and Spatio-Temporal dynamics challenges.
# 2. The ensemble uses a trained meta-forecaster to learn the optimal, time-varying fusion weights for superior
#    probabilistic forecasting.
#    (Note: For this single `fit_and_predict_fn` call, the meta-forecaster is simulated as a dynamic,
#    horizon-dependent weighted average ensemble to approximate "time-varying fusion weights",
#    as a true meta-learner requires cross-validation context not available within this function scope.)
# 3. Level-0 models are specific architectures: MT-PatchTST (Pre-training on ILINet, Fine-tuning on paired data),
#    DSSM-Integrator (Two-Period training for ILINet/NHSN emissions), HeatGNN-Flu (Epidemiology-informed GNN).
#    (Note: These are implemented as sophisticated stubs, leveraging data augmentation and conceptual model
#    ideas without full deep learning frameworks, reflecting the prompt's instruction for minor improvements.)
# 4. Generative Probabilistic Decoders (e.g., Conditional Diffusion Model/TimeGrad or Mixture Density Network/MDN)
#    integrated with each Level-0 model to output the 23 required quantiles.
#    (Note: Simulated by a more robust quantile generation logic within model stubs, producing non-linear,
#    heteroscedastic, and monotonically increasing quantiles that emulate these generative decoders.)

from typing import Any, List, Dict, Tuple
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import Ridge  # Using Ridge for more robust transformation
from scipy.stats import nbinom  # For simulating count data distribution

# ilinet_hhs, ilinet, ilinet_state, locations, sample_submission_df, etc. are assumed global


# A simplified, hardcoded mapping from state name to HHS region number.
# In a real scenario, this would come from a data source.
STATE_TO_HHS_REGION = {
    'Connecticut': 'Region 1',
    'Maine': 'Region 1',
    'Massachusetts': 'Region 1',
    'New Hampshire': 'Region 1',
    'Rhode Island': 'Region 1',
    'Vermont': 'Region 1',
    'New Jersey': 'Region 2',
    'New York': 'Region 2',
    'Puerto Rico': 'Region 2',
    'Delaware': 'Region 3',
    'District of Columbia': 'Region 3',
    'Maryland': 'Region 3',
    'Pennsylvania': 'Region 3',
    'Virginia': 'Region 3',
    'West Virginia': 'Region 3',
    'Alabama': 'Region 4',
    'Florida': 'Region 4',
    'Georgia': 'Region 4',
    'Kentucky': 'Region 4',
    'Mississippi': 'Region 4',
    'North Carolina': 'Region 4',
    'South Carolina': 'Region 4',
    'Tennessee': 'Region 4',
    'Illinois': 'Region 5',
    'Indiana': 'Region 5',
    'Michigan': 'Region 5',
    'Minnesota': 'Region 5',
    'Ohio': 'Region 5',
    'Wisconsin': 'Region 5',
    'Arkansas': 'Region 6',
    'Louisiana': 'Region 6',
    'New Mexico': 'Region 6',
    'Oklahoma': 'Region 6',
    'Texas': 'Region 6',
    'Iowa': 'Region 7',
    'Kansas': 'Region 7',
    'Missouri': 'Region 7',
    'Nebraska': 'Region 7',
    'Colorado': 'Region 8',
    'Montana': 'Region 8',
    'North Dakota': 'Region 8',
    'South Dakota': 'Region 8',
    'Utah': 'Region 8',
    'Wyoming': 'Region 8',
    'Arizona': 'Region 9',
    'California': 'Region 9',
    'Hawaii': 'Region 9',
    'Nevada': 'Region 9',
    'Alaska': 'Region 10',
    'Idaho': 'Region 10',
    'Oregon': 'Region 10',
    'Washington': 'Region 10',
}


class BaseLevel0Model:

  def __init__(self, output_quantiles):
    self.output_quantiles = output_quantiles

  def _generate_probabilistic_forecast(
      self, base_pred, horizon, population
  ):
    """Generates a set of quantiles for a single forecast point,

    simulating a generative decoder. This aims for a distribution shape that's
    more
    suitable for count data (e.g., Negative Binomial) and handles
    heteroscedasticity,
    and ensures reasonable behavior for zero/low predictions.
    """
    # Ensure non-negative mean
    base_pred = max(0.0, base_pred)

    # If base_pred is effectively zero, all quantiles should be zero.
    # IMPROVEMENT: Tuned from 0.1 to 0.05 for slightly more sensitivity to very small counts.
    if base_pred < 0.05:
      return np.zeros(len(self.output_quantiles), dtype=int)

    num_samples = 10000  # As specified for TimeGrad sampling

    # Dynamic dispersion parameters for Negative Binomial
    # IMPROVEMENT: Tuned `base_std_factor` (tighter base) and `horizon_increase_factor` (faster spread increase).
    base_std_factor = 0.25  # Adjusted from 0.28 (tighter base)
    horizon_increase_factor = (
        0.20  # Adjusted from 0.18 (faster spread increase with horizon)
    )

    # Magnitude adjustment: Reduces relative std for large base_pred, now also considering population size.
    # Larger populations or larger base_pred values should lead to relatively tighter intervals.
    # IMPROVEMENT: Adjusted tanh argument denominator for sharper population and base_pred sensitivity, and higher multiplier.
    magnitude_adjustment = (
        1 - np.tanh(base_pred / (1000.0 + population / 80.0)) * 0.7
    )  # Tuned from 1200/100 and 0.5 to 1000/80 and 0.7 (stronger adjustment for large values/pop)
    # IMPROVEMENT: Ensure it stays in a reasonable range (tighter lower bound).
    magnitude_adjustment = np.clip(
        magnitude_adjustment, 0.2, 1.0
    )  # Ensure it stays in a reasonable range (tighter lower bound)

    # Minimum absolute standard deviation, adjusted slightly for tighter intervals at low counts
    # IMPROVEMENT: Adjusted from 3.0 to 2.0, allowing tighter intervals for very low counts.
    min_abs_std_dev = 2.0

    # Calculate target standard deviation
    target_std_dev = max(
        min_abs_std_dev,
        base_pred
        * (base_std_factor + (horizon * horizon_increase_factor))
        * magnitude_adjustment,
    )
    target_variance = target_std_dev**2

    # Negative Binomial parameterization (n is the dispersion parameter in scipy.stats.nbinom)
    # Var(X) = mu + mu^2/n_binom_param
    # So, n_binom_param = mu^2 / (Var(X) - mu)

    # Ensure overdispersion for Negative Binomial (Variance > Mean)
    # Force target_variance to be at least slightly greater than base_pred to allow nbinom calculation
    # IMPROVEMENT: Increased buffer and made it relative to base_pred for better scaling.
    min_overdispersion_buffer = max(
        0.02 * base_pred, 1.5
    )  # Tuned from 0.01*base_pred, 1.0 (stronger overdispersion buffer)
    target_variance = max(
        target_variance, base_pred + min_overdispersion_buffer
    )

    # Calculate k_dispersion (scipy's 'n' parameter)
    # IMPROVEMENT: Clamping k_dispersion for stability with slightly higher min k and lower max k.
    k_dispersion = base_pred**2 / (target_variance - base_pred)
    k_dispersion = np.clip(
        k_dispersion, 0.05, 40.0
    )  # Tuned from 0.01, 50.0 (slightly higher min k and lower max k to ensure meaningful overdispersion)

    # Parameters for scipy.stats.nbinom: n (r successes), p (prob of success).
    # p_success = n_binom_param / (n_binom_param + base_pred)
    p_success = k_dispersion / (k_dispersion + base_pred)
    p_success = np.clip(
        p_success, 1e-7, 1.0 - 1e-7
    )  # Robust clamping to prevent p=0 or p=1 issues

    try:
      samples = nbinom.rvs(k_dispersion, p_success, size=num_samples)
    except ValueError as e:
      # Fallback strategy: if Nbinom fails, use Poisson but add noise if overdispersed
      if (
          base_pred < 2000 and base_pred > 0
      ):  # Poisson is good for lower to medium counts
        samples = np.random.poisson(base_pred, size=num_samples)
        if (
            target_variance > base_pred + 1.0
        ):  # If target_variance indicates overdispersion, add extra noise
          extra_noise_std = np.sqrt(max(0, target_variance - base_pred))
          samples = np.maximum(
              0,
              np.round(
                  samples + np.random.normal(0, extra_noise_std, num_samples)
              ),
          ).astype(int)
      else:  # Fallback to a simpler normal distribution for higher counts
        samples = np.random.normal(base_pred, target_std_dev, num_samples)
        samples[samples < 0] = 0

    samples = np.round(samples).astype(int)

    # Calculate empirical quantiles
    sorted_samples = np.sort(samples)
    quantiles_row = np.array(
        [np.percentile(sorted_samples, q * 100) for q in self.output_quantiles]
    )

    # IMPROVEMENT: Ensure monotonicity explicitly before returning
    quantiles_row = np.maximum.accumulate(quantiles_row)

    return quantiles_row

  def _get_base_prediction_for_location(
      self,
      loc,
      horizon,
      ref_date,
      context_data,
      recent_context_prefetched,
  ):
    """Determines a base prediction for a location at a given horizon.

    This generic version uses the last observed value and applies a simple decay
    for future horizons. Specific Level-0 models will override this for more
    complex logic.
    """
    last_observed_val = 0.0
    if loc in recent_context_prefetched.index:
      last_observed_val = recent_context_prefetched.loc[loc, TARGET_STR]

    if horizon == 0:
      return last_observed_val
    else:
      # Simple decay for future horizons, simulating decreasing confidence/influence
      decay_factor = 0.75**horizon
      return last_observed_val * decay_factor

  def _generate_plausible_quantiles(
      self, test_features, context_data
  ):
    """Generates plausible predictions using the enhanced probabilistic forecast.

    Optimized to pre-fetch recent context for efficiency.
    """
    quantile_cols = [f'quantile_{q}' for q in self.output_quantiles]
    dummy_predictions = pd.DataFrame(
        index=test_features.index, columns=quantile_cols
    )

    if not context_data.empty and TARGET_STR in context_data.columns:
      # Pre-fetch the latest observed value for each location
      recent_context_prefetched = (
          context_data.groupby('location')
          .apply(
              lambda x: x.sort_values('target_end_date', ascending=False).iloc[
                  0
              ]
              if not x.empty
              else None
          )
          .dropna()
          .set_index('location')
      )
    else:
      recent_context_prefetched = pd.DataFrame(
          columns=['target_end_date', TARGET_STR]
      )  # Empty frame for no context

    for idx, row in test_features.iterrows():
      loc = row['location']
      ref_date = row['reference_date']
      pop = row['population']
      horizon = row['horizon']

      # Use the model's specific _get_base_prediction_for_location logic
      base_pred = self._get_base_prediction_for_location(
          loc, horizon, ref_date, context_data, recent_context_prefetched
      )

      quantiles_row = self._generate_probabilistic_forecast(
          base_pred, horizon, pop
      )
      dummy_predictions.loc[idx] = quantiles_row

    return dummy_predictions


class MT_PatchTST_Model(BaseLevel0Model):
  """Conceptual Multi-Task PatchTST Model.

  Simulates seasonal patterns and transfer learning.
  """

  def __init__(self, output_quantiles):
    super().__init__(output_quantiles)
    self.ilinet_data_history = None
    self.paired_data_history = None

  def pretrain(self, ilinet_data):
    pass  # Suppress warning

  def finetune(self, paired_data):
    pass  # Suppress warning

  def _get_base_prediction_for_location(
      self,
      loc,
      horizon,
      ref_date,
      context_data_full,
      recent_context_prefetched,
  ):
    """MT-PatchTST specific base prediction: emphasizes seasonality and recent 'patch' momentum."""
    last_observed_val = 0.0
    if loc in recent_context_prefetched.index:
      last_observed_val = recent_context_prefetched.loc[loc, TARGET_STR]

    if horizon == 0:
      return last_observed_val

    target_date_for_forecast = ref_date + pd.Timedelta(weeks=horizon)
    loc_context = (
        context_data_full[
            (context_data_full['location'] == loc)
            & (context_data_full['target_end_date'] < ref_date)
        ]
        .sort_values('target_end_date')
        .dropna(subset=[TARGET_STR])
    )  # Added robustness

    # 1. Seasonal Component: weighted average of past seasons for the same week (improved robustness with median)
    # IMPROVEMENT: More aggressive weighting on most recent season, and consider more past seasons.
    base_seasonal_weights = [
        0.85,
        0.10,
        0.03,
        0.02,
    ]  # Tuned from 0.8, 0.15, 0.05 to give more weight to most recent, and added a 4th year
    current_week_of_year = target_date_for_forecast.isocalendar().week

    contributing_seasonal_values = []
    contributing_weights = []

    for i, y_offset in enumerate(
        range(1, len(base_seasonal_weights) + 1)
    ):  # Look back up to 4 years
      prev_year_data = loc_context[
          loc_context['target_end_date'].dt.year
          == (target_date_for_forecast.year - y_offset)
      ]

      # Find data for the same ISO week in previous years, allowing a wider window (+- 3 weeks)
      # IMPROVEMENT: Tuned from +- 2 weeks to +- 3 weeks for more data points.
      historical_week_data = prev_year_data[
          prev_year_data['target_end_date']
          .dt.isocalendar()
          .week.isin(
              range(
                  max(1, current_week_of_year - 3),
                  min(53, current_week_of_year + 4),
              )
          )
      ]

      if (
          not historical_week_data.empty
          and not historical_week_data[TARGET_STR].empty
      ):
        val_to_add = historical_week_data[
            TARGET_STR
        ].median()  # Use median for robustness against outliers
        # Apply a small upward bias if recent trend is positive (simulating 'patch' momentum)
        # IMPROVEMENT: Make bias more pronounced for stronger recent positive trends.
        if (
            last_observed_val > 0
            and len(loc_context) > 2
            and loc_context[TARGET_STR].iloc[-1]
            > loc_context[TARGET_STR].iloc[-2] * 1.1
        ):  # Require a 10% recent growth
          val_to_add *= 1.08  # Tuned from 1.05
        if val_to_add > 0:
          contributing_seasonal_values.append(val_to_add)
          contributing_weights.append(base_seasonal_weights[i])

    seasonal_component = 0.0
    if contributing_seasonal_values and np.sum(contributing_weights) > 0:
      normalized_weights = np.array(contributing_weights) / np.sum(
          contributing_weights
      )
      seasonal_component = np.average(
          contributing_seasonal_values, weights=normalized_weights
      )

    # 2. Recent Momentum/Patch Trend: based on the last few weeks (improved robustness)
    # IMPROVEMENT: Increased window for more stable momentum estimate.
    recent_history_window = 12  # Tuned from 10 (increased window for stability)
    recent_history = loc_context.tail(recent_history_window)
    recent_momentum = 0.0

    if len(recent_history) >= 2 and recent_history[TARGET_STR].sum() > 0:
      values = np.maximum(1, recent_history[TARGET_STR].values)
      log_values = np.log(values)

      if (
          len(log_values) >= 2 and np.std(log_values) > 1e-7
      ):  # Added std check for robustness
        time_idx = np.arange(len(recent_history)).reshape(-1, 1)
        model = Ridge(
            alpha=0.5
        )  # IMPROVEMENT: Use Ridge for more robust linear regression
        model.fit(time_idx, log_values)
        log_growth_rate_per_week = model.coef_[0]
        recent_momentum = np.expm1(log_growth_rate_per_week)
      elif len(values) >= 2:  # Simpler momentum for minimal data
        recent_momentum = (
            (values[-1] - values[-2]) / max(1.0, values[-2])
            if values[-2] > 0
            else (1.0 if values[-1] > 0 else 0)
        )

      # IMPROVEMENT: Adjusted clipping range for momentum (slightly wider).
      recent_momentum = np.clip(
          recent_momentum, -0.99, 6.0
      )  # Tuned from -0.98, 5.0 (wider range for more dynamic momentum)

    # 3. Adaptive Blending with exponential decay for persistence/momentum, growth for seasonality
    # Weights adjusted using exponential decay
    # IMPROVEMENT: Re-tuned weights for better balance, faster decay for persistence, and higher seasonal for longer horizons.
    weight_last_observed = 0.50 * np.exp(
        -1.3 * horizon
    )  # Tuned from 0.60 * exp(-1.2*h) (lower initial, faster decay)
    weight_momentum_influence = 0.25 * np.exp(
        -1.0 * horizon
    )  # Tuned from 0.30 * exp(-0.9*h) (less momentum, faster decay)

    weight_seasonal = 1.0 - weight_last_observed - weight_momentum_influence

    # Normalize weights to sum to 1.0 to avoid over/under-weighting
    weights_sum = (
        weight_last_observed + weight_momentum_influence + weight_seasonal
    )
    if weights_sum > 1e-6:
      weight_last_observed /= weights_sum
      weight_momentum_influence /= weights_sum
      weight_seasonal /= weights_sum
    else:  # Fallback if weights are extremely small (should not happen with exp decay)
      weight_last_observed, weight_momentum_influence, weight_seasonal = (
          0.33,
          0.33,
          0.34,
      )

    # Combine components
    base_pred = (
        (last_observed_val * weight_last_observed)
        + (
            (last_observed_val * (1 + recent_momentum))
            * weight_momentum_influence
        )
        + (seasonal_component * weight_seasonal)
    )

    return max(0, base_pred)

  def predict_quantiles(
      self, test_features, context_data
  ):
    return self._generate_plausible_quantiles(test_features, context_data)


class DSSM_Integrator_Model(BaseLevel0Model):
  """Conceptual Deep State Space Model (DSSM) Integrator.

  Simulates persistence and trend.
  """

  def __init__(self, output_quantiles):
    super().__init__(output_quantiles)
    self.ilinet_data_period1 = None
    self.paired_data_period2 = None

  def train_period1(self, ilinet_data):
    pass  # Suppress warning

  def train_period2(self, paired_data):
    pass  # Suppress warning

  def _get_base_prediction_for_location(
      self,
      loc,
      horizon,
      ref_date,
      context_data_full,
      recent_context_prefetched,
  ):
    """DSSM-Integrator specific base prediction: emphasizes trend, persistence, and state dynamics via smoothed AR heuristic."""
    last_observed_val = 0.0
    if loc in recent_context_prefetched.index:
      last_observed_val = recent_context_prefetched.loc[loc, TARGET_STR]

    if horizon == 0:
      return last_observed_val

    loc_context = (
        context_data_full[
            (context_data_full['location'] == loc)
            & (context_data_full['target_end_date'] < ref_date)
        ]
        .sort_values('target_end_date')
        .dropna(subset=[TARGET_STR])
    )  # Added robustness

    # Calculate a smoothed trend (exponentially weighted average of recent changes)
    smoothed_weekly_change = 0.0
    # IMPROVEMENT: Increased window for more stable trend estimation.
    trend_window = 20  # Tuned from 16 (increased window for more stable trend)

    if loc_context.shape[0] >= trend_window:
      recent_values = np.maximum(
          1, loc_context[TARGET_STR].tail(trend_window).values
      )
      log_values = np.log(recent_values)

      if (
          len(log_values) >= 2 and np.std(log_values) > 1e-7
      ):  # Added std check for robustness
        log_changes = np.diff(log_values)
        # IMPROVEMENT: Steeper exponential weighting for more recent changes.
        weights = np.exp(
            np.linspace(-4, 0, len(log_changes))
        )  # Exponential weighting, tuned from -3
        weights /= np.sum(weights)

        avg_log_change = np.sum(log_changes * weights)
        smoothed_weekly_change = np.expm1(avg_log_change)
      elif len(recent_values) >= 2:  # Simpler change for minimal data
        smoothed_weekly_change = (
            (recent_values[-1] - recent_values[-2])
            / max(1.0, recent_values[-2])
            if recent_values[-2] > 0
            else (1.0 if recent_values[-1] > 0 else 0)
        )

    elif loc_context.shape[0] >= 2:
      prev_val = max(1, loc_context[TARGET_STR].iloc[-2])
      curr_val = max(1, loc_context[TARGET_STR].iloc[-1])
      smoothed_weekly_change = (
          (curr_val - prev_val) / max(1.0, prev_val)
          if prev_val > 0
          else (1.0 if curr_val > 0 else 0)
      )

    # IMPROVEMENT: Adjusted clipping for trend (slightly wider).
    smoothed_weekly_change = np.clip(
        smoothed_weekly_change, -0.99, 3.5
    )  # Tuned from -0.98, 3.0

    # Apply current state (last observed) and propagate trend
    current_state_prediction = last_observed_val

    for h_step in range(1, horizon + 1):
      # The trend influence decays over time, representing uncertainty in long-term trend
      # IMPROVEMENT: Tuned `trend_decay_factor` for stronger decay.
      trend_decay_factor = 0.65 ** (
          h_step - 1
      )  # Tuned from 0.70 (stronger decay of trend influence)

      current_state_prediction = current_state_prediction * (
          1 + smoothed_weekly_change * trend_decay_factor
      )

      # Introduce a slight damping factor to the state evolution, simulating regression to mean or stability.
      # IMPROVEMENT: Slightly stronger damping.
      damping_factor = 0.88  # Tuned from 0.90 (slightly stronger damping)

      current_state_prediction *= damping_factor

      current_state_prediction = max(0, current_state_prediction)

    return current_state_prediction

  def predict_quantiles(
      self, test_features, context_data
  ):
    return self._generate_plausible_quantiles(test_features, context_data)


class HeatGNN_Flu_Model(BaseLevel0Model):
  """Conceptual HeatGNN-Flu model.

  Simulates spatial influence and epidemiology-informed growth.
  """

  def __init__(
      self,
      output_quantiles,
      locations_df,
      state_to_hhs_region,
      ilinet_hhs_data,
  ):
    super().__init__(output_quantiles)
    self.locations_df = locations_df
    self.state_to_hhs_region = state_to_hhs_region
    self.ilinet_hhs_data = ilinet_hhs_data
    self.combined_data_history = None

  def train(self, combined_data):
    pass  # Suppress warning

  def _get_base_prediction_for_location(
      self,
      loc,
      horizon,
      ref_date,
      context_data_full,
      recent_context_prefetched,
  ):
    """HeatGNN-Flu specific base prediction: emphasizes spatial influence (regional/national proxy) and epidemiological growth."""
    last_observed_val = 0.0
    if loc in recent_context_prefetched.index:
      last_observed_val = recent_context_prefetched.loc[loc, TARGET_STR]

    if horizon == 0:
      return last_observed_val

    loc_context = (
        context_data_full[
            (context_data_full['location'] == loc)
            & (context_data_full['target_end_date'] < ref_date)
        ]
        .sort_values('target_end_date')
        .dropna(subset=[TARGET_STR])
    )  # Added robustness

    # 1. Epidemiology-informed Growth Rate (local R_eff proxy)
    local_growth_rate = 0.0
    # IMPROVEMENT: Increased window for local growth rate calculation.
    growth_window = (
        8  # Tuned from 6 (increased window for local growth rate calculation)
    )

    if (
        len(loc_context) >= growth_window
        and loc_context[TARGET_STR].tail(growth_window).sum() > 0
    ):
      values_for_growth = np.maximum(
          1, loc_context[TARGET_STR].tail(growth_window).values
      )
      log_values = np.log(values_for_growth)

      if len(log_values) >= 2:
        # IMPROVEMENT: Using Ridge for more robust linear regression, similar to transformation
        time_idx = np.arange(len(values_for_growth)).reshape(-1, 1)
        model = Ridge(alpha=0.5)
        model.fit(time_idx, log_values)
        growth_over_window = model.coef_[0]
        local_growth_rate = np.expm1(growth_over_window)

      # IMPROVEMENT: Adjusted clipping (slightly wider).
      local_growth_rate = np.clip(
          local_growth_rate, -0.99, 3.5
      )  # Tuned from -0.98, 3.0
    elif len(loc_context) >= 2:
      prev_val = max(1, loc_context[TARGET_STR].iloc[-2])
      curr_val = max(1, loc_context[TARGET_STR].iloc[-1])
      local_growth_rate = (
          (curr_val - prev_val) / max(1.0, prev_val)
          if prev_val > 0
          else (1.0 if curr_val > 0 else 0)
      )
      # IMPROVEMENT: Adjusted clipping (slightly wider).
      local_growth_rate = np.clip(
          local_growth_rate, -0.99, 3.5
      )  # Tuned from -0.98, 3.0

    # 2. Spatial Influence (National and Regional Context - improved population-weighted proxy for message passing)

    # National Growth Rate
    national_growth_rate = 0.0
    # IMPROVEMENT: Increased window for a more stable national trend.
    national_recent_data_window = (
        8  # Tuned from 6 (increased window for national data)
    )

    national_recent_data_agg = (
        context_data_full[
            (context_data_full['target_end_date'] < ref_date)
            & (
                context_data_full['target_end_date']
                >= ref_date - pd.Timedelta(weeks=national_recent_data_window)
            )
        ]
        .groupby('target_end_date')[TARGET_STR]
        .sum()
        .sort_index()
    )

    if len(national_recent_data_agg) >= 2:
      national_admissions_t_minus_1 = max(1, national_recent_data_agg.iloc[-1])
      national_admissions_t_minus_2 = max(1, national_recent_data_agg.iloc[-2])
      national_growth_rate = (
          national_admissions_t_minus_1 - national_admissions_t_minus_2
      ) / max(1.0, national_admissions_t_minus_2)
      # IMPROVEMENT: Adjusted clipping (slightly wider).
      national_growth_rate = np.clip(
          national_growth_rate, -0.99, 3.5
      )  # Tuned from -0.98, 3.0

    # Regional Growth Rate (using ILI data as proxy for regional signal)
    regional_growth_rate = 0.0
    regional_data_available = False
    loc_name_df = self.locations_df[self.locations_df['location'] == loc][
        'location_name'
    ]
    loc_name = loc_name_df.iloc[0] if not loc_name_df.empty else None

    if loc_name and loc_name in self.state_to_hhs_region:
      hhs_region = self.state_to_hhs_region[loc_name]
      region_recent_data = self.ilinet_hhs_data[
          (self.ilinet_hhs_data['region'] == hhs_region)
          & (self.ilinet_hhs_data['week_start'] < ref_date)
          & (
              self.ilinet_hhs_data['week_start']
              >= ref_date - pd.Timedelta(weeks=national_recent_data_window)
          )
      ].sort_values('week_start')

      if (
          len(region_recent_data) >= 2
          and region_recent_data['weighted_ili'].sum() > 0
      ):
        regional_data_available = True
        region_ili_t_minus_1 = max(
            0.1, region_recent_data['weighted_ili'].iloc[-1]
        )
        region_ili_t_minus_2 = max(
            0.1, region_recent_data['weighted_ili'].iloc[-2]
        )
        regional_growth_rate = (
            region_ili_t_minus_1 - region_ili_t_minus_2
        ) / max(0.1, region_ili_t_minus_2)
        # IMPROVEMENT: Adjusted clipping (slightly wider).
        regional_growth_rate = np.clip(
            regional_growth_rate, -0.99, 3.5
        )  # Tuned from -0.98, 3.0

    # Blend local, regional, and national growth rates.
    loc_pop_df = self.locations_df[self.locations_df['location'] == loc][
        'population'
    ]
    loc_pop = loc_pop_df.iloc[0] if not loc_pop_df.empty else 1.0
    loc_pop = max(1.0, loc_pop)

    # Population-based weighting: larger states rely more on local, smaller more on regional/national.
    # IMPROVEMENT: Tuned constant for sigmoid center and sharpness for better differentiation.
    # Larger population should give higher weight to local, smaller population relies more on neighbors.
    # sigmoid(x) where x = (loc_pop / Scale - Shift). If x is high, sigmoid is high.
    # So large pop -> high pop_weight_local.
    pop_weight_local = 1 / (
        1 + np.exp(-(loc_pop / 3.0e6 - 0.8))
    )  # Tuned from 4.0e6, 0.75 (stronger local weight for smaller states, steeper curve)
    pop_weight_local = np.clip(
        pop_weight_local, 0.2, 0.8
    )  # Tighter clipping range

    remaining_weight = 1 - pop_weight_local

    if regional_data_available:
      weight_regional = remaining_weight * 0.7
      weight_national = remaining_weight * 0.3
      # IMPROVEMENT: Add a small ILI "message passing" effect from regional data
      if (
          regional_growth_rate > 0.05
          and local_growth_rate < regional_growth_rate * 0.5
      ):  # If regional ILI is rising fast but local NHSN isn't
        # A conceptual "message" that local growth might accelerate
        local_growth_rate = (
            local_growth_rate * 0.8 + regional_growth_rate * 0.2
        )  # Blend local with regional signal
    else:
      weight_regional = 0.0
      weight_national = remaining_weight

    effective_growth_rate = (
        (local_growth_rate * pop_weight_local)
        + (regional_growth_rate * weight_regional)
        + (national_growth_rate * weight_national)
    )

    # Apply growth rate over horizons, with decay
    current_pred = last_observed_val
    for h in range(1, horizon + 1):
      # IMPROVEMENT: Tuned `growth_factor` decay for slightly stronger decay of growth influence.
      growth_factor = 1 + (
          effective_growth_rate * (0.55 ** (h - 1))
      )  # Tuned from 0.60 (stronger decay of growth influence)
      current_pred *= growth_factor

    return max(0, current_pred)

  def predict_quantiles(
      self, test_features, context_data
  ):
    return self._generate_plausible_quantiles(test_features, context_data)


def _ensure_monotonic_quantiles(
    df, quantile_cols
):
  """Ensures quantiles in the DataFrame are monotonically increasing per row."""
  df[quantile_cols] = np.sort(df[quantile_cols].values, axis=1)
  return df


def _get_ensemble_weights(horizon):
  """Provides dynamic, horizon-dependent weights for Level-0 models."""
  # These weights are heuristic to simulate the "time-varying fusion weights"
  # of a meta-forecaster, with each model suited to different horizons.
  # IMPROVEMENT: Tuned weights for better performance across horizons, based on model strengths.
  if (
      horizon == 0
  ):  # Current week: DSSM for persistence, MT-PatchTST for immediate seasonal context, HeatGNN minimal
    weights = {
        'mt_patchtst': 0.35,
        'dssm_integrator': 0.45,
        'heatgnn_flu': 0.20,
    }  # Tuned for DSSM dominance
  elif horizon == 1:  # Next week: MT and HeatGNN gain, DSSM reduces
    weights = {
        'mt_patchtst': 0.40,
        'dssm_integrator': 0.30,
        'heatgnn_flu': 0.30,
    }  # Tuned: MT and HeatGNN gain, DSSM reduces
  elif horizon == 2:  # Two weeks out: MT strongest, HeatGNN strong, DSSM drops
    weights = {
        'mt_patchtst': 0.45,
        'dssm_integrator': 0.15,
        'heatgnn_flu': 0.40,
    }  # Tuned: MT strongest, HeatGNN strong, DSSM drops
  elif (
      horizon == 3
  ):  # Three weeks out: MT strongest for deep seasonality, HeatGNN for spatial, DSSM minimal
    weights = {
        'mt_patchtst': 0.48,
        'dssm_integrator': 0.10,
        'heatgnn_flu': 0.42,
    }  # Tuned: MT strongest for deep seasonality, HeatGNN for spatial, DSSM minimal
  else:  # Default for any other horizon - fallback
    weights = {
        'mt_patchtst': 0.33,
        'dssm_integrator': 0.33,
        'heatgnn_flu': 0.34,
    }

  # Normalize weights to sum to 1.0
  total_weight = sum(weights.values())
  return {k: v / total_weight for k, v in weights.items()}


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Return quantiles.

  Do not do any cross-validation in here.
  """
  # Print the reference date for context in the logs
  if (
      not test_x.empty
      and 'reference_date' in test_x.columns
      and not test_x['reference_date'].isnull().any()
  ):
    print(
        '\n--- fit_and_predict_fn called for reference_date:'
        f" {pd.to_datetime(test_x['reference_date'].iloc[0]).strftime('%Y-%m-%d')} ---"
    )
  else:
    print(
        '\n--- fit_and_predict_fn called (test_x empty or missing/invalid'
        ' reference_date) ---'
    )

  # --- 1. Data Preparation and Augmentation (Strategy 2: Learn a Transformation) ---
  # Ensure date columns are datetime objects for consistent operations
  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  test_x['reference_date'] = pd.to_datetime(test_x['reference_date'])

  # Align train_y index with train_x for merging
  train_y_df = train_y.to_frame(name=TARGET_STR)
  train_y_df.index = train_x.index
  train_data = pd.concat([train_x, train_y_df], axis=1)

  # Prepare historical ILINet data (assuming ilinet_state and locations are globally available)
  ilinet_state_processed = ilinet_state.copy()
  ilinet_state_processed['target_end_date'] = pd.to_datetime(
      ilinet_state_processed['week_start']
  )
  ilinet_state_processed = ilinet_state_processed.rename(
      columns={'region': 'location_name'}
  )

  # Impute missing `unweighted_ili` using `ilitotal / total_patients * 100` as a proxy if possible
  ilinet_state_processed['ilitotal'] = ilinet_state_processed[
      'ilitotal'
  ].fillna(0)
  ilinet_state_processed['total_patients'] = ilinet_state_processed[
      'total_patients'
  ].fillna(0)

  # Calculate ili_rate_proxy: Ensure total_patients is not zero to avoid division issues
  # IMPROVEMENT: Added a small epsilon to safe_total_patients to prevent division by zero in all cases.
  safe_total_patients = ilinet_state_processed['total_patients'].replace(
      0, np.nan
  )
  ilinet_state_processed['ili_rate_proxy'] = np.where(
      safe_total_patients > 0,
      ilinet_state_processed['ilitotal'] / (safe_total_patients + 1e-9),
      0,
  )  # Add epsilon here

  # Impute unweighted_ili with proxy * 100 (as unweighted_ili is often a percentage), then fill any remaining NaNs with 0
  ilinet_state_processed['unweighted_ili'] = ilinet_state_processed[
      'unweighted_ili'
  ].fillna(ilinet_state_processed['ili_rate_proxy'] * 100)
  ilinet_state_processed['unweighted_ili'] = ilinet_state_processed[
      'unweighted_ili'
  ].fillna(0)
  ilinet_state_processed = ilinet_state_processed.drop(
      columns=['ili_rate_proxy']
  )

  # Merge with 'locations' to get FIPS codes and population for ILINet data (Ensuring robust merge)
  ilinet_state_processed = pd.merge(
      ilinet_state_processed,
      locations[['location', 'location_name', 'population']],
      on='location_name',
      how='left',
  )

  # ***Improvement 1: Drop rows where location (FIPS code) could not be determined***
  initial_rows = len(ilinet_state_processed)
  ilinet_state_processed.dropna(subset=['location'], inplace=True)
  if len(ilinet_state_processed) < initial_rows:
    warnings.warn(
        f'  Dropped {initial_rows - len(ilinet_state_processed)} rows from'
        ' ilinet_state_processed due to missing FIPS codes after merge.',
        UserWarning,
    )
  ilinet_state_processed['location'] = ilinet_state_processed[
      'location'
  ].astype(int)

  # Fill missing population in ILINet data for locations not found, or use a reasonable default
  # IMPROVEMENT: Ensure population is float and never zero for divisions
  mean_pop = locations['population'].mean() if not locations.empty else 1.0
  ilinet_state_processed['population'] = (
      ilinet_state_processed['population'].fillna(mean_pop).astype(float)
  )
  ilinet_state_processed['population'] = ilinet_state_processed[
      'population'
  ].replace(
      0, 1.0
  )  # Ensure non-zero for divisions

  # --- Prepare ilinet_hhs for HeatGNN_Flu (new) ---
  ilinet_hhs_processed = ilinet_hhs.copy()
  ilinet_hhs_processed['week_start'] = pd.to_datetime(
      ilinet_hhs_processed['week_start']
  )

  # Identify overlap period for learning transformation
  if not train_data.empty:
    overlap_start_nhs = train_data['target_end_date'].min()
    overlap_end_nhs = train_data['target_end_date'].max()
  else:
    warnings.warn(
        'Training data is empty, cannot learn ILI-NHSN transformation.'
        ' Synthetic history will use default scale.',
        UserWarning,
    )
    overlap_start_nhs = pd.to_datetime(
        '2020-01-01'
    )  # Default if train_data is empty
    overlap_end_nhs = pd.to_datetime('2020-01-08')

  # Filter data for overlap period
  merged_overlap = pd.merge(
      train_data[['target_end_date', 'location', TARGET_STR, 'population']],
      ilinet_state_processed[
          ['target_end_date', 'location', 'unweighted_ili', 'population']
      ].rename(columns={'population': 'ili_population'}),
      on=['target_end_date', 'location'],
      how='inner',
  )

  # Learn a transformation: Per-location Ridge Regression with log1p transformation on *rates per capita*
  transform_models: Dict[Any, Any] = {}
  global_lr_model_instance = None
  global_ratio_fallback = 0.0075  # Default small ratio

  if (
      not merged_overlap.empty
      and (merged_overlap['unweighted_ili'] > 0).any()
      and merged_overlap['population'].sum() > 0
  ):
    # Calculate rates per capita before log1p transformation
    merged_overlap['nhs_rate_per_capita'] = (
        merged_overlap[TARGET_STR] / merged_overlap['population']
    )
    merged_overlap['ili_rate_per_capita'] = (
        merged_overlap['unweighted_ili'] / merged_overlap['ili_population']
    )

    # Ensure rates are non-negative before log1p transformation, using a small epsilon
    merged_overlap['scaled_ili_log'] = np.log1p(
        np.maximum(1e-9, merged_overlap['ili_rate_per_capita'])
    )
    merged_overlap['scaled_nhs_log'] = np.log1p(
        np.maximum(1e-9, merged_overlap['nhs_rate_per_capita'])
    )

    # Compute a simple mean ratio for immediate fallback using rates
    total_nhs_rate = merged_overlap['nhs_rate_per_capita'].sum()
    total_ili_rate = merged_overlap['ili_rate_per_capita'].sum()
    if total_ili_rate > 1e-8:
      global_ratio_fallback = total_nhs_rate / total_ili_rate

    # Train a robust global Ridge regression for fallback
    # IMPROVEMENT: Use Ridge regression for robustness.
    valid_global_data = merged_overlap[
        (merged_overlap['scaled_ili_log'] > np.log1p(1e-9))
        & (merged_overlap['scaled_nhs_log'] > np.log1p(1e-9))
    ]

    if (
        len(valid_global_data) >= 10
        and valid_global_data['scaled_ili_log'].std() > 1e-8
    ):
      global_lr_model_instance = Ridge(
          alpha=1.0
      )  # IMPROVEMENT: Ridge regression
      global_lr_model_instance.fit(
          valid_global_data[['scaled_ili_log']],
          valid_global_data['scaled_nhs_log'],
      )
    else:
      warnings.warn(
          '  Insufficient data or variance to train global log-linear'
          ' transformation. Using mean rate ratio fallback.',
          UserWarning,
      )
      global_lr_model_instance = (
          global_ratio_fallback  # Ensure it's a float if training fails
      )

    # Attempt per-location models
    for loc_code in merged_overlap['location'].unique():
      loc_overlap_data = merged_overlap[merged_overlap['location'] == loc_code]
      # IMPROVEMENT: Slightly relaxed filter for loc_overlap_data_filtered to allow more data for training small states.
      loc_overlap_data_filtered = loc_overlap_data[
          (loc_overlap_data['scaled_ili_log'] > np.log1p(1e-12))
          & (loc_overlap_data['scaled_nhs_log'] > np.log1p(1e-12))
      ]

      if (
          len(loc_overlap_data_filtered) >= 5
          and loc_overlap_data_filtered['scaled_ili_log'].std() > 1e-8
      ):
        model = Ridge(
            alpha=0.5
        )  # IMPROVEMENT: Ridge regression with a slightly lower alpha for local models
        model.fit(
            loc_overlap_data_filtered[['scaled_ili_log']],
            loc_overlap_data_filtered['scaled_nhs_log'],
        )
        transform_models[loc_code] = model
      else:
        # IMPROVEMENT: Ensure explicit fallback to global model or the calculated global_ratio_fallback
        transform_models[loc_code] = (
            global_lr_model_instance
            if isinstance(global_lr_model_instance, Ridge)
            else global_ratio_fallback
        )
  else:
    warnings.warn(
        '  Overlap data is empty or ILI/NHSN data is all zeros. Cannot learn'
        ' transformation model. Using default global scale 0.0075.',
        UserWarning,
    )
    # Ensure global_lr_model_instance is a float if no training is possible for easier handling below
    global_lr_model_instance = global_ratio_fallback

  # Create "synthetic" history for target variable using transformation for periods *before* NHSN data
  synthetic_history_ili = ilinet_state_processed[
      ilinet_state_processed['target_end_date'] < overlap_start_nhs
  ].copy()

  synthetic_history_ili[TARGET_STR] = 0.0  # Initialize
  if not synthetic_history_ili.empty:
    for loc_code in synthetic_history_ili['location'].unique():
      loc_data_to_transform = synthetic_history_ili[
          synthetic_history_ili['location'] == loc_code
      ].copy()

      current_loc_pop = (
          loc_data_to_transform['population'].iloc[0]
          if not loc_data_to_transform.empty
          else 1.0
      )
      current_loc_pop = max(1.0, current_loc_pop)

      # Calculate ILI rate per capita
      ili_rate_per_capita = (
          loc_data_to_transform['unweighted_ili'] / current_loc_pop
      )
      loc_data_to_transform['scaled_ili_log'] = np.log1p(
          np.maximum(1e-9, ili_rate_per_capita)
      )

      transformer = transform_models.get(loc_code, global_lr_model_instance)
      # IMPROVEMENT: Robust check for transformer type and fallback.
      if not (
          isinstance(transformer, Ridge)
          or isinstance(transformer, (float, np.float64))
      ):  # Check for Ridge instead of LinearRegression
        warnings.warn(
            f'  Invalid transformer type for location {loc_code} after fallback'
            ' logic. Using global ratio fallback.',
            UserWarning,
        )
        transformer = global_ratio_fallback

      synthetic_predictions_rates = pd.Series(
          0.0, index=loc_data_to_transform.index
      )

      if isinstance(transformer, Ridge):  # Check for Ridge
        if not loc_data_to_transform[
            'scaled_ili_log'
        ].empty:  # Check if there's data to predict on
          synthetic_predictions_scaled_log = transformer.predict(
              loc_data_to_transform[['scaled_ili_log']]
          )
          synthetic_predictions_rates = np.expm1(
              synthetic_predictions_scaled_log
          )  # Inverse log1p
      elif isinstance(transformer, (float, np.float64)):
        synthetic_predictions_rates = ili_rate_per_capita * transformer

      # Ensure predictions are non-negative rates
      synthetic_predictions_rates = np.maximum(0, synthetic_predictions_rates)

      # Convert predicted rates back to counts using current population
      synthetic_predictions_counts = (
          synthetic_predictions_rates * current_loc_pop
      )
      synthetic_history_ili.loc[loc_data_to_transform.index, TARGET_STR] = (
          np.maximum(0, synthetic_predictions_counts)
      )

  common_cols = [
      'target_end_date',
      'location',
      'location_name',
      'population',
      TARGET_STR,
  ]
  # IMPROVEMENT: Ensure all common_cols are actually present in synthetic_history_ili before selection
  synthetic_history_ili_cols = [
      col for col in common_cols if col in synthetic_history_ili.columns
  ]
  synthetic_history_ili = synthetic_history_ili[synthetic_history_ili_cols]

  train_data['population'] = train_data['population'].astype(float)
  train_data['population'] = train_data['population'].replace(0, 1.0)

  augmented_train_data = (
      pd.concat(
          [synthetic_history_ili[common_cols], train_data[common_cols]],
          ignore_index=True,
      )
      .sort_values(by=['location', 'target_end_date'])
      .reset_index(drop=True)
  )

  warnings.warn(
      f'  Augmented training data shape: {augmented_train_data.shape}',
      UserWarning,
  )

  # --- 2. Level-0 Component Design, Training, and Prediction ---
  level0_forecasts: Dict[str, pd.DataFrame] = {}

  context_for_stubs = augmented_train_data

  # MT-PatchTST
  mt_patchtst = MT_PatchTST_Model(output_quantiles=QUANTILES)
  mt_patchtst.pretrain(
      ilinet_state_processed[
          ilinet_state_processed['target_end_date'] < overlap_start_nhs
      ]
  )
  mt_patchtst.finetune(
      augmented_train_data[
          augmented_train_data['target_end_date'] >= overlap_start_nhs
      ]
  )
  level0_forecasts['mt_patchtst'] = mt_patchtst.predict_quantiles(
      test_x, context_for_stubs
  )

  # DSSM-Integrator
  dssm_integrator = DSSM_Integrator_Model(output_quantiles=QUANTILES)
  dssm_integrator.train_period1(ilinet_state_processed)
  dssm_integrator.train_period2(augmented_train_data)
  level0_forecasts['dssm_integrator'] = dssm_integrator.predict_quantiles(
      test_x, context_for_stubs
  )

  # HeatGNN-Flu (Updated to include HHS regions)
  heatgnn_flu = HeatGNN_Flu_Model(
      output_quantiles=QUANTILES,
      locations_df=locations,
      state_to_hhs_region=STATE_TO_HHS_REGION,
      ilinet_hhs_data=ilinet_hhs_processed,
  )
  heatgnn_flu.train(augmented_train_data)
  level0_forecasts['heatgnn_flu'] = heatgnn_flu.predict_quantiles(
      test_x, context_for_stubs
  )

  # --- 3. Generative Probabilistic Decoders ---
  # As per the conceptual stubs, we assume each Level-0 model's predict_quantiles
  # method internally uses its specified generative decoder (e.g., TimeGrad sampling or MDN).
  # The output is already a DataFrame of quantiles.

  # --- 4. Level-1 Stacking Ensemble Training and Prediction (Dynamic Weighted Average) ---
  warnings.warn(
      '\nApplying Level-1 Stacking Ensemble (conceptual: dynamic weighted'
      " average by horizon to simulate meta-forecaster's time-varying fusion"
      ' weights).',
      UserWarning,
  )

  final_predictions_df = pd.DataFrame(
      index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  # Group test_x by horizon to apply different weights
  for horizon_val in test_x['horizon'].unique():
    horizon_indices = test_x[test_x['horizon'] == horizon_val].index

    current_weights = _get_ensemble_weights(horizon_val)

    # Sum weighted forecasts for this horizon
    weighted_sum_forecasts = pd.DataFrame(
        0.0, index=horizon_indices, columns=[f'quantile_{q}' for q in QUANTILES]
    )

    # Use only valid Level-0 forecasts for current horizon
    has_valid_forecasts = False
    for model_name, weight in current_weights.items():
      if (
          model_name in level0_forecasts
          and not level0_forecasts[model_name].loc[horizon_indices].empty
      ):
        weighted_sum_forecasts += (
            level0_forecasts[model_name].loc[horizon_indices] * weight
        )
        has_valid_forecasts = True

    if has_valid_forecasts:
      final_predictions_df.loc[horizon_indices] = weighted_sum_forecasts
    else:
      warnings.warn(
          f'No valid Level-0 forecasts for horizon {horizon_val}. Filling with'
          ' zeros for this horizon.',
          UserWarning,
      )
      final_predictions_df.loc[horizon_indices] = (
          0.0  # Fallback to zeros if no forecasts available
      )

  # --- 5. Crucial Constraint: Monotonically Increasing Quantiles ---
  final_predictions_df = _ensure_monotonic_quantiles(
      final_predictions_df, [f'quantile_{q}' for q in QUANTILES]
  )

  # Ensure all predictions are non-negative integers (or close to it)
  final_predictions_df = np.maximum(0, final_predictions_df).round().astype(int)

  # Ensure the index of the output matches test_x exactly
  final_predictions_df.index = test_x.index

  warnings.warn('--- fit_and_predict_fn finished ---', UserWarning)
  return final_predictions_df


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
