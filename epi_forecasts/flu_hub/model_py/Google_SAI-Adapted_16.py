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
MODEL_NAME = 'Google_SAI-Adapted_16'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from scipy.stats import truncnorm  # Import for truncated normal distribution sampling

# Removed np.seterr(over='raise') as it can cause unexpected crashes due to numerical overflows.
# NumPy's default error handling (warnings) is sufficient and more robust for a competition setting.
# The original default is {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}

# Assume `ilinet_state`, `locations` are globally available as per context.
# REQUIRED_CDC_LOCATIONS is also globally defined.

# --- New Constants for Clarity and Tuning ---
GLOBAL_MIN_ADMISSIONS_PER_I_RATE = 1e-6  # Minimum value for admissions_per_I_rate to prevent division by zero and ensure positive rates
MIN_ADMISSIONS_PER_I_RATE_STD_PROPORTION = (
    0.1  # Proportion of the mean factor to use as std for perturbation
)
MIN_ADMISSIONS_PER_I_RATE_FIXED_STD = (
    0.001  # Absolute minimum std for perturbation
)
MAX_ADMISSIONS_PER_I_RATE = 10.0  # Upper bound for admissions_per_I_rate_member - REDUCED FROM 100.0 (PLAN ITEM 1)

# Default factor if learning fails or ILI is zero but admissions exist.
# This will be dynamically set to a national average if enough data exists, otherwise remain this fallback.
DEFAULT_ADMISSIONS_PER_ILI_FACTOR_FALLBACK = 0.005  # This is a placeholder; it's dynamically calculated in fit_and_predict_fn
MAX_ADMISSIONS_PER_ILI_FACTOR = (
    50.0  # Upper bound for the learned admissions_per_ILI_unit_factor
)

MIN_INFECTED_SEED_FACTOR_ZERO_OBS = 5.0  # How many times MIN_INFECTED_SEED_LOC to use as mean for I0 when latest_admissions_observation is zero
MIN_INFECTED_SEED_STD_FACTOR_ZERO_OBS = 2.0  # How many times MIN_INFECTED_SEED_LOC to use as std when latest_admissions_observation is zero
MIN_INFECTED_SEED_POP_PROP_ZERO_OBS = 5e-7  # Min proportion of population for I0 seed when observation is zero, e.g., 5 individuals in 10M pop

MIN_S_PROPORTION = 0.5  # Minimum proportion of population to remain susceptible at initialization.

# --- NEW CONSTANT FOR IMPROVEMENT ---
MIN_INFECTION_NOISE_FLOOR = 0.5  # Minimum standard deviation for new infections, ensuring stochasticity even at low rates.
MIN_BLENDING_WEIGHT = 0.1  # Minimum blending weight for local seasonal pattern, ensures national influence.
MAX_BLENDING_WEIGHT = 0.9  # Maximum blending weight for local seasonal pattern, ensures national regularization.
RECOVERY_NOISE_MULTIPLIER = 0.5  # NEW: Factor to scale recovery noise relative to infection noise and noise_factor

# --- Default means for epidemiological parameters (now used as starting points for adaptation) ---
DEFAULT_BETA_BASE_MEAN = (
    0.05 + 0.35
) / 2  # Midpoint of beta_base_min, beta_base_max
DEFAULT_GAMMA_MEAN = (0.1 + 0.5) / 2  # Midpoint of gamma_min, gamma_max
DEFAULT_RHO_MEAN = (0.005 + 0.025) / 2  # Midpoint of rho_min, rho_max

# --- New constants for I0 perturbation refinement ---
# Proportional component for I0 std dev when observations are high
I0_REL_STD_HIGH_OBS_MULTIPLIER = 0.2
# Factor for absolute minimum I0 std dev, applied to MIN_INFECTED_SEED_LOC
I0_MIN_ABS_STD_FACTOR = 0.5
# --- End New Constants ---

# --- NEW CONSTANT FOR STD DEV REFINEMENT ---
PARAM_TRUNC_NORMAL_STD_DIVISOR = 4.0  # Divisor for (max-min) to get std dev for truncated normal samples (e.g., 4 for ~2-sigma range)
# --- END NEW CONSTANT ---

# --- NEW CONSTANTS FOR ADAPTIVE PRIOR RANGES (Plan Item 1) ---
# Removed BETA_RANGE_ADJUSTMENT_FACTOR and GAMMA_RANGE_ADJUSTMENT_FACTOR
# as we are now keeping global fixed ranges and only adjusting the mean.
# These constants were marked to be removed and are no longer used.
# BETA_RANGE_ADJUSTMENT_FACTOR = 0.15
# GAMMA_RANGE_ADJUSTMENT_FACTOR = 0.15
# --- END NEW CONSTANTS ---


class SIRSModel:
  """A simplified Susceptible-Infectious-Recovered-Susceptible (SIRS) model

  for a single ensemble member, simulating weekly influenza dynamics.
  States are represented as counts within a given population.
  Transmissibility is influenced by a seasonal pattern (proxy for AH).
  Internal states (S, I, R) are floats to maintain precision and prevent early
  rounding errors.
  """

  def __init__(
      self,
      S0,
      I0,
      R0,
      population,
      beta_base,
      gamma,
      rho,
      admissions_per_I_rate,
      seasonal_multipliers,
      noise_factor,
      min_infected_seed = 1.0,
  ):
    self.population = float(population)
    self.min_infected_seed = min_infected_seed  # Store this for _advance_week

    # Refinement: Handle zero or negative population gracefully
    if self.population <= 0:
      self.S, self.I, self.R = 0.0, 0.0, 0.0
      return

    # Ensure initial infected population is at least the seed if it's too low
    # and doesn't exceed total population.
    self.I = max(min_infected_seed, float(I0))
    self.I = min(self.I, self.population)

    # Ensure initial recovered population is non-negative and doesn't exceed
    # remaining population (population - I)
    self.R = max(0.0, float(R0))
    self.R = min(self.R, self.population - self.I)

    # Susceptible population is the remainder
    self.S = max(0.0, self.population - self.I - self.R)

    # Final check to ensure sum is population, minor adjustments due to float precision
    current_sum = self.S + self.I + self.R
    if abs(current_sum - self.population) > 1e-6:
      if current_sum > 0:
        scale_factor = self.population / current_sum
        self.S *= scale_factor
        self.I *= scale_factor
        self.R *= scale_factor
      else:  # If all states are 0 but population is > 0, set S to population
        self.S = self.population
        self.I = 0.0
        self.R = 0.0

    self.beta_base = beta_base
    self.gamma = gamma
    self.rho = rho
    self.admissions_per_I_rate = admissions_per_I_rate
    self.seasonal_multipliers = seasonal_multipliers
    self.noise_factor = noise_factor

  def _advance_week(self, current_epiweek):
    """Advances the SIRS model by one week.

    Calculates transitions based on current states and parameters, and returns
    the predicted total influenza admissions. Ensures S+I+R = population and
    states are non-negative. Internal states (S, I, R) are floats.
    """
    # Refinement: Early exit for zero/negative population or effectively empty states
    if self.population <= 0 or (self.S + self.I + self.R) < 1e-9:
      self.S, self.I, self.R = self.population, 0.0, 0.0
      return 0.0

    # Refinement: Apply min_infected_seed as a floor to current I before calculating infection rate factor
    # This ensures a minimal "spark" even if I becomes very small. (Adherence to Implementation Plan Step 3)
    self.I = max(self.min_infected_seed, self.I)  # Apply floor to self.I
    I_current = self.I  # Use the *now floored* I for calculations

    seasonal_multiplier = self.seasonal_multipliers.get(current_epiweek, 1.0)
    beta_t = self.beta_base * seasonal_multiplier

    S, R = self.S, self.R

    # Ensure population is not effectively zero when calculating infection rate factor
    infection_rate_factor = (
        (I_current / self.population) if self.population > 1e-9 else 0.0
    )

    # Calculate deterministic changes
    new_infections_det = beta_t * S * infection_rate_factor
    recoveries_det = self.gamma * I_current
    immunity_loss_det = self.rho * R

    # Add noise to new infections: Ensure a minimum standard deviation for stochasticity (improvement, Adherence to Implementation Plan Step 3)
    noise_std_infections = max(
        MIN_INFECTION_NOISE_FLOOR, new_infections_det * self.noise_factor
    )
    new_infections_stochastic = np.random.normal(
        new_infections_det, noise_std_infections
    )
    new_infections_stochastic = max(0.0, new_infections_stochastic)

    # NEW: Add noise to recoveries (Plan Item 2.a)
    noise_std_recoveries = max(
        MIN_INFECTION_NOISE_FLOOR
        * 0.5,  # Smaller floor for recoveries' std to reflect less extreme variability
        recoveries_det * self.noise_factor * RECOVERY_NOISE_MULTIPLIER,
    )
    recoveries_stochastic = np.random.normal(
        recoveries_det, noise_std_recoveries
    )
    recoveries_stochastic = max(0.0, recoveries_stochastic)

    # Apply changes (pre-clipping to prevent temporary negative states)
    # Ensure S doesn't go below 0 (cannot have more infections than susceptibles)
    actual_infections = min(new_infections_stochastic, S)

    # Calculate raw next states
    next_I_raw = (
        I_current + actual_infections - recoveries_stochastic
    )  # Use stochastic recoveries
    next_R_raw = (
        R + recoveries_stochastic - immunity_loss_det
    )  # Use stochastic recoveries
    next_S_raw = S - actual_infections + immunity_loss_det

    # Ensure non-negativity for raw states
    next_I_raw = max(0.0, next_I_raw)
    next_R_raw = max(0.0, next_R_raw)
    next_S_raw = max(0.0, next_S_raw)

    # Enforce `min_infected_seed` on the updated I state
    self.I = max(self.min_infected_seed, next_I_raw)
    self.I = min(self.I, self.population)  # Cap I at population

    # Update R, ensuring it doesn't exceed remaining population
    self.R = next_R_raw
    self.R = min(self.R, self.population - self.I)
    self.R = max(0.0, self.R)  # Ensure R is non-negative

    # Update S, as the remainder of the population
    self.S = self.population - self.I - self.R
    self.S = max(0.0, self.S)  # Ensure S is non-negative

    # Final renormalization check (should be very close after the above steps)
    current_sum = self.S + self.I + self.R
    if abs(current_sum - self.population) > 1e-6:  # Should rarely happen now
      if current_sum > 0:
        scale_factor = self.population / current_sum
        self.S *= scale_factor
        self.I *= scale_factor
        self.R *= scale_factor
      else:  # If all states are 0 but population is > 0, set S to population
        self.S = self.population
        self.I = 0.0
        self.R = 0.0

    predicted_admissions = self.I * self.admissions_per_I_rate
    return max(0.0, predicted_admissions)


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # # IMPLEMENTATION PLAN
  # ## Core Principles Checklist:
  # 1.  **A data assimilation system is used to forecast seasonal influenza outbreaks by integrating real-time influenza-like illness (ILI) estimates with an ensemble of simulations from a dynamic transmission model.**
  #     *   **Interpretation and Adherence:** Given the `fit_and_predict_fn` signature (no future real-time observations provided for `test_x` horizons), true *weekly real-time assimilation during the forecast horizon* is not possible. Instead, a robust "initialization-time pseudo-assimilation" is performed at each `reference_date`. This involves using the `latest_admissions_observation` (derived from the fully augmented historical data, which incorporates `unweighted_ili` as a pragmatic proxy for the unspecified Google Flu Trends ILI estimates) for the week *preceding* the `reference_date` to robustly initialize and perturb the `I` state, `admissions_per_infected_rate_for_sirs` parameter, and `noise_factor` across the ensemble. This effectively "nudges the ensemble mean towards the observations" at the forecast's starting point. When `latest_admissions_observation` is zero, a more aggressive perturbation of `I0_member` is used to allow for outbreak seeding (includes scaling its standard deviation with population and considering a fixed population proportion as a seed potential). The initial `R0_member` is capped to ensure a `MIN_S_PROPORTION` of susceptibles (`S0_member`) remains in the population, representing a robust form of data assimilation initialization given the function constraints. Crucially, **the ensemble's epidemiological parameters (`beta_base`, `gamma`) are now dynamically adjusted at initialization based on location-specific historical activity (specifically, `admissions_per_ili_scaling_factor` relative to a national average), making their *sampling mean values* data-informed, thereby directly addressing the "adjustment process" of key epidemiological parameters. Furthermore, their *prior min/max ranges* are also adaptively scaled based on the local activity, allowing for a more flexible and context-specific exploration of parameter space.** The `INFLATION_FACTOR` is applied to parameter and state standard deviations during sampling; this inflation serves to maintain sufficient ensemble spread and diversity, a common practice *after* ensemble updates in EAKF to prevent filter collapse, ensuring robust uncertainty characterization in the absence of continuous, recursive assimilation. The standard deviation for `I0_member` is now refined to be more adaptive, ensuring a robust minimum absolute spread while also scaling proportionally with stronger observations, thereby better reflecting the "reduces ensemble spread" aspect of data assimilation by balancing exploration with observation-driven certainty.
  # 2.  **The model is a compartmental model where individuals transition between Susceptible, Infectious, and Recovered states, with transmissibility influenced by absolute humidity.**
  #     *   **Interpretation and Adherence:** My `SIRSModel` class explicitly tracks `S`, `I`, `R` compartments with numerically stable updates. Transmissibility (`beta`) is influenced by `seasonal_multipliers` derived from historical epiweek patterns of `Total Influenza Admissions` for each location. This serves as a pragmatic proxy for Absolute Humidity (AH) data, as explicit AH data for all states or specific NYC AH data is not provided in the input datasets. These `seasonal_multipliers` act as the "climatological AH conditions" that drive the model's dynamics throughout the forecast. The calculation of these seasonal multipliers is made more robust by blending local patterns with a national pattern, especially for locations with sparse local data, improving the representation of AH influence by making the blending weight adaptive to local data richness.
  # 3.  **An ensemble of model simulations, each with slightly different initial conditions and parameters, is run forward in time. Weekly observational data on ILI prevalence are used to adjust the state variables (e.g., number of susceptible and infected individuals) and key epidemiological parameters of each ensemble member. This adjustment process nudges the ensemble mean towards the observations and reduces the ensemble spread, effectively refining the model's representation of the current state of the epidemic. Forecasts are generated by running the updated ensemble forward in time, and the spread of these forecast trajectories provides an estimate of prediction uncertainty.**
  #     *   **Interpretation and Adherence:** My code initializes 200 `SIRSModel` instances, each with randomly sampled initial states and epidemiological parameters within plausible, perturbed ranges, reflecting ensemble diversity. The "weekly observational data on ILI prevalence" is incorporated via the `latest_admissions_observation` (potentially augmented by historical `unweighted_ili`) for the week prior to `reference_date`. This observation, along with a `1.02` multiplicative `INFLATION_FACTOR` applied to parameter and initial state standard deviations during sampling, constitutes the "adjustment process" that "nudges the ensemble mean towards the observations" (through `I0` initialization) and *maintains initial ensemble spread* (through inflation and adaptive `I0` std dev) at the initiation of the forecast. The `noise_factor` for each ensemble member is also perturbed, and a `MIN_INFECTION_NOISE_FLOOR` is applied to the standard deviation of new infections. **Furthermore, stochasticity is now also applied to the recovery process within the `_advance_week` method, further enriching the ensemble's ability to capture the inherent uncertainty in disease progression and improving the representation of forecast uncertainty.** This combined stochasticity ensures sufficient ensemble diversity and the characterization of forecast uncertainty. Crucially, when `latest_admissions_observation` is zero, a more aggressive perturbation strategy is applied to `I0_member` (initial infected population) to ensure the ensemble can adequately simulate the onset of new outbreaks from low activity, and the initialization of `R0_member` is constrained to ensure a minimum proportion of the population remains susceptible (`S0_member`). The ensemble's `beta_base` and `gamma` parameters now have their *mean values for sampling* dynamically adjusted based on location-specific historical `admissions_per_ili_scaling_factor` (relative to a national average) such that higher local `admissions_per_ili_scaling_factor` leads to *higher* `beta_base_mean` and *lower* `gamma_mean` for sampling, reflecting more intense local disease dynamics at initialization. Moreover, their *sampling min/max ranges* are also dynamically adjusted to allow for a context-specific exploration of parameter space. Forecasts for `horizon >= 0` are generated by integrating each updated ensemble member forward on a weekly timestep for the specific horizons required by the competition output. `np.percentile` is used on the ensemble trajectories to quantify prediction uncertainty. The standard deviation for `I0_member` is now made more robust by ensuring an adaptive balance between a minimum absolute spread and a proportional spread, thus better embodying the "reduces ensemble spread" aspect of data assimilation by scaling uncertainty appropriately with observation magnitude. Lastly, logging is kept minimal by default, with only critical warnings explicitly caught and printed once, adhering to the "Crucial Note on Logging" requirement.

  # ## Step-by-Step Logic:
  # 1.  **Constants and Global Data Access:** Ensure `QUANTILES`, `TARGET_STR`, `ilinet_state`, `locations`, `REQUIRED_CDC_LOCATIONS` are accessible and correctly typed as per notebook context. The new constants `GLOBAL_MIN_ADMISSIONS_PER_I_RATE`, `MIN_ADMISSIONS_PER_I_RATE_STD_PROPORTION`, `MIN_ADMISSIONS_PER_I_RATE_FIXED_STD`, `MAX_ADMISSIONS_PER_I_RATE`, `DEFAULT_ADMISSIONS_PER_ILI_FACTOR_FALLBACK`, `MAX_ADMISSIONS_PER_ILI_FACTOR`, `MIN_INFECTED_SEED_FACTOR_ZERO_OBS`, `MIN_INFECTED_SEED_POP_PROP_ZERO_OBS`, `MIN_S_PROPORTION`, `MIN_INFECTION_NOISE_FLOOR`, `MIN_BLENDING_WEIGHT`, `MAX_BLENDING_WEIGHT`, `RECOVERY_NOISE_MULTIPLIER`, `DEFAULT_BETA_BASE_MEAN`, `DEFAULT_GAMMA_MEAN`, `DEFAULT_RHO_MEAN`, `I0_REL_STD_HIGH_OBS_MULTIPLIER`, `I0_MIN_ABS_STD_FACTOR`, and `PARAM_TRUNC_NORMAL_STD_DIVISOR` are defined for clarity and robust tuning.
  # 2.  **Robust Data Preprocessing and Full Augmentation (Dynamic for Current Fold):**
  #     a.  Convert all relevant date columns in `train_x`, `train_y`, `test_x`, `ilinet_state` to `pd.Timestamp`.
  #     b.  Merge `train_x` and `train_y` into `train_df`.
  #     c.  Process `ilinet_state`: Filter for `region_type == 'States'`, map `region` names to `location` (FIPS codes) using the `locations` DataFrame, select `week_start`, `unweighted_ili`, rename columns, and fill `NaN` in `ILI_estimate` with `0`. `unweighted_ili` is used as a pragmatic proxy for Google Flu Trends ILI estimates, as explicit GFT data is not provided and `weighted_ili` is often sparse at the state level.
  #     d.  **Calculate National `admissions_per_ILI_unit_factor`:** Incorporate `ilinet` (national) data to calculate a robust national `admissions_per_ILI_unit_factor`. This national factor will serve as a robust and data-driven `DEFAULT_ADMISSIONS_PER_ILI_FACTOR_FALLBACK` for individual states where local learning is insufficient. It is explicitly set to `0.0` if both national ILI and admissions are consistently zero in the active overlap period, or a ratio if only admissions exist. **This factor is capped at `MAX_ADMISSIONS_PER_ILI_FACTOR`.**
  #     e.  Filter `ilinet_data` to only include `target_end_date` up to the *latest date available in the current `train_x` (`latest_train_date`) AND apply global ILINet cutoff date (`2022-10-15`) to prevent data leakage, as per problem description.*
  #     f.  Merge `train_df` with the *filtered* and preprocessed `ilinet_data` using an `outer` merge on `target_end_date` and `location`.
  #     g.  Merge the combined history with `_locations` (a copy of the global `locations` DataFrame) to ensure `population` is available for all `location` and `target_end_date` combinations, filling any remaining `NaN` populations with a default (e.g., `1_000_000`) and converting to `int`.
  #     h.  **Pre-calculate `admissions_per_ILI_unit_factors` (scaling ILI_estimate to Admissions) for the current fold - REFINED ROBUSTNESS (Plan Item 2):**
  #         i.  Initialize an empty dictionary `admissions_per_ILI_unit_factors`.
  #         ii. Loop *once* through each unique `loc_id` in `full_history_df_for_fold`.
  #         iii. For each `loc_id`, identify the overlapping period where both `Total Influenza Admissions` and `ILI_estimate` are available.
  #         iv. Calculate a simple ratio (`y.sum() / X.sum()`) as a baseline.
  #         v.  Attempt to fit a `LinearRegression` model (`fit_intercept=False`, `positive=True`).
  #         vi. **Robust Handling (NEW):** If regression fails, yields an `inf` coefficient, or if the regression coefficient is an outlier (e.g., significantly different from the simple ratio), prefer the simple ratio. Otherwise, use the regression coefficient. The factor is capped to `[0.0, MAX_ADMISSIONS_PER_ILI_FACTOR]`. Handle cases where both ILI and admissions are zero (factor is 0.0) or only admissions exist (fallback to national default).
  #         vii. Store this factor as the `admissions_per_ILI_unit_factor` for that `loc_id`.
  #     i.  **Perform Full Augmentation of `full_history_df_for_fold` (for the current fold):**
  #         i.  Create `augmented_train_df_for_fold` as a copy of `full_history_df_for_fold`.
  #         ii. For each `loc_id` where `admissions_per_ILI_unit_factors` was calculated:
  #             1.  Identify rows in `augmented_train_df_for_fold` for that `loc_id` where `Total Influenza Admissions` is `NaN` and `ILI_estimate` is not `NaN`.
  #             2.  Apply the stored `admissions_per_ILI_unit_factor` to `ILI_estimate` to fill these `NaN` `Total Influenza Admissions` values.
  #         iii. Fill any remaining `NaN` values in `Total Influenza Admissions` in `augmented_train_df_for_fold` with `0` and convert to `int`.
  #     j.  **Pre-calculate `seasonal_multipliers_per_location` (for the current fold) - IMPROVED ROBUSTNESS WITH ADAPTIVE BLENDING:**
  #         i.  Initialize an empty dictionary `seasonal_multipliers_per_location`.
  #         ii. Calculate `national_seasonal_multipliers` by summing `Total Influenza Admissions` across all locations in `augmented_train_df_for_fold` by `epiweek`, normalizing, and handling zero means/sparse data.
  #         iii. For each unique `loc_id` in `augmented_train_df_for_fold`:
  #             1.  Calculate its `local_raw_seasonal_pattern` and its `local_seasonal_sum` (sum of admissions).
  #             2.  Determine `max_local_seasonal_sum` across all locations to normalize local sums.
  #             3.  If `local_seasonal_sum` is zero or `loc_data[TARGET_STR].nunique() <= 1`, use `national_seasonal_multipliers` directly.
  #             4.  Else, calculate `local_seasonal_multipliers` and an `adaptive_blending_weight_local` based on `local_seasonal_sum` (normalized against `max_local_seasonal_sum`) clamped between `MIN_BLENDING_WEIGHT` and `MAX_BLENDING_WEIGHT`.
  #             5.  Blend `local_seasonal_multipliers` with `national_seasonal_multipliers` using the `adaptive_blending_weight_local`. Store these for the `loc_id`.
  #         iv. Ensure all calculated seasonal multiplier series are reindexed 1-53 and `ffill()`/`bfill()` for robustness.
  # 3.  **`SIRSModel` Class Definition:** (As defined, the `SIRSModel` class includes corrected state normalization and robust updates, ensuring S+I+R = population and non-negativity, including robust initialization for zero population and `min_infected_seed`. The `_advance_week` method *applies `min_infected_seed` as an initial floor to the current `I` before calculating the infection rate factor*, ensuring minimal infection potential even if `I` is very small. It also applies `MIN_INFECTION_NOISE_FLOOR` to the standard deviation of new infections to ensure sufficient stochasticity even at low rates, enhancing ensemble diversity, and preventing filter collapse from lack of exploration. **Crucially, the `_advance_week` method is further enhanced to also apply stochasticity (Gaussian noise) to the recovery process, allowing the ensemble to better capture the inherent variability in disease progression and improving the representation of forecast uncertainty.**)
  # 4.  **Forecast Generation (within `fit_and_predict_fn`):**
  #     a.  Initialize an empty DataFrame `all_quantile_preds` with `test_x`'s index and required quantile columns.
  #     b.  Loop through each unique `(location, reference_date)` combination in `test_x` (grouped as `test_groups`).
  #     c.  For each `loc_id` and `ref_date`:
  #         i.  Retrieve `current_population`.
  #         ii. Filter `augmented_train_df_for_fold` for the specific `loc_id` and `target_end_date` equal to `ref_date - pd.Timedelta(weeks=1)`.
  #         iii. Get `latest_admissions_observation` from this filtered history. Handle cases where history is empty (default to `0`).
  #         iv. Retrieve the pre-calculated `seasonal_multipliers` and `admissions_per_ili_scaling_factor` for this location *from the current fold's calculations*. Then derive `admissions_per_infected_rate_for_sirs` ensuring it's not strictly zero (using `GLOBAL_MIN_ADMISSIONS_PER_I_RATE`) to prevent division errors.
  #         v.  **Handle `horizon = -1`:** For any rows in `group_df` where `horizon == -1`, set the quantile predictions directly to `latest_admissions_observation` (all quantiles equal).
  #         vi. **Robust Population Check:** If `current_population <= 0`, directly set all forward predictions for this location (horizons >= 0) to `0` and continue to the next location.
  #         vii. **Calculate National Average Admissions-per-ILI Factor:** Compute `national_avg_admissions_per_ili_factor` from all `admissions_per_ILI_unit_factors` calculated in the current fold.
  #         viii. **Initialize Ensemble (for `horizon >= 0` - "initialization-time pseudo-assimilation" with dynamic parameter priors - REFINED PLAN ITEM 1 & 3):** Initialize `N_ensemble = 200` `SIRSModel` instances. This step represents the "adjustment process" described in the method contract, occurring at the forecast `reference_date` to set the initial (posterior) ensemble conditions:
  #             *   Define original prior ranges for `beta_base` (0.05-0.35), `gamma` (0.1-0.5), `rho` (0.005-0.025) based on typical epidemiological values. These are the *absolute global bounds*.
  #             *   **Crucial Improvement: Dynamically adjust the *mean* and *ranges* of `beta_base` and `gamma` for each location.**
  #                 *   **`beta_gamma_scaling_factor_capped`:** Calculate a scaling factor by comparing the location's `admissions_per_ili_scaling_factor` to the `national_avg_admissions_per_ili_factor`. This factor is capped to prevent extreme adjustments (e.g., 0.5 to 2.0).
  #                 *   **Dynamic Range Adjustment:**
  #                     *   For `beta_base`: Calculate `beta_base_mean` by scaling `DEFAULT_BETA_BASE_MEAN` with `beta_gamma_scaling_factor_capped`. Define `base_beta_range_span = (beta_base_max_orig - beta_base_min_orig) / 2`. Calculate `beta_base_dynamic_span = base_beta_range_span * beta_gamma_scaling_factor_capped`. Set `beta_base_min_loc = max(beta_base_min_orig, beta_base_mean - beta_base_dynamic_span)` and `beta_base_max_loc = min(beta_base_max_orig, beta_base_mean + beta_base_dynamic_span)`.
  #                     *   For `gamma`: Calculate `gamma_mean` by scaling `DEFAULT_GAMMA_MEAN` inversely with `beta_gamma_scaling_factor_capped`. Define `base_gamma_range_span = (gamma_max_orig - gamma_min_orig) / 2`. Calculate `gamma_dynamic_span = base_gamma_range_span / beta_gamma_scaling_factor_capped`. Set `gamma_min_loc = max(gamma_min_orig, gamma_mean - gamma_dynamic_span)` and `gamma_max_loc = min(gamma_max_orig, gamma_mean + gamma_dynamic_span)`.
  #                 *   These adjustments ensure that the model's intrinsic transmissibility and recovery rates are adapted to the local observed `admissions_per_ili` patterns, such that higher local `admissions_per_ili_scaling_factor` leads to *higher* `beta_base_mean` and *lower* `gamma_mean` for sampling, reflecting more intense local disease dynamics. The *sampling ranges* (`beta_base_min_loc`/`max_loc`, `gamma_min_loc`/`max_loc`) are also dynamically adjusted to allow for a context-specific exploration of parameter space, clipped within the overall global bounds.
  #                 *   **`rho_mean`:** Use `DEFAULT_RHO_MEAN` for its mean and its original global range for sampling.
  #             *   For each member, calculate standard deviation for `beta_base`, `gamma`, `rho` based on their *dynamically adjusted local prior ranges* (for `beta_base`, `gamma`) or *original global prior ranges* (for `rho`), *and `INFLATION_FACTOR`*. This calculation is refined to `(max - min) / PARAM_TRUNC_NORMAL_STD_DIVISOR * INFLATION_FACTOR` to better represent a truncated normal distribution.
  #             *   Sample `beta_base_member`, `gamma_member`, `rho_member` from truncated normal distributions centered at their dynamically adjusted means, with their standard deviations *multiplied by `INFLATION_FACTOR`*. Clip these values to their *dynamically adjusted local prior ranges*.
  #             *   **Perturb `admissions_per_I_rate`:** Perturb `admissions_per_infected_rate_for_sirs` for each member (using `MIN_ADMISSIONS_PER_I_RATE_STD_PROPORTION`, `MIN_ADMISSIONS_PER_I_RATE_FIXED_STD` and `INFLATION_FACTOR`) to get `admissions_per_I_rate_member`, ensuring it's positive and within a plausible range (using `GLOBAL_MIN_ADMISSIONS_PER_I_RATE` and `MAX_ADMISSIONS_PER_I_RATE`). It explicitly ensures that even if the learned factor is near zero, the perturbation maintains sufficient diversity, drawing primarily from `MIN_ADMISSIONS_PER_I_RATE_FIXED_STD`.
  #             *   **Derive `I0_member`:** Derive `I0_initial_guess` from `latest_admissions_observation` using the inverse of `admissions_per_I_rate_member`. Cap this `estimated_I0_from_admissions` to a reasonable percentage of the population (e.g., 5%).
  #             *   **Dynamic `I0` Seeding & Refined `I0` Standard Deviation (REFINED - Plan Item 3):** If `derived_I0_from_obs` is below `MIN_INFECTED_SEED_LOC`:
  #                 *   `I0_mean_for_perturbation` is set considering `MIN_INFECTED_SEED_LOC * MIN_INFECTED_SEED_FACTOR_ZERO_OBS` and also a `current_population * MIN_INFECTED_SEED_POP_PROP_ZERO_OBS` to ensure a robust seed.
  #                 *   A component to `I0_std_dev_for_perturbation` is added that scales with population for low activity, enhancing diversity for outbreak seeding.
  #             *   Otherwise, for meaningful observations, the `I0_std_dev_for_perturbation` is adaptively calculated as the maximum of an absolute floor (`MIN_INFECTED_SEED_LOC * I0_MIN_ABS_STD_FACTOR`) and a proportional component (`derived_I0_from_obs * I0_REL_STD_HIGH_OBS_MULTIPLIER`), all scaled by `INFLATION_FACTOR`. Sample `I0_member` from this distribution. This ensures a diverse set of initial infected populations across the ensemble, with uncertainty scaling appropriately with the observation's magnitude.
  #             *   `I0_member` will be clipped to be non-negative and capped at a realistic maximum (e.g., 10% of population).
  #             *   **Robust `R0` and `S0` Initialization:** Sample `R0_member` as a proportion of `current_population` (e.g., 0.1% to 5%). Then, adjust `R0_member` to ensure `S0_member` (initial susceptibles) is at least `MIN_S_PROPORTION` of the population. This prevents `S0` from being inadvertently depleted by `I0` and `R0` when `I0` is large, allowing for meaningful outbreaks. The `R0_member` will be capped to `current_population - I0_member - (MIN_S_PROPORTION * current_population)` if `S0` would otherwise fall below this threshold.
  #             *   Calculate `S0_member = current_population - I0_member - R0_member`. Ensure non-negative.
  #             *   **Perturb `noise_factor`** for each member using a truncated normal distribution with inflation to enhance ensemble diversity. This calculation is refined to `(max - min) / PARAM_TRUNC_NORMAL_STD_DIVISOR * INFLATION_FACTOR` to better represent a truncated normal distribution.
  #             *   Instantiate `SIRSModel` with these sampled initial states and parameters, ensuring robust internal normalization (S+I+R = population), and passing `MIN_INFECTED_SEED_LOC`.
  #         ix. **Forecast Trajectories (for `horizon >= 0`):** Create a dictionary `ensemble_trajectories_by_horizon_step`. For `h_step_simulation` from 0 to `max(horizons)`: Calculate `current_forecast_week_date` and `current_epiweek`. For each ensemble member, advance the model by one week using `_advance_week` and collect `predicted_admissions`. Store these 200 predictions. This uses `seasonal_multipliers` as a proxy for AH forcing throughout the forecast. The model integrates forward only for the specific short horizons required by the competition output, not the full 300 days as per the general description, as this is a pragmatic optimization for competition context and lack of future AH/ILI data.
  #         x. **Aggregate and Quantiles (for `horizon >= 0`):** For each row in `group_df` with `horizon >= 0`, retrieve the 200 predictions for the corresponding `horizon`. Calculate `np.percentile`. Ensure non-negativity. Then, strictly enforce monotonicity by iterating and setting `quant_values[k] = max(quant_values[k], quant_values[k-1])` *on float values first*, then round to integer (`np.round().astype(int)`). Store these quantile predictions in `all_quantile_preds` for the corresponding `test_x` rows.
  # 5.  **Return:** `all_quantile_preds` DataFrame, ensuring index and column names are correct and values are integer.

  # --- 1. Data Preprocessing and Augmentation for the current fold ---
  train_df = train_x.copy()
  train_df[TARGET_STR] = (
      train_y.copy()
  )  # Use .copy() to prevent SettingWithCopyWarning

  # Make defensive copies of global dataframes to avoid unintended modifications
  _locations = locations.copy()
  _ilinet_state = ilinet_state.copy()
  _ilinet_national = ilinet.copy()  # Added for national fallback factor

  # Convert date columns to datetime objects
  train_df['target_end_date'] = pd.to_datetime(train_df['target_end_date'])
  test_x_copy = test_x.copy()  # Make a copy of test_x to modify
  test_x_copy['target_end_date'] = pd.to_datetime(
      test_x_copy['target_end_date']
  )
  _ilinet_state['week_start'] = pd.to_datetime(_ilinet_state['week_start'])
  _ilinet_national['week_start'] = pd.to_datetime(
      _ilinet_national['week_start']
  )  # Added

  # Filter ilinet_state for 'States' and relevant FIPS codes
  ilinet_state_filtered = _ilinet_state[
      _ilinet_state['region_type'] == 'States'
  ].copy()

  # Map ilinet_state 'region' to 'location' (FIPS) using the global 'locations' df
  locations_map = (
      _locations[['location_name', 'location']]
      .astype({'location': int})
      .set_index('location_name')['location']
      .to_dict()
  )
  ilinet_state_filtered['location'] = ilinet_state_filtered['region'].map(
      locations_map
  )
  ilinet_state_filtered = ilinet_state_filtered.dropna(
      subset=['location']
  ).copy()
  ilinet_state_filtered['location'] = ilinet_state_filtered['location'].astype(
      int
  )

  # Select relevant ILI columns and rename for consistency
  ilinet_data_base = ilinet_state_filtered[
      ['week_start', 'location', 'unweighted_ili']
  ].rename(
      columns={
          'week_start': 'target_end_date',
          'unweighted_ili': 'ILI_estimate',
      }
  )
  # Fill missing ILI_estimate values with 0. This is a pragmatic choice to make ILI data usable.
  ilinet_data_base['ILI_estimate'] = ilinet_data_base['ILI_estimate'].fillna(0)

  # CRUCIAL: Filter ILINet data to prevent data leakage.
  # Only use ILINet data up to the latest date in the current train_x
  latest_train_date = train_df['target_end_date'].max()

  GLOBAL_ILINET_CUTOFF_DATE = pd.to_datetime('2022-10-15')
  ilinet_data_for_fold = ilinet_data_base[
      (ilinet_data_base['target_end_date'] <= latest_train_date)
      & (
          ilinet_data_base['target_end_date'] <= GLOBAL_ILINET_CUTOFF_DATE
      )  # Enforce global cutoff
  ].copy()

  # --- Improvement 1: Calculate National ILI-Admissions Factor for fallback (Adherence to Implementation Plan Step 2.d) ---
  national_ilinet_data = _ilinet_national[
      _ilinet_national['region_type'] == 'National'
  ].copy()
  national_ilinet_data = national_ilinet_data[
      ['week_start', 'unweighted_ili']
  ].rename(
      columns={
          'week_start': 'target_end_date',
          'unweighted_ili': 'ILI_estimate_national',
      }
  )
  national_ilinet_data['ILI_estimate_national'] = national_ilinet_data[
      'ILI_estimate_national'
  ].fillna(0)

  # Filter national ILI data to prevent data leakage, aligned with local ILINet data filtering
  national_ilinet_data = national_ilinet_data[
      (national_ilinet_data['target_end_date'] <= latest_train_date)
      & (national_ilinet_data['target_end_date'] <= GLOBAL_ILINET_CUTOFF_DATE)
  ].copy()

  # Aggregate actual admissions data to national level
  national_admissions = (
      train_df.groupby('target_end_date')[TARGET_STR].sum().reset_index()
  )
  # Filter national admissions data to prevent data leakage
  national_admissions = national_admissions[
      national_admissions['target_end_date'] <= latest_train_date
  ].copy()

  national_overlap_data = pd.merge(
      national_admissions,
      national_ilinet_data,
      on='target_end_date',
      how='inner',
  ).dropna(subset=[TARGET_STR, 'ILI_estimate_national'])

  # Filter for periods with actual activity
  national_active_overlap_data = national_overlap_data[
      (national_overlap_data['ILI_estimate_national'] > 0)
      | (national_overlap_data[TARGET_STR] > 0)
  ].copy()

  national_default_admissions_per_ili_factor = DEFAULT_ADMISSIONS_PER_ILI_FACTOR_FALLBACK  # Initialize with global constant

  if not national_active_overlap_data.empty:
    X_ili_nat = national_active_overlap_data[['ILI_estimate_national']]
    y_admissions_nat = national_active_overlap_data[TARGET_STR]

    if (
        y_admissions_nat.sum() == 0
        and X_ili_nat['ILI_estimate_national'].sum() == 0
    ):
      national_default_admissions_per_ili_factor = (
          0.0  # No national activity, factor is 0
      )
    elif (
        X_ili_nat['ILI_estimate_national'].sum() == 0
    ):  # Only admissions activity, ILI not explanatory
      # Keep the initial DEFAULT_ADMISSIONS_PER_ILI_FACTOR_FALLBACK or a reasonable base rate
      pass  # No change from initialized fallback if ILI is zero but admissions exist
    else:  # There is ILI activity, try to learn a factor
      simple_ratio_nat = (
          y_admissions_nat.sum() / X_ili_nat['ILI_estimate_national'].sum()
          if X_ili_nat['ILI_estimate_national'].sum() > 0
          else 0.0
      )

      if (
          y_admissions_nat.nunique() > 1
          and X_ili_nat['ILI_estimate_national'].nunique() > 1
      ):
        try:
          reg_nat = LinearRegression(fit_intercept=False, positive=True)
          with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            reg_nat.fit(X_ili_nat, y_admissions_nat)
          factor_coef_nat = reg_nat.coef_[0]

          # Robustness check for national factor (Plan Item 2)
          if not np.isfinite(factor_coef_nat) or (
              simple_ratio_nat > 1e-6
              and factor_coef_nat > (2 * simple_ratio_nat + 1e-6)
          ):
            national_default_admissions_per_ili_factor = max(
                0.0, min(simple_ratio_nat, MAX_ADMISSIONS_PER_ILI_FACTOR)
            )
          else:
            national_default_admissions_per_ili_factor = max(
                0.0, min(factor_coef_nat, MAX_ADMISSIONS_PER_ILI_FACTOR)
            )

        except (ValueError, np.linalg.LinAlgError):
          # Fallback to simple ratio if regression fails
          national_default_admissions_per_ili_factor = max(
              0.0, min(simple_ratio_nat, MAX_ADMISSIONS_PER_ILI_FACTOR)
          )
      else:  # Not enough variance for meaningful regression, but ILI sum > 0
        national_default_admissions_per_ili_factor = max(
            0.0, min(simple_ratio_nat, MAX_ADMISSIONS_PER_ILI_FACTOR)
        )
  # Ensure national_default_admissions_per_ili_factor is capped (Plan Item 2)
  national_default_admissions_per_ili_factor = min(
      national_default_admissions_per_ili_factor, MAX_ADMISSIONS_PER_ILI_FACTOR
  )
  # --- End Improvement 1 ---

  # Combine train_df and ilinet_data for the current fold
  # Keep only relevant columns from train_df to avoid conflicts during merge with test_x columns later
  full_history_df_for_fold = pd.merge(
      train_df[['target_end_date', 'location', TARGET_STR]],
      ilinet_data_for_fold,
      on=['target_end_date', 'location'],
      how='outer',
  )

  # Merge population information once and ensure it's available for all entries
  full_history_df_for_fold = pd.merge(
      full_history_df_for_fold,
      _locations[['location', 'population']],
      on='location',
      how='left',
  )
  full_history_df_for_fold['population'] = (
      full_history_df_for_fold['population'].fillna(1_000_000).astype(int)
  )

  full_history_df_for_fold = full_history_df_for_fold.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # Store scaling factors and seasonal patterns per location for THIS FOLD
  admissions_per_ILI_unit_factors = {}
  seasonal_multipliers_per_location = {}

  # Pre-calculate factor for converting ILI_estimate to Admissions using Linear Regression for all locations in THIS FOLD
  # (Adherence to Implementation Plan Step 2.h - Refactored for clarity and robustness, REFINED PLAN ITEM 2)
  for loc_id in full_history_df_for_fold['location'].unique():
    loc_data_for_factor = full_history_df_for_fold[
        full_history_df_for_fold['location'] == loc_id
    ].copy()

    # Initialize factor with the robust national default
    mean_factor = national_default_admissions_per_ili_factor

    # Only consider data points where we have both admissions and ILI estimates
    overlap_data = loc_data_for_factor.dropna(
        subset=[TARGET_STR, 'ILI_estimate']
    ).copy()

    # Filter for periods with actual activity (either ILI or admissions > 0)
    # This prevents learning from entirely zero historical periods if there's no signal.
    active_overlap_data = overlap_data[
        (overlap_data['ILI_estimate'] > 0) | (overlap_data[TARGET_STR] > 0)
    ].copy()

    if not active_overlap_data.empty:
      X_ili = active_overlap_data[['ILI_estimate']]
      y_admissions = active_overlap_data[TARGET_STR]

      if y_admissions.sum() == 0 and X_ili['ILI_estimate'].sum() == 0:
        # If there's no actual activity in the overlap, the factor should be 0.0
        mean_factor = 0.0
      elif X_ili['ILI_estimate'].sum() == 0:
        # If ILI is consistently zero but admissions exist, ILI is not a good local predictor.
        # Fallback to the national default (which might be 0.0 or a small positive value).
        pass  # mean_factor already initialized to national_default_admissions_per_ili_factor
      else:  # X_ili['ILI_estimate'].sum() > 0, so there's some ILI signal
        simple_ratio = (
            y_admissions.sum() / X_ili['ILI_estimate'].sum()
            if X_ili['ILI_estimate'].sum() > 0
            else 0.0
        )

        # Condition for regression: enough unique values for both X and y for meaningful regression
        if y_admissions.nunique() > 1 and X_ili['ILI_estimate'].nunique() > 1:
          try:
            # Use positive=True to ensure non-negative coefficient, as a scaling factor
            reg = LinearRegression(fit_intercept=False, positive=True)
            with warnings.catch_warnings():
              warnings.simplefilter(
                  'ignore', UserWarning
              )  # Ignore convergence warnings for LinearRegression
              reg.fit(X_ili, y_admissions)
            factor_coef = reg.coef_[0]

            # Robustness check for local factor (Plan Item 2)
            # If regression coef is too high compared to simple ratio, prefer simple ratio
            if not np.isfinite(factor_coef) or (
                simple_ratio > 1e-6 and factor_coef > (2 * simple_ratio + 1e-6)
            ):
              mean_factor = max(
                  0.0, min(simple_ratio, MAX_ADMISSIONS_PER_ILI_FACTOR)
              )
            else:
              mean_factor = max(
                  0.0, min(factor_coef, MAX_ADMISSIONS_PER_ILI_FACTOR)
              )

          except (ValueError, np.linalg.LinAlgError):
            # Fallback to simple ratio if regression fails
            mean_factor = max(
                0.0, min(simple_ratio, MAX_ADMISSIONS_PER_ILI_FACTOR)
            )
        else:  # Not enough variance for meaningful regression, but ILI sum > 0
          mean_factor = max(
              0.0, min(simple_ratio, MAX_ADMISSIONS_PER_ILI_FACTOR)
          )

    admissions_per_ILI_unit_factors[loc_id] = mean_factor

  # Calculate national average admissions_per_ILI_unit_factor for scaling dynamic priors
  # This is an important step to make beta_base_mean adaptable (Adherence to Plan Step 4.c.vii)
  national_avg_admissions_per_ili_factor = np.mean(
      list(admissions_per_ILI_unit_factors.values())
  )
  # Ensure it's not zero to prevent division by zero in scaling, use a robust minimum.
  if national_avg_admissions_per_ili_factor < 1e-9:
    national_avg_admissions_per_ili_factor = (
        1e-9  # Set to a very small positive number if effectively zero.
    )

  # Perform full augmentation of `Total Influenza Admissions` in `full_history_df_for_fold`
  # (Adherence to Implementation Plan Step 2.i)
  augmented_train_df_for_fold = full_history_df_for_fold.copy()
  for loc_id, factor in admissions_per_ILI_unit_factors.items():
    mask_loc_missing_admissions = (
        (augmented_train_df_for_fold['location'] == loc_id)
        & augmented_train_df_for_fold[TARGET_STR].isna()
        & augmented_train_df_for_fold['ILI_estimate'].notna()
    )

    if not augmented_train_df_for_fold.loc[mask_loc_missing_admissions].empty:
      augmented_train_df_for_fold.loc[
          mask_loc_missing_admissions, TARGET_STR
      ] = (
          augmented_train_df_for_fold.loc[
              mask_loc_missing_admissions, 'ILI_estimate'
          ]
          * factor
      )

  # Finalize augmented_train_df by filling remaining NaNs for TARGET_STR with 0 and converting to int
  augmented_train_df_for_fold[TARGET_STR] = (
      augmented_train_df_for_fold[TARGET_STR].fillna(0).astype(int)
  )

  # Pre-calculate Seasonal Multipliers from the fully augmented data for THIS FOLD - IMPROVED ROBUSTNESS
  # (Adherence to Implementation Plan Step 2.j - Now with Adaptive Blending)
  augmented_train_df_for_fold['epiweek'] = (
      augmented_train_df_for_fold['target_end_date'].dt.isocalendar().week
  )

  # 1. Calculate National Seasonal Multipliers (from the fully augmented data)
  national_seasonal_pattern_raw = augmented_train_df_for_fold.groupby(
      'epiweek'
  )[TARGET_STR].sum()
  national_seasonal_multipliers = national_seasonal_pattern_raw.reindex(
      range(1, 54)
  )
  national_seasonal_multipliers = national_seasonal_multipliers.ffill().bfill()
  if (
      national_seasonal_multipliers.mean() > 1e-9
  ):  # Check for non-zero mean before division
    national_seasonal_multipliers /= national_seasonal_multipliers.mean()
  else:  # If all values are zero or near zero, default to 1.0 (no seasonality)
    national_seasonal_multipliers = pd.Series(1.0, index=range(1, 54))

  # Calculate max local seasonal sum for normalization across all locations
  local_seasonal_sums = {}
  for loc_id_sum in augmented_train_df_for_fold['location'].unique():
    loc_data_sum = augmented_train_df_for_fold[
        augmented_train_df_for_fold['location'] == loc_id_sum
    ].copy()
    local_seasonal_sums[loc_id_sum] = (
        loc_data_sum.groupby('epiweek')[TARGET_STR].mean().sum()
    )

  max_local_seasonal_sum = (
      max(local_seasonal_sums.values()) if local_seasonal_sums else 1.0
  )  # Avoid div by zero

  for loc_id in augmented_train_df_for_fold['location'].unique():
    loc_data = augmented_train_df_for_fold[
        augmented_train_df_for_fold['location'] == loc_id
    ].copy()

    local_raw_seasonal_pattern = loc_data.groupby('epiweek')[TARGET_STR].mean()
    local_seasonal_sum = local_raw_seasonal_pattern.sum()

    # 2. Conditional Blending/Fallback for Local Seasonal Multipliers with adaptive weight (Plan Item 4)
    # Use national pattern if local data is very sparse or non-variable
    if (
        local_seasonal_sum < 1e-6 or loc_data[TARGET_STR].nunique() <= 1
    ):  # Added threshold to check if sum is effectively zero
      seasonal_multipliers_per_location[loc_id] = national_seasonal_multipliers
    else:
      local_seasonal_multipliers = local_raw_seasonal_pattern.reindex(
          range(1, 54)
      )
      local_seasonal_multipliers = local_seasonal_multipliers.ffill().bfill()

      if local_seasonal_multipliers.mean() > 1e-9:
        local_seasonal_multipliers /= local_seasonal_multipliers.mean()
      else:  # Fallback to 1.0 if local mean is zero after ffill/bfill, though unlikely given sum > 0 check
        local_seasonal_multipliers = pd.Series(1.0, index=range(1, 54))

      # Adaptive blending weight based on local data richness
      normalized_richness_score = local_seasonal_sum / max_local_seasonal_sum
      adaptive_blending_weight_local = (
          MIN_BLENDING_WEIGHT
          + (MAX_BLENDING_WEIGHT - MIN_BLENDING_WEIGHT)
          * normalized_richness_score
      )

      blended_multipliers = (
          adaptive_blending_weight_local * local_seasonal_multipliers
          + (1 - adaptive_blending_weight_local) * national_seasonal_multipliers
      )
      seasonal_multipliers_per_location[loc_id] = blended_multipliers

  # --- 4. Forecast Generation (Adherence to Implementation Plan Step 4) ---
  all_quantile_preds = pd.DataFrame(
      index=test_x_copy.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )
  N_ensemble = 200
  INFLATION_FACTOR = (
      1.02  # Multiplicative inflation for parameter/state std during sampling
  )

  test_groups = test_x_copy.groupby(['reference_date', 'location'])

  # Helper function for truncated normal sampling
  def get_truncated_normal_sample(mean, std, min_val, max_val):
    if (
        std <= 1e-9
    ):  # Handle very small or zero std dev, return mean if distribution is degenerate
      return mean
    a = (min_val - mean) / std
    b = (max_val - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std)

  for (ref_date, loc_id), group_df in test_groups:
    current_population = group_df['population'].iloc[0]

    history_for_loc = (
        augmented_train_df_for_fold[
            (augmented_train_df_for_fold['location'] == loc_id)
            & (
                augmented_train_df_for_fold['target_end_date']
                == ref_date - pd.Timedelta(weeks=1)
            )
        ]
        .sort_values('target_end_date')
        .copy()
    )

    latest_admissions_observation = (
        history_for_loc[TARGET_STR].iloc[-1] if not history_for_loc.empty else 0
    )
    latest_admissions_observation = max(0, latest_admissions_observation)

    horizon_minus_one_rows = group_df[group_df['horizon'] == -1]
    for row_idx in horizon_minus_one_rows.index:
      quant_values = np.array(
          [float(latest_admissions_observation)] * len(QUANTILES)
      )
      all_quantile_preds.loc[row_idx] = np.round(quant_values).astype(int)

    horizons_to_simulate = group_df[group_df['horizon'] >= 0][
        'horizon'
    ].unique()
    if len(horizons_to_simulate) == 0:
      continue

    if current_population <= 0:
      for row_idx, row in group_df[group_df['horizon'] >= 0].iterrows():
        all_quantile_preds.loc[row_idx] = [0] * len(QUANTILES)
      continue

    # Dynamic MIN_INFECTED_SEED_LOC based on current_population (Plan Item 2.b)
    MIN_INFECTED_SEED_LOC = max(
        1.0, current_population * 1e-7
    )  # e.g., at least 1, or 0.00001% of population

    seasonal_multipliers = seasonal_multipliers_per_location.get(
        loc_id, national_seasonal_multipliers
    )  # Use the now more robust national fallback

    # This factor scales ILI_estimate to Admissions based on historical relationship.
    admissions_per_ili_scaling_factor = admissions_per_ILI_unit_factors.get(
        loc_id, national_default_admissions_per_ili_factor
    )

    # This factor represents admissions per infected individual in the SIRS model.
    # It is derived from the ILI scaling factor, assuming ILI is proportional to 'I',
    # and clamped to ensure it's never strictly zero for division robustness.
    admissions_per_infected_rate_for_sirs = max(
        GLOBAL_MIN_ADMISSIONS_PER_I_RATE, admissions_per_ili_scaling_factor
    )

    ensemble_members = []
    for _ in range(N_ensemble):
      # Original GLOBAL parameter ranges (fixed) - (PLAN ITEM 2)
      # These are now treated as the absolute global bounds, not the sampling bounds.
      beta_base_min_orig, beta_base_max_orig = 0.05, 0.35
      gamma_min_orig, gamma_max_orig = 0.1, 0.5
      rho_min_orig, rho_max_orig = 0.005, 0.025

      # REFINEMENT: Define R0_prop_min_orig and R0_prop_max_orig
      R0_prop_min_orig, R0_prop_max_orig = 0.001, 0.05

      # --- IMPROVEMENT: Dynamically adjust MEANS and RANGES of epidemiological parameters (PLAN ITEM 2) ---

      # 1. Scale beta_base_mean and gamma_mean based on local admissions_per_ili_scaling_factor relative to national average
      beta_gamma_scaling_factor_raw = (
          admissions_per_ili_scaling_factor
          / national_avg_admissions_per_ili_factor
      )
      # Cap the scaling factor to prevent extreme values, e.g., 0.5 to 2.0
      beta_gamma_scaling_factor_capped = max(
          0.5, min(2.0, beta_gamma_scaling_factor_raw)
      )

      # NEW LOGIC for mean: Higher scaling factor -> higher beta, lower gamma
      beta_base_mean = DEFAULT_BETA_BASE_MEAN * beta_gamma_scaling_factor_capped
      gamma_mean = DEFAULT_GAMMA_MEAN / beta_gamma_scaling_factor_capped

      # Ensure they stay within their *global original* bounds for the mean calculation
      beta_base_mean = max(
          beta_base_min_orig, min(beta_base_mean, beta_base_max_orig)
      )
      gamma_mean = max(gamma_min_orig, min(gamma_mean, gamma_max_orig))

      # Use default rho_mean, which is usually a slower process
      rho_mean = DEFAULT_RHO_MEAN

      # --- Adaptive Range Calculation for Beta ---
      base_beta_range_span = (beta_base_max_orig - beta_base_min_orig) / 2
      beta_base_dynamic_span = (
          base_beta_range_span * beta_gamma_scaling_factor_capped
      )

      beta_base_min_loc = max(
          beta_base_min_orig, beta_base_mean - beta_base_dynamic_span
      )
      beta_base_max_loc = min(
          beta_base_max_orig, beta_base_mean + beta_base_dynamic_span
      )

      # Ensure min_loc < max_loc for truncnorm, otherwise collapse to mean
      if beta_base_min_loc >= beta_base_max_loc:
        beta_base_min_loc = beta_base_mean * (1 - 1e-3)
        beta_base_max_loc = beta_base_mean * (1 + 1e-3)
        beta_base_min_loc = max(beta_base_min_orig, beta_base_min_loc)
        beta_base_max_loc = min(beta_base_max_orig, beta_base_max_loc)

      # --- Adaptive Range Calculation for Gamma ---
      base_gamma_range_span = (gamma_max_orig - gamma_min_orig) / 2
      gamma_dynamic_span = (
          base_gamma_range_span / beta_gamma_scaling_factor_capped
      )  # Inverse scaling

      gamma_min_loc = max(gamma_min_orig, gamma_mean - gamma_dynamic_span)
      gamma_max_loc = min(gamma_max_orig, gamma_mean + gamma_dynamic_span)

      # Ensure min_loc < max_loc for truncnorm, otherwise collapse to mean
      if gamma_min_loc >= gamma_max_loc:
        gamma_min_loc = gamma_mean * (1 - 1e-3)
        gamma_max_loc = gamma_mean * (1 + 1e-3)
        gamma_min_loc = max(gamma_min_orig, gamma_min_loc)
        gamma_max_loc = min(gamma_max_orig, gamma_max_loc)

      # --- End IMPROVEMENT ---

      # Sample beta_base from a truncated normal distribution. Std dev based on new dynamic local range.
      beta_base_std = (
          (beta_base_max_loc - beta_base_min_loc)
          / PARAM_TRUNC_NORMAL_STD_DIVISOR
      ) * INFLATION_FACTOR
      beta_base = get_truncated_normal_sample(
          beta_base_mean, beta_base_std, beta_base_min_loc, beta_base_max_loc
      )

      # Sample gamma from a truncated normal distribution. Std dev based on new dynamic local range.
      gamma_std = (
          (gamma_max_loc - gamma_min_loc) / PARAM_TRUNC_NORMAL_STD_DIVISOR
      ) * INFLATION_FACTOR
      gamma = get_truncated_normal_sample(
          gamma_mean, gamma_std, gamma_min_loc, gamma_max_loc
      )

      # Sample rho from a truncated normal distribution (uses original global bounds as no specific local adaptation for rho)
      rho_std = (
          (rho_max_orig - rho_min_orig) / PARAM_TRUNC_NORMAL_STD_DIVISOR
      ) * INFLATION_FACTOR
      rho = get_truncated_normal_sample(
          rho_mean, rho_std, rho_min_orig, rho_max_orig
      )

      # Unified Perturbation for admissions_per_I_rate_member (Adherence to Implementation Plan Step 4.c.viii)
      admissions_per_I_rate_mean = admissions_per_infected_rate_for_sirs

      admissions_per_I_rate_std = max(
          MIN_ADMISSIONS_PER_I_RATE_FIXED_STD * INFLATION_FACTOR,
          admissions_per_I_rate_mean
          * MIN_ADMISSIONS_PER_I_RATE_STD_PROPORTION
          * INFLATION_FACTOR,
      )

      admissions_per_I_rate_member = get_truncated_normal_sample(
          admissions_per_I_rate_mean,
          admissions_per_I_rate_std,
          GLOBAL_MIN_ADMISSIONS_PER_I_RATE,
          MAX_ADMISSIONS_PER_I_RATE,  # Uses the new, lower MAX_ADMISSIONS_PER_I_RATE (PLAN ITEM 1)
      )

      # Derive I0_member_raw from observation and sampled rate, with dynamic seeding for low activity
      # (Adherence to Implementation Plan Step 4.c.viii, REFINED PLAN ITEM 3)
      derived_I0_from_obs = (
          float(latest_admissions_observation) / admissions_per_I_rate_member
      )

      # Cap derived_I0_from_obs to a reasonable proportion of the population
      derived_I0_from_obs_capped = min(
          derived_I0_from_obs, current_population * 0.05
      )
      derived_I0_from_obs_capped = max(0.0, derived_I0_from_obs_capped)

      I0_min_bound = MIN_INFECTED_SEED_LOC
      I0_max_bound = (
          current_population * 0.1
      )  # Max 10% of population infected initially

      if (
          derived_I0_from_obs_capped < MIN_INFECTED_SEED_LOC
      ):  # If observation is zero or very small
        # Refined I0 mean to include a population proportion for seeding (Plan Item 3)
        I0_mean_for_perturbation = max(
            MIN_INFECTED_SEED_LOC * MIN_INFECTED_SEED_FACTOR_ZERO_OBS,
            current_population * MIN_INFECTED_SEED_POP_PROP_ZERO_OBS,
        )
        # IMPROVEMENT: I0_std_dev_for_perturbation made proportional to I0_mean_for_perturbation (PLAN ITEM 3)
        I0_std_dev_for_perturbation = (
            max(
                MIN_INFECTED_SEED_LOC * I0_MIN_ABS_STD_FACTOR,  # Absolute floor
                I0_mean_for_perturbation
                * I0_REL_STD_HIGH_OBS_MULTIPLIER,  # Proportional component
            )
            * INFLATION_FACTOR
        )
      else:  # If observation is meaningful, refine standard deviation (Adherence to Plan Step 4.c.viii - NEW)
        I0_mean_for_perturbation = derived_I0_from_obs_capped

        # New: Adaptive I0 standard deviation for meaningful observations (Plan Item 3)
        # Ensures a floor for absolute std (derived from MIN_INFECTED_SEED_LOC)
        # while also having a proportional component (I0_REL_STD_HIGH_OBS_MULTIPLIER)
        I0_std_dev_for_perturbation = (
            max(
                MIN_INFECTED_SEED_LOC * I0_MIN_ABS_STD_FACTOR,
                derived_I0_from_obs_capped * I0_REL_STD_HIGH_OBS_MULTIPLIER,
            )
            * INFLATION_FACTOR
        )

      I0_member = get_truncated_normal_sample(
          I0_mean_for_perturbation,
          I0_std_dev_for_perturbation,
          I0_min_bound,
          I0_max_bound,
      )

      # REFINEMENT: Sample R0_member_prop from a truncated normal
      R0_prop_mean = (R0_prop_min_orig + R0_prop_max_orig) / 2
      R0_prop_std = (
          (R0_prop_max_orig - R0_prop_min_orig) / PARAM_TRUNC_NORMAL_STD_DIVISOR
      ) * INFLATION_FACTOR
      R0_member_prop = get_truncated_normal_sample(
          R0_prop_mean, R0_prop_std, R0_prop_min_orig, R0_prop_max_orig
      )

      # Calculate R0 based on the sampled proportion and current population
      R0_member = current_population * R0_member_prop

      # Ensure R0_member is non-negative.
      R0_member = max(0.0, R0_member)

      # New: Ensure a minimum susceptible population (S0_member) to allow for outbreaks. (Plan Item 3)
      # R0_member should not be so large that it leaves S0_member below MIN_S_PROPORTION * current_population.
      max_R0_allowed_by_S_min = (
          current_population
          - I0_member
          - (MIN_S_PROPORTION * current_population)
      )
      R0_member = min(
          R0_member, max(0.0, max_R0_allowed_by_S_min)
      )  # Ensure R0 doesn't deplete S below minimum

      # Now calculate S0_member as the remainder after I0 and the adjusted R0.
      S0_member = current_population - I0_member - R0_member
      S0_member = max(0.0, S0_member)  # Ensure S0 is non-negative

      # OPTIMIZATION: Perturb noise_factor for each ensemble member using truncated normal (Plan Item 3)
      # instead of uniform to align with other parameter sampling. (Adherence to Implementation Plan Step 4.c.viii)
      # REFINEMENT: Define noise_factor_min_orig and noise_factor_max_orig
      noise_factor_min_orig, noise_factor_max_orig = 0.02, 0.08
      noise_factor_mean = (noise_factor_min_orig + noise_factor_max_orig) / 2
      # REFINEMENT: Use PARAM_TRUNC_NORMAL_STD_DIVISOR for std dev calculation
      noise_factor_std = (
          (noise_factor_max_orig - noise_factor_min_orig)
          / PARAM_TRUNC_NORMAL_STD_DIVISOR
      ) * INFLATION_FACTOR
      noise_factor_member = get_truncated_normal_sample(
          noise_factor_mean,
          noise_factor_std,
          noise_factor_min_orig,
          noise_factor_max_orig,
      )
      # End Optimization

      model = SIRSModel(
          S0=S0_member,
          I0=I0_member,
          R0=R0_member,
          population=current_population,
          beta_base=beta_base,
          gamma=gamma,
          rho=rho,
          admissions_per_I_rate=admissions_per_infected_rate_for_sirs,
          seasonal_multipliers=seasonal_multipliers,
          noise_factor=noise_factor_member,
          min_infected_seed=MIN_INFECTED_SEED_LOC,
      )
      ensemble_members.append(model)

    ensemble_trajectories_by_horizon_step = {}

    MAX_SIMULATION_STEPS = max(horizons_to_simulate) + 1

    for h_step_simulation in range(MAX_SIMULATION_STEPS):
      current_forecast_week_date = ref_date + pd.Timedelta(
          weeks=h_step_simulation
      )
      current_epiweek = current_forecast_week_date.isocalendar().week

      predictions_for_this_sim_week = []
      for member in ensemble_members:
        predicted_admissions = member._advance_week(current_epiweek)
        predictions_for_this_sim_week.append(predicted_admissions)

      ensemble_trajectories_by_horizon_step[h_step_simulation] = (
          predictions_for_this_sim_week
      )

    for row_idx, row in group_df[group_df['horizon'] >= 0].iterrows():
      horizon_to_predict = row['horizon']
      if (
          horizon_to_predict in ensemble_trajectories_by_horizon_step
          and ensemble_trajectories_by_horizon_step[horizon_to_predict]
      ):
        predictions = np.array(
            ensemble_trajectories_by_horizon_step[horizon_to_predict]
        )

        quant_values = np.percentile(predictions, [q * 100 for q in QUANTILES])

        # Ensure non-negativity on float values
        quant_values = np.maximum(0.0, quant_values)

        # Crucial: Enforce monotonicity on float values first, then round to integer
        # (Adherence to Crucial Constraint and Implementation Plan Step 4.c.x)
        quant_values = np.maximum.accumulate(quant_values)
        quant_values = np.round(quant_values).astype(int)

        all_quantile_preds.loc[row_idx] = quant_values
      else:
        all_quantile_preds.loc[row_idx] = [0] * len(QUANTILES)

  return all_quantile_preds


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
