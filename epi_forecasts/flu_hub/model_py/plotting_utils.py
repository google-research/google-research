# coding=utf-8
# Copyright 2025 The Google Research Authors.
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


"""Plotting utilities for epi models and evaluation."""


from matplotlib import dates as mdates
from matplotlib import pyplot as plt
import pandas as pd

import constant_defs

QUANTILES = constant_defs.QUANTILES


def plot_season_forecasts(
    season_start,
    season_end,
    season_name,
    all_forecasts_df,
    national_observed_df,
    step_size = 4,
):
  """Generates and displays a consolidated forecast plot for a single flu season.

  Args:
      season_start (str): The start date of the season (e.g., '2023-10-15').
      season_end (str): The end date of the season (e.g., '2024-05-15').
      season_name (str): The name of the season for the plot title (e.g.,
        '2023-2024').
      all_forecasts_df (pd.DataFrame): DataFrame containing all forecast data.
      national_observed_df (pd.DataFrame): Pre-aggregated DataFrame of national
        observed data.
      step_size (int): The interval for plotting forecast plumes (e.g., 4 means
        every 4th forecast).
  """
  # --- Filter data for the specified season ---
  season_forecasts_df = all_forecasts_df[
      (all_forecasts_df['reference_date'] >= season_start)
      & (all_forecasts_df['reference_date'] <= season_end)
  ].copy()

  # --- Plotting ---
  plt.style.use('seaborn-v0_8-whitegrid')
  _, ax = plt.subplots(figsize=(13, 8))

  quantile_cols = [f'quantile_{q}' for q in QUANTILES]
  # Plot the continuous observed data line
  if national_observed_df is not None:
    national_observed_season = national_observed_df[
        (national_observed_df['target_end_date'] >= season_start)
        & (national_observed_df['target_end_date'] <= season_end)
    ]
    ax.plot(
        national_observed_season['target_end_date'],
        national_observed_season['Total Influenza Admissions'],
        color='black',
        marker='o',
        linestyle='-',
        linewidth=2,
        label='Observed Data',
    )

  # Get unique forecast dates and apply the step size
  unique_reference_dates = sorted(
      season_forecasts_df['reference_date'].unique()
  )
  dates_to_plot = unique_reference_dates[::step_size]

  # Loop through each selected reference date and plot its forecast plume
  for i, forecast_date in enumerate(dates_to_plot):
    single_forecast_df = season_forecasts_df[
        season_forecasts_df['reference_date'] == forecast_date
    ].copy()
    for q_col in quantile_cols:
      single_forecast_df[q_col] = pd.to_numeric(
          single_forecast_df[q_col], errors='coerce'
      )
      single_forecast_df[q_col] = single_forecast_df[q_col].fillna(1000)

    national_forecast = single_forecast_df.groupby('target_end_date').sum(
        numeric_only=True
    )

    is_first_forecast = i == 0  # Flag to ensure legend is only created once

    # Plot Prediction Intervals
    ax.fill_between(
        national_forecast.index,
        national_forecast['quantile_0.025'],
        national_forecast['quantile_0.975'],
        color='steelblue',
        alpha=0.1,
        label='95% Prediction Interval' if is_first_forecast else None,
    )
    ax.fill_between(
        national_forecast.index,
        national_forecast['quantile_0.1'],
        national_forecast['quantile_0.9'],
        color='steelblue',
        alpha=0.2,
        label='80% Prediction Interval' if is_first_forecast else None,
    )
    ax.fill_between(
        national_forecast.index,
        national_forecast['quantile_0.25'],
        national_forecast['quantile_0.75'],
        color='steelblue',
        alpha=0.4,
        label='50% Prediction Interval' if is_first_forecast else None,
    )

    # Plot Median Forecast line
    ax.plot(
        national_forecast.index,
        national_forecast['quantile_0.5'],
        color='tab:blue',
        marker='o',
        linestyle='-',
        linewidth=2.0,
        label='TS Forecast' if is_first_forecast else None,
    )

  # --- Final Formatting and Labels ---
  ax.set_title(
      f'National Flu Hospitalizations Forecasts ({season_name})',
      fontsize=18,
  )
  ax.set_xlabel('Date', fontsize=12)
  ax.set_ylabel('Weekly Hospital Admissions', fontsize=12)
  ax.legend(loc='upper right', frameon=True, edgecolor='black')

  # Format the x-axis to show each forecast date explicitly
  ax.xaxis.set_major_locator(
      mdates.WeekdayLocator(byweekday=mdates.SATURDAY)
  )  # Set ticks at each Saturday
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
  plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

  # Set axis limits
  ax.set_xlim(pd.to_datetime(season_start), pd.to_datetime(season_end))
  ax.set_ylim(bottom=0)

  plt.tight_layout()
  plt.savefig(f'/tmp/{season_name}.png')
