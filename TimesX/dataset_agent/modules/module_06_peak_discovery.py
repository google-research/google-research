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

"""Module 06: Peak Discovery via Trends

Peak discovery using Google Trends time series.
"""

from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .module_01_trends import fetch_google_trends

logger = logging.getLogger(__name__)


def get_significant_peaks_via_trends(
    keyword, time_range, geography, config
):
  """Args:

      keyword:
      time_range:
      geography:
      config:

  Returns:
      List of peaks with structure:
      {
          "peak_date": datetime,
          "peak_value": float,
          "percentile_rank": float,
          "suppression_radius": int,
          "detection_method": "absolute|percentile"
      }
  """
  logger.info(f"'{keyword}'")

  try:

    trends_df = fetch_google_trends([keyword], config)

    if trends_df is None or trends_df.empty:
      logger.warning(
          "No Google Trends data returned for keyword '%s'.", keyword
      )
      return []

    if keyword in trends_df.columns:
      trends_df = trends_df.rename(columns={keyword: "target"})
    elif len(trends_df.columns) > 0:

      trends_df = trends_df.rename(columns={trends_df.columns[0]: "target"})
    else:
      logger.error("Trend dataframe for keyword '%s' has no columns.", keyword)
      return []

    if "target" not in trends_df.columns:
      logger.error(
          "Trend dataframe for keyword '%s' does not contain a target column.",
          keyword,
      )
      return []

    logger.info(f"Trend dataframe rows: {len(trends_df)}")
    logger.info(
        f"Trend statistics: min={trends_df['target'].min():.1f},"
        f" max={trends_df['target'].max():.1f},"
        f" mean={trends_df['target'].mean():.1f}"
    )

    dual_channel_config = config.get("dual_channel_discovery", {})
    trends_config = dual_channel_config.get("trends_channel", {})

    absolute_threshold = trends_config.get("detection_thresholds", {}).get(
        "absolute_value", 75
    )
    percentile_threshold = trends_config.get("detection_thresholds", {}).get(
        "percentile", 0.90
    )
    suppression_radius = trends_config.get("suppression_radius_days", 3)

    logger.info(
        f"Peak thresholds: absolute={absolute_threshold},"
        f" percentile={percentile_threshold},"
        f" suppression_radius={suppression_radius}"
    )

    candidate_peaks = []

    absolute_peaks = trends_df[trends_df["target"] > absolute_threshold]
    for idx, row in absolute_peaks.iterrows():
      candidate_peaks.append({
          "peak_date": idx,
          "peak_value": row["target"],
          "percentile_rank": (trends_df["target"] <= row["target"]).mean(),
          "suppression_radius": suppression_radius,
          "detection_method": "absolute",
      })

    logger.info(f"Absolute-threshold peak candidates: {len(absolute_peaks)}")

    percentile_value = trends_df["target"].quantile(percentile_threshold)
    percentile_peaks = trends_df[trends_df["target"] >= percentile_value]

    for idx, row in percentile_peaks.iterrows():

      if not any(p["peak_date"] == idx for p in candidate_peaks):
        candidate_peaks.append({
            "peak_date": idx,
            "peak_value": row["target"],
            "percentile_rank": (trends_df["target"] <= row["target"]).mean(),
            "suppression_radius": suppression_radius,
            "detection_method": "percentile",
        })

    logger.info(
        f"Percentile-threshold peak candidates: {len(percentile_peaks)}"
    )
    logger.info(
        f"Candidate peak count before suppression: {len(candidate_peaks)}"
    )

    if not candidate_peaks:
      logger.info("No candidate peaks detected after thresholding.")
      return []

    suppressed_peaks = apply_suppression_radius(
        candidate_peaks, suppression_radius
    )

    logger.info(f"Peak candidates after suppression: {len(suppressed_peaks)}")

    suppressed_peaks.sort(key=lambda x: x["peak_date"])

    for i, peak in enumerate(suppressed_peaks, 1):
      logger.info(
          f"Suppressed peak {i}: date={peak['peak_date'].date()}, "
          f"value={peak['peak_value']:.1f}, method={peak['detection_method']}"
      )

    return suppressed_peaks

  except Exception as e:
    logger.error(f"Peak suppression failed: {str(e)}")
    return []


def apply_suppression_radius(
    candidate_peaks, radius_days
):
  """Apply a suppression radius to avoid reporting overlapping peaks.

  Args:
      candidate_peaks: Candidate peaks sorted by significance.
      radius_days: Suppression window size in days.

  Returns:
      A filtered peak list with overlapping peaks removed.
  """
  if not candidate_peaks:
    return []

  sorted_peaks = sorted(
      candidate_peaks, key=lambda x: x["peak_value"], reverse=True
  )

  suppressed_peaks = []
  used_dates = set()

  for peak in sorted_peaks:
    peak_date = peak["peak_date"]

    is_suppressed = False
    for used_date in used_dates:
      days_diff = abs((peak_date - used_date).days)
      if days_diff <= radius_days:
        is_suppressed = True
        break

    if not is_suppressed:
      suppressed_peaks.append(peak)
      used_dates.add(peak_date)

  logger.info(
      f"Suppression radius applied: {len(candidate_peaks)} ->"
      f" {len(suppressed_peaks)} peaks"
  )

  return suppressed_peaks


def analyze_peak_characteristics(
    trends_df, peak_date
):
  """Args:

      trends_df:
      peak_date:

  Returns:
  """
  try:
    if peak_date not in trends_df.index:
      return {}

    peak_value = trends_df.loc[peak_date, "target"]

    window_size = 7
    start_window = peak_date - timedelta(days=window_size)
    end_window = peak_date + timedelta(days=window_size)

    window_data = trends_df[
        (trends_df.index >= start_window) & (trends_df.index <= end_window)
    ]["target"]

    if len(window_data) > 1:
      baseline = (
          window_data.drop(peak_date).mean()
          if peak_date in window_data.index
          else window_data.mean()
      )
      relative_increase = (
          (peak_value - baseline) / baseline if baseline > 0 else 0
      )
    else:
      relative_increase = 0

    return {
        "peak_value": peak_value,
        "baseline_7d": baseline,
        "relative_increase": relative_increase,
        "percentile_rank": (trends_df["target"] <= peak_value).mean(),
        "window_max": window_data.max(),
        "window_min": window_data.min(),
    }

  except Exception as e:
    logger.warning(f"Peak extraction fallback triggered: {str(e)}")
    return {}


def filter_peaks_by_significance(
    peaks, min_significance = 0.5
):
  """Args:

      peaks:
      min_significance:

  Returns:
  """
  if not peaks:
    return []

  significant_peaks = []

  for peak in peaks:
    percentile_rank = peak.get("percentile_rank", 0)

    significance_score = percentile_rank

    if significance_score >= min_significance:
      peak["significance_score"] = significance_score
      significant_peaks.append(peak)

  logger.info(
      f"Significant peak filter: {len(peaks)} -> {len(significant_peaks)} peaks"
  )

  return significant_peaks
