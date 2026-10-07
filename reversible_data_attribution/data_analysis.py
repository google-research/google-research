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

#!/usr/bin/env python3
"""Unified data analysis module for data attribution and counterfactuals.

This module consolidates data analysis, evaluation, and visualization tools
across the data attribution pipeline:
1. Counterfactual loss scatterplots & layout grids (by file, by method, matrix).
2. Counterfactual attribution score merging and condition filtering.
3. Estimator correlation against ground-truth counterfactuals.
4. Data cleansing corruption detection evaluation (Precision, Recall, ROC-AUC).
5. Data cleansing statistics reporting and trajectory visualization.
"""

import argparse
from collections.abc import Mapping, Sequence
from functools import reduce
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics

# Default method renaming dictionary for reports and LaTeX tables.
DEFAULT_RENAME_METHODS: dict[str, str] = {
    "baseline": "Full sample",
    "random": "Random",
    "remove_corrupted": "Oracle",
    "actual_cf": "GT-CF",
    "ae": "Autoencoder",
    "iso": "Iso-forest",
    "icml": "IF",
    "tracin_sgd": "TracIn-SGD",
    "tracin_adam": "Trac-Adam",
    "sgd_all": "TSLOO-SGD",
    "adam_masked_5%": "TSLOO-Adam-5%",
    "adam_recursive_nodv": "TSLOO-Adam-nodv",
    "adam_recursive": "TSLOO-Adam",
}

# Default display mapping for cleansing trajectory curves.
DEFAULT_METHOD_MAPPING: dict[str, str] = {
    "tracin_adam": "Tracin Adam",
    "tracin_sgd": "Tracin SGD",
    "adam_recursive": "TSLOO Adam",
    "sgd_all": "SGD",
    "actual_cf": "Exact CF",
}

# Default color palette for cleansing trajectory curves.
DEFAULT_PALETTE: dict[str, str] = {
    "TSLOO Adam": "tab:blue",
    "SGD": "tab:orange",
    "Tracin Adam": "tab:green",
    "Tracin SGD": "tab:red",
    "Exact CF": "black",
}

# Default candidate estimators for correlation analysis.
DEFAULT_CANDIDATE_ESTIMATORS: list[str] = [
    "tracin_sgd",
    "tracin_adam",
    "sgd_all",
    "adam_masked_5%",
    "adam_recursive_nodv",
    "adam_recursive",
]

# Default candidate estimator columns for score merging.
DEFAULT_ESTIMATOR_COLS: list[str] = [
    "adam_recursive",
    "sgd_all",
    "tracin_adam",
    "tracin_sgd",
    "adam_masked_5%",
    "adam_recursive_nodv",
    "actual_cf",
]

__all__ = [
    "DEFAULT_RENAME_METHODS",
    "DEFAULT_METHOD_MAPPING",
    "DEFAULT_PALETTE",
    "DEFAULT_CANDIDATE_ESTIMATORS",
    "DEFAULT_ESTIMATOR_COLS",
    "auto_select_y_cols",
    "create_scatterplots",
    "merge_clean_scores",
    "merge_scores_on_cond",
    "merge_scores_all",
    "get_score_correlations",
    "compute_score_correlations",
    "plot_score_correlations",
    "evaluate_data_cleansing",
    "print_cleansing_stats",
    "plot_cleansing_stats",
]


# ==============================================================================
# 1. Counterfactual Loss Scatterplot Utilities
# ==============================================================================


def auto_select_y_cols(
    df: pd.DataFrame, ignore_cols: Sequence[str]
) -> list[str]:
  """Auto-selects numeric columns to plot on Y-axis."""
  ignore_set = {c.lower() for c in ignore_cols}
  y_cols = []
  for col in df.columns:
    if col.lower() not in ignore_set:
      converted = pd.to_numeric(df[col], errors="coerce")
      if converted.notna().sum() > 0:
        y_cols.append(col)
  return y_cols


def create_scatterplots(
    df: pd.DataFrame,
    x_col: str = "True adam CF test loss",
    y_cols: Sequence[str] | None = None,
    group_col: str = "File",
    base_loss_col: str = "Base adam test loss",
    layout: str = "by_file",
    show_identity: bool = True,
    show_base_lines: bool = True,
    output_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
  """Generates scatterplots grouped by file/method with base loss lines."""
  df = df.copy()
  df.columns = [c.strip() for c in df.columns]

  if x_col not in df.columns:
    raise ValueError(
        f"X column '{x_col}' not found in DataFrame. Available:"
        f" {list(df.columns)}"
    )
  if group_col not in df.columns:
    raise ValueError(
        f"Group column '{group_col}' not found in DataFrame. Available:"
        f" {list(df.columns)}"
    )
  if base_loss_col not in df.columns:
    raise ValueError(
        f"Base loss column '{base_loss_col}' not found in DataFrame. Available:"
        f" {list(df.columns)}"
    )

  df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
  df[base_loss_col] = pd.to_numeric(df[base_loss_col], errors="coerce")

  if not y_cols:
    y_cols = auto_select_y_cols(
        df, ignore_cols=["Index", x_col, group_col, base_loss_col]
    )

  if not y_cols:
    raise ValueError("No valid numeric Y-columns found for plotting.")

  for col in y_cols:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
      print(
          f"Warning: Requested Y column '{col}' not found in dataframe.",
          file=sys.stderr,
      )

  y_cols = [c for c in y_cols if c in df.columns]
  if not y_cols:
    raise ValueError("None of the requested Y-columns exist in the DataFrame.")

  groups = list(df.groupby(group_col))
  num_groups = len(groups)

  print(f"Loaded {len(df)} rows across {num_groups} group(s).")
  print(f"X-axis: '{x_col}'")
  print(f"Y-axis columns ({len(y_cols)}): {y_cols}")
  print(f"Base loss column: '{base_loss_col}'")

  if layout == "by_file":
    fig = _plot_by_file(
        groups=groups,
        x_col=x_col,
        y_cols=y_cols,
        base_loss_col=base_loss_col,
        show_identity=show_identity,
        show_base_lines=show_base_lines,
        title=title,
    )
  elif layout == "by_method":
    fig = _plot_by_method(
        df=df,
        groups=groups,
        x_col=x_col,
        y_cols=y_cols,
        base_loss_col=base_loss_col,
        show_identity=show_identity,
        show_base_lines=show_base_lines,
        title=title,
    )
  elif layout == "matrix":
    fig = _plot_matrix(
        groups=groups,
        x_col=x_col,
        y_cols=y_cols,
        base_loss_col=base_loss_col,
        show_identity=show_identity,
        show_base_lines=show_base_lines,
        title=title,
    )
  else:
    raise ValueError(
        f"Unknown layout: {layout}. Choose from ['by_file', 'by_method',"
        " 'matrix']"
    )

  plt.tight_layout()

  if output_path:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

  return fig


def _plot_by_file(
    groups, x_col, y_cols, base_loss_col, show_identity, show_base_lines, title
) -> plt.Figure:
  """Subplot per File group, plotting all Y-columns in each subplot."""
  num_groups = len(groups)
  ncols = min(3, num_groups)
  nrows = (num_groups + ncols - 1) // ncols

  fig, axes = plt.subplots(
      nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
  )
  fig.suptitle(
      title or "Counterfactual Loss Scatterplots Grouped by File",
      fontsize=14,
      y=1.02,
  )

  colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(y_cols))))

  for idx, (group_name, group_df) in enumerate(groups):
    row, col = divmod(idx, ncols)
    ax = axes[row, col]

    base_loss_vals = group_df[base_loss_col].dropna()
    base_loss = base_loss_vals.iloc[0] if not base_loss_vals.empty else None

    x_vals = group_df[x_col]
    all_x = x_vals.dropna()
    all_y_list = [group_df[y].dropna() for y in y_cols]
    all_y = pd.concat(all_y_list) if all_y_list else pd.Series(dtype=float)

    min_val = min(
        all_x.min() if not all_x.empty else 0,
        all_y.min() if not all_y.empty else 0,
    )
    max_val = max(
        all_x.max() if not all_x.empty else 1,
        all_y.max() if not all_y.empty else 1,
    )
    if base_loss is not None:
      min_val = min(min_val, base_loss)
      max_val = max(max_val, base_loss)

    for y_idx, y_col_name in enumerate(y_cols):
      y_vals = group_df[y_col_name]
      ax.scatter(
          x_vals,
          y_vals,
          color=colors[y_idx],
          label=y_col_name,
          alpha=0.8,
          edgecolors="k",
          linewidths=0.5,
          s=40,
      )

    if show_base_lines and base_loss is not None:
      ax.axvline(
          x=base_loss,
          color="crimson",
          linestyle="--",
          linewidth=1.5,
          alpha=0.85,
          label=f"Base Loss ({base_loss:.4f})",
      )
      ax.axhline(
          y=base_loss,
          color="crimson",
          linestyle="--",
          linewidth=1.5,
          alpha=0.85,
      )

    if show_identity:
      padding = (max_val - min_val) * 0.05
      line_range = np.linspace(min_val - padding, max_val + padding, 100)
      ax.plot(
          line_range,
          line_range,
          color="gray",
          linestyle=":",
          linewidth=1.2,
          label="y = x (Ideal)",
      )

    clean_file_name = os.path.basename(str(group_name))
    ax.set_title(f"File: {clean_file_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel("Estimated Test Loss", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)

  for idx in range(num_groups, nrows * ncols):
    row, col = divmod(idx, ncols)
    fig.delaxes(axes[row, col])

  return fig


def _plot_by_method(
    df,
    groups,
    x_col,
    y_cols,
    base_loss_col,
    show_identity,
    show_base_lines,
    title,
) -> plt.Figure:
  """Subplot per Y method, showing data points from all file groups."""
  num_methods = len(y_cols)
  ncols = min(3, num_methods)
  nrows = (num_methods + ncols - 1) // ncols

  fig, axes = plt.subplots(
      nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
  )
  fig.suptitle(
      title or "Counterfactual Loss Comparison per Method",
      fontsize=14,
      y=1.02,
  )

  group_colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(groups))))

  for y_idx, y_col_name in enumerate(y_cols):
    row, col = divmod(y_idx, ncols)
    ax = axes[row, col]

    for g_idx, (group_name, group_df) in enumerate(groups):
      clean_name = os.path.basename(str(group_name))
      x_vals = group_df[x_col]
      y_vals = group_df[y_col_name]
      base_loss_vals = group_df[base_loss_col].dropna()
      base_loss = base_loss_vals.iloc[0] if not base_loss_vals.empty else None

      ax.scatter(
          x_vals,
          y_vals,
          color=group_colors[g_idx],
          label=clean_name,
          alpha=0.8,
          edgecolors="k",
          linewidths=0.5,
          s=40,
      )

      if show_base_lines and base_loss is not None:
        ax.axvline(
            x=base_loss,
            color=group_colors[g_idx],
            linestyle="--",
            alpha=0.5,
        )
        ax.axhline(
            y=base_loss,
            color=group_colors[g_idx],
            linestyle="--",
            alpha=0.5,
        )

    if show_identity:
      all_x = df[x_col].dropna()
      all_y = df[y_col_name].dropna()
      min_val = min(
          all_x.min() if not all_x.empty else 0,
          all_y.min() if not all_y.empty else 0,
      )
      max_val = max(
          all_x.max() if not all_x.empty else 1,
          all_y.max() if not all_y.empty else 1,
      )
      padding = (max_val - min_val) * 0.05
      line_range = np.linspace(min_val - padding, max_val + padding, 100)
      ax.plot(
          line_range,
          line_range,
          color="gray",
          linestyle=":",
          label="y = x",
      )

    ax.set_title(f"Method: {y_col_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col_name, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)

  for idx in range(num_methods, nrows * ncols):
    row, col = divmod(idx, ncols)
    fig.delaxes(axes[row, col])

  return fig


def _plot_matrix(
    groups,
    x_col,
    y_cols,
    base_loss_col,
    show_identity,
    show_base_lines,
    title,
) -> plt.Figure:
  """Grid matrix of subplots (Rows = File Groups, Cols = Y Methods)."""
  nrows = len(groups)
  ncols = len(y_cols)

  fig, axes = plt.subplots(
      nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False
  )
  fig.suptitle(
      title or "Counterfactual Loss Matrix (Files x Methods)",
      fontsize=14,
      y=1.01,
  )

  for r_idx, (group_name, group_df) in enumerate(groups):
    clean_file_name = os.path.basename(str(group_name))
    base_loss_vals = group_df[base_loss_col].dropna()
    base_loss = base_loss_vals.iloc[0] if not base_loss_vals.empty else None
    x_vals = group_df[x_col]

    for c_idx, y_col_name in enumerate(y_cols):
      ax = axes[r_idx, c_idx]
      y_vals = group_df[y_col_name]

      ax.scatter(
          x_vals,
          y_vals,
          color="navy",
          alpha=0.75,
          edgecolors="k",
          linewidths=0.5,
          s=35,
      )

      if show_base_lines and base_loss is not None:
        ax.axvline(x=base_loss, color="crimson", linestyle="--", alpha=0.7)
        ax.axhline(y=base_loss, color="crimson", linestyle="--", alpha=0.7)

      if show_identity:
        valid_x = x_vals.dropna()
        valid_y = y_vals.dropna()
        if not valid_x.empty and not valid_y.empty:
          min_v = min(valid_x.min(), valid_y.min())
          max_v = max(valid_x.max(), valid_y.max())
          if base_loss is not None:
            min_v = min(min_v, base_loss)
            max_v = max(max_v, base_loss)
          line_range = np.linspace(min_v, max_v, 50)
          ax.plot(
              line_range,
              line_range,
              color="gray",
              linestyle=":",
              alpha=0.8,
          )

      if r_idx == 0:
        ax.set_title(y_col_name, fontsize=10, fontweight="bold")
      if c_idx == 0:
        ax.set_ylabel(f"{clean_file_name}\nLoss", fontsize=9)
      if r_idx == nrows - 1:
        ax.set_xlabel(x_col, fontsize=9)

      ax.grid(True, linestyle="--", alpha=0.3)

  return fig


# ==============================================================================
# 2. Score Merging Utilities
# ==============================================================================


def merge_clean_scores(
    scores_list: Sequence[pd.DataFrame],
    batch_size: int | None = None,
    noise_rate: float | None = None,
    learning_rate: float | None = None,
    num_epochs: int | None = None,
    merge_keys: Sequence[str] | None = None,
    estimator_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
  """Filters and merges multiple counterfactual score DataFrames cleanly.

  Prevents column name collisions by retaining metadata keys and only adding
  unique estimator columns from subsequent DataFrames.

  Args:
    scores_list: List of DataFrames containing attribution and CF scores.
    batch_size: Optional batch size filter.
    noise_rate: Optional label noise rate filter.
    learning_rate: Optional learning rate filter.
    num_epochs: Optional number of training epochs filter.
    merge_keys: Explicit columns to join on. Defaults to standard metadata keys.
    estimator_cols: Candidate estimator columns to preserve from subsequent
      DataFrames. Defaults to DEFAULT_ESTIMATOR_COLS.

  Returns:
    Merged pandas DataFrame containing all unique estimators across the inputs.
  """
  if not scores_list:
    return pd.DataFrame()

  if merge_keys is None:
    merge_keys = [
        "sample_index",
        "is_corrupted",
        "noise_seed",
        "batch_size",
        "noise_rate",
        "learning_rate",
        "num_epochs",
        "dataset",
        "num_train",
        "num_val",
        "num_test",
    ]

  if estimator_cols is None:
    estimator_cols = DEFAULT_ESTIMATOR_COLS

  # 1. Filter each DataFrame based on parameters if provided
  scores_filtered = []
  for score in scores_list:
    mask = pd.Series(True, index=score.index)
    if batch_size is not None and "batch_size" in score.columns:
      mask &= score["batch_size"] == batch_size
    if noise_rate is not None and "noise_rate" in score.columns:
      mask &= score["noise_rate"] == noise_rate
    if learning_rate is not None and "learning_rate" in score.columns:
      mask &= score["learning_rate"] == learning_rate
    if num_epochs is not None and "num_epochs" in score.columns:
      mask &= score["num_epochs"] == num_epochs
    scores_filtered.append(score[mask])

  # Identify keys present across all DataFrames
  common_keys = [
      k for k in merge_keys if all(k in df.columns for df in scores_filtered)
  ]
  if not common_keys:
    raise ValueError(
        f"No common merge keys found among candidate keys: {merge_keys}"
    )

  # 2. Select only common keys + unique estimator columns from subsequent frames
  cleaned_list = []
  for i, df in enumerate(scores_filtered):
    if i == 0:
      cleaned_list.append(df)
    else:
      # Keep join keys + specified estimator columns present in this df
      valid_estimators = [
          col
          for col in df.columns
          if col not in common_keys
          and (col in estimator_cols or pd.api.types.is_numeric_dtype(df[col]))
      ]
      cols_to_keep = list(dict.fromkeys(common_keys + valid_estimators))
      cleaned_list.append(df[[c for c in cols_to_keep if c in df.columns]])

  # 3. Inner merge
  merged = reduce(
      lambda left, right: pd.merge(left, right, on=common_keys, how="inner"),
      cleaned_list,
  )
  return merged


def merge_scores_on_cond(
    scores_list: Sequence[pd.DataFrame],
    batch_size: int | None = None,
    noise_rate: float | None = None,
    learning_rate: float | None = None,
    num_epochs: int | None = None,
) -> pd.DataFrame:
  """Convenience alias for merge_clean_scores matching data_analysis.ipynb."""
  return merge_clean_scores(
      scores_list,
      batch_size=batch_size,
      noise_rate=noise_rate,
      learning_rate=learning_rate,
      num_epochs=num_epochs,
  )


def merge_scores_all(
    scores_list: Sequence[pd.DataFrame],
    merge_keys: Sequence[str] | None = None,
    estimator_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
  """Merges all score DataFrames across common keys without filtering.

  Matches merge_scores_all in data_analysis.ipynb.

  Args:
    scores_list: Sequence of score DataFrames.
    merge_keys: Explicit columns to join on.
    estimator_cols: Estimator columns to preserve from subsequent DataFrames.

  Returns:
    Merged pandas DataFrame.
  """
  return merge_clean_scores(
      scores_list,
      merge_keys=merge_keys,
      estimator_cols=estimator_cols,
  )


# ==============================================================================
# 3. Estimator Correlation Analysis
# ==============================================================================


def get_score_correlations(
    scores_cond: pd.DataFrame,
    max_cf_val: float | None = None,
    rename_methods: Mapping[str, str] | None = None,
    add_latex_support: bool = False,
    actual_col: str = "actual_cf",
    estimators: Sequence[str] | None = None,
    group_col: str | None = "noise_seed",
    return_raw: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
  """Computes and aggregates Pearson and Spearman estimator correlations.

  Evaluates estimator correlation against ground truth (actual_cf) across
  different seeds. Matches get_score_correlations in data_analysis.ipynb.

  Args:
    scores_cond: DataFrame containing scores and actual CF ground truth.
    max_cf_val: Optional maximum absolute value threshold on actual_col.
    rename_methods: Optional dictionary mapping estimator names to display
      names.
    add_latex_support: If True, wraps formatted entries in '$...$'.
    actual_col: Ground truth column name (default: 'actual_cf').
    estimators: Optional list of estimators to evaluate. Defaults to
      DEFAULT_CANDIDATE_ESTIMATORS.
    group_col: Column to group by across seeds (default: 'noise_seed').
    return_raw: If True, returns numerical means and SEMs in the summary
      DataFrame.

  Returns:
    Tuple of (raw per-seed correlation DataFrame, aggregated summary DataFrame).
  """
  if actual_col not in scores_cond.columns:
    print(
        f"Ground truth '{actual_col}' not found in scores. Correlation"
        " skipped.",
        file=sys.stderr,
    )
    return None, None

  has_group = group_col is not None and group_col in scores_cond.columns
  if has_group:
    seeds = scores_cond[group_col].unique()
  else:
    seeds = [None]

  corr_records = []
  if estimators is None:
    candidate_estimators = DEFAULT_CANDIDATE_ESTIMATORS
  else:
    candidate_estimators = list(estimators)

  available_estimators = [
      m for m in candidate_estimators if m in scores_cond.columns
  ]

  for seed in seeds:
    if has_group:
      df_this = scores_cond[scores_cond[group_col] == seed]
    else:
      df_this = scores_cond

    if max_cf_val is not None:
      df_this = df_this[np.abs(df_this[actual_col]) <= max_cf_val]

    for est in available_estimators:
      sub_df = df_this[[actual_col, est]].dropna()
      if len(sub_df) > 1:
        p_corr = sub_df[actual_col].corr(sub_df[est], method="pearson")
        s_corr = sub_df[actual_col].corr(sub_df[est], method="spearman")
        record = {
            "estimator": est,
            "Pearson": p_corr,
            "Spearman": s_corr,
        }
        if has_group:
          record[group_col] = seed
        corr_records.append(record)

  if not corr_records:
    return None, None

  corr_df = pd.DataFrame(corr_records)

  # Aggregate across seeds
  grouped = corr_df.groupby("estimator")
  means = grouped[["Pearson", "Spearman"]].mean()
  sems = grouped[["Pearson", "Spearman"]].sem().fillna(0.0)

  if return_raw:
    raw_summary = pd.DataFrame(index=means.index)
    for col in ["Pearson", "Spearman"]:
      raw_summary[f"{col}_mean"] = means[col]
      raw_summary[f"{col}_sem"] = sems[col]
    existing_estimators = [
        m for m in candidate_estimators if m in raw_summary.index
    ]
    raw_summary = raw_summary.reindex(existing_estimators)
    if rename_methods is not None:
      raw_summary = raw_summary.rename(index=rename_methods)
    return corr_df, raw_summary

  summary_df = pd.DataFrame(index=means.index)
  for col in ["Pearson", "Spearman"]:
    if len(seeds) > 1:
      formatted_col = [
          f"{m:.3f} \\pm {1.96 * s:.3f}" for m, s in zip(means[col], sems[col])
      ]
    else:
      formatted_col = [f"{m:.3f}" for m in means[col]]

    if add_latex_support:
      formatted_col = [f"${val}$" for val in formatted_col]

    summary_df[col] = formatted_col

  # Reorder according to candidate order first (while keys match index)
  existing_estimators = [
      m for m in candidate_estimators if m in summary_df.index
  ]
  summary_df = summary_df.reindex(existing_estimators)

  # Rename indices using rename_methods dictionary at the very end
  if rename_methods is not None:
    summary_df = summary_df.rename(index=rename_methods)

  return corr_df, summary_df


def compute_score_correlations(
    scores_df: pd.DataFrame,
    actual_col: str = "actual_cf",
    estimators: Sequence[str] | None = None,
    max_cf_val: float | None = None,
    group_col: str | None = "noise_seed",
) -> pd.DataFrame:
  """Computes correlations between estimators and actual CF.

  Args:
    scores_df: DataFrame containing actual and estimated counterfactual scores.
    actual_col: Name of column containing ground-truth counterfactuals.
    estimators: List of estimator columns to evaluate. If None, auto-selects.
    max_cf_val: Optional maximum absolute value threshold on actual_col.
    group_col: Optional column to group by (e.g. 'noise_seed'). If None or not
      present, computes overall correlations.

  Returns:
    DataFrame with correlation metrics (Pearson, Spearman) per estimator.
  """
  corr_df, _ = get_score_correlations(
      scores_cond=scores_df,
      max_cf_val=max_cf_val,
      actual_col=actual_col,
      estimators=estimators,
      group_col=group_col,
  )
  return corr_df if corr_df is not None else pd.DataFrame()


def plot_score_correlations(
    scores_df: pd.DataFrame,
    max_cf_val: float | None = None,
    actual_col: str = "actual_cf",
    estimators: Sequence[str] | None = None,
    group_col: str | None = "noise_seed",
    show_identity: bool = True,
    output_path: str | None = None,
    save_file: str | None = None,
    title: str | None = None,
    show: bool = False,
) -> tuple[pd.DataFrame | None, plt.Figure | None]:
  """Visualizes estimator correlation against actual CF across seeds/groups.

  Matches plot_score_correlations in data_analysis.ipynb.

  Args:
    scores_df: DataFrame containing counterfactual scores.
    max_cf_val: Optional maximum absolute value threshold on actual_col.
    actual_col: Name of column containing ground-truth counterfactuals.
    estimators: Estimator columns to plot. If None, plots adam_recursive and
      sgd_all if available, or available candidate estimators.
    group_col: Column to group subplots by (e.g. 'noise_seed').
    show_identity: Whether to draw y = x ideal diagonal line.
    output_path: Optional path to save image file.
    save_file: Optional alias for output_path matching data_analysis.ipynb.
    title: Optional figure super-title.
    show: Whether to call plt.show() at the end.

  Returns:
    Tuple of (correlation summary DataFrame, matplotlib Figure).
  """
  if actual_col not in scores_df.columns:
    print(
        f"Ground truth column '{actual_col}' not found in DataFrame."
        " Correlation plot skipped.",
        file=sys.stderr,
    )
    return None, None

  if estimators is None:
    preferred = [
        m for m in ["adam_recursive", "sgd_all"] if m in scores_df.columns
    ]
    if preferred:
      target_estimators = preferred
    else:
      target_estimators = [
          m for m in DEFAULT_CANDIDATE_ESTIMATORS if m in scores_df.columns
      ]
  else:
    target_estimators = [m for m in estimators if m in scores_df.columns]

  if not target_estimators:
    print("No valid estimator columns found for correlation plot.")
    return None, None

  corr_df = compute_score_correlations(
      scores_df=scores_df,
      actual_col=actual_col,
      estimators=target_estimators,
      max_cf_val=max_cf_val,
      group_col=group_col,
  )

  has_group = group_col is not None and group_col in scores_df.columns
  if has_group:
    seeds = scores_df[group_col].unique()
  else:
    seeds = [None]
  num_seeds = len(seeds)

  fig, axs = plt.subplots(
      1, max(1, num_seeds), figsize=(8 * max(1, num_seeds), 7), squeeze=False
  )
  if title:
    fig.suptitle(title, fontsize=16, y=1.02)

  estimator_labels = {
      "adam_recursive": "TSLOO-Adam",
      "sgd_all": "TSLOO-SGD",
      "tracin_adam": "TracIn-Adam",
      "tracin_sgd": "TracIn-SGD",
      "adam_masked_5%": "TSLOO-Adam-5%",
      "adam_recursive_nodv": "TSLOO-Adam-nodv",
  }
  colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(target_estimators))))

  for i, seed in enumerate(seeds):
    ax = axs[0, i]
    if has_group:
      df_curr = scores_df[scores_df[group_col] == seed].copy()
    else:
      df_curr = scores_df.copy()

    if max_cf_val is not None:
      df_curr = df_curr[np.abs(df_curr[actual_col]) <= max_cf_val]

    for est_idx, est in enumerate(target_estimators):
      if est in df_curr.columns:
        label = estimator_labels.get(est, est)
        ax.scatter(
            df_curr[actual_col],
            df_curr[est],
            label=label,
            color=colors[est_idx],
            alpha=0.15,
        )

    if show_identity:
      ax.axline((0, 0), slope=1, color="black", linestyle="--")

    ax.set_xlabel("Actual CF change", fontsize=16)
    ax.set_ylabel("Estimated CF change", fontsize=16)
    group_title = f"Seed {seed}" if has_group else "All Seeds"
    ax.set_title(group_title, fontsize=18)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.grid(True, linestyle="--", alpha=0.3)

    leg = ax.legend(fontsize=14)
    if leg:
      for handle in leg.legend_handles:
        handle.set_alpha(1.0)
      plt.setp(ax.get_legend().get_texts(), fontsize=14)
      plt.setp(ax.get_legend().get_title(), fontsize=14)

  plt.tight_layout()

  save_dest = save_file or output_path
  if save_dest:
    os.makedirs(os.path.dirname(os.path.abspath(save_dest)), exist_ok=True)
    plt.savefig(save_dest, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_dest}")

  if show:
    plt.show()

  return corr_df, fig


# ==============================================================================
# 4. Data Cleansing Evaluation Metrics
# ==============================================================================


def evaluate_data_cleansing(
    scores_df: pd.DataFrame,
    methods: Sequence[str],
    num_removal: int,
    label_col: str = "is_corrupted",
    group_col: str | None = "noise_seed",
    rename_methods: Mapping[str, str] | None = None,
    add_latex_support: bool = True,
    return_raw: bool = False,
) -> pd.DataFrame:
  """Calculates Precision, Recall, F1-Score, and ROC-AUC metrics.

  Identifies predicted corrupted indices by ranking attribution scores in
  ascending order (lowest scores = most harmful / likely corrupted). Matches
  evaluate_data_cleansing in data_analysis.ipynb.

  Args:
    scores_df: DataFrame containing attribution scores and corruption labels.
    methods: Sequence of column names representing attribution methods.
    num_removal: Number of candidate samples to remove/classify as corrupted.
    label_col: Name of boolean column indicating actual corrupted samples.
    group_col: Name of grouping column across which to aggregate (e.g.
      'noise_seed').
    rename_methods: Optional dictionary mapping method names to display names.
    add_latex_support: If True, wraps formatted entries in '$...$'.
    return_raw: If True, returns unformatted numerical means and stds.

  Returns:
    DataFrame indexed by method containing evaluation metrics.
  """
  if label_col not in scores_df.columns:
    raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

  methods = [m for m in methods if m in scores_df.columns]
  if not methods:
    raise ValueError("None of the specified methods exist in DataFrame.")

  has_group = group_col is not None and group_col in scores_df.columns
  groups = scores_df.groupby(group_col) if has_group else [(None, scores_df)]

  seed_results = []
  for _, df in groups:
    num_corrupted_total = df[label_col].sum()
    if num_corrupted_total == 0:
      continue

    # Rank ascending: lowest scores correspond to most likely corrupted
    est_corrupted = df[methods].rank(ascending=True, method="first") <= int(
        num_removal
    )

    tp = est_corrupted.apply(lambda col: col & df[label_col])
    est_sum = est_corrupted.sum()
    precision = tp.sum() / est_sum.replace(0, np.nan)
    recall = tp.sum() / num_corrupted_total
    denom = precision + recall
    f1 = 2 * precision * recall / denom.replace(0, np.nan)

    roc_auc_vals = []
    for m in methods:
      # Use -score so that lower scores produce higher probability of corruption
      try:
        score_val = sklearn.metrics.roc_auc_score(df[label_col], -df[m])
      except ValueError:
        score_val = np.nan
      roc_auc_vals.append(score_val)

    roc_auc = pd.Series(roc_auc_vals, index=methods)

    res = pd.DataFrame(
        [precision, recall, f1, roc_auc],
        index=["Precision", "Recall", "F1 Score", "ROC AUC"],
    ).T
    seed_results.append(res)

  if not seed_results:
    return pd.DataFrame()

  all_seeds_df = pd.concat(seed_results)
  means = all_seeds_df.groupby(all_seeds_df.index).mean()
  stds = all_seeds_df.groupby(all_seeds_df.index).std().fillna(0.0)

  if return_raw:
    raw_df = pd.DataFrame(index=means.index)
    for col in ["Precision", "Recall", "F1 Score", "ROC AUC"]:
      raw_df[f"{col}_mean"] = means[col]
      raw_df[f"{col}_std"] = stds[col]
    res_df = raw_df.reindex(methods)
    if rename_methods is not None:
      res_df = res_df.rename(index=rename_methods)
    return res_df

  formatted_df = pd.DataFrame(index=means.index)
  for col in ["Precision", "Recall", "F1 Score", "ROC AUC"]:
    if len(seed_results) > 1:
      entries = [f"{m:.3f} \\pm {s:.3f}" for m, s in zip(means[col], stds[col])]
    else:
      entries = [f"{m:.3f}" for m in means[col]]

    if add_latex_support:
      entries = [f"${e}$" for e in entries]

    formatted_df[col] = entries

  formatted_df = formatted_df.reindex(methods)
  if rename_methods is not None:
    formatted_df = formatted_df.rename(index=rename_methods)

  return formatted_df


# ==============================================================================
# 5. Data Cleansing Statistics & Plotting
# ==============================================================================


def print_cleansing_stats(
    cleaning_df: pd.DataFrame,
    batch_size: int | None = None,
    noise_rate: float | None = None,
    learning_rate: float | None = None,
    num_epochs: int | None = None,
    num_samples: int = 50000,
    num_remove: int | None = None,
    methods: Sequence[str] | None = None,
    rename_methods: Mapping[str, str] | None = None,
    add_latex_support: bool = False,
    return_raw: bool = False,
) -> pd.DataFrame:
  """Returns a DataFrame of (mean ± 1.96*SEM) for data cleansing methods.

  Aggregates validation and test loss/accuracy for baseline, oracle, and custom
  methods at the target removal threshold k = num_remove (and k = 0 for
  baseline). Matches print_cleansing_stats in data_analysis.ipynb.

  Args:
    cleaning_df: DataFrame containing data cleansing results across seeds.
    batch_size: Optional batch size filter.
    noise_rate: Optional noise rate filter.
    learning_rate: Optional learning rate filter.
    num_epochs: Optional num epochs filter.
    num_samples: Total training samples (default: 50000).
    num_remove: Number of samples removed. Defaults to int(noise_rate *
      num_samples).
    methods: Optional sequence of custom methods to include.
    rename_methods: Optional mapping from method names to display names.
    add_latex_support: If True, wraps entries in '$...$'.
    return_raw: If True, returns numerical means and SEMs.

  Returns:
    Formatted or raw DataFrame indexed by method with columns:
    ['Validation loss', 'Test loss', 'Validation accuracy', 'Test accuracy'].
  """
  cols = ["method", "k", "val_loss", "test_loss", "val_acc", "test_acc"]
  if num_remove is None:
    if noise_rate is not None:
      num_remove = int(noise_rate * num_samples)
    else:
      num_remove = 0

  # Default methods (baseline and oracles) to place at the top
  top_methods = ["baseline", "remove_corrupted"]
  if "actual_cf" in cleaning_df["method"].values:
    top_methods.append("actual_cf")

  custom_methods = list(methods) if methods is not None else []

  # Combine all methods: top_methods first, then custom methods (deduplicated)
  all_methods = []
  for m in top_methods + custom_methods:
    if m not in all_methods:
      all_methods.append(m)

  cond = pd.Series(True, index=cleaning_df.index)
  if batch_size is not None and "batch_size" in cleaning_df.columns:
    cond &= cleaning_df["batch_size"] == batch_size
  if noise_rate is not None and "noise_rate" in cleaning_df.columns:
    cond &= cleaning_df["noise_rate"] == noise_rate
  if learning_rate is not None and "learning_rate" in cleaning_df.columns:
    cond &= cleaning_df["learning_rate"] == learning_rate
  if num_epochs is not None and "num_epochs" in cleaning_df.columns:
    cond &= cleaning_df["num_epochs"] == num_epochs

  available_cols = [c for c in cols if c in cleaning_df.columns]
  df_filtered = cleaning_df[cond].copy()
  if "k" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["k"] != "auto") & df_filtered["k"].notna()
    ]
    df_seeds = df_filtered[available_cols].copy()
    df_seeds["k"] = (
        pd.to_numeric(df_seeds["k"], errors="coerce").fillna(-1).astype(int)
    )
  else:
    df_seeds = df_filtered[available_cols].copy()

  # Filter criteria:
  # 1) k == num_remove for selected non-baseline methods
  # 2) k == 0 specifically for 'baseline'
  is_target_k = (df_seeds["k"] == num_remove) & (
      df_seeds["method"] != "baseline"
  )
  is_baseline = (df_seeds["k"] == 0) & (df_seeds["method"] == "baseline")

  sub_df = df_seeds[is_target_k | is_baseline]
  sub_df = sub_df[sub_df["method"].isin(all_methods)]

  metric_cols = ["val_loss", "test_loss", "val_acc", "test_acc"]
  available_metrics = [c for c in metric_cols if c in sub_df.columns]

  # Calculate mean and SEM
  grouped = sub_df.groupby("method")[available_metrics]
  means = grouped.mean()
  sems = grouped.sem().fillna(0.0)

  column_names = [
      "Validation loss",
      "Test loss",
      "Validation accuracy",
      "Test accuracy",
  ]
  metric_to_col = dict(zip(metric_cols, column_names))

  if return_raw:
    raw_df = pd.DataFrame(index=all_methods)
    for m_col in available_metrics:
      display_name = metric_to_col.get(m_col, m_col)
      raw_df[f"{display_name}_mean"] = [
          means.loc[m, m_col] if m in means.index else np.nan
          for m in all_methods
      ]
      raw_df[f"{display_name}_sem"] = [
          sems.loc[m, m_col] if m in sems.index else np.nan for m in all_methods
      ]
    if rename_methods is not None:
      raw_df = raw_df.rename(index=rename_methods)
    return raw_df

  formatted_df = pd.DataFrame(index=all_methods)
  for m_col in metric_cols:
    display_name = metric_to_col.get(m_col, m_col)
    formatted_col = []
    for method in all_methods:
      if m_col in available_metrics and method in means.index:
        m = means.loc[method, m_col]
        s = sems.loc[method, m_col]
        entry_str = f"{m:.3f} \\pm {1.96 * s:.3f}"
        if add_latex_support:
          entry_str = f"${entry_str}$"
        formatted_col.append(entry_str)
      else:
        formatted_col.append("N/A")
    formatted_df[display_name] = formatted_col

  if rename_methods is not None:
    formatted_df = formatted_df.rename(index=rename_methods)

  return formatted_df


def plot_cleansing_stats(
    cleaning_df: pd.DataFrame,
    batch_size: int | None = None,
    noise_rate: float | None = None,
    learning_rate: float | None = None,
    num_epochs: int | None = None,
    num_samples: int = 50000,
    methods_to_plot: Sequence[str] | None = None,
    method_mapping: Mapping[str, str] | None = None,
    palette: Mapping[str, Any] | None = None,
    output_path: str | None = None,
    save_file: str | None = None,
    title: str | None = None,
) -> plt.Figure:
  """Plots Validation & Test Loss/Accuracy with respect to removed samples (k).

  Generates a 2x2 grid of line plots comparing data cleansing methods against
  the baseline model. Matches plot_cleansing_stats in data_analysis.ipynb.

  Args:
    cleaning_df: DataFrame containing data cleansing evaluation results.
    batch_size: Optional batch size filter.
    noise_rate: Optional noise rate filter.
    learning_rate: Optional learning rate filter.
    num_epochs: Optional num epochs filter.
    num_samples: Total number of training samples (to compute corruption line).
    methods_to_plot: Sequence of methods to include in the plot.
    method_mapping: Mapping from raw method names to human-readable labels.
    palette: Color mapping for method labels.
    output_path: Optional path to save image file.
    save_file: Optional alias for output_path matching data_analysis.ipynb.
    title: Optional figure super-title.

  Returns:
    matplotlib Figure object.
  """
  cols = ["method", "k", "val_loss", "test_loss", "val_acc", "test_acc"]
  available_cols = [c for c in cols if c in cleaning_df.columns]

  cond = pd.Series(True, index=cleaning_df.index)
  if batch_size is not None and "batch_size" in cleaning_df.columns:
    cond &= cleaning_df["batch_size"] == batch_size
  if noise_rate is not None and "noise_rate" in cleaning_df.columns:
    cond &= cleaning_df["noise_rate"] == noise_rate
  if learning_rate is not None and "learning_rate" in cleaning_df.columns:
    cond &= cleaning_df["learning_rate"] == learning_rate
  if num_epochs is not None and "num_epochs" in cleaning_df.columns:
    cond &= cleaning_df["num_epochs"] == num_epochs

  df_filtered = cleaning_df[cond].copy()
  if "k" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["k"] != "auto") & df_filtered["k"].notna()
    ]
    df_seeds = df_filtered[available_cols].copy()
    df_seeds["k"] = (
        pd.to_numeric(df_seeds["k"], errors="coerce").fillna(-1).astype(int)
    )
  else:
    df_seeds = df_filtered[available_cols].copy()

  if methods_to_plot is None:
    methods_to_plot = [
        "adam_recursive",
        "sgd_all",
        "tracin_adam",
        "tracin_sgd",
        "actual_cf",
    ]

  if method_mapping is None:
    method_mapping = DEFAULT_METHOD_MAPPING

  if palette is None:
    palette = DEFAULT_PALETTE

  plot_cond = (df_seeds["k"] >= 100) & (
      df_seeds["method"].isin(methods_to_plot)
  )
  baseline_cond = (df_seeds["k"] == 0) & (df_seeds["method"] == "baseline")
  baseline_stats = df_seeds[baseline_cond]

  num_corrupt = (
      int(noise_rate * num_samples) if noise_rate is not None else None
  )
  ticks = [100, 1000]
  if num_corrupt and num_corrupt > 0:
    ticks.append(num_corrupt)
  xticklabels = sorted(list(set(ticks)))

  fig, axs = plt.subplots(2, 2, figsize=(24, 16))
  if title:
    fig.suptitle(title, fontsize=16, y=1.01)

  metrics = [["val_loss", "test_loss"], ["val_acc", "test_acc"]]
  labels = [["Val Loss", "Test Loss"], ["Val Acc", "Test Acc"]]

  mapped_hue = df_seeds.loc[plot_cond, "method"].map(
      lambda m: method_mapping.get(m, m)
  )

  for row_idx in range(2):
    for col_idx in range(2):
      metric = metrics[row_idx][col_idx]
      label = labels[row_idx][col_idx]
      ax = axs[row_idx, col_idx]

      if metric in df_seeds.columns:
        sns.lineplot(
            data=df_seeds[plot_cond],
            x="k",
            y=metric,
            hue=mapped_hue,
            ax=ax,
            palette=palette,
            marker="o",
            markersize=8,
        )

        ax.set_xscale("log")
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=14)
        ax.set_xlabel("Number of removed samples", fontsize=20)
        ax.set_ylabel(label, fontsize=18)
        ax.tick_params(axis="y", labelsize=14)

        if num_corrupt and num_corrupt > 0:
          ax.axvline(
              num_corrupt,
              color="black",
              linestyle="--",
          )

        if not baseline_stats.empty and metric in baseline_stats.columns:
          m = baseline_stats[metric].mean()
          se = baseline_stats[metric].sem()
          ax.axhline(
              m, color="black", linestyle="--", label=f"Baseline ({m:.4f})"
          )
          if pd.notna(se) and se > 0:
            ax.axhspan(m - 1.96 * se, m + 1.96 * se, color="black", alpha=0.15)

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=14, loc="best", framealpha=0.8)

  plt.tight_layout()

  save_dest = save_file or output_path
  if save_dest:
    os.makedirs(os.path.dirname(os.path.abspath(save_dest)), exist_ok=True)
    plt.savefig(save_dest, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_dest}")

  return fig


# ==============================================================================
# 6. CLI Entry Point
# ==============================================================================


def parse_args() -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description="Data analysis and plotting tools for data attribution."
  )
  parser.add_argument(
      "csv_file",
      nargs="?",
      default=None,
      help="Path to input CSV file (e.g. combined_files_272763300.csv)",
  )
  parser.add_argument(
      "--input_csv",
      "-i",
      type=str,
      default=None,
      help="Alternative flag for input CSV file path",
  )
  parser.add_argument(
      "--x_col",
      type=str,
      default="True adam CF test loss",
      help="Column name for X-axis (default: 'True adam CF test loss')",
  )
  parser.add_argument(
      "--y_cols",
      nargs="+",
      default=None,
      help=(
          "Columns to plot on Y-axis (space separated). Default: auto-select"
          " numeric columns."
      ),
  )
  parser.add_argument(
      "--group_col",
      type=str,
      default="File",
      help="Column name to group data by (default: 'File')",
  )
  parser.add_argument(
      "--base_loss_col",
      type=str,
      default="Base adam test loss",
      help=(
          "Column name containing base test loss (default: 'Base adam test"
          " loss')"
      ),
  )
  parser.add_argument(
      "--layout",
      type=str,
      choices=["by_file", "by_method", "matrix"],
      default="by_file",
      help="Subplot layout mode: 'by_file' (default), 'by_method', or 'matrix'",
  )
  parser.add_argument(
      "--no_identity",
      action="store_true",
      help="Disable drawing the y = x identity line",
  )
  parser.add_argument(
      "--no_base_lines",
      action="store_true",
      help="Disable drawing horizontal and vertical base loss lines",
  )
  parser.add_argument(
      "--output",
      "-o",
      type=str,
      default=None,
      help=(
          "Output image file path (e.g. scatter_plot.png). Default:"
          " <csv_name>_scatter.png"
      ),
  )
  parser.add_argument(
      "--title",
      type=str,
      default=None,
      help="Custom main title for figure",
  )
  return parser.parse_args()


def main():
  args = parse_args()

  csv_path = args.csv_file or args.input_csv
  if not csv_path:
    print(
        "Error: Please provide a CSV file path as positional argument or"
        " via --input_csv.",
        file=sys.stderr,
    )
    sys.exit(1)

  if not os.path.isfile(csv_path):
    print(f"Error: File '{csv_path}' does not exist.", file=sys.stderr)
    sys.exit(1)

  output_path = args.output
  if not output_path:
    base_no_ext = os.path.splitext(csv_path)[0]
    output_path = f"{base_no_ext}_scatter.png"

  df = pd.read_csv(csv_path)

  create_scatterplots(
      df=df,
      x_col=args.x_col,
      y_cols=args.y_cols,
      group_col=args.group_col,
      base_loss_col=args.base_loss_col,
      layout=args.layout,
      show_identity=not args.no_identity,
      show_base_lines=not args.no_base_lines,
      output_path=output_path,
      title=args.title,
  )


if __name__ == "__main__":
  main()
