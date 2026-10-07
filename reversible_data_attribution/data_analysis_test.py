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
"""Tests for data_analysis."""

import os
import tempfile

from absl.testing import absltest
import numpy as np
import pandas as pd

from reversible_data_attribution import data_analysis


class DataAnalysisTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # 1. Test data for scatterplots
    self.scatter_df = pd.DataFrame({
        "File": [
            "wu_1.txt",
            "wu_1.txt",
            "wu_1.txt",
            "wu_2.txt",
            "wu_2.txt",
        ],
        "Index": ["01", "02", "03", "01", "02"],
        "Base adam test loss": [1.5, 1.5, 1.5, 2.0, 2.0],
        "True adam CF test loss": [1.6, 1.4, 1.7, 2.1, 1.9],
        "Matched adam": [1.58, 1.42, 1.69, 2.08, 1.92],
        "Mismatched SGD": [1.55, 1.45, 1.65, 2.05, 1.95],
    })

    # 2. Test data for score merging, correlations, and cleansing evaluation
    np.random.seed(42)
    n_samples = 40
    self.scores_df1 = pd.DataFrame({
        "sample_index": list(range(n_samples)) * 2,
        "is_corrupted": ([True] * 8 + [False] * 32) * 2,
        "noise_seed": [1] * n_samples + [2] * n_samples,
        "batch_size": [100] * (2 * n_samples),
        "noise_rate": [0.2] * (2 * n_samples),
        "learning_rate": [1e-3] * (2 * n_samples),
        "num_epochs": [10] * (2 * n_samples),
        "actual_cf": np.concatenate([
            np.linspace(-0.5, 0.5, n_samples),
            np.linspace(-0.6, 0.6, n_samples),
        ]),
        "adam_recursive": np.concatenate([
            np.linspace(-0.45, 0.48, n_samples)
            + np.random.normal(0, 0.02, n_samples),
            np.linspace(-0.55, 0.58, n_samples)
            + np.random.normal(0, 0.02, n_samples),
        ]),
    })

    self.scores_df2 = pd.DataFrame({
        "sample_index": list(range(n_samples)) * 2,
        "is_corrupted": ([True] * 8 + [False] * 32) * 2,
        "noise_seed": [1] * n_samples + [2] * n_samples,
        "batch_size": [100] * (2 * n_samples),
        "noise_rate": [0.2] * (2 * n_samples),
        "learning_rate": [1e-3] * (2 * n_samples),
        "num_epochs": [10] * (2 * n_samples),
        "tracin_adam": np.concatenate([
            np.linspace(-0.4, 0.45, n_samples),
            np.linspace(-0.5, 0.52, n_samples),
        ]),
        "sgd_all": np.concatenate([
            np.linspace(-0.35, 0.4, n_samples),
            np.linspace(-0.45, 0.48, n_samples),
        ]),
    })

    # 3. Test data for cleansing stats
    self.cleansing_df = pd.DataFrame({
        "batch_size": [100] * 12,
        "noise_rate": [0.2] * 12,
        "learning_rate": [1e-3] * 12,
        "num_epochs": [10] * 12,
        "method": [
            "baseline",
            "baseline",
            "adam_recursive",
            "adam_recursive",
            "sgd_all",
            "sgd_all",
            "tracin_adam",
            "tracin_adam",
            "tracin_sgd",
            "tracin_sgd",
            "actual_cf",
            "actual_cf",
        ],
        "k": [0, 0, 100, 500, 100, 500, 100, 500, 100, 500, 100, 500],
        "val_loss": [
            0.5,
            0.52,
            0.48,
            0.42,
            0.49,
            0.44,
            0.51,
            0.46,
            0.5,
            0.45,
            0.45,
            0.38,
        ],
        "test_loss": [
            0.55,
            0.57,
            0.52,
            0.45,
            0.53,
            0.47,
            0.54,
            0.49,
            0.53,
            0.48,
            0.48,
            0.41,
        ],
        "val_acc": [
            0.85,
            0.84,
            0.86,
            0.88,
            0.85,
            0.87,
            0.84,
            0.86,
            0.85,
            0.87,
            0.87,
            0.9,
        ],
        "test_acc": [
            0.83,
            0.82,
            0.84,
            0.86,
            0.83,
            0.85,
            0.82,
            0.84,
            0.83,
            0.85,
            0.85,
            0.88,
        ],
    })

  # --- Scatterplot Tests ---

  def test_auto_select_y_cols(self):
    y_cols = data_analysis.auto_select_y_cols(
        self.scatter_df,
        ignore_cols=[
            "Index",
            "True adam CF test loss",
            "File",
            "Base adam test loss",
        ],
    )
    self.assertEqual(y_cols, ["Matched adam", "Mismatched SGD"])

  def test_missing_x_col_raises(self):
    with self.assertRaises(ValueError):
      data_analysis.create_scatterplots(
          df=self.scatter_df.copy(),
          x_col="Nonexistent Column",
      )

  def test_missing_group_col_raises(self):
    with self.assertRaises(ValueError):
      data_analysis.create_scatterplots(
          df=self.scatter_df.copy(),
          group_col="Nonexistent Column",
      )

  def test_create_scatterplots_layouts(self):
    for layout in ["by_file", "by_method", "matrix"]:
      fig = data_analysis.create_scatterplots(
          df=self.scatter_df.copy(),
          x_col="True adam CF test loss",
          y_cols=["Matched adam", "Mismatched SGD"],
          group_col="File",
          base_loss_col="Base adam test loss",
          layout=layout,
      )
      self.assertIsNotNone(fig)

  def test_create_scatterplots_save_output(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      out_path = os.path.join(tmp_dir, "test_plot.png")
      data_analysis.create_scatterplots(
          df=self.scatter_df.copy(),
          output_path=out_path,
      )
      self.assertTrue(os.path.exists(out_path))

  # --- Score Merging Tests ---

  def test_merge_clean_scores(self):
    merged = data_analysis.merge_clean_scores(
        [self.scores_df1, self.scores_df2],
        batch_size=100,
        noise_rate=0.2,
        learning_rate=1e-3,
        num_epochs=10,
    )
    self.assertIn("adam_recursive", merged.columns)
    self.assertIn("tracin_adam", merged.columns)
    self.assertIn("sgd_all", merged.columns)
    self.assertEqual(len(merged), len(self.scores_df1))

  def test_merge_scores_on_cond_alias(self):
    merged1 = data_analysis.merge_clean_scores(
        [self.scores_df1, self.scores_df2],
        batch_size=100,
    )
    merged2 = data_analysis.merge_scores_on_cond(
        [self.scores_df1, self.scores_df2],
        batch_size=100,
    )
    pd.testing.assert_frame_equal(merged1, merged2)

  def test_merge_scores_all(self):
    merged = data_analysis.merge_scores_all([self.scores_df1, self.scores_df2])
    self.assertIn("adam_recursive", merged.columns)
    self.assertIn("tracin_adam", merged.columns)
    self.assertEqual(len(merged), len(self.scores_df1))

  # --- Score Correlation Tests ---

  def test_compute_score_correlations(self):
    merged = data_analysis.merge_clean_scores(
        [self.scores_df1, self.scores_df2]
    )
    corr_df = data_analysis.compute_score_correlations(
        scores_df=merged,
        actual_col="actual_cf",
        estimators=["adam_recursive", "tracin_adam"],
        group_col="noise_seed",
    )
    self.assertEqual(len(corr_df), 4)  # 2 seeds x 2 estimators
    self.assertIn("Pearson", corr_df.columns)
    self.assertIn("Spearman", corr_df.columns)
    # Check that high-correlation synthetic data gives Pearson > 0.9
    adam_corr = corr_df[corr_df["estimator"] == "adam_recursive"]["Pearson"]
    for val in adam_corr:
      self.assertGreater(val, 0.9)

  def test_get_score_correlations(self):
    merged = data_analysis.merge_clean_scores(
        [self.scores_df1, self.scores_df2]
    )
    corr_df, summary_df = data_analysis.get_score_correlations(
        scores_cond=merged,
        actual_col="actual_cf",
        estimators=["adam_recursive", "tracin_adam"],
        rename_methods={"adam_recursive": "TSLOO-Adam"},
        add_latex_support=True,
    )
    self.assertIsNotNone(corr_df)
    self.assertIsNotNone(summary_df)
    self.assertIn("TSLOO-Adam", summary_df.index)
    self.assertIn("Pearson", summary_df.columns)
    self.assertIn("Spearman", summary_df.columns)
    self.assertTrue(summary_df.loc["TSLOO-Adam", "Pearson"].startswith("$"))

  def test_plot_score_correlations(self):
    merged = data_analysis.merge_clean_scores(
        [self.scores_df1, self.scores_df2]
    )
    corr_df, fig = data_analysis.plot_score_correlations(
        scores_df=merged,
        actual_col="actual_cf",
        estimators=["adam_recursive"],
        group_col="noise_seed",
    )
    self.assertIsNotNone(corr_df)
    self.assertIsNotNone(fig)

  def test_plot_score_correlations_notebook_style(self):
    merged = data_analysis.merge_clean_scores(
        [self.scores_df1, self.scores_df2]
    )
    corr_df, fig = data_analysis.plot_score_correlations(
        merged,
        max_cf_val=0.5,
    )
    self.assertIsNotNone(corr_df)
    self.assertIsNotNone(fig)

  # --- Data Cleansing Evaluation Tests ---

  def test_evaluate_data_cleansing(self):
    merged = data_analysis.merge_clean_scores(
        [self.scores_df1, self.scores_df2]
    )
    # Total corrupted per seed is 8
    res_formatted = data_analysis.evaluate_data_cleansing(
        scores_df=merged,
        methods=["adam_recursive", "tracin_adam"],
        num_removal=8,
        label_col="is_corrupted",
        group_col="noise_seed",
        return_raw=False,
    )
    self.assertIn("Precision", res_formatted.columns)
    self.assertIn("ROC AUC", res_formatted.columns)
    self.assertEqual(len(res_formatted), 2)

    res_raw = data_analysis.evaluate_data_cleansing(
        scores_df=merged,
        methods=["adam_recursive", "tracin_adam"],
        num_removal=8,
        label_col="is_corrupted",
        group_col="noise_seed",
        return_raw=True,
    )
    self.assertIn("Precision_mean", res_raw.columns)
    self.assertIn("ROC AUC_mean", res_raw.columns)

  def test_evaluate_data_cleansing_latex_and_rename(self):
    merged = data_analysis.merge_clean_scores(
        [self.scores_df1, self.scores_df2]
    )
    res = data_analysis.evaluate_data_cleansing(
        scores_df=merged,
        methods=["adam_recursive", "tracin_adam"],
        num_removal=8,
        rename_methods={"adam_recursive": "TSLOO-Adam"},
        add_latex_support=True,
    )
    self.assertIn("TSLOO-Adam", res.index)
    self.assertIn("Precision", res.columns)
    self.assertTrue(res.loc["TSLOO-Adam", "Precision"].startswith("$"))

  # --- Data Cleansing Statistics & Plotting Tests ---

  def test_print_cleansing_stats(self):
    stats = data_analysis.print_cleansing_stats(
        cleaning_df=self.cleansing_df,
        batch_size=100,
        noise_rate=0.2,
        learning_rate=1e-3,
        num_epochs=10,
        # 0.2 * 2500 = 500 => matches k=500 in self.cleansing_df
        num_samples=2500,
        methods=["adam_recursive", "sgd_all"],
        rename_methods={"adam_recursive": "TSLOO-Adam"},
        add_latex_support=True,
    )
    self.assertIn("Validation loss", stats.columns)
    self.assertIn("Test loss", stats.columns)
    self.assertIn("TSLOO-Adam", stats.index)
    self.assertIn("baseline", stats.index)

  def test_plot_cleansing_stats(self):
    fig = data_analysis.plot_cleansing_stats(
        cleaning_df=self.cleansing_df,
        batch_size=100,
        noise_rate=0.2,
        learning_rate=1e-3,
        num_epochs=10,
        num_samples=1000,
    )
    self.assertIsNotNone(fig)

  def test_plot_cleansing_stats_save_file(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      out_path = os.path.join(tmp_dir, "clean_plot.png")
      fig = data_analysis.plot_cleansing_stats(
          cleaning_df=self.cleansing_df,
          batch_size=100,
          noise_rate=0.2,
          learning_rate=1e-3,
          num_epochs=10,
          num_samples=1000,
          save_file=out_path,
      )
      self.assertIsNotNone(fig)
      self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
  absltest.main()
