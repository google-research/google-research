# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python3
"""Script for running PPCA on the target dataset and generating plots.

The goal of this analysis is to understand which dimensions of the target label
space are significant via Principal Preserved Component Analysis
(Cowen et al., 2019). PPCA seeks to identify dimensions of the latent space
that maximally covary across two datasets (in our case, randomly split raters).

Reference:
  Cowen, A. S., Laukka, P., Elfenbein, H. A., Liu, R., & Keltner, D. (2019).
  The primacy of categories in the recognition of 12 emotions in speech prosody
  across two cultures. Nature human behaviour, 3(4), 369.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import random

from absl import app
from absl import flags
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import wilcoxon
import seaborn as sns
from sklearn.manifold import TSNE
from statsmodels.stats.multitest import multipletests

sns.set_style("whitegrid")
plt.rcParams["xtick.major.size"] = 0
plt.rcParams["ytick.major.size"] = 0
plt.rcParams["figure.dpi"] = 1000

FLAGS = flags.FLAGS

flags.DEFINE_string("data", None, "Input data.")

flags.DEFINE_string("plot_dir", "plots", "Directory for saving plots.")

flags.DEFINE_string("target_file", "data/targets.txt",
                    "File containing list of targets.")

flags.DEFINE_string("rgb_colors", "plots/colors.tsv",
                    "File containing list of distinctive rgb colors.")

flags.DEFINE_string(
    "target_color_order", "plots/color_order.txt",
    "File containing targets in order for coloring based on FLAGS.rgb_colors.")


def PPCA(x, y):
  """Function that returns PPCA weights for x and y."""
  x = x - x.mean(axis=0)  # demean x
  y = y - y.mean(axis=0)  # demean y
  crosscov = np.matmul(x.transpose(), y) + np.matmul(
      y.transpose(), x)  # symmetrized cross-covariance

  # v is the eigenvalues (or component covariances)
  # w is the eigenvectors (or PPCs)
  v, w = LA.eigh(crosscov)
  w = np.flip(w, 1)  # reverse w so it is in descending order of eigenvalue
  v = np.flip(v)  # reverse v so it is in descending order
  return w, v


def Demean(m):
  return m - m.mean(axis=0)


def PartialCorr(x, y, covar):
  """Calculate partial correlation."""
  cvar = np.atleast_2d(covar)
  beta_x = np.linalg.lstsq(cvar, x, rcond=None)[0]
  beta_y = np.linalg.lstsq(cvar, y, rcond=None)[0]
  res_x = x - np.dot(cvar, beta_x)
  res_y = y - np.dot(cvar, beta_y)
  return spearmanr(res_x, res_y)


def Varimax(phi, gamma=1, q=20, tol=1e-6):
  """Source: https://stackoverflow.com/questions/17628589/perform-varimax-rotation-in-python-using-numpy."""
  p, k = phi.shape
  r = np.eye(k)
  d = 0
  for _ in range(q):
    d_old = d
    l = np.dot(phi, r)
    u, s, vh = LA.svd(
        np.dot(
            phi.T,
            np.asarray(l)**3 -
            (gamma / p) * np.dot(l, np.diag(np.diag(np.dot(l.T, l))))))
    r = np.dot(u, vh)
    d = np.sum(s)
    if d / d_old < tol:
      break
  return np.dot(phi, r)


def LeaveOut(ratings, rater_msk, worker2examples, worker_id):
  """Calculate correlations and partial correlations for a particular rater."""
  examples = worker2examples[worker_id]
  use_examples = copy.deepcopy(ratings[[idx for idx, _ in examples]])
  use_examples_msk = copy.deepcopy(rater_msk[[idx for idx, _ in examples
                                             ]]).sum(axis=1).astype(int)
  rater_indices = [rater_idx for _, rater_idx in examples]

  # Take remaining raters and split them randomly
  x = []
  y = []
  exclude = []
  average_insample = []
  for i, ex in enumerate(use_examples):

    # Separate leave-out rater from others
    keep = []
    for j, worker_rating in enumerate(ex[:use_examples_msk[i]]):
      if j != rater_indices[i]:
        keep.append(list(worker_rating))
      else:
        exclude.append(list(worker_rating))

    # Calculate average of insample ratings
    avg_insample_ex = np.array(keep).mean(axis=0)
    assert np.isnan(avg_insample_ex).sum() == 0
    average_insample.append(list(avg_insample_ex))

    # Shuffle raters randomly
    random.shuffle(keep)

    # If there are two in-sample raters, just separate them
    num_raters = len(keep)
    if num_raters == 2:
      x.append(keep[0])
      y.append(keep[1])
    else:
      x.append(list(np.array(keep[:int(num_raters / 2)]).mean(axis=0)))
      y.append(list(np.array(keep[int(num_raters / 2):]).mean(axis=0)))

  assert np.isnan(x).sum() == 0
  assert np.isnan(y).sum() == 0
  x = np.array(x)
  y = np.array(y)

  w, _ = PPCA(x, y)  # get PPCs
  exclude = np.array(exclude)
  assert np.isnan(exclude).sum() == 0
  average_insample = np.array(average_insample)
  exclude = Demean(exclude)  # demean held out rater's values
  average_insample = Demean(average_insample)  # demean in sample rater's values
  assert np.isnan(exclude).sum() == 0
  assert np.isnan(average_insample).sum() == 0
  left_out_scores = exclude.dot(w)  # get scores for leave-out rater
  insample_scores = average_insample.dot(w)  # scores for in-sample raters
  assert np.isnan(left_out_scores).sum() == 0
  assert np.isnan(insample_scores).sum() == 0

  # Run partial Spearman correlation for each component, doing a regular
  # Spearman correlation for the first dimension
  first_corr = spearmanr(left_out_scores[:, 0], insample_scores[:, 0])[0]
  partial_corrs = [first_corr]
  corrs = [first_corr]
  for i in range(1, left_out_scores.shape[1]):
    # each column represents a component
    # partial out insample raters' scores for previous components
    pc = PartialCorr(left_out_scores[:, i], insample_scores[:, i],
                     insample_scores[:, :i])[0]

    # regular spearman correlation
    c = spearmanr(left_out_scores[:, i], insample_scores[:, i])[0]

    # if no correlation (i.e. the standard deviation of the vectors is 0, this
    # happens when the rater only labeled a 1-2 examples) ignore that rater
    if np.isnan(pc):
      break
    partial_corrs.append(pc)
    corrs.append(c)
  return partial_corrs, corrs


def PlotCovar(v):
  var_explained = v / np.sum(v) * 100
  num_components = 28
  fig = plt.figure()
  plt.plot(np.arange(num_components), var_explained, marker="o")
  plt.ylabel("Percentage of Covariance Explained", fontsize="x-large")
  plt.xlabel("Component", fontsize="x-large")
  plt.xlim(-1, num_components)
  fig.savefig(
      FLAGS.plot_dir + "/covar_explained.pdf",
      dpi=600,
      format="pdf",
      bbox_inches="tight")


def main(_):
  print("Loading data...")
  data = pd.read_csv(FLAGS.data, encoding="utf-8")
  print("%d Examples" % (len(set(data["id"]))))
  print("%d Annotations" % len(data))
  os.makedirs(FLAGS.plot_dir, exist_ok=True)

  with open(FLAGS.target_file, "r") as f:
    all_targets = f.read().splitlines()
  all_targets_neutral = all_targets + ["neutral"]
  target2idx = {e: i for i, e in enumerate(all_targets)}
  print("%d Target Categories" % len(all_targets))

  print("Processing data...")

  # Remove neutral labels
  data = data[data["neutral"] == 0]

  # Remove examples with no ratings (difficult examples)
  data = data[data[all_targets_neutral].sum(axis=1) != 0]

  # Convert into num_examples x num_raters x num_ratings format
  data = data.groupby("id").filter(lambda x: len(x) >= 3)
  id_groups = data.groupby("id")

  worker2examples = {}  # dict mapping worker ids to (example, rater id) tuples
  max_num_raters = data.groupby("id").size().max()
  ratings = np.zeros(
      (len(id_groups), max_num_raters, len(all_targets)))  # ignore "neutral"
  rater_msk = np.zeros(
      (len(id_groups), max_num_raters))  # for masking out non-existent raters
  print("Ratings shape", ratings.shape)

  # Get ratings and rater mask
  texts = []
  for ex_idx, (_, g) in enumerate(id_groups):
    texts.append(g.iloc[0]["text"])
    rater_count = 0

    # iterate through workers
    for _, row in g.iterrows():
      for e in all_targets:
        ratings[ex_idx, rater_count, target2idx[e]] = row[e]
        rater_msk[ex_idx, rater_count] = 1

      worker_id = row["rater_id"]
      if worker_id in worker2examples:
        worker2examples[worker_id].append((ex_idx, rater_count))
      else:
        worker2examples[worker_id] = [(ex_idx, rater_count)]
      rater_count += 1

  print("Calculating leave-out (partial) correlations...")
  partial_corr_per_rater = []
  corr_per_rater = []
  for worker_id in worker2examples:
    partial_corrs, corrs = LeaveOut(ratings, rater_msk, worker2examples,
                                    worker_id)
    if len(partial_corrs) < len(all_targets):
      continue

    partial_corr_per_rater.append(partial_corrs)
    corr_per_rater.append(corrs)
  corr_per_rater = np.array(corr_per_rater)
  partial_corr_per_rater = np.array(partial_corr_per_rater)

  # Verify that there are no NaN values
  assert np.isnan(corr_per_rater).sum() == 0

  # Apply Wilcoxon signed rank test to test significance of each dimension
  p_vals = np.apply_along_axis(wilcoxon, 0, partial_corr_per_rater)[1]

  # Apply Bonferroni correction
  reject, corr_pvals, _, newalpha = multipletests(
      p_vals, alpha=0.05, method="bonferroni")
  print("Which dimensions to keep?")
  print(reject)
  print(corr_pvals)
  print(newalpha)

  print("Running PPCA on all the data...")
  # Take all raters and split them randomly
  x = []
  y = []
  rater_counts = rater_msk.sum(axis=1).astype(int)
  all_ratings_avg = []
  for i, ex in enumerate(ratings):
    # Get actual raters based on mask
    keep = []
    for worker_rating in ex[:rater_counts[i]]:
      keep.append(list(worker_rating))
    all_ratings_avg.append(list(np.array(keep).mean(axis=0)))

    # Shuffle raters randomly
    random.shuffle(keep)

    num_raters = len(keep)
    x.append(list(np.array(keep[:int(num_raters / 2)]).mean(axis=0)))
    y.append(list(np.array(keep[int(num_raters / 2):]).mean(axis=0)))

  x = np.array(x)
  y = np.array(y)
  all_ratings_avg = np.array(all_ratings_avg)
  w, v = PPCA(x, y)  # final components (p-values determine which ones to keep)

  print("Plotting percentage of covariance explained...")
  PlotCovar(v)

  # Apply varimax rotation
  w_vari = Varimax(w)

  # Get mapping between ppcs and targets
  map_df = pd.DataFrame(
      w_vari, index=all_targets, columns=np.arange(len(all_targets))).round(4)
  # Sort to move values to diagonal
  map_df = map_df[list(
      np.argsort(map_df.apply(lambda x: pd.Series.nonzero(x)[0]).values)[0])]
  f = plt.figure(figsize=(10, 6), dpi=300)
  sns.heatmap(
      map_df,
      center=0,
      cmap=sns.diverging_palette(240, 10, n=50),
      yticklabels=all_targets)
  plt.xlabel("Component")
  plt.savefig(
      FLAGS.plot_dir + "/component_loadings.pdf",
      dpi=600,
      format="pdf",
      bbox_inches="tight")
  ppc2target = map_df.abs().idxmax().to_dict()
  target2ppc = {e: i for i, e in ppc2target.items()}
  print(ppc2target)

  print("Plotting frequency and mean left-out rater correlations...")
  corr_mean = corr_per_rater.mean(axis=0)
  corr_mean_ordered = [corr_mean[target2ppc[e]] for e in all_targets]
  df_plot = pd.DataFrame({
      "target": all_targets,
      "agreement": corr_mean_ordered
  })
  df_plot["count"] = df_plot["target"].map(
      data[all_targets].sum(axis=0).to_dict())
  df_plot.sort_values("count", ascending=False, inplace=True)
  df_plot.to_csv(FLAGS.plot_dir + "/target_agreements.csv", index=False)

  # Get colors
  norm = plt.Normalize(df_plot["agreement"].min(), df_plot["agreement"].max())
  sm = plt.cm.ScalarMappable(cmap="BuPu", norm=norm)
  sm.set_array([])

  # Generate figure
  fig = plt.figure(dpi=600, figsize=(5, 6))
  ax = sns.barplot(
      data=df_plot,
      y="target",
      x="count",
      orient="h",
      hue="agreement",
      palette="BuPu",
      dodge=False,
      edgecolor="black",
      linewidth=1)
  ax.get_legend().remove()
  ax.figure.colorbar(sm)
  plt.text(18000, 31, "Interrater\nCorrelation", ha="center")
  plt.xlabel("Number of Examples")
  plt.ylabel("")
  plt.draw()
  labels = [item.get_text() for item in ax.get_xticklabels()]
  ax.set_xticklabels(["%dk" % (int(int(label) / 1000)) for label in labels])
  plt.tight_layout()
  fig.savefig(
      FLAGS.plot_dir + "/label_distr_agreement.pdf",
      dpi=600,
      format="pdf",
      bbox_inches="tight")

  print("Generating t-SNE plot...")
  # Get PPC scores for all examples
  all_ratings_avg = Demean(all_ratings_avg)  # demean all ratings
  ppc_scores = all_ratings_avg.dot(w_vari)  # project onto ppcs
  ppc_scores_abs = np.absolute(ppc_scores)

  # Load maximally distinct colors
  colors = pd.read_csv(
      FLAGS.rgb_colors, sep="\t", header=None, names=np.arange(3))

  # Set colors (todo(ddemszky): add names to colors in file)
  palette_rgb = colors.values
  with open(FLAGS.target_color_order) as f:
    color_order = f.read().splitlines()
  ppc2color = {target2ppc[e]: i for i, e in enumerate(color_order)}
  # get rgb value for each example based on weighted average of top targets
  rgb_vals = []
  hex_vals = []
  top_categories = []
  threshold = 0.5  # exclude points not loading on any of the top 10 categories
  counter = 0
  rgb_max = 255
  other_color = palette_rgb[len(all_targets), :]
  for i, scores in enumerate(ppc_scores_abs):

    top_ppcs = [
        idx for idx in (-scores).argsort()[:2] if scores[idx] > threshold
    ]
    top_targets = ",".join([ppc2target[idx] for idx in top_ppcs
                           ]) if top_ppcs else "other"
    top_categories.append(top_targets)
    if len(top_ppcs) < 1:  # doesn't have top targets from list
      color = other_color  # use grey
      counter += 1
    else:
      # Weighted average of top targets (square->weighted average->square root)
      color_ids = [ppc2color[idx] for idx in top_ppcs]
      weights = [scores[idx] for idx in top_ppcs]
      # Need to round, otherwise floating point precision issues will result
      # in values slightly above 1
      avg = np.round(
          np.sqrt(
              np.average(
                  np.power(palette_rgb[color_ids] * rgb_max, 2),
                  axis=0,
                  weights=weights)) / rgb_max, 4)
      if (avg > 1).sum() > 0:
        print(avg)
      color = avg
    rgb_vals.append(list(color))
    hex_vals.append("#%02x%02x%02x" %
                    tuple(np.array(color * rgb_max, dtype=int)))
  rgb_vals = np.array(rgb_vals)

  # Create t-SNE model
  tsne_model = TSNE(
      perplexity=30,
      n_components=2,
      n_iter=1000,
      random_state=23,
      learning_rate=500,
      init="pca")
  new_values = tsne_model.fit_transform(ppc_scores)
  x = []
  y = []
  for value in new_values:
    x.append(value[0])
    y.append(value[1])
  # Put data in dataframe
  df = pd.DataFrame({
      "x": x,
      "y": y,
      "color": hex_vals,
      "label(s)": top_categories,
      "text": texts
  })

  df = df[df["label(s)"] != "other"]
  df["top_label"] = df["label(s)"].str.split(",").str[0]

  # Two selections:
  # - a brush that is active on the top panel
  # - a multi-click that is active on the bottom panel
  brush = alt.selection(type="interval")
  click = alt.selection_multi(encodings=["color"])

  sample = df.sample(5000)  # max 5000 examples can be plotted
  points = alt.Chart(sample).mark_point(
      filled=True, size=50).encode(
          x="x:Q",
          y="y:Q",
          color=alt.Color("color", scale=None),
          tooltip=["label(s)", "text"]).properties(
              width=700, height=600).add_selection(brush)

  # Bottom panel is a bar chart
  bars = alt.Chart(sample).mark_bar().encode(
      x="count()",
      y="top_label:N",
      color=alt.condition(click, alt.Color("color:N", scale=None),
                          alt.value("lightgray")),
  ).transform_filter(brush.ref()).properties(
      width=700, selection=click)

  chart = alt.vconcat(
      points, bars, data=sample, title="t-SNE Projection of Examples")

  chart.save(FLAGS.plot_dir + "/tsne.html", format="html")


if __name__ == "__main__":
  app.run(main)
