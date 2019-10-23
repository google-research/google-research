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
"""Script for analyzing dataset annotations.

The analysis includes calculating high-level statistics as well as correlation
among labels.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import seaborn as sns

FLAGS = flags.FLAGS

flags.DEFINE_string("data", None, "Input data.")

flags.DEFINE_string("plot_dir", "plots",
                    "Directory for saving plots and analyses.")

flags.DEFINE_string("target_file", "data/targets.txt",
                    "File containing list of targets.")
flags.DEFINE_string("sentiment_dict", None, "Sentiment dictionary.")


def CheckAgreement(ex, min_agreement, all_targets, max_agreement=100):
  """Return the labels that at least min_agreement raters agree on."""
  sum_ratings = ex[all_targets].sum(axis=0)
  agreement = ((sum_ratings >= min_agreement) & (sum_ratings <= max_agreement))
  return ",".join(sum_ratings.index[agreement].tolist())


def CountLabels(labels):
  if (not isinstance(labels, float)) and labels:
    return len(labels.split(","))
  return 0


def main(_):
  print("Loading data...")
  data = pd.read_csv(FLAGS.data, encoding="utf-8")
  print("%d Examples" % (len(set(data["id"]))))
  print("%d Annotations" % len(data))

  with open(FLAGS.target_file, "r") as f:
    all_targets = f.read().splitlines()
  all_targets_neutral = all_targets + ["neutral"]
  print("%d Target Categories" % len(all_targets))

  print("%d unique raters" % len(data["rater_id"].unique()))
  print("%.3f marked unclear" %
        (data["example_very_unclear"].sum() / len(data)))

  # Since the ones marked as difficult have no labels, exclude those
  data = data[data[all_targets_neutral].sum(axis=1) != 0]

  print("Distribution of number of labels per example:")
  print(data[all_targets_neutral].sum(axis=1).value_counts() / len(data))
  print("%.2f with more than 3 labels" %
        ((data[all_targets_neutral].sum(axis=1) > 3).sum() /
         len(data)))  # more than 3 labels

  print("Label distributions:")
  print((data[all_targets_neutral].sum(axis=0).sort_values(ascending=False) /
         len(data) * 100).round(2))

  print("Plotting label correlations...")
  ratings = data.groupby("id")[all_targets].mean()

  # Compute the correlation matrix
  corr = ratings.corr()

  # Generate a mask for the upper triangle
  mask = np.zeros_like(corr, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True

  # Set up the matplotlib figure
  fig, _ = plt.subplots(figsize=(11, 9))

  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(220, 10, as_cmap=True)

  # Draw the heatmap with the mask and correct aspect ratio
  sns.heatmap(
      corr,
      mask=mask,
      cmap=cmap,
      vmax=.3,
      center=0,
      square=True,
      linewidths=.5,
      cbar_kws={"shrink": .5})
  fig.savefig(
      FLAGS.plot_dir + "/correlations.pdf",
      dpi=500,
      format="pdf",
      bbox_inches="tight")

  print("Plotting hierarchical relations...")
  z = linkage(
      pdist(ratings.T, metric="correlation"),
      method="ward",
      optimal_ordering=True)
  fig = plt.figure(figsize=(11, 4), dpi=400)
  plt.xlabel("")
  plt.ylabel("")
  dendrogram(
      z,
      labels=ratings.columns,
      leaf_rotation=90.,  # rotates the x axis labels
      leaf_font_size=12,  # font size for the x axis labels
      color_threshold=1.05,
  )
  fig.savefig(
      FLAGS.plot_dir + "/hierarchical_clustering.pdf",
      dpi=600,
      format="pdf",
      bbox_inches="tight")

  sent_color_map = {
      "positive": "#BEECAF",
      "negative": "#EE5E5E",
      "ambiguous": "#FFFC9E"
  }
  with open(FLAGS.sentiment_dict) as f:
    sent_dict = json.loads(f.read())
  sent_colors = {}
  for e in all_targets:
    if e in sent_dict["positive"]:
      sent_colors[e] = sent_color_map["positive"]
    elif e in sent_dict["negative"]:
      sent_colors[e] = sent_color_map["negative"]
    else:
      sent_colors[e] = sent_color_map["ambiguous"]

  # Generate a mask for the upper triangle
  mask = np.zeros_like(corr, dtype=np.bool)
  mask[np.diag_indices(mask.shape[0])] = True

  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(220, 10, as_cmap=True)

  row_colors = pd.Series(
      corr.columns, index=corr.columns, name="sentiment").map(sent_colors)

  # Draw the heatmap with the mask and correct aspect ratio
  g = sns.clustermap(
      corr,
      mask=mask,
      cmap=cmap,
      vmax=.3,
      vmin=-0.3,
      center=0,
      row_linkage=z,
      col_linkage=z,
      col_colors=row_colors,
      linewidths=.1,
      cbar_kws={
          "ticks": [-.3, -.15, 0, .15, .3],
          "use_gridspec": False,
          "orientation": "horizontal",
      },
      figsize=(10, 10))

  g.ax_row_dendrogram.set_visible(False)
  g.cax.set_position([.34, -0.05, .5, .03])

  for label in sent_color_map:
    g.ax_col_dendrogram.bar(
        0, 0, color=sent_color_map[label], label=label, linewidth=0)

  g.ax_col_dendrogram.legend(
      title="Sentiment", loc="center", bbox_to_anchor=(1.1, .5))

  g.savefig(FLAGS.plot_dir + "/hierarchical_corr.pdf", dpi=600, format="pdf")

  print("Calculating agreements...")
  unique_labels = data.groupby("id").apply(CheckAgreement, 1,
                                           all_targets_neutral).to_dict()
  data["unique_labels"] = data["id"].map(unique_labels)
  agree_dict_2 = data.groupby("id").apply(CheckAgreement, 2,
                                          all_targets_neutral).to_dict()
  data["agree_2"] = data["id"].map(agree_dict_2)
  agree_dict = data.groupby("id").apply(CheckAgreement, 3,
                                        all_targets_neutral).to_dict()
  data["agree_3"] = data["id"].map(agree_dict)
  agree_dict = data.groupby("id").apply(CheckAgreement, 1, all_targets_neutral,
                                        1).to_dict()
  data["no_agree"] = data["id"].map(agree_dict)

  filtered_2 = data[data["agree_2"].str.len() > 0]
  print(
      "%d (%d%%) of the examples have 2+ raters agreeing on at least one target label"
      % (len(filtered_2["id"].unique()), (len(filtered_2) / len(data) * 100)))

  filtered_3 = data[data["agree_3"].str.len() > 0]
  print(
      "%d (%d%%) of the examples have 3+ raters agreeing on at least one target label"
      % (len(filtered_3["id"].unique()), (len(filtered_3) / len(data) * 100)))

  print("Plotting number of labels...")
  data["num_unique_prefilter"] = data["unique_labels"].apply(CountLabels)
  data["num_unique_postfilter"] = data["agree_2"].apply(CountLabels)
  unique_ex = data.drop_duplicates("id")
  df = pd.DataFrame({
      "count":
          unique_ex["num_unique_prefilter"].tolist() +
          unique_ex["num_unique_postfilter"].tolist(),
      "type": ["pre-filter"] * len(unique_ex) + ["post-filter"] * len(unique_ex)
  })

  fig = plt.figure(dpi=600)
  ax = sns.countplot(
      data=df, x="count", hue="type", palette=["skyblue", "navy"])
  plt.xlim(-.5, 7.5)
  plt.legend(loc="center right", fontsize="x-large")
  plt.ylabel("Number of Examples")
  plt.xlabel("Number of Labels")
  plt.draw()
  labels = [item.get_text() for item in ax.get_yticklabels()]
  ax.set_yticklabels(["%dk" % (int(int(label) / 1000)) for label in labels])
  plt.tight_layout()

  fig.savefig(
      FLAGS.plot_dir + "/number_of_labels.pdf",
      dpi=600,
      format="pdf",
      bbox_inches="tight")

  print("Proportion of agreement per label:")
  print(
      filtered_2[all_targets_neutral].sum(axis=0).sort_values(ascending=False)
      / len(data))


if __name__ == "__main__":
  app.run(main)
