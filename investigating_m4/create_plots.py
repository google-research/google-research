# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Functions to create plots in paper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bokeh.layouts import row
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import LabelSet
from bokeh.palettes import Category10
from bokeh.palettes import Category20
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn import manifold


def get_subset_of_df(df, lp_subset, aname):
  """Get subset of the dataframe corresponding to these language pairs."""
  if lp_subset is not None:
    curr_df = df[aname].copy()
    curr_df = curr_df.loc[lp_subset][zip(["SVCCA Score"] * len(lp_subset),
                                         lp_subset)]
  else:
    curr_df = df[aname]
  return curr_df


def non_english_lang(lp):
  """Returns the non-English language from the supplied language pair.

  Args:
    lp: a string representing a language pair, e.g. "en-te"

  Returns:
    A string representing the non-English language in the language-pair, e.g.
    "te".

  Raises:
    ValueError if `lp` does not have two parts separated by a dash or if one of
    the languages is not 'en'.
  """
  if len(lp.split("-")) != 2:
    raise ValueError("ERROR: Purported language pair '{}' does not have exactly"
                     "two parts separated by a dash (like 'en-ml').".format(lp))

  src, tgt = lp.split("-")
  if src != "en":
    return src
  elif tgt != "en":
    return tgt

  raise ValueError("ERROR: Neither code in purported language pair '{}'"
                   "is 'en'.".format(lp))


def finetuning_block(data,
                     int_lps,
                     task_lps,
                     selected_lps,
                     order,
                     props,
                     values,
                     color,
                     figure_size=(50, 20)):
  """Plots finetuning results."""
  # Filter the languages for finetuning set and finetunee set.
  ft_col = "Finetuning Language Pair"
  lp_col = "Language Pair"
  df_a = data[data[ft_col].isin(int_lps) & data[lp_col].isin(task_lps)
              & data[lp_col].isin(selected_lps)
              & data[ft_col].isin(selected_lps)]
  df_a[lp_col] = pd.Categorical(df_a[lp_col], order)
  df_a[ft_col] = pd.Categorical(df_a[ft_col], order)

  df_a = df_a.sort_values([lp_col, ft_col])

  # Coloring based on sub-family.
  ft_list = list(df_a[ft_col].dropna().unique())
  task_list = list(df_a[lp_col].dropna().unique())

  def get_lp_props(langs):
    return props[props["Language Code"].isin(langs)]["Sub-Family"].unique()

  selected_langs = [non_english_lang(lp) for lp in selected_lps]
  col_labels = get_lp_props(selected_langs)
  network_pal = seaborn.hls_palette(len(set(col_labels)))

  network_lut = dict(zip(map(str, list(set(col_labels))), network_pal))

  def get_lp_colors(lps, color_dict):
    ret = []

    for lp in lps:
      ret.append(color_dict[get_lp_props([non_english_lang(lp)])[0]])
    return ret

  row_lut = get_lp_colors(ft_list, network_lut)
  col_lut = get_lp_colors(task_list, network_lut)

  # Plot
  seaborn.set(font_scale=2, style="white")
  g = seaborn.clustermap(
      df_a,
      pivot_kws={
          "index": "Finetuning Language Pair",
          "columns": "Language Pair",
          "values": values
      },
      row_cluster=False,
      col_cluster=False,
      row_colors=row_lut,
      col_colors=col_lut,
      cmap=seaborn.color_palette(color, 1000),
      linewidths=.5,
      figsize=figure_size)

  # Setting axes, labels and legends.
  for label in list(set(col_labels)):
    g.ax_col_dendrogram.bar(
        0, 0, color=network_lut[label], label=label, linewidth=0)
  g.ax_col_dendrogram.legend(
      loc="center",
      ncol=5,
      bbox_to_anchor=(0.54, 0.94),
      bbox_transform=plt.gcf().transFigure,
      fontsize=40)
  g.cax.set_position([.15, .2, .02, .45])
  g.cax.tick_params(labelsize=40)

  g.ax_heatmap.set_ylabel("Finetuning Language Pair", fontsize=45)
  g.ax_heatmap.set_xlabel("Language Pair", fontsize=45)
  plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=20)

  return g


def layerwise_boxplot(data, key, ax, activation_list):
  """Plot distribution of SVCCA Scores of each layer in activation_list."""
  b = seaborn.boxplot(
      x="Layer",
      y="SVCCA Score",
      data=data,
      order=activation_list,
      saturation=1.0,
      ax=ax)

  # Setting axes and labels.
  b.set_xlabel("Layers of Transformer Model %s" % key, fontsize=30)
  b.set_ylabel("SVCCA Score", fontsize=30)
  b.set_xticklabels(activation_list, rotation=45)
  b.tick_params(labelsize=25)
  b.axes.set(ylim=(0.0, 1.01))
  return b


def show_embedding(sources,
                   activation_list,
                   dotsize=8,
                   plot_width=800,
                   plot_height=600):
  """Create scatter plot from data source."""
  p = {}
  # For each layer in activation_list, plot a scatter plot based on sources.
  for aname in activation_list:
    print("\taname: %s" % aname)
    hover = HoverTool(tooltips=[("x",
                                 "@x"), ("y",
                                         "@y"), ("lang_pair",
                                                 "@lang_pair"), ("prop",
                                                                 "@prop")])

    p[aname] = figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=aname,
        tools=[hover, "tap", "box_zoom", "reset", "save"])
    p[aname].scatter(
        x="x", y="y", size=dotsize, source=sources[aname], color="color")
    labels = LabelSet(
        x="x",
        y="y",
        text="lang_pair",
        level="overlay",
        x_offset=4,
        y_offset=4,
        source=sources[aname],
        render_mode="canvas",
        text_font_size=str(dotsize + 2) + "pt")

    p[aname].add_layout(labels, "right")
    p[aname].xgrid.visible = False
    p[aname].ygrid.visible = False
    p[aname].xaxis.axis_label_text_font_size = "14pt"
    p[aname].yaxis.axis_label_text_font_size = "14pt"

  return row([p[a] for a in activation_list])


def get_visualizations(
    cross_dfs,
    activation_list,
    lp_subset=None,
    property_mapper=None,
    dotsize=8,
    plot_height=600,
    plot_width=800,
):
  """Plots a subset of language pairs, coloring by language properties.

  Args:
    cross_dfs: Pandas DataFrame containing SVCCA scores.
    activation_list: List of layer activations to visualize, e.g. ['enc/tok',
      'enc/out_5']
    lp_subset: Which of the language pairs in `cross_dfs` to visualize. List of
      language pairs, e.g. ["en-ku", "en-et", ... ]
    property_mapper: a mapper from language code to string, e.g.
      property_mapper("fr") returns "Romance".
    dotsize: the size of the dots representing the languages
    plot_height: Height of plot to return.
    plot_width: Width of plot to return.

  Returns:
    Row of figures corresponding to activation_list.
  """
  emb_dfs = {}
  emb_sources = {}

  def get_color_mappings(property_mapper, lang_pairs):
    """Get the colors for each language_pair given property_mapping."""
    all_props = [property_mapper(lp) for lp in lang_pairs]
    propset = sorted(list(set(all_props)))  # Set of properties.
    # Number of properties. Cap at 20 because Bokeh can't support more colors.
    n_props = min(len(list(set(all_props))), 20)
    # Use at least three colors. If there are more than 10 categories use the
    # finer-grained categories. Note that if there are more than 20 categories
    # multiple categories will have the same color.
    palette = Category10[max(3,
                             n_props)] if n_props <= 10 else Category20[n_props]
    prop_to_color = {p: palette[i % n_props] for i, p in enumerate(propset)}
    colors = [prop_to_color[p] for p in all_props]
    return all_props, colors

  # Create a plot for each activation in `activation_list`.
  for aname in activation_list:
    print("aname: %s" % aname)

    curr_df = get_subset_of_df(cross_dfs, lp_subset, aname)
    # SVCCA values.
    x = curr_df.values  # SVCCA values.
    x = np.maximum(x, x.T)
    # Language Pairs.
    y = list(curr_df.index)  # Language Pairs.

    mds = manifold.SpectralEmbedding(affinity="precomputed")
    emb = mds.fit_transform(x)

    props, colors = get_color_mappings(property_mapper, y)

    labels = [non_english_lang(lp) for lp in y]
    emb_dfs[aname] = pd.DataFrame({
        "x": emb[:, 0],
        "y": emb[:, 1],
        "lang_pair": labels,
        "prop": props,
        "color": colors
    })
    emb_sources[aname] = ColumnDataSource(emb_dfs[aname])

  return show_embedding(
      emb_sources,
      dotsize=dotsize,
      activation_list=activation_list,
      plot_height=plot_height,
      plot_width=plot_width)
