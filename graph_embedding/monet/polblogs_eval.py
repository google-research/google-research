# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Runs eval metrics for the political blogs experiment in Section 4."""

# pylint: disable=use-symbolic-message-instead
# pylint: disable=C6204

import collections
import copy
import functools
import json
import operator
import os
from eval_utils import load_numpy_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

################################################################################
# User-defined hyperparameters for the experiment. These should match the first
# user parameters in polblogs_experiment.py
DATA_DIR = 'polblogs'
SAVE_DIR = 'experiment_data/polblogs'
NUM_RUNS = 10
PLOT_LINE_WIDTH = 5.0
PLOT_MARKER_SIZE = 17.0
################################################################################

EVAL_SAVE_DIR = os.path.join(SAVE_DIR, 'exp_results')
if not os.path.isdir(EVAL_SAVE_DIR):
  os.mkdir(EVAL_SAVE_DIR)

# Load and save metadata leakages & importances
leakage_dict = collections.defaultdict(list)
monet_importances = [None] * NUM_RUNS
monet0_importances = [None] * NUM_RUNS
for exp_no in range(NUM_RUNS):
  exp_save_path = os.path.join(SAVE_DIR, str(exp_no))
  monet_importances[exp_no] = load_numpy_matrix(
      os.path.join(exp_save_path, 'monet_importances'))
  monet0_importances[exp_no] = load_numpy_matrix(
      os.path.join(exp_save_path, 'monet0_importances'))
  with open(os.path.join(exp_save_path, 'leakage_dict')) as f:
    for k, v in json.loads(f.read()).items():
      leakage_dict[k].append(v)

with open(os.path.join(EVAL_SAVE_DIR, 'metadata_importances'), 'w') as f:
  pr = functools.partial(print, file=f)
  pr('# monet0 importance mean/std')
  pr(numpy.array2string(numpy.mean(monet0_importances, axis=0)))
  pr(numpy.array2string(numpy.std(monet0_importances, axis=0)))
  pr('# monet importance mean/std')
  pr(numpy.array2string(numpy.mean(monet_importances, axis=0)))
  pr(numpy.array2string(numpy.std(monet_importances, axis=0)))

with open(os.path.join(EVAL_SAVE_DIR, 'leakages'), 'w') as f:
  for method, metrics in leakage_dict.items():
    print_str = '%s & %0.0f $\pm$ %0.0f\n'  # pylint: disable=anomalous-backslash-in-string
    if method == 'monet':
      print_str = '%s & %0.3f $\pm$ %0.3f\n'  # pylint: disable=anomalous-backslash-in-string
    f.write(print_str % (method, numpy.mean(metrics), numpy.std(metrics)))

# Prep eval curves display
lin_data = []
rbf_data = []
for rep in range(NUM_RUNS):
  rep_save_path = os.path.join(SAVE_DIR, str(rep))
  score_dict_path = os.path.join(rep_save_path, 'score_dict')
  with open(score_dict_path) as f:
    score_dict = json.loads(f.read())
  for method in score_dict:
    if (score_dict[method] is not None and method != 'monet_dw' and
        method != 'glove_fairwalks'):
      for i, v in enumerate(score_dict[method]['linear']):
        lin_data.append({
            'Method': method,
            'Avg Accuracy': v,
            'Size of training set (%)': (i + 1) * 10,
            'rep': rep})
      for i, v in enumerate(score_dict[method]['rbf']):
        rbf_data.append({
            'Method': method,
            'Avg Accuracy': v,
            'Size of training set (%)': (i + 1) * 10,
            'rep': rep})
lin_df = pd.DataFrame(lin_data)
rbf_df = pd.DataFrame(rbf_data)


# Method plot params
def replace_method_column(df, input_replace_dict):
  new_df = copy.deepcopy(df)
  for m in input_replace_dict:
    new_df = new_df.replace(m, input_replace_dict[m])
  return new_df


replace_dict = {
    'adv1': 'Adversarial Debiasing',
    'deepwalk': 'DeepWalk',
    'monet0': 'GloVe_meta',
    'monet': 'MONET_G',
    'random': 'Random',
    'glove': 'GloVe',
    'deepwalk_fairwalks': 'FairWalk'
}

lin_df = replace_method_column(lin_df, replace_dict)
rbf_df = replace_method_column(rbf_df, replace_dict)

acceptable_points = ['^', 's', 'v', 'o', 'p', 'D', 'X', '*']
acceptable_lines = [(1, 0), (1, 1), (2, 2), (2, 2, 1, 2), (2, 1, 1, 1), (2, 4),
                    (2, 2, 1, 2, 1, 2), (1, 3)]
acceptable_colors = sns.color_palette()
sorted_names = sorted(replace_dict.values())
colors_palette = {n: acceptable_colors[i] for i, n in enumerate(sorted_names)}
points_palette = {n: acceptable_points[i] for i, n in enumerate(sorted_names)}
lines_palette = {n: acceptable_lines[i] for i, n in enumerate(sorted_names)}

colors_palette['GloVe'] = (0.99609375, 0.83984375, 0.)
colors_palette['GloVe_meta'] = colors_palette['MONET_G']
lines_palette['Adversarial Debiasing'] = lines_palette['MONET_G']
lines_palette['MONET_G'] = ''
colors_palette['MONET_G'] = [i / float(255.0) for i in [225, 0, 0]]
colors_palette['Random'] = [0.0, 0.0, 0.0]

ORDER = [0, 7, 2, 3, 5, 6, 4, 1]
ORDER_RBF = [0, 6, 2, 3, 4, 5, 1]


def results_lineplot(df,
                     cpalette,
                     mpalette,
                     title,
                     input_lines_palette=None,
                     figsize=(10, 10),
                     x_var='Size of training set (%)',
                     y_var='Avg Accuracy',
                     x_title='Size of training set (%)',
                     y_title='Avg Accuracy\n Debias Target: 0.5',
                     linewidth=2.0,
                     markersize=20,
                     title_size=25.0,
                     ax_label_size=20.0,
                     ax_tick_size=18.0,
                     order=None,
                     legend_text_size=15.0,
                     xlim=(0, 100),
                     ylim=(0.4, 1.0),
                     filename=None):
  """Makes results lineplot.

  Args:
    df: dataframe
    cpalette: color palette
    mpalette: marker palette
    title: plot title
    input_lines_palette: lines palette
    figsize: figure size
    x_var: column lookup name for x variable in df
    y_var: column lookup name for y variable in df
    x_title: x axis display name
    y_title: y axis display name
    linewidth: line width
    markersize: marker size
    title_size: title size
    ax_label_size: axis label size
    ax_tick_size: axis tick size
    order: list giving order of hues in legend
    legend_text_size: legend text size
    xlim: x axis plot limits
    ylim: y axis plot limits
    filename: filename
  Returns: (none)
  """
  # General figure specs
  _ = plt.figure(figsize=figsize)
  plt.rc('axes', labelsize=ax_label_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=ax_tick_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=ax_tick_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=legend_text_size)  # legend fontsize
  plt.xlim(xlim)
  plt.ylim(ylim)
  plt.title(' ')
  dashes_order, dashes_values = zip(*input_lines_palette.items())
  ax = sns.lineplot(
      x=x_var,
      y=y_var,
      hue='Method',
      palette=cpalette,
      data=df,
      lw=linewidth,
      markers=mpalette,
      style='Method',
      style_order=dashes_order,
      dashes=dashes_values,
      markersize=markersize)
  plt.xlabel(x_title)
  plt.ylabel(y_title)
  if order is None:
    leg = plt.legend(loc=0, bbox_to_anchor=(1, 1), markerscale=3)
    handles = leg.legendHandles
  else:
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    assert len(handles) == len(order)
    handles = [handles[order[i]] for i in range(len(order))]
    labels = [labels[order[i]] for i in range(len(order))]
    ax.legend(handles=handles, labels=labels,
              loc=0, bbox_to_anchor=(1, 1), markerscale=3)
  for legobj in handles:
    legobj.set_linewidth(linewidth)
    legobj.set_markersize(markersize)
  plt.suptitle(title, fontsize=title_size)
  plt.title(' ')
  if filename is not None:
    plt.savefig(filename, bbox_inches='tight')

sns.set_style('whitegrid')
results_lineplot(
    lin_df,
    colors_palette,
    points_palette,
    'Blog Affiliation - Linear SVM',
    lines_palette,
    y_title='Avg Accuracy',
    ylim=(0.4, 1.0),
    figsize=(13, 10),
    title_size=40,
    order=ORDER,
    ax_tick_size=26,
    ax_label_size=28,
    legend_text_size=26,
    linewidth=PLOT_LINE_WIDTH,
    markersize=PLOT_MARKER_SIZE,
    filename=os.path.join(EVAL_SAVE_DIR, 'polblogs_lin_svm.png'))

sns.set_style('whitegrid')
results_lineplot(
    rbf_df[~rbf_df.Method.isin(['Random'])],
    {n: p for n, p in colors_palette.items() if n != 'Random'},
    {n: p for n, p in points_palette.items() if n != 'Random'},
    'Blog Affiliation - RBF SVM',
    {n: p for n, p in lines_palette.items() if n != 'Random'},
    y_title='Avg Accuracy',
    ylim=(0.8, 1.0),
    figsize=(13, 10),
    title_size=40,
    order=ORDER_RBF,
    ax_tick_size=26,
    ax_label_size=28,
    legend_text_size=26,
    linewidth=PLOT_LINE_WIDTH,
    markersize=PLOT_MARKER_SIZE,
    filename=os.path.join(EVAL_SAVE_DIR, 'polblogs_rbf_svm.png'))


def plot_2d_embeddings(input_embeddings,
                       label_matrix,
                       title='Title Here',
                       top=10,
                       reverse=True,
                       plot_size=12,
                       pntsize=6,
                       do_legend=False,
                       show_axes=True,
                       wrap_points=False,
                       titlesize=4,
                       subtitle='',
                       ticksize=16,
                       filename=None):
  """Plot 2d embeddings.

  Args:
    input_embeddings: input embeddings
    label_matrix: label matrix
    title: plot title
    top: number of categories to take
    reverse: reverse category sort
    plot_size: plot size
    pntsize: scatterplot point size
    do_legend: plot legend
    show_axes: show axes
    wrap_points: wrap points
    titlesize: title size
    subtitle: subtitle
    ticksize: tick size
    filename: filename
  Returns: (none)
  """
  # Filter samples with no labels
  retained_samples = numpy.argwhere(numpy.sum(label_matrix, axis=1))[:, 0]
  x1 = input_embeddings[retained_samples, 0]
  x2 = input_embeddings[retained_samples, 1]
  label_matrix = label_matrix[retained_samples, :]
  labels = [p[1] for p in list(numpy.argwhere(label_matrix))]

  # Filter the label set if necessary
  if len(set(labels)) > top:
    item_counts = dict([(label, labels.count(label)) for label in set(labels)])
    sorted_counts = sorted(
        item_counts.items(), key=operator.itemgetter(1), reverse=reverse)
    good_labels = set()
    for entry in sorted_counts[:top]:
      good_labels.add(entry[0])

    x1 = numpy.array(
        [x1[ii] for ii in range(len(labels)) if labels[ii] in good_labels])
    x2 = numpy.array(
        [x2[ii] for ii in range(len(labels)) if labels[ii] in good_labels])
    good_example_labels = [label for label in labels if label in good_labels]
    labels = good_example_labels

  # Split the data into groups
  label_set = set(labels)
  data_groups = [None] * len(label_set)
  for ii, label in enumerate(label_set):
    indx = [j for j in range(len(labels)) if labels[j] == label]
    data_groups[ii] = (x1[indx], x2[indx])

  # Make the plot
  fig = plt.figure(figsize=(plot_size, plot_size))
  if wrap_points:
    plt.xlim(numpy.min(x1), numpy.max(x1))
    plt.ylim(numpy.min(x2), numpy.max(x2))
  ax = fig.add_subplot(1, 1, 1)
  for ii, data_group in enumerate(data_groups):
    x, y = data_group
    ax.scatter(x, y, s=pntsize, c=COLORS[ii], edgecolors='none', label=ii)
  if not subtitle:
    plt.title(title, fontsize=titlesize)
  else:
    plt.suptitle(title, fontsize=titlesize)
    plt.title(subtitle)

  if do_legend:
    plt.legend(loc=1)

  # Modify axes
  frame1 = plt.gca()
  frame1.axes.get_xaxis().set_visible(show_axes)
  frame1.axes.get_yaxis().set_visible(show_axes)
  plt.rc('xtick', labelsize=ticksize)
  plt.rc('ytick', labelsize=ticksize)

  # Save or plot
  if filename is not None:
    plt.savefig(filename, bbox_inches='tight')
  else:
    plt.show()


# Load embeddings from last run
methods = ['adv1', 'deepwalk', 'glove', 'monet', 'monet0', 'random']
embeddings = {}
default_m_types = ['E', 'W', 'Z', 'H1', 'H2']


def load_weights_object(savedir):
  weights = {}
  for m_type in default_m_types:
    weights[m_type] = load_numpy_matrix(os.path.join(savedir, m_type))
  return weights


for method in methods:
  method_save_path = os.path.join(rep_save_path, method)
  embeddings[method] = load_weights_object(method_save_path)


def get_keyed_vector(input_line):
  numbers = input_line.strip().split()
  return {str(int(numbers[0])): [float(x) for x in numbers[1:]]}


def load_embeddings(fileobj):
  _ = fileobj.readline()
  model = {}
  for l in fileobj:
    model.update(get_keyed_vector(l))
  return model

# Load the blog attributes
with open(os.path.join(DATA_DIR, 'party_cvrt.txt')) as f:
  party_cvrt_data = load_embeddings(f)

# Load the memberships and get tokens
memships = {}
with open(os.path.join(DATA_DIR, 'memberships.txt')) as f:
  for line in f:
    line_split = line.strip().split()
    memships.update({line_split[0]: int(line_split[1])})
tokens = sorted(memships.keys())

# Construct party labels
party_labels = numpy.zeros(shape=(len(memships), 2))
for i, node in enumerate(tokens):
  party_labels[i, memships[node]] = 1.0

# Plot TSNE for standard glove
glove_tsne = PCA(n_components=2).fit_transform(embeddings['glove']['W'])
sns.set_style('whitegrid')
COLORS = ['red', 'blue', 'orange', 'green']
plot_2d_embeddings(
    glove_tsne,
    party_labels,
    title='Affiliation - GloVe PCA',
    subtitle=' ',
    pntsize=14,
    plot_size=10,
    filename=os.path.join(EVAL_SAVE_DIR, 'PartyOnGloVePCA.png'),
    show_axes=True,
    wrap_points=False,
    ticksize=28,
    titlesize=40)

# Plot TSNE for naive MONET
monet0_tsne = PCA(n_components=2).fit_transform(embeddings['monet0']['W'])
sns.set_style('whitegrid')
plot_2d_embeddings(
    monet0_tsne,
    party_labels,
    title='Affiliation - GloVe_meta PCA',
    subtitle=' ',
    pntsize=14,
    plot_size=10,
    filename=os.path.join(EVAL_SAVE_DIR, 'PartyOnMonet0PCA.png'),
    show_axes=True,
    wrap_points=False,
    ticksize=28,
    titlesize=40)

# Plot TSNE for MONET
monet_tsne = PCA(n_components=2).fit_transform(embeddings['monet']['W'])
sns.set_style('whitegrid')
plot_2d_embeddings(
    monet_tsne,
    party_labels,
    title='Affiliation - MONET_G PCA',
    subtitle=' ',
    pntsize=14,
    plot_size=10,
    filename=os.path.join(EVAL_SAVE_DIR, 'PartyOnMonetPCA.png'),
    show_axes=True,
    wrap_points=False,
    ticksize=28,
    titlesize=40)
