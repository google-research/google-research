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

"""Computes model performance metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import itertools
import os
from typing import Any, Dict, List, Optional, Text

from absl import logging
import blast_utils
import blundell_constants
import db
import hmmer_utils
import inference_lib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parallel
import pfam_utils
import protein_task
import seaborn as sns
import sklearn.decomposition
from statsmodels.stats import contingency_tables
from statsmodels.stats import proportion
import tensorflow.compat.v1 as tf
import util as classification_util

INTERVAL_DATAFRAME_KEY = 'interval'

Subsequence = collections.namedtuple('Subsequence', ['name', 'range'])

# A part of an amino acid sequence that is of particular interest.
ATPASE_START_INDEX_OF_DOMAIN = 160

# https://pfam.xfam.org/protein/AT1A1_PIG, residues 161-352 (1-indexed)
ATPASE_PIG_SEQUENCE = 'NMVPQQALVIRNGEKMSINAEEVVVGDLVEVKGGDRIPADLRIISANGCKVDNSSLTGESEPQTRSPDFTNENPLETRNIAFFSTNCVEGTARGIVVYTGDRTVMGRIATLASGLEGGQTPIAAEIEHFIHIITGVAVFLGVSFFILSLILEYTWLEAVIFLIGIIVANVPEGLLATVTVCLTLTAKRMARK'  # pylint: disable=line-too-long

# https://pfam.xfam.org/protein/AT1A1_PIG
ATPASE_ANNOTATED_SUBSEQUENCES = (
    Subsequence('disordered', slice(213, 231)),
    Subsequence('helical', slice(288, 313)),
    Subsequence('helical', slice(318, 342)),
)

# Since our v2r_human is only the domain (not the whole protein),
# we have have to shift by 54 - 1 (because of zero-index); 54 is the start site
# according to http://pfam.xfam.org/protein/P30518.
V2R_START_INDEX_OF_DOMAIN = 53

# http://pfam.xfam.org/protein/P30518
V2R_HUMAN_SEQUENCE = 'SNGLVLAALARRGRRGHWAPIHVFIGHLCLADLAVALFQVLPQLAWKATDRFRGPDALCRAVKYLQMVGMYASSYMILAMTLDRHRAICRPMLAYRHGSGAHWNRPVLVAWAFSLLLSLPQLFIFAQRNVEGGSGVTDCWACFAEPWGRRTYVTWIALMVFVAPTLGIAACQVLIFREIHASLVPGPSERPGGRRRGRRTGSPGEGAHVSAAVAKTVRMTLVIVVVYVLCWAPFFLVQLWAAWDPEAPLEGAPFVLLMLLASLNSCTNPWIY'  # pylint: disable=line-too-long

# http://pfam.xfam.org/protein/P30518
V2R_ANNOTATED_SUBSEQUENCES = (
    Subsequence('helical', slice(41, 63)),
    Subsequence('helical', slice(74, 95)),
    Subsequence('helical', slice(114, 136)),
    Subsequence('helical', slice(156, 180)),
    Subsequence('helical', slice(205, 230)),
    Subsequence('helical', slice(274, 297)),
    Subsequence('helical', slice(308, 329)),
)

_PRECISION_RECALL_PERCENTILE_THRESHOLDS = np.arange(0, 1., .05)

RECALL_PRECISION_RECALL_KEY = 'recall'
PRECISION_PRECISION_RECALL_KEY = 'precision'
THRESHOLD_PRECISION_RECALL_KEY = 'threshold'
_PRECISION_RECALL_COLUMNS = [
    THRESHOLD_PRECISION_RECALL_KEY, PRECISION_PRECISION_RECALL_KEY,
    RECALL_PRECISION_RECALL_KEY
]

GATHERING_THRESHOLDS_PATH = 'testdata/gathering_thresholds_v32.0.csv'

TMP_TABLE_NAME = 'seed_train'

ACCURACY_KEY = 'accuracy'
FAMILY_ACCESSION_KEY = 'family_accession'
NUM_EXAMPLES_KEY = 'num_examples'
AVERAGE_SEQUENCE_LENGTH_KEY = 'average_length'
OUT_OF_VOCABULARY_FAMILY_ACCESSION = 'PF00000.0'

# Container for basic accuracy computations: unweighted accuracy, mean per class
# accuracy, and mean per clan accuracy.
BasicAccuracyComputations = collections.namedtuple(
    'BasicAccuracyComputations', [
        'unweighted_accuracy', 'mean_per_class_accuracy',
        'mean_per_clan_accuracy'
    ])


def get_latest_prediction_file_from_checkpoint_dir(prediction_dir):
  """Return path to prediction file that is for the most recent global_step.

  Args:
    prediction_dir: Path to directory containing csv-formatted predictions.

  Returns:
    string. Path to csv file containing latest predictions.
  """
  files = tf.io.gfile.Glob(os.path.join(prediction_dir, '*.csv'))

  def get_global_step_from_filename(filename):
    return int(os.path.basename(filename).replace('.csv', ''))

  return max(files, key=get_global_step_from_filename)


def load_prediction_file(filename, idx_to_family_accession):
  """Load csv file containing predictions into pandas dataframe.

  Args:
    filename: string. Path to csv file outputted when training a
      pfam model. The csv file should contain 3 columns,
      `classification_util.PREDICTION_FILE_COLUMN_NAMES`.
    idx_to_family_accession: dict from int to string. The keys
      are indices of the families in the dataset descriptor used in training
      (which correspond to the logits of the classes). The values are the
      accession ids corresponding to that index.

  Returns:
    pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES.
  """
  # Read in csv.
  with tf.io.tf.io.gfile.GFile(filename, 'r') as f:
    dataframe = pd.read_csv(
        f, names=classification_util.PREDICTION_FILE_COLUMN_NAMES)

  # Convert true and predicted labels from class indexes to accession ids.
  dataframe[classification_util.TRUE_LABEL_KEY] = dataframe[
      classification_util.TRUE_LABEL_KEY].apply(
          lambda true_class_idx: idx_to_family_accession[true_class_idx])
  dataframe[classification_util.PREDICTED_LABEL_KEY] = dataframe[
      classification_util.PREDICTED_LABEL_KEY].apply(
          # pylint: disable=g-long-lambda
          lambda predicted_class_idx: (idx_to_family_accession.get(
              predicted_class_idx, OUT_OF_VOCABULARY_FAMILY_ACCESSION)))

  return dataframe


def mean_per_class_accuracy(predictions_dataframe):
  """Compute accuracy of predictions, giving equal weight to all classes.

  Args:
    predictions_dataframe: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES.

  Returns:
    float. The average of all class-level accuracies.
  """
  grouped_predictions = collections.defaultdict(list)
  for row in predictions_dataframe.itertuples():
    grouped_predictions[row.true_label].append(row.predicted_label)

  accuracy_per_class = {
      true_label: np.mean(predicted_label == np.array(true_label))
      for true_label, predicted_label in grouped_predictions.items()
  }

  return np.mean(list(accuracy_per_class.values()))


def raw_unweighted_accuracy(
    predictions_dataframe,
    true_label=classification_util.TRUE_LABEL_KEY,
    predicted_label=classification_util.PREDICTED_LABEL_KEY):
  """Compute accuracy, regardless of which class each prediction corresponds to.

  Args:
    predictions_dataframe: pandas DataFrame with at least 2 columns, true_label
      and predicted_label.
    true_label: str. Column name of true labels.
    predicted_label: str. Column name of predicted labels.

  Returns:
    float. Accuracy.
  """
  num_correct = (predictions_dataframe[true_label] ==
                 predictions_dataframe[predicted_label]).sum()
  total = len(predictions_dataframe)
  return num_correct / total


def number_correct(predictions_dataframe,
                   true_label=classification_util.TRUE_LABEL_KEY,
                   predicted_label=classification_util.PREDICTED_LABEL_KEY):
  """Computes the number of correct predictions.

  Args:
    predictions_dataframe: pandas DataFrame with at least 2 columns, true_label
      and predicted_label.
    true_label: str. Column name of true labels.
    predicted_label: str. Column name of predicted labels.

  Returns:
    int.
  """
  return (predictions_dataframe[true_label] ==
          predictions_dataframe[predicted_label]).sum()


def family_predictions_to_clan_predictions(predictions_dataframe,
                                           family_to_clan_dict):
  """Convert family predictions to clan predictions.

  If a true label has no clan, it is omitted from the returned dataframe.
  If a predicted label has no clan, it is *included* in the outputted dataframe
  with clan prediction `None`.

  Args:
    predictions_dataframe: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES. The true and
      predicted label are allowed to have version numbers on the accession ids
      (like PF12345.x).
    family_to_clan_dict: dictionary from string to string, like
      {'PF12345': 'CL1234'}, (where PF stands for protein family, and CL stands
      for clan. No version information is on the accession numbers (like
      PF12345.x).

  Returns:
    pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES, where
      the true and predicted labels are converted from family labels to clan
      labels.
  """
  # Avoid mutating original dataframe by making a copy.
  clan_prediction_dataframe = predictions_dataframe.copy(deep=True)

  # Since the family_to_clan_dict has no versioning on accession numbers,
  # we strip those versions from the predictions.
  clan_prediction_dataframe[classification_util
                            .TRUE_LABEL_KEY] = clan_prediction_dataframe[
                                classification_util.TRUE_LABEL_KEY].apply(
                                    pfam_utils.parse_pfam_accession)
  clan_prediction_dataframe[classification_util
                            .PREDICTED_LABEL_KEY] = clan_prediction_dataframe[
                                classification_util.PREDICTED_LABEL_KEY].apply(
                                    pfam_utils.parse_pfam_accession)

  # Filter to only predictions for which the true labels are in clans.
  clan_prediction_dataframe = clan_prediction_dataframe[
      clan_prediction_dataframe.true_label.isin(family_to_clan_dict.keys())]

  # Convert family predictions to clan predictions for true labels.
  clan_prediction_dataframe[classification_util
                            .TRUE_LABEL_KEY] = clan_prediction_dataframe[
                                classification_util.TRUE_LABEL_KEY].apply(
                                    lambda label: family_to_clan_dict[label])

  # Convert family predictions to clan predictions for predicted labels.
  # Use `None` when there is no clan for our predicted label.
  clan_prediction_dataframe[classification_util
                            .PREDICTED_LABEL_KEY] = clan_prediction_dataframe[
                                classification_util.PREDICTED_LABEL_KEY].apply(
                                    family_to_clan_dict.get)

  return clan_prediction_dataframe


def families_with_more_than_n_examples(size_of_training_set_by_family, n):
  """Return list of family accession ids with more than n training examples.

  Args:
    size_of_training_set_by_family: pandas DataFrame with two columns,
      NUM_EXAMPLES_KEY and FAMILY_ACCESSION_KEY
    n: int.

  Returns:
    list of string: accession ids for large families.
  """
  filtered_dataframe = size_of_training_set_by_family[
      size_of_training_set_by_family.num_examples > n]
  return filtered_dataframe[FAMILY_ACCESSION_KEY].values


def mean_class_per_accuracy_for_only_large_classes(
    all_predictions_dataframe, class_minimum_size,
    size_of_training_set_by_family):
  """Compute mean per class accuracy on classes with lots of training data.

  Args:
    all_predictions_dataframe: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES. The true and predicted
      label are allowed to have version numbers on the accession ids
      (like PF12345.x).
    class_minimum_size: int.
    size_of_training_set_by_family: pandas DataFrame with two columns,
      NUM_EXAMPLES_KEY and FAMILY_ACCESSION_KEY

  Returns:
    float.
  """
  qualifying_families = families_with_more_than_n_examples(
      size_of_training_set_by_family, class_minimum_size)
  qualifying_predictions = all_predictions_dataframe[
      all_predictions_dataframe.true_label.isin(qualifying_families)]

  return mean_per_class_accuracy(qualifying_predictions)


def accuracy_by_family(family_predictions):
  """Return DataFrame that has accuracy by classification_util.TRUE_LABEL_KEY.

  Args:
    family_predictions: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES. The true and predicted
      label are allowed to have version numbers on the accession ids (like
      PF12345.x).

  Returns:
    pandas DataFrame with two columns, classification_util.TRUE_LABEL_KEY and
    ACCURACY_KEY.
  """
  return family_predictions.groupby([
      classification_util.TRUE_LABEL_KEY
  ]).apply(raw_unweighted_accuracy).reset_index(name=ACCURACY_KEY)


def pca_embedding_for_sequences(list_of_seqs, inferrer, num_pca_dims=2):
  """Take top num_pca_dims of an embedding of each sequence in list_of_seqs.

  Args:
    list_of_seqs: list of string. Amino acid characters only.
    inferrer: inference_lib.Inferrer instance.
    num_pca_dims: the number of prinicple components to retain.

  Returns:
    np.array of shape (len(list_of_seqs), num_pca_dims).
  """
  activations_batch = inferrer.get_activations(list_of_seqs=list_of_seqs)

  pca = sklearn.decomposition.PCA(n_components=num_pca_dims, whiten=True)

  return pca.fit_transform(np.stack(activations_batch, axis=0))


def accuracy_by_size_of_family(family_predictions, size_of_family):
  """Return DataFrame with the accuracy computed, segmented by family.

  Args:
    family_predictions: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES. The true and predicted
      label are allowed to have version numbers on the accession ids (like
      PF12345.x).
    size_of_family: pandas DataFrame with two columns, NUM_EXAMPLES_KEY and
      FAMILY_ACCESSION_KEY

  Returns:
    pandas DataFrame with two columns, NUM_EXAMPLES_KEY and ACCURACY_KEY.
  """
  return pd.merge(
      accuracy_by_family(family_predictions),
      size_of_family,
      left_on=classification_util.TRUE_LABEL_KEY,
      right_on=FAMILY_ACCESSION_KEY)[[NUM_EXAMPLES_KEY, ACCURACY_KEY]]


def accuracy_by_sequence_length(family_predictions,
                                length_of_examples_by_family):
  """Return DataFrame with accuracy computed by avg sequence length per family.

  Args:
    family_predictions: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES. The true and predicted
      label are allowed to have version numbers on the accession ids (like
      PF12345.x).
    length_of_examples_by_family: pandas DataFrame with two columns,
      AVERAGE_SEQUENCE_LENGTH_KEY and FAMILY_ACCESSION_KEY

  Returns:
    pandas DataFrame with two columns, AVERAGE_SEQUENCE_LENGTH_KEY and
      ACCURACY_KEY.
  """
  return pd.merge(
      accuracy_by_family(family_predictions),
      length_of_examples_by_family,
      left_on=classification_util.TRUE_LABEL_KEY,
      right_on=FAMILY_ACCESSION_KEY)[[
          AVERAGE_SEQUENCE_LENGTH_KEY, ACCURACY_KEY
      ]]


def num_examples_per_class(connection, table_name):
  """Compute number of examples per class.

  Args:
    connection: a connection that has temporary table `table_name`
      in it's session.
    table_name: name of table to query. The table should have a family_accession
      column.

  Returns:
    pandas DataFrame with two columns: FAMILY_ACCESSION_KEY and num_examples.
      num_examples is the number of examples with that family_accession.
  """
  connection.ExecuteQuery(r"""SELECT
                                         family_accession,
                                         COUNT(*) AS num_examples
                                     FROM
                                         """ + table_name + r"""
                                     GROUP BY
                                         family_accession""")
  result = db.ResultToFrame(connection)
  return result


def average_sequence_length_per_class(connection, table_name):
  """Compute average length of sequence per class.

  Args:
    connection: a connection that has temporary table `table_name`
      in it's session.
    table_name: name of table to query. The table should have a family_accession
      column.

  Returns:
    pandas DataFrame with two columns: FAMILY_ACCESSION_KEY and
      AVERAGE_LENGTH_KEY. average_length is the average length of sequences
      that have that family_accession.
  """
  connection.ExecuteQuery(r"""SELECT
                                         family_accession,
                                         AVG(LENGTH(sequence)) as average_length
                                     FROM
                                         """ + table_name + r"""
                                     GROUP BY
                                         family_accession""")
  result = db.ResultToFrame(connection)
  return result


def _pad_front_of_all_mutations(all_mutation_measurements, pad_amount,
                                pad_value):
  """Pad all_mutation_measurements with pad_value in the front.

  Adds a "mutation measurement" of pad_value for each amino acid,
  pad_amount times.

  For example, if the input shape of all_mutation_measurements is (20, 100),
  and pad_amount is 17, the output shape is (20, 117)

  Args:
    all_mutation_measurements: np.array of float, with shape
      (len(pfam_utils.AMINO_ACID_VOCABULARY), len(amino_acid_sequence)). The
      output of `measure_all_mutations`.
    pad_amount: the amount to pad the front of the amino acid measurements.
    pad_value: float.

  Returns:
    np.array of shape (len(pfam_utils.AMINO_ACID_VOCABULARY),
      all_mutation_measurements.shape[1] + pad_amount).
  """
  padded_acids = []
  for acid_index in range(all_mutation_measurements.shape[0]):
    front_pad = np.full(pad_amount, pad_value)
    padded_acid = np.append(front_pad, all_mutation_measurements[acid_index])
    padded_acids.append(padded_acid)
  padded_acids = np.array(padded_acids)

  return padded_acids


def _round_to_base(x, base=5):
  """Round to nearest multiple of `base`."""
  return int(base * round(float(x) / base))


def plot_all_mutations(all_mutation_measurements_excluding_pad, subsequences,
                       start_index_of_mutation_predictions):
  """Plot all mutations, annotated with domains, along with average values.

  Args:
    all_mutation_measurements_excluding_pad: np.array of float, with shape
      (len(pfam_utils.AMINO_ACID_VOCABULARY), len(amino_acid_sequence)).
      The output of `measure_all_mutations`.
    subsequences: List of `Subsequence`, which annotates the particular areas
      of interest of a protein/domain.
    start_index_of_mutation_predictions: Start index of the amino acid sequence
      that generated parameter `all_mutation_measurements_excluding_pad`.
      Since we often only predict mutations for a domain of a protein, which
      doesn't necessarily start at amino acid index 0, we have to offset the
      plot to appropriately line up indices.
  """
  sns.set_style('whitegrid')

  min_x_index = min(
      min([subsequence.range.start for subsequence in subsequences]),
      start_index_of_mutation_predictions)

  # https://www.compoundchem.com/2014/09/16/aminoacids/ explains this ordering.
  amino_acid_semantic_grouping_reordering = list('CMGAVILPFWYSTNQDERKH')
  amino_acid_reordering_indexes = [
      pfam_utils.AMINO_ACID_VOCABULARY.index(aa)
      for aa in amino_acid_semantic_grouping_reordering
  ]
  all_mutation_measurements_excluding_pad = (
      all_mutation_measurements_excluding_pad[amino_acid_reordering_indexes])

  all_mutation_measurements_including_pad = _pad_front_of_all_mutations(
      all_mutation_measurements=all_mutation_measurements_excluding_pad,
      pad_amount=start_index_of_mutation_predictions,
      pad_value=np.min(all_mutation_measurements_excluding_pad))

  ### FIGURE
  plt.figure(figsize=(35, 10))
  gs = plt.GridSpec(2, 1, height_ratios=[1, 10, 1], hspace=0)

  ## PLOT 0
  share_axis = plt.subplot(gs[0])
  share_axis.set_xlim(left=min_x_index)

  share_axis.get_xaxis().set_visible(False)
  share_axis.get_yaxis().set_visible(False)

  for subsequence in subsequences:
    plt.hlines(
        0,
        subsequence.range.start,
        subsequence.range.stop,
        linewidth=10,
        color='k')
    text_x_location = (subsequence.range.stop + subsequence.range.start) / 2
    plt.text(
        text_x_location,
        .03,
        subsequence.name,
        horizontalalignment='center',
        fontsize=18)

  share_axis.set_yticks([])

  share_axis.grid(False, axis='y')

  ### PLOT 1
  plt.subplot(gs[1], sharex=share_axis)
  plt.imshow(
      all_mutation_measurements_including_pad,
      cmap='Blues',
      interpolation='none',
      clim=[
          np.min(all_mutation_measurements_excluding_pad),
          np.percentile(all_mutation_measurements_excluding_pad.flatten(), 80)
      ])
  plt.axis('tight')

  ax = plt.gca()
  ax.set_xticks([x[1].start for x in subsequences] +
                [x[1].stop for x in subsequences])
  ax.tick_params(
      axis='x',
      which='both',  # both major and minor ticks are affected
      top=False)
  ax.tick_params(
      axis='y',
      which='both',  # both major and minor ticks are affected
      right=False)
  ax.set_yticks(
      np.arange(0, len(amino_acid_semantic_grouping_reordering), 1.),
      minor=True)
  ax.set_yticklabels(
      amino_acid_semantic_grouping_reordering,
      minor=True,
      fontdict={'fontsize': 16})
  ax.set_yticks([], minor=False)
  plt.tick_params(labelsize=16)

  sns.despine(top=True, left=True, right=True, bottom=False)
  plt.xlim(xmin=min_x_index)


def get_basic_accuracy_computations(all_family_predictions):
  """Returns unweighted, mean-per-class, and clan-level accuracy.

  Args:
    all_family_predictions: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES.

  Returns:
    BasicAccuracyComputations object with unweighted_accuracy (float),
    mean_per_class_accuracy (float), and mean_per_clan_accuracy (float) fields.
  """
  family_to_clan_dict = pfam_utils.family_to_clan_mapping()
  clan_predictions = family_predictions_to_clan_predictions(
      all_family_predictions, family_to_clan_dict)

  return BasicAccuracyComputations(
      unweighted_accuracy=raw_unweighted_accuracy(all_family_predictions),
      mean_per_class_accuracy=mean_per_class_accuracy(all_family_predictions),
      mean_per_clan_accuracy=mean_per_class_accuracy(clan_predictions),
  )


def print_basic_accuracy_computations(all_family_predictions):
  """Print unweighted, mean-per-class, and clan-level accuracy.

  Args:
    all_family_predictions: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES.
  """
  basic_accuracy_computations = get_basic_accuracy_computations(
      all_family_predictions)

  print('Unweighted accuracy: {:.5f}'.format(
      basic_accuracy_computations.unweighted_accuracy))
  print('Mean per class accuracy: {:.5f}'.format(
      basic_accuracy_computations.mean_per_class_accuracy))
  print('Mean per clan accuracy: {:.5f}'.format(
      basic_accuracy_computations.mean_per_clan_accuracy))


def show_size_of_family_accuracy_comparisons(all_family_predictions):
  """Compare, using charts and measurements, effect of size on accuracy.

  Args:
    all_family_predictions: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES.
  """
  cxn = pfam_utils.connection_with_tmp_table(
      TMP_TABLE_NAME, blundell_constants.RANDOM_SPLIT_ALL_SEED_DATA_PATH)
  size_by_family = num_examples_per_class(cxn, TMP_TABLE_NAME)

  accuracy_by_size_of_family_dataframe = accuracy_by_size_of_family(
      all_family_predictions, size_by_family)

  imperfect_classes = accuracy_by_size_of_family_dataframe[
      accuracy_by_size_of_family_dataframe[ACCURACY_KEY] != 1.0]
  grid = sns.JointGrid(
      x=NUM_EXAMPLES_KEY,
      y=ACCURACY_KEY,
      data=imperfect_classes,
      xlim=(-10, 2000),
      ylim=(.01, 1.01),
  )
  grid = grid.plot_joint(plt.scatter, color='k', s=10)
  grid = grid.plot_marginals(
      sns.distplot,
      kde=False,
      color='.5',
  )
  grid = grid.set_axis_labels(
      xlabel='Number of examples in unsplit seed dataset', ylabel='Accuracy')

  # pytype incorrectly decides that this is not an attribute, but it is.
  # https://seaborn.pydata.org/generated/seaborn.JointGrid.html
  grid.ax_marg_x.set_axis_off()  # pytype: disable=attribute-error
  grid.ax_marg_y.set_axis_off()  # pytype: disable=attribute-error

  plt.show()

  print('Correlation between number of seed examples and accuracy: '
        '{:.5f}'.format(accuracy_by_size_of_family_dataframe[ACCURACY_KEY].corr(
            accuracy_by_size_of_family_dataframe[NUM_EXAMPLES_KEY])))


def show_sequence_length_accuracy_comparisons(all_family_predictions):
  """Compare, using charts and measurements, effect of length on accuracy.

  Args:
    all_family_predictions: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES.
  """
  cxn = pfam_utils.connection_with_tmp_table(
      TMP_TABLE_NAME, blundell_constants.RANDOM_SPLIT_ALL_SEED_DATA_PATH)
  length_of_examples_by_family = average_sequence_length_per_class(
      cxn, TMP_TABLE_NAME)

  accuracy_by_sequence_length_dataframe = accuracy_by_sequence_length(
      all_family_predictions, length_of_examples_by_family)

  imperfect_classes = accuracy_by_sequence_length_dataframe[
      accuracy_by_sequence_length_dataframe[ACCURACY_KEY] != 1.0]
  grid = sns.JointGrid(
      x=AVERAGE_SEQUENCE_LENGTH_KEY,
      y=ACCURACY_KEY,
      data=imperfect_classes,
      xlim=(-10, 2000),
      ylim=(.01, 1.01),
  )
  grid = grid.plot_joint(plt.scatter, color='k', s=10)
  grid = grid.set_axis_labels(
      xlabel='Avg sequence length per family (incl train and test)',
      ylabel='Accuracy of predictions')
  grid = grid.plot_marginals(sns.distplot, kde=False, color='.5')

  # pytype incorrectly decides that this is not an attribute, but it is.
  # https://seaborn.pydata.org/generated/seaborn.JointGrid.html
  grid.ax_marg_x.set_axis_off()  # pytype: disable=attribute-error
  grid.ax_marg_y.set_axis_off()  # pytype: disable=attribute-error

  plt.show()
  print(
      'Correlation between sequence length of seed examples and accuracy: '
      '{:.5f}'.format(accuracy_by_sequence_length_dataframe[ACCURACY_KEY].corr(
          accuracy_by_sequence_length_dataframe[AVERAGE_SEQUENCE_LENGTH_KEY])))


def show_transmembrane_mutation_matrices(savedmodel_dir_path):
  """Compute and plot mutation matrices for various transmembrane domains.

  Args:
    savedmodel_dir_path: path to directory where a SavedModel pb or
      pbtxt is stored. The SavedModel must only have one input per signature
      and only one output per signature.
  """
  inferrer = inference_lib.Inferrer(
      savedmodel_dir_path=savedmodel_dir_path,
      activation_type=protein_task.LOGITS_SAVEDMODEL_SIGNATURE_KEY)

  show_mutation_matrix(
      sequence=V2R_HUMAN_SEQUENCE,
      subsequences=V2R_ANNOTATED_SUBSEQUENCES,
      start_index_of_domain=V2R_START_INDEX_OF_DOMAIN,
      inferrer=inferrer)

  atpase_pig_subsequences = ATPASE_ANNOTATED_SUBSEQUENCES
  show_mutation_matrix(
      sequence=ATPASE_PIG_SEQUENCE,
      subsequences=atpase_pig_subsequences,
      start_index_of_domain=ATPASE_START_INDEX_OF_DOMAIN,
      inferrer=inferrer)


def _filter_hmmer_first_pass_by_gathering_threshold(
    hmmer_prediction_with_scores_df, gathering_thresholds_df):
  """Filters predictions to only include those above the gathering thresholds.

  Args:
    hmmer_prediction_with_scores_df: pandas DataFrame with 4 columns,
      hmmer_util.HMMER_OUTPUT_CSV_COLUMN_HEADERS. The true and predicted label
      are allowed to have version numbers on the accession ids (like PF12345.x).
    gathering_thresholds_df: pandas DataFrame with 2 columns,
      classification_util.TRUE_LABEL_KEY and
      hmmer_utils.DATAFRAME_SCORE_NAME_KEY. The true label is allowed to have
      version numbers on the accession ids (like PF12345.x).

  Returns:
    pandas DataFrame with columns hmmer_util.HMMER_OUTPUT_CSV_COLUMN_HEADERS.
  Raises:
    KeyError: If there is a true label that's not in
      gathering_thresholds_df that is in hmmer_prediction_with_scores_df.
    ValueError: If there is a true_label that is repeated in
      gathering_thresholds_df.

  """
  # Avoid modifying passed arguments.
  hmmer_prediction_with_scores_df = hmmer_prediction_with_scores_df.copy(
      deep=True)
  gathering_thresholds_df = gathering_thresholds_df.copy(deep=True)

  # Sanitize family accessions to not have version numbers.
  hmmer_scores_key = classification_util.PREDICTED_LABEL_KEY + '_sanitized'
  gathering_thresholds_key = classification_util.TRUE_LABEL_KEY + '_sanitized'
  hmmer_prediction_with_scores_df[
      hmmer_scores_key] = hmmer_prediction_with_scores_df[
          classification_util.PREDICTED_LABEL_KEY].apply(
              pfam_utils.parse_pfam_accession)
  gathering_thresholds_df[gathering_thresholds_key] = gathering_thresholds_df[
      classification_util.TRUE_LABEL_KEY].apply(pfam_utils.parse_pfam_accession)

  if np.any(
      gathering_thresholds_df.duplicated(
          classification_util.TRUE_LABEL_KEY).values):
    raise ValueError('One or more of the true labels in the gathering '
                     'thresholds dataframe was duplicated: {}'.format(
                         gathering_thresholds_df.groupby(
                             classification_util.TRUE_LABEL_KEY).size() > 1))

  gathering_thresholds_dict = pd.Series(
      gathering_thresholds_df.score.values,
      index=gathering_thresholds_df[gathering_thresholds_key]).to_dict()
  threshold_key = hmmer_utils.DATAFRAME_SCORE_NAME_KEY + '_thresh'
  # Will raise KeyError if the family is not found in the gathering thresholds
  # dict.
  hmmer_prediction_with_scores_df[
      threshold_key] = hmmer_prediction_with_scores_df[hmmer_scores_key].apply(
          lambda x: gathering_thresholds_dict[x])

  filtered = hmmer_prediction_with_scores_df[
      hmmer_prediction_with_scores_df[hmmer_utils.DATAFRAME_SCORE_NAME_KEY] >
      hmmer_prediction_with_scores_df[threshold_key]]

  logging.info('Size before filtering by gathering thresh: %d',
               len(hmmer_prediction_with_scores_df))
  logging.info('Size after filtering: %d', len(filtered))
  assert len(filtered) <= len(hmmer_prediction_with_scores_df)

  # Get rid of extra columns.
  return filtered[hmmer_utils.HMMER_OUTPUT_CSV_COLUMN_HEADERS]


def _group_by_size_histogram_data(dataframe, group_by_key):
  """Returns a histogram of the number of elements per group.

    If you group by sequence_name, the dictionary you get returned is:
      key: number of predictions a sequence has
      value: number of sequences with `key` many predictions.

  Args:
    dataframe: pandas DataFrame that has column group_by_key.
    group_by_key: string. The column to group dataframe by.

  Returns:
    dict from int to int.
  """
  return dataframe.groupby(group_by_key).size().to_frame('size').groupby(
      'size').size().to_dict()


def _had_more_than_one_prediction_and_in_clan(predictions_df, family_to_clan):
  """Returns the number of sequences with >1 prediction and also in a clan.

  Args:
    predictions_df: pandas DataFrame with 3 columns:
      hmmer_utils.DATAFRAME_SCORE_NAME_KEY, classification_util.TRUE_LABEL_KEY,
      and classification_util.PREDICTED_LABEL_KEY. Version numbers are
      acceptable on the family names.
    family_to_clan: dict from string to string, e.g. {'PF12345': 'CL9999'}.
      Version numbers are acceptable on the family names.

  Returns:
    int.
  """
  # Avoid mutating original object.
  predictions_df = predictions_df.copy(deep=True)

  number_of_predictions = predictions_df.groupby(
      classification_util.DATAFRAME_SEQUENCE_NAME_KEY).size().to_frame(
          'number_of_predictions')
  number_of_predictions.reset_index(inplace=True)

  predictions_df = pd.merge(
      predictions_df,
      number_of_predictions,
      left_on=classification_util.DATAFRAME_SEQUENCE_NAME_KEY,
      right_on=classification_util.DATAFRAME_SEQUENCE_NAME_KEY)
  multiple_predictions = predictions_df.copy(deep=True)
  multiple_predictions = multiple_predictions[
      multiple_predictions['number_of_predictions'] > 1]
  multiple_predictions['parsed_true_label'] = multiple_predictions[
      classification_util.TRUE_LABEL_KEY].map(pfam_utils.parse_pfam_accession)
  family_to_clan_parsed = {
      pfam_utils.parse_pfam_accession(k): v for k, v in family_to_clan.items()
  }
  in_clans = multiple_predictions[
      multiple_predictions['parsed_true_label'].isin(family_to_clan_parsed)]

  return len(in_clans.groupby(classification_util.DATAFRAME_SEQUENCE_NAME_KEY))


def show_hmmer_first_pass_gathering_threshold_statistics(
    hmmer_prediction_with_scores_csv_path):
  """Print number of predictions per sequence over gathering threshold.

  Args:
    hmmer_prediction_with_scores_csv_path: string. Path to csv file that has
      columns hmmer_utils.HMMER_OUTPUT_CSV_COLUMN_HEADERS.
  """
  with tf.io.gfile.GFile(
      GATHERING_THRESHOLDS_PATH) as gathering_thresholds_file:
    gathering_thresholds_df = pd.read_csv(
        gathering_thresholds_file,
        names=[
            classification_util.TRUE_LABEL_KEY,
            hmmer_utils.DATAFRAME_SCORE_NAME_KEY
        ])

  with tf.io.gfile.GFile(
      hmmer_prediction_with_scores_csv_path) as hmmer_output_file:
    hmmer_scores_df = pd.read_csv(hmmer_output_file)

  filtered = _filter_hmmer_first_pass_by_gathering_threshold(
      hmmer_scores_df, gathering_thresholds_df)

  counted_by_num_predictions = _group_by_size_histogram_data(
      filtered, classification_util.DATAFRAME_SEQUENCE_NAME_KEY)

  family_to_clan_dict = pfam_utils.family_to_clan_mapping()
  meet_reporting_criteria = _had_more_than_one_prediction_and_in_clan(
      filtered, family_to_clan_dict)
  print('Count of seqs that had more than one prediction, '
        'and also were in a clan {}'.format(meet_reporting_criteria))

  print('Count of sequences by number of predictions: {}'.format(
      counted_by_num_predictions))


def show_mutation_matrix(sequence, subsequences, start_index_of_domain,
                         inferrer):
  """Compute and display predicted effects of mutating sequence everywhere.

  Args:
    sequence: string of amino acid characters.
    subsequences: list of Subsequence. These areas will be displayed alongside
      the mutation predictions.
    start_index_of_domain: int. Because most domains do not begin at index 0 of
      the protein, but Subsequence.slice indexing does, we have to offset where
      we display the start of the mutation predictions in the plot. This
      argument is 0-indexed.
    inferrer: inference_lib.Inferrer instance.
  """
  mutation_measurements = inference_lib.measure_all_mutations(
      sequence, inferrer)
  plot_all_mutations(
      mutation_measurements,
      start_index_of_mutation_predictions=start_index_of_domain,
      subsequences=subsequences)


def precision_recall_dataframe(
    predictions_df,
    percentile_thresholds=_PRECISION_RECALL_PERCENTILE_THRESHOLDS):
  """Return dataframe with precision and recall for each percentile in list.

  Args:
    predictions_df: pandas DataFrame with 3 columns:
      hmmer_utils.DATAFRAME_SCORE_NAME_KEY, classification_util.TRUE_LABEL_KEY,
      and classification_util.PREDICTED_LABEL_KEY.
    percentile_thresholds: list of float between 0 and 1. These values will be
      used as percentiles for varying the thresholding of
      hmmer_utils.DATAFRAME_SCORE_NAME_KEY to compute precision and recall.

  Returns:
    pandas dataframe with columns PRECISION_RECALL_COLUMNS.
  """
  # Avoid mutating original object.
  predictions_df = predictions_df.copy(deep=True)
  precision_recall_df = pd.DataFrame(columns=_PRECISION_RECALL_COLUMNS)

  for percentile in percentile_thresholds:
    percentile_cutoff = predictions_df[
        hmmer_utils.DATAFRAME_SCORE_NAME_KEY].quantile(
            percentile, interpolation='nearest')

    called_elements = predictions_df[predictions_df[
        hmmer_utils.DATAFRAME_SCORE_NAME_KEY] >= percentile_cutoff]

    true_positive = len(called_elements[called_elements[
        classification_util.TRUE_LABEL_KEY] == called_elements[
            classification_util.PREDICTED_LABEL_KEY]])
    false_positive = len(called_elements[
        called_elements[classification_util.TRUE_LABEL_KEY] != called_elements[
            classification_util.PREDICTED_LABEL_KEY]])

    if true_positive == 0 and false_positive == 0:
      # Avoid division by zero error; we called zero elements.
      precision = 0
    else:
      precision = float(true_positive) / (true_positive + false_positive)

    uncalled_elements = predictions_df[predictions_df[
        hmmer_utils.DATAFRAME_SCORE_NAME_KEY] < percentile_cutoff]
    false_negative = len(uncalled_elements[uncalled_elements[
        classification_util.TRUE_LABEL_KEY] == uncalled_elements[
            classification_util.PREDICTED_LABEL_KEY]])

    if true_positive == 0 and false_negative == 0:
      # Avoid division by zero error.
      recall = 0.
    else:
      recall = float(true_positive) / len(predictions_df)

    precision_recall_df = precision_recall_df.append(
        {
            THRESHOLD_PRECISION_RECALL_KEY: percentile_cutoff,
            PRECISION_PRECISION_RECALL_KEY: precision,
            RECALL_PRECISION_RECALL_KEY: recall,
        },
        ignore_index=True,
    )

  return precision_recall_df


def show_precision_recall(predictions_df):
  """Compute precision and recall for predictions, and graph.

  Args:
    predictions_df: pandas DataFrame with 3 columns:
      hmmer_utils.DATAFRAME_SCORE_NAME_KEY, classification_util.TRUE_LABEL_KEY,
      and classification_util.PREDICTED_LABEL_KEY.
  """
  precision_recall_df = precision_recall_dataframe(predictions_df)
  ax = sns.scatterplot(
      x=PRECISION_PRECISION_RECALL_KEY,
      y=RECALL_PRECISION_RECALL_KEY,
      data=precision_recall_df,
      color='k')
  ax.set_xlim(left=0, right=1)
  ax.set_ylim(bottom=0, top=1)
  plt.show()


def output_basic_measurements_and_figures(all_family_predictions):
  """Show basic charts and graphs about the given predictions' accuracy.

  Args:
    all_family_predictions: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES. The true and predicted
      label are allowed to have version numbers on the accession ids
      (like PF12345.x).
  """
  print_basic_accuracy_computations(all_family_predictions)
  show_size_of_family_accuracy_comparisons(all_family_predictions)
  show_sequence_length_accuracy_comparisons(all_family_predictions)


def output_all_measurements_and_figures(savedmodel_dir_path,
                                        prediction_dirs_and_names,
                                        dataset_descriptor_path):
  """Output all measurements for trained model needed for benchmark paper.

  Args:
    savedmodel_dir_path: path to directory where a SavedModel pb or pbtxt is
      stored. The SavedModel must only have one input per signature and only one
      output per signature.
    prediction_dirs_and_names: dictionary from str to str. Keys are human
      readable descriptions of keys. Keys are paths to csv prediction files
      from a model
    dataset_descriptor_path: Path to dataset descriptor used when training the
      classification model.
  """
  for name, prediction_dir in prediction_dirs_and_names.items():
    prediction_csv_path = get_latest_prediction_file_from_checkpoint_dir(
        prediction_dir)

    print('Predictions for {} in path {} '.format(name, prediction_csv_path))

    index_to_family_accession = classification_util.idx_to_family_id_dict(
        dataset_descriptor_path)
    all_family_predictions = load_prediction_file(prediction_csv_path,
                                                  index_to_family_accession)

    output_basic_measurements_and_figures(all_family_predictions)
    print('\n')

  show_transmembrane_mutation_matrices(savedmodel_dir_path=savedmodel_dir_path)


def read_blundell_style_csv(
    fin,
    all_sequence_names=None,
    names=None,
    raise_if_missing=True,
    missing_sequence_name=hmmer_utils.NO_SEQUENCE_MATCH_SEQUENCE_NAME_SENTINEL,
    deduplicate=True):
  """Reads a blundell-style CSV of predictions.

  This function is a similar but more general function than
  `load_prediction_file()` as it makes fewer assumptions about the encoding of
  the family names, directly handles deduplication, verifies the completeness
  of the predictions, and handles NA or missing sequence sentinel values.

  This function handles missing predictions with the all_sequence_names and
  raise_if_missing arguments. For example, blast's output CSV is expected to
  have missing predictions because not all sequences can be aligned with blast
  using the default parameters. In such cases, we want to fill in NaN
  predictions for all of these missing sequences. We do so by providing
  all_sequence_names and raise_if_missing=False, which will result in the
  returned dataframe having NaN values filled in for all of the missing
  sequences. Not that if raise_if_missing=True, this function would raise a
  ValueError if any sequences are missing.

  In some situations (e.g., with hmmer) the CSV isn't outright missing
  predictions for some sequences but rather there are special output rows that
  are marked with a sentinel value indicating they aren't real predictions but
  are rather missing values. To support such situation, any row in our dataframe
  whose sequence name is equal to `missing_sequence_name` will be removed and
  treated just like a genuinely missing row, where the response columns (
  basically everything except classification_util.DATAFRAME_SEQUENCE_NAME_KEY)
  set to NaN values.

  If deduplicate=True, `yield_top_el_by_score_for_each_sequence_name` will be
  run on the raw dataframe read from fin to select only the best scoring
  prediction for each sequence.

  Args:
    fin: file-like. Where we will read the CSV.
    all_sequence_names: Option set/list of expected sequence names. If not None,
      this set of strings should contain all of the sequence names expected to
      occur in fin.
    names: optional list of strings. Defaults to None. If provided, will be
      passed as the `names` argument to pandas.read_csv, allowing us to read a
      CSV from fin without a proper header line.
    raise_if_missing: bool. Defaults to True. Should we raise an exception if
      any sequence in all_sequence_names is missing in the read in dataframe?
    missing_sequence_name: string. Defaults to
      `hmmer_utils.NO_SEQUENCE_MATCH_SEQUENCE_NAME_SENTINEL`. If provided, any
      sequence name in the dataframe matching this string will be treated as a
      missing value marker.
    deduplicate: bool. Defaults to True. If True, we will deduplicate the
      dataframe, selecting only the best scoring prediction when there are
      multiple predictions per sequence.

  Returns:
    A Pandas dataframe containing four columns: sequence_name, true_label,
    predicted_label, score, and domain_evalue.
    The dataframe is sorted by sequence_name.
  """

  df = pd.read_csv(fin, names=names)
  if deduplicate:
    df = pd.concat(
        hmmer_utils.yield_top_el_by_score_for_each_sequence_name(df),
        ignore_index=True)

  no_call_sequences = df.sequence_name == missing_sequence_name
  if no_call_sequences.sum() > 0:
    logging.info('Removing %d no-called sequences', no_call_sequences.sum())
    df = df[np.logical_not(no_call_sequences)]

  if all_sequence_names is not None:
    pfam_utils.validate_no_extra_sequence_names(df.sequence_name,
                                                all_sequence_names)
    missing = pfam_utils.find_missing_sequence_names(
        df.sequence_name, all_sequence_names, raise_if_missing=raise_if_missing)
    if missing:
      df = df.append(pd.DataFrame(dict(sequence_name=list(missing))))

  df.sort_values(
      by=[classification_util.DATAFRAME_SEQUENCE_NAME_KEY], inplace=True)
  df.reset_index(drop=True, inplace=True)
  return df.reindex(sorted(df.columns), axis=1)


def _split_df_get_agg_stats_dict(df, split_interval
                                ):
  """Computes a dictionary of statistics about a DataFrame that's been split.

  Args:
    df: pdDataFrame containing labels 'true_label' and 'predicted_label'.
    split_interval: interval describing how this dataframe was split.

  Returns:
    Dictionary containing interval, accuracy, and error_rate information.
  """
  accuracy = raw_unweighted_accuracy(df)
  mpca = mean_per_class_accuracy(df)
  return dict(
      lower_bound=split_interval.left,
      midpoint=split_interval.mid,
      upper_bound=split_interval.right,
      n=len(df),
      accuracy=accuracy,
      mean_per_class_accuracy=mpca,
      error_rate=1 - accuracy,
      mean_per_class_error_rate=1 - mpca,
  )


def _agg_stats_df_for_cuts(df):
  """Given df with an interval column, group by that column and report stats.

  Args:
    df: pd.DataFrame containing columns 'true_label', 'predicted_label', and
      'interval'

  Returns:
    pd.DataFrame with aggregate accuracy statistics for every interval.
  """
  accuracy_by_cut = []
  for split_interval, df_subset in df.groupby(INTERVAL_DATAFRAME_KEY):
    if len(df_subset) > 0:  # pylint: disable=g-explicit-length-test
      cur_row = _split_df_get_agg_stats_dict(
          df=df_subset, split_interval=split_interval)
      accuracy_by_cut.append(cur_row)

  cuts_df = pd.DataFrame(accuracy_by_cut)

  return cuts_df


def _to_lookup_dict(df, cut_df,
                    cut_column_name):
  """Computes dict from sequence name to interval.

  E.g. if you want to stratify accuracy analysis by family size, you'd pass
  in `df` that has all your test sequences, `cut_df` is the result of
  pd.cut(df, 'num_examples'), and cut_column_name 'num_examples', or similar.

  Args:
    df: pd.DataFrame containing columns sequence_name.
    cut_df: pd.Categorical; probably the result of pd.cut(df) or pd.qcut(df).
    cut_column_name: str describing the column by which df was cut.

  Returns:
    dict of sequence name to pd.Interval, describing the group into which
    each sequence is put for stratified analysis.
  """
  joined_cut = pd.concat([cut_df, df.sequence_name],
                         axis='columns',
                         ignore_index=False)
  return {
      k: v
      for (k, v) in zip(joined_cut.sequence_name, joined_cut[cut_column_name])
  }


def accuracy_by_family_size_equal_sized(df,
                                        num_quantiles = 10
                                       ):
  """Computes dataframe of aggregate statistics for each equal sized bin.

  Bins sequences into equal-sized bins (equal number of sequences) and then
  computes aggregate statistics for each bin.

  Args:
    df: pd.DataFrame containing columns 'true_label', `predicted_label_key`,
      'sequence_name', and 'num_examples'.
    num_quantiles: number of quantiles to cut family size by.

  Returns:
    dict of sequence name to pd.Interval, describing the group into which
    each sequence is put for stratified analysis.
  """
  cut = pd.qcut(
      df[NUM_EXAMPLES_KEY], num_quantiles, duplicates='drop', precision=1)

  return _to_lookup_dict(df, cut, NUM_EXAMPLES_KEY)


def accuracy_by_sequence_identity_equal_sized(
    df,
    distance_metric = blast_utils.BLAST_SEQUENCE_PERCENT_IDENTITY,
    num_bins = 10):
  """Computes accuracy for each split into equal sized bins of `df`.

  Each bin will have roughly the same number of elements (but the intervals
  corresponding to these bins may not be equal width).

  Args:
    df: pd.DataFrame containing columns 'true_label', `predicted_label_key`, and
      'sequence_name'
    distance_metric: string column name of value by which to slice df.
    num_bins: int.

  Returns:
    dict of sequence name to pd.Interval, describing the group into which
    each sequence is put for stratified analysis.
  """
  cut = pd.qcut(df[distance_metric], num_bins, duplicates='drop', precision=1)

  return _to_lookup_dict(df, cut, distance_metric)


def accuracy_by_metric_equal_width(
    df,
    metric_col_name = blast_utils.BLAST_SEQUENCE_PERCENT_IDENTITY,
    bin_width = 10,
    min_metric_value = 0,
    max_metric_value = 101):
  """Computes accuracy for each split into equal width bins of `df`.

  Each bin will have the same width (but the intervals corresponding to these
  bins may not have equal numbers of elements).

  Bins are computed based on the value of `sequence_identity_col_name`.

  Bins are left-inclusive, right-exclusive.

  Args:
    df: pd.DataFrame containing columns 'true_label', `predicted_label_key`, and
      'sequence_name'
    metric_col_name: string column name of value by which to slice
      df.
    bin_width: float.
    min_metric_value: minimum metric_value to plot.
    max_metric_value: maximum metric_value to plot. Values of metric_value in
      df up to but not including this value are plotted.

  Returns:
    dict of sequence name to pd.Interval, describing the group into which
    each sequence is put for stratified analysis.
  """
  cut = pd.cut(
      df[metric_col_name],
      np.arange(min_metric_value, max_metric_value + 1, bin_width),
      duplicates='drop',
      precision=1,
      # include_lowest=True along with right=False makes these
      # left-inclusive, right-exclusive intervals.
      right=False,
      include_lowest=True)

  return _to_lookup_dict(df, cut, cut_column_name=metric_col_name)


def _stable_hash(s):
  m = hashlib.sha256()
  m.update(s.encode('UTF-8'))
  return int(m.hexdigest(), 16)


def matplotlib_color_for(method):
  """Returns a stable coloring for a given method.

  Args:
    method: Text describing which method.

  Returns:
    rgb tuple.
  """
  if 'HMM' in method:
    index_str = 'Top pick HMM'
  elif 'phmmer' in method:
    index_str = 'phmmer'
  elif 'ProtCNN' in method:
    index_str = 'ProtCNN'
  elif 'ProtENN' in method:
    index_str = 'ProtENN'
  elif 'BLASTp' in method:
    index_str = 'BLASTp'
  elif 'ProtREP' in method:
    # Extra spaces to get a unique color.
    index_str = 'ProtREP '
  elif 'Per-Instance ProtREP' in method:
    index_str = 'Per-Instance ProtREP'
  else:
    index_str = method

  palette = sns.color_palette('colorblind')

  return palette[_stable_hash(index_str) % len(palette)]


def plot_by_cut(df,
                interval_lookup,
                lineplot_label,
                y_axis = 'error_rate',
                bin_boundaries_on_axis = 'below_every'):
  """Computes and plots `y_axis` for each split into equal width bins of `df`.

  Each bin will have the same width (but the intervals corresponding to these
  bins may not have equal numbers of elements).

  Bins are computed based on the value of `sequence_identity_col_name`.

  Bins are left-inclusive, right-exclusive.

  Args:
    df: pd.DataFrame containing columns 'true_label', `predicted_label_key`, and
      'sequence_name'
    interval_lookup: dict of sequence name to pd.Interval, describing the group
      into which each sequence is put for stratified analysis.
    lineplot_label: string label for line in graph.
    y_axis: the column values to use for the y-axis in the plot, e.g.
      'accuracy', 'error_rate', 'mean_per_class_accuracy',
      'mean_per_class_error_rate'.
    bin_boundaries_on_axis: str. Either 'below_every' or 'just_boundary_edges'.
      If below_every, then upper and lower bounds will be printed between each
      tick mark on the x axis. If just_boundary_edges, just the bin boundaries
      will be printed as the x axis labels.

  Returns:
    plt.Axes on which the graph is plotted.
  """
  df_copy = df.copy(deep=True)
  df_copy[INTERVAL_DATAFRAME_KEY] = df_copy[
      classification_util.DATAFRAME_SEQUENCE_NAME_KEY].apply(
          interval_lookup.__getitem__)

  cut_df = _agg_stats_df_for_cuts(df_copy)

  axis_to_plot = y_axis
  if y_axis == 'error_rate' or y_axis == 'mean_per_class_error_rate':
    axis_to_plot = 'Error rate (log scale)'
    cut_df[axis_to_plot] = cut_df[y_axis] + 10**-5
    plt.yscale('log')

  # Get equidistant x-axis values, then change ticks labels to indicate the
  # value.
  cut_df = cut_df.sort_values('lower_bound')
  cut_df['x_axis'] = np.arange(0, len(cut_df)) + .5

  color = matplotlib_color_for(lineplot_label)

  g = sns.lineplot(
      data=cut_df,
      x='x_axis',
      y=axis_to_plot,
      label=lineplot_label,
      marker='o',
      lw=3,
      color=color,
      markersize=10)

  ax = plt.gca()
  ax.set_xticks(cut_df.x_axis, minor=True)
  # Add a tick to the right of the rightmost point for a nice-looking margin.
  ax.set_xticks(np.arange(0, len(cut_df) + 1), minor=False)
  ax.tick_params(top=False, bottom=False, which='minor')
  ax.grid(which='minor', axis='x', color='w', linewidth=8)

  if bin_boundaries_on_axis == 'below_every':
    ax.set_xticklabels([
        '{:.0f}-{:.0f}'.format(lower_bound, upper_bound)
        for (lower_bound,
             upper_bound) in zip(cut_df.lower_bound, cut_df.upper_bound)
    ],
                       minor=True)
    ax.set_xticklabels([], minor=False)

  elif bin_boundaries_on_axis == 'just_boundary_edges':
    ax.set_xticklabels(
        ['{:.0f}'.format(min(cut_df.lower_bound))] +
        ['{:.0f}'.format(upper_bound) for upper_bound in cut_df.upper_bound],
        minor=False)
    ax.tick_params(axis='x', which='minor', labelsize=11)

  else:
    raise ValueError(
        'Unsupported value for bin_boundaries_on_axis. Was {}, expected \'below_every\' or \'just_boundary_edges\''
        .format(bin_boundaries_on_axis))

  return g


def format_accuracy_graph():
  """Formats graph containing accuracy or error_rate line plots."""
  sns.set_style('whitegrid')
  sns.set_palette('colorblind')
  ax = plt.gca()
  ax.set_title(ax.title.get_text(), fontsize=18)
  plt.gcf().set_size_inches((12, 6))
  plt.setp(ax.get_legend().get_texts(), fontsize='18')
  plt.setp(ax.get_legend().get_title(), fontsize='0')

  ax = plt.gca()
  ax.tick_params(axis='x', which='both', labelsize=16, top=False)
  ax.tick_params(
      axis='y',
      labelsize=16,
      which='both',  # both major and minor ticks are affected
      right=False)

  ax.xaxis.label.set_size(16)
  ax.yaxis.label.set_size(16)

  if ax.get_yscale() == 'log':
    plt.ylim(10**-5.5, 1.5)
  else:
    plt.ylim(plt.ylim()[0], 1.05)


def filter_to_is_in_clan(df,
                         family_to_clan_dict):
  """Filters df to only families that are in clans.

  Args:
    df: DataFrame with column true_label. Version numbers are acceptable on the
      family names.
    family_to_clan_dict: dict from string to string, e.g. {'PF12345': 'CL9999'}.
      Version numbers are NOT acceptable on the family names.

  Returns:
    pd.DataFrame.
  """
  return df[df.apply(
      lambda row: pfam_utils.parse_pfam_accession(row.true_label) in  # pylint: disable=g-long-lambda
      family_to_clan_dict,
      axis='columns')]


def filter_to_not_in_clan(df,
                          family_to_clan_dict
                         ):
  """Filters df to only families that are NOT in clans.

  Args:
    df: DataFrame with column true_label. Version numbers are acceptable on the
      family names.
    family_to_clan_dict: dict from string to string, e.g. {'PF12345': 'CL9999'}.
      Version numbers are NOT acceptable on the family names.

  Returns:
    pd.DataFrame.
  """
  return df[df.apply(
      lambda row: pfam_utils.parse_pfam_accession(row.true_label) not in  # pylint: disable=g-long-lambda
      family_to_clan_dict,
      axis='columns')]


def joint_correctness_contingency_table(df1, df2, uid_key, prediction_key,
                                        true_label_key):
  """Makes a contingency table comparing correctness of model 1 and model 2.

  Args:
    df1: a DataFrame for the predictions of model 1.
    df2: a DataFrame for the predictions of model 2.
    uid_key: unique identifier used when joining df1 and df2
    prediction_key: column in df1 and df2 containing models' predictions.
    true_label_key: column in df1 and df2 containing ground truth labels.

  Returns:
    A 2x2 contingency table of counts for when each model made a correct
    prediction. The rows index model 1 and the columns index model 2.
    Table entry table[0][1], for example, is the number of times that model 1
    was incorrect and model 2 was correct.
  """

  df = df1.merge(df2, on=uid_key, suffixes=('_1', '_2'))
  df.reset_index()
  assert np.all(df[true_label_key + '_1'] == df[true_label_key + '_2'])
  true_labels = df[true_label_key + '_1']

  pred1 = df[prediction_key + '_1']
  pred2 = df[prediction_key + '_2']
  pred1_correct = pred1 == true_labels
  pred2_correct = pred2 == true_labels

  contingency_table = np.zeros(shape=[2, 2])
  contingency_table[0, 0] = len(df[~pred1_correct & ~pred2_correct])
  contingency_table[0, 1] = len(df[~pred1_correct & pred2_correct])
  contingency_table[1, 0] = len(df[pred1_correct & ~pred2_correct])
  contingency_table[1, 1] = len(df[pred1_correct & pred2_correct])
  return contingency_table


def _run_binomial_test(contingency_table, verbose):
  """Runs one-sided binomial test using contingency_table.

  The null hypothesis is that both models have the same error rate. The
  alternative hypothesis is that model 1 has a higher error rate.

  See: https://en.wikipedia.org/wiki/Sign_test

  Args:
    contingency_table: A 2x2 contingency table of counts for when each model
      made a correct prediction. The rows index model 1 and the columns index
      model 2. Table entry table[0][1], for example, is the number of times that
      model 1 was correct and model 2 was incorrect.
    verbose: whether to print the results of the test.

  Returns:
    The p_value of the test.
  """

  def print_if_verbose(msg):
    if verbose:
      logging.info(msg)

  print_if_verbose('Null hypothesis: the models have the same error rate')
  print_if_verbose('Alternative hypothesis: model 1 has a higher error rate')
  num_errors_for_model_1 = contingency_table[0][1]
  num_errors_for_model_2 = contingency_table[1][0]
  print_if_verbose('num model 1 wrong and model 2 '
                   'correct: %d' % num_errors_for_model_1)
  print_if_verbose('num model 2 wrong and model 1 '
                   'correct: %d' % num_errors_for_model_2)
  num_total_errors = num_errors_for_model_1 + num_errors_for_model_2
  p_value = proportion.binom_test(
      num_errors_for_model_1, num_total_errors, prop=0.5, alternative='larger')
  print_if_verbose('p_value: %f' % p_value)
  return p_value


def one_sided_binomial_test(df1,
                            df2,
                            uid_key,
                            prediction_key,
                            true_label_key,
                            verbose=True):
  """One-sided binomial test for model 1 having higher error rate than model 2.

  The null hypothesis is that both models have the same error rate. The
  alternative hypothesis is that model 1 has a higher error rate. For each model
  we have a random variable for each prediction corresponding to whether the
  it was correct. When comparing two models, we have paired random variables for
  each example. Given that these random variables disagree, we have that either
  model 1 was correct and model 2 was wrong or vice-versa. Under the
  null hypothesis, these occur with probability 0.5.

  See: https://en.wikipedia.org/wiki/Sign_test

  Note that the two-sided version of this test (where the alternative hypothesis
  is that model 1 and model 2 have different error rates)
  is also known as the McNemar test.

  Args:
    df1: a DataFrame for the predictions of model 1 containing columns for
      uid_key, prediction_key and true_label_key.
    df2: a DataFrame with the same format as df1, but for model 2.
    uid_key: unique identifier used when joining df1 and df2
    prediction_key: column in df1 and df2 containing models' predictions.
    true_label_key: column in df1 and df2 containing ground truth labels.
    verbose: whether to print the results of the test.

  Returns:
    The p_value of the test.
  """
  contingency_table = joint_correctness_contingency_table(
      df1, df2, uid_key, prediction_key, true_label_key)
  return _run_binomial_test(contingency_table, verbose)


def _run_mcnemar_test(contingency_table, verbose):
  """Runs McNemar test for signficance of difference of models' accuracy.

  See: https://en.wikipedia.org/wiki/McNemar%27s_test

  Results of the test are printed.

  Args:
    contingency_table: A 2x2 contingency table of counts for when each model
      made a correct prediction. The rows index model 1 and the columns index
      model 2. Table entry table[0][1], for example, is the number of times that
      model 1 was correct and model 2 was incorrect.
    verbose: whether to print the results of the test.

  Returns:
    The p_value of the test.
  """

  def print_if_verbose(msg):
    if verbose:
      logging.info(msg)

  print_if_verbose('Null hypothesis: the models have the same error rate')
  print_if_verbose('Alternative hypothesis: the models have different '
                   'error rates')
  num_errors_for_model_1 = contingency_table[0][1]
  num_errors_for_model_2 = contingency_table[1][0]
  print_if_verbose('num errors by model 1: %d' % num_errors_for_model_1)
  print_if_verbose('num errors by model 2: %d' % num_errors_for_model_2)
  results = contingency_tables.mcnemar(contingency_table, exact=True)
  print_if_verbose(results)
  p_value = results.pvalue
  print_if_verbose('p_value: %f' % p_value)
  return p_value


def mcnemar_test(df1,
                 df2,
                 uid_key,
                 prediction_key,
                 true_label_key,
                 verbose=True):
  """Runs McNemar test for signficance of difference of models' accuracy.

  See: https://en.wikipedia.org/wiki/McNemar%27s_test

  Args:
    df1: a DataFrame for the predictions of model 1 containing columns for
      uid_key, prediction_key and true_label_key.
    df2: a DataFrame with the same format as df1, but for model 2.
    uid_key: unique identifier used when joining df1 and df2
    prediction_key: column in df1 and df2 containing models' predictions.
    true_label_key: column in df1 and df2 containing ground truth labels.
    verbose: whether to print the results of the test.

  Returns:
    The p_value of the test.
  """
  contingency_table = joint_correctness_contingency_table(
      df1, df2, uid_key, prediction_key, true_label_key)
  return _run_mcnemar_test(contingency_table, verbose)


def statistical_significance_per_bin(df1,
                                     df2,
                                     name1,
                                     name2,
                                     interval_lookup,
                                     p_value_thresh=0.05,
                                     hypothesis_test_fn=mcnemar_test):
  """Perform hypothesis test for each bin.

  Args:
    df1: DataFrame with PREDICTED_LABEL_KEY, TRUE_LABEL_KEY, and
      DATAFRAME_SEQUENCE_NAME_KEY.
    df2: DataFrame with PREDICTED_LABEL_KEY, TRUE_LABEL_KEY, and
      DATAFRAME_SEQUENCE_NAME_KEY.
    name1: name for the method corresponding to df1.
    name2: name for the method corresponding to df2.
    interval_lookup: dict from DATAFRAME_SEQUENCE_NAME_KEY field to a
      pd.Interval for each item in the input DataFrames.
    p_value_thresh: the null hypothesis will not be rejected if the p_value is
      above this threshold.
    hypothesis_test_fn: a function obeying the API of mcnemar_test or
      one_sided_binomial_test.
  """

  # Drop the true label column from df2 so that it doesn't get a suffix added to
  # it when joining.
  df2 = df2.drop([classification_util.TRUE_LABEL_KEY], axis='columns')
  joined = df1.merge(df2, on='sequence_name', suffixes=('_1', '_2'))
  joined['interval'] = joined[
      classification_util.DATAFRAME_SEQUENCE_NAME_KEY].apply(
          interval_lookup.__getitem__)
  logging.info('Assessing statistical significance using %s',
               hypothesis_test_fn.__name__)
  for split_interval, sdf in joined.groupby(INTERVAL_DATAFRAME_KEY):
    n = len(sdf)
    if n:
      df1 = sdf[[
          classification_util.TRUE_LABEL_KEY,
          classification_util.DATAFRAME_SEQUENCE_NAME_KEY
      ]].copy()
      df1[classification_util.PREDICTED_LABEL_KEY] = sdf[
          classification_util.PREDICTED_LABEL_KEY + '_1']

      df2 = sdf[[
          classification_util.TRUE_LABEL_KEY,
          classification_util.DATAFRAME_SEQUENCE_NAME_KEY
      ]].copy()
      df2[classification_util.PREDICTED_LABEL_KEY] = sdf[
          classification_util.PREDICTED_LABEL_KEY + '_2']
      acc1 = raw_unweighted_accuracy(df1)
      acc2 = raw_unweighted_accuracy(df2)

      p_value = hypothesis_test_fn(
          df1,
          df2,
          uid_key=classification_util.DATAFRAME_SEQUENCE_NAME_KEY,
          prediction_key=classification_util.PREDICTED_LABEL_KEY,
          true_label_key=classification_util.TRUE_LABEL_KEY,
          verbose=False)
      if p_value > p_value_thresh:
        logging.info(
            'Insignificant difference: %s vs. %s @ cut = %s, '
            'p_value = %f, n = %d, acc1: %f, acc2: %f', name1, name2,
            str(split_interval), p_value, n, acc1, acc2)


def binwise_statistical_significance_of_model_performances(
    lhs_dfs, rhs_dfs, interval_lookup, hypothesis_test_fn=mcnemar_test):
  """Does hypothesis tests for signficance of accuracy differences.

  We compare each model in lhs_dfs to each model in rhs_dfs. For each comparison
  we compute whether their difference in accuracy is statistically
  significant at each bin provided by interval_lookup.

  Args:
    lhs_dfs: dict with names of methods as keys and values that are DataFrames
      with PREDICTED_LABEL_KEY, TRUE_LABEL_KEY, and DATAFRAME_SEQUENCE_NAME_KEY.
    rhs_dfs: dict with names of methods as keys and values that are DataFrames
      with PREDICTED_LABEL_KEY, TRUE_LABEL_KEY, and DATAFRAME_SEQUENCE_NAME_KEY.
    interval_lookup: dict from DATAFRAME_SEQUENCE_NAME_KEY field to a
      pd.Interval for each item in the input DataFrames.
    hypothesis_test_fn: a function obeying the API of mcnemar_test or
      one_sided_binomial_test.
  """
  for lhs_name, lhs_df in lhs_dfs.items():
    for rhs_name, rhs_df in rhs_dfs.items():
      statistical_significance_per_bin(
          lhs_df,
          rhs_df,
          lhs_name,
          rhs_name,
          interval_lookup,
          p_value_thresh=0.05,
          hypothesis_test_fn=hypothesis_test_fn)


def load_sharded_df_csvs(
    csv_shard_dir,
    *,
    use_given_header = False,
    ignore_first_line = False,
    column_names = None):
  """Loads sharded CSVs into a pd.DataFrame.

  Runs concurrently to speed IO-bound operations.
  Assumes no headers are present on the CSVs.

  Args:
    csv_shard_dir: Path to directory where (only the) sharded csv files are
      found.
    use_given_header: Whether to use the header of each csv shard. Incompatible
      with `column_names` and `ignore_first_line`.
    ignore_first_line: Whether to ignore the first (header) line of the csv.
      Incompatible with `use_given_header`.
    column_names: List of column names inside csvs. Incompatible with
      `use_given_header`.

  Returns:
    pd.DataFrame, containing sharded contents, with rows in no particular order.

  Raises:
    ValueError if both use_given_header and column_names are passed.
    ValueError if both use_given_header and ignore_first_line are passed.
    ValueError if parsed shards have nonuniform column names.
  """
  if use_given_header and column_names:
    raise ValueError('Cannot pass both `use_given_header` and `column_names`. '
                     'Was given {} and {} respectively'.format(
                         use_given_header, column_names))
  if use_given_header and ignore_first_line:
    raise ValueError(
        'Cannot pass both `use_given_header` and `ignore_first_line`.')

  def read_pred_file(p):
    kwargs = {'skiprows': [0]} if ignore_first_line else {}
    with tf.io.gfile.GFile(p) as f:
      if column_names:
        return pd.read_csv(f, names=column_names, **kwargs)
      else:
        assert use_given_header
        return pd.read_csv(f, **kwargs)

  csv_shard_paths = [
      os.path.join(csv_shard_dir, f) for f in tf.io.gfile.ListDir(csv_shard_dir)
  ]

  shard_dfs = parallel.RunInParallel(
      read_pred_file, [{
          'p': p
      } for p in csv_shard_paths],
      10,
      cancel_futures=True)

  for shard1_cols, shard2_cols in itertools.combinations(
      [x.columns for x in shard_dfs], 2):
    set_difference = set(shard1_cols).symmetric_difference(set(shard2_cols))
    if set_difference:
      raise ValueError(
          'All shards must have the same columns. '
          'Saw {} and {} with symmetric set difference of {}.'.format(
              shard1_cols, shard2_cols, set_difference))

  return pd.concat(shard_dfs)
