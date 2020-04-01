# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Functions to evaluate and summarize metadata by their embeddings.
"""

import collections

import numpy as np
import pandas as pd
from scipy.spatial import distance
from six.moves import range
from six.moves import zip
from sklearn import metrics
from sklearn import model_selection as model_sel

from correct_batch_effects_wdn import distance as distance_analysis
from correct_batch_effects_wdn import metadata

CORRECT_NSC = 'Correct NSC'
MISMATCH_NSC = 'Mismatch NSC'
ACCURACY_NSC = 'Accuracy NSC'
CORRECT_NSC_NSB = 'Correct NSC NSB'
MISMATCH_NSC_NSB = 'Mismatch NSC NSB'
ACCURACY_NSC_NSB = 'Accuracy NSC NSB'

SCORE_MEAN = 'Score Mean'
SCORE_STD = 'Score Std'

# maximum sample size for each batch classifier
MAX_SAMPLE_SIZE = 10000
SEED = 1234


def _index_to_dict(idx, row):
  """Convert an index (a tuple) into a dict mapping columns to index values.

  Helper method used below to apply labels to the unlabeled index tuples
  emitted by iterrows().  Relies on the fact that in a distance matrix, the
  row indices and column indices are the same.

  Args:
    idx: a distance matrix index value (a tuple); should have the same columns
      as the row's index.
    row: a Series containing a row of a distance matrix.

  Returns:
    A dict mapping index columns to index values.
  """
  return dict(list(zip(row.index.names, idx)))


def not_same_metadata_filter(idx, row, metadatas):
  """Test if a row has any metadata overlap with a set specified by an index.

  Args:
    idx: a distance matrix index value (a tuple); should be the same columns
      as the row.
    row: a Series containing a row of a distance matrix.
    metadatas: List of metadata names that should return false if the values
      match.

  Returns:
    An array of booleans, with True indicating that the item does not match any
    of the metadata.
  """
  idx_dict = _index_to_dict(idx, row)
  retval = np.ones(row.shape, dtype=bool)
  for m in metadatas:
    retval &= row.index.get_level_values(m) != idx_dict[m]
  return retval


def not_same_compound_filter(idx, row):
  """Test whether a row of compounds differ from a value specified in an index.

  Args:
    idx: a distance matrix index value (a tuple); should be the same columns
      as the row
    row: a Series containing a row of a distance matrix.  The MultiIndex for
      the series should contain a COMPOUND level.

  Returns:
    An array of booleans, with True indicating that the item has a different
    compound.
  """
  return not_same_metadata_filter(idx, row, [metadata.COMPOUND])


def not_same_compound_or_batch_filter(idx, row):
  """Test whether a row of compounds and batches differ from those in an index.

  Args:
    idx: a distance matrix index value (a tuple); should be the same columns
      as the row
    row: a Series containing a row of a distance matrix.  The MultiIndex for
      the series should contain COMPOUND and BATCH levels.

  Returns:
    An array of booleans, with True indicating that the item has both a
    different compound and batch.
  """
  return not_same_metadata_filter(idx, row, [metadata.COMPOUND, metadata.BATCH])


def one_nearest_neighbor(dist, row_filter, match_metadata=metadata.MOA):
  """Compute fraction of compounds whose nearest neighbor has the same metadata.

  Args:
    dist: DataFrame containing a distance matrix.  Column and row indices
      should be the same.
    row_filter: filter function taking as arguments a dist matrix index (as a
      tuple) and a row (as a Series containing a row from the distance matrix).
    match_metadata: String of the index value used to evaluate matches.

  Returns:
    A tuple of (1) a list of correct (actual, predicted) pairs of metadata
    dictionaries and (2) a list of incorrect (actual, predicted) pairs of
    metadata dictionaries.
  """
  return k_nearest_neighbors(dist, 1, row_filter, match_metadata)


def k_nearest_neighbors(dist, k, row_filter,
                        match_metadata=metadata.MOA,
                        by_match_metadata=False):
  """Compute fraction of compounds whose k nearest neighbors have same metadata.

  Args:
    dist: DataFrame containing a distance matrix.  Column and row indices
      should be the same.
    k: number of nearest neighbors
    row_filter: filter function taking as arguments a dist matrix index (as a
      tuple) and a row (as a Series containing a row from the distance matrix).
    match_metadata: String of the index value used to evaluate matches.
    by_match_metadata: Boolean, whether the returned pairs of metadata
      dictionaries are separated by match_metadata.

  Returns:
    If by_match_metadata is False, returns a tuple of (1) a list of correct
    (actual, predicted) pairs of metadata dictionaries and (2) a list of
    incorrect (actual, predicted) pairs of metadata dictionaries. If
    by_match_metadata is True, returns a tuple of (1) a dictionary whose keys
    are match_metadata values, and whose values are lists of correct
    (actual, predicted) pairs of metadata dictionaries and (2) a dictionary
    whose keys are match_metadata values, and whose values are lists of
    incorrect (actual, predicted) pairs of metadata dictionaries.
  """
  if by_match_metadata:
    correct, mismatch = (collections.defaultdict(list),
                         collections.defaultdict(list))
  else:
    correct, mismatch = [], []

  for idx, row in dist.iterrows():
    filtered_row = row[row_filter(idx, row)]
    idx_dict = _index_to_dict(idx, row)
    idx_meta = idx_dict[match_metadata]
    # count filtered values with the same metadata
    n_meta = np.sum(
        filtered_row.index.get_level_values(match_metadata) == idx_meta)
    filtered_row.sort_values(ascending=True, inplace=True)
    for i in range(min(k, n_meta)):
      match_dict = _index_to_dict(filtered_row.index[i], row)
      closest_meta = match_dict[match_metadata]
      if closest_meta == idx_meta:
        if by_match_metadata:
          correct[idx_meta].append((idx_dict, match_dict))
        else:
          correct.append((idx_dict, match_dict))
      else:
        if by_match_metadata:
          mismatch[idx_meta].append((idx_dict, match_dict))
        else:
          mismatch.append((idx_dict, match_dict))
  return correct, mismatch


def _convert_pairs_to_df(pair_list):
  """Converts list of pairs to dataframe.

  Args:
    pair_list: A list of (actual, predicted) pairs of metadata dictionaries.

  Returns:
    Two DataFrames recording the metadata of actual and predicted samples,
    respectively. Each row of the DataFrame represents a sample, and each column
    represents a metadata.
  """
  actual = [pairs[0] for pairs in pair_list]
  classified = [pairs[1] for pairs in pair_list]
  return pd.DataFrame(actual), pd.DataFrame(classified)


def get_confusion_matrix(correct, mismatch, match_metadata,
                         match_metadata_values):
  """Gets confusion matrix.

  Args:
    correct: A list of correct (actual, predicted) pairs of metadata
      dictionaries.
    mismatch: A list of incorrect (actual, predicted) pairs of metadata
      dictionaries.
    match_metadata: String, metadata that we would like to match.
    match_metadata_values: List of match_metadata values to index the confusion
      matrix.

  Returns:
    A 2-D NumPy array of the confusion matrix.
  """
  actual_df_from_correct, classified_df_from_correct = _convert_pairs_to_df(
      correct)
  actual_df_from_mismatch, classified_df_from_mismatch = _convert_pairs_to_df(
      mismatch)
  actual_df = pd.concat([actual_df_from_correct, actual_df_from_mismatch])
  classified_df = pd.concat(
      [classified_df_from_correct, classified_df_from_mismatch])
  confusion_matrix = metrics.confusion_matrix(
      actual_df[match_metadata].values,
      classified_df[match_metadata].values,
      labels=match_metadata_values)
  return confusion_matrix


def make_knn_moa_dataframe(means, max_k=4):
  """Make a dataframe of k-NN classification accuracy for MOA.

  Args:
    means: Pandas dataframe computed from a dataframe of embedding vectors by
      aggregating the cell-level embedding vectors to a higher level (e.g.,
      batch-level) averaged embedding vectors
    max_k: (optional) An integer giving the maximum number of neighbors under
      consideration in k-NN

  Returns:
    A Pandas dataframe consisting of the k-NN classification accuracy.  Each row
      represents a record of the accuracy.
  """
  dist = distance_analysis.matrix(distance.cosine, means)
  correct_nsc_list, mismatch_nsc_list, accuracy_nsc_list = [], [], []
  (correct_nsc_nsb_list, mismatch_nsc_nsb_list,
   accuracy_nsc_nsb_list) = [], [], []
  for k in range(1, max_k + 1):
    correct_nsc, mismatch_nsc = k_nearest_neighbors(
        dist, k, not_same_compound_filter)
    correct_nsc_nsb, mismatch_nsc_nsb = k_nearest_neighbors(
        dist, k, not_same_compound_or_batch_filter)
    correct_nsc_list.append(len(correct_nsc))
    mismatch_nsc_list.append(len(mismatch_nsc))
    accuracy_nsc_list.append(
        round(100.0 * len(correct_nsc) / (len(correct_nsc) + len(mismatch_nsc)),
              1))
    correct_nsc_nsb_list.append(len(correct_nsc_nsb))
    mismatch_nsc_nsb_list.append(len(mismatch_nsc_nsb))
    accuracy_nsc_nsb_list.append(
        round(
            100.0 * len(correct_nsc_nsb) /
            (len(correct_nsc_nsb) + len(mismatch_nsc_nsb)), 1))
  dict_knn = {
      CORRECT_NSC: correct_nsc_list,
      MISMATCH_NSC: mismatch_nsc_list,
      ACCURACY_NSC: accuracy_nsc_list,
      CORRECT_NSC_NSB: correct_nsc_nsb_list,
      MISMATCH_NSC_NSB: mismatch_nsc_nsb_list,
      ACCURACY_NSC_NSB: accuracy_nsc_nsb_list,
  }
  return pd.DataFrame(data=dict_knn)


def make_batch_classifier_score_dataframe(df,
                                          classifier,
                                          matching_conc=False,
                                          label_to_predict=metadata.BATCH,
                                          comp_considered=None,
                                          n_fold=3,
                                          downsample=True,
                                          seed=SEED):
  """Make a dataframe of batch classifier cross-validation scores.

  The purpose of this function is to check whether batch effects have been
  successfully removed. A classifier is used to classify batches (or plates,
  etc) for each compound (or compound, concentration pair), and the
  classification is conducted in a cross-validation procedure. A significant
  drop of the classification accuracy is regarded as a SUCCESS. Note that we
  mainly care about the change of the classification accuracy instead of its
  absolute value.

  Args:
    df: Pandas dataframe with complete multi-index of metadata and embeddings
    classifier: A callable function giving the batch classifier
    matching_conc: (optional) A boolean giving whether the batch classifier is
      applied to each compound (False) or each compound, concentration pair
      (True)
    label_to_predict: (optional) A string giving the label to predict for the
      batch classifier, which can be batch, plate, etc
    comp_considered: (optional) A list of compound names that are under
      consideration for the batch classifier.  None means all compounds are
      considered
    n_fold: (optional) An integer giving the number of folds in the
      cross-validation
    downsample: (optional) A boolean giving whether downsampling is conducted if
      the number of samples exceeds the maximum sample size for each batch
      classifier
    seed: (optional) Integer random seed for generating cross-validation folds.

  Returns:
    A Pandas dataframe consisting of means and stds of cross-validation scores
      for each compound (or compound, concentration pair).  Each row represents
      a record of mean and std.
  """
  comp_list, score_mean_list, score_std_list = [], [], []
  levels = [metadata.COMPOUND]
  if matching_conc:
    conc_list = []
    levels.append(metadata.CONCENTRATION)
  for item, df in df.groupby(level=levels):
    if matching_conc:
      comp, conc = item[0], item[1]
    else:
      comp = item
    # if this compound is not under consideration
    if (comp_considered is not None) and (comp not in comp_considered):
      continue
    labels = df.index.get_level_values(level=label_to_predict).values
    # if there is only one class, don't do any classification
    if len(np.unique(labels)) == 1:
      continue
    feature = df.values
    comp_list.append(comp)
    if matching_conc:
      conc_list.append(conc)
    # if there are too many samples, do downsampling
    if downsample and (len(labels) > MAX_SAMPLE_SIZE):
      _, feature, _, labels = model_sel.train_test_split(
          feature,
          labels,
          stratify=labels,
          test_size=MAX_SAMPLE_SIZE,
          random_state=seed)
    scores = model_sel.cross_val_score(classifier, feature, labels, cv=n_fold)
    score_mean, score_std = round(np.mean(scores), 3), round(np.std(scores), 3)
    score_mean_list.append(score_mean)
    score_std_list.append(score_std)
    dict_score = {
        metadata.COMPOUND: comp_list,
        SCORE_MEAN: score_mean_list,
        SCORE_STD: score_std_list
    }
    if matching_conc:
      dict_score[metadata.CONCENTRATION] = conc_list
  return pd.DataFrame(data=dict_score)
