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

"""Utilities for the spectre paper."""

from collections.abc import Mapping, Sequence
import operator
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy.stats
import sklearn.manifold
import statsmodels.stats.multitest
import tree

from covid_vhh_design import helper

# High-level annotations.
PARENT = 'Parent'
BASELINE = 'Baseline'
SHUFFLED = 'Shuffled'
ML = 'ML'
RANDOM = 'Random'

# Detailed model annotation
CNN = 'CNN'
LGB = 'LGB'
DBR = 'DBR'
VAE = 'VAE'
VAE_RANDOM = 'VAE (random)'

AMINO_ACIDS = (
    'A',
    'R',
    'N',
    'D',
    'C',
    'E',
    'Q',
    'G',
    'H',
    'I',
    'L',
    'K',
    'M',
    'F',
    'P',
    'S',
    'T',
    'W',
    'Y',
    'V',
)


class ProteinVocab:
  """Protein vocabulary."""

  def __init__(self):
    indices = range(len(AMINO_ACIDS))
    self._aa_to_index = dict(zip(AMINO_ACIDS, indices))
    self._index_to_aa = dict(zip(indices, AMINO_ACIDS))

  def encode(self, sequence):
    """Integer-encodes an amino acid sequence."""
    return [self._aa_to_index[a] for a in sequence]

  def decode(self, indices):
    """Decodes indices to a list of amino acids."""
    return [self._index_to_aa[i] for i in indices]


def get_imgt_mapping(parent_df):
  """Returns a dictionary mapping from amino acid index to IMGT numbering."""
  return dict(zip(parent_df['ipos'], parent_df['pos']))


def extract_parent_df(df):
  return df[df['source_num_mutations'] == 0]


def sequence_has_invalid_mutations(
    seq,
    parent_seq,
    pos_to_imgt,
    allowed_pos,
):
  """Checks whether `seq` only mutates `parent_seq` at allowed positions."""
  if len(seq) != len(parent_seq):
    return True
  return any(
      seq_aa != parent_aa and pos_to_imgt[ipos] not in allowed_pos
      for ipos, (seq_aa, parent_aa) in enumerate(zip(seq, parent_seq))
  )


def get_source_annotations(
    dfs_by_round, column
):
  """Returns the mapping from a source sequence to its corresponding group.

  This is necessary since the same sequence was sometimes annotated differently
  across rounds (e.g., "MBO" in the first library, and then "best_prior" in the
  second). When this occurs, the final annotation is chosen to be the annotation
  from the earliest library ("MBO" for the previous example.)

  Args:
    dfs_by_round: Dictionary mapping a round to the corresponding raw alphaseq
      measurements.
    column: Column defining the annotation, such as 'source_group' or
      'source_std_group'.

  Returns:
    A dictionary mapping each source sequence to its canonical annotation.
  """
  return (
      pd.concat([dfs_by_round[i].assign(round=i) for i in dfs_by_round])
      .sort_values(by='round', ascending=True)
      .groupby('source_seq', sort=False)[column]
      .first()
      .to_dict()
  )


def annotate_alphaseq_data(
    raw_data,
    parent_seq,
    parent_df,
    allowed_pos,
    model_by_group,
    design_by_group,
    target_short_name_mapping,
):
  """Augments dataframe with metadata about the nanobody sequences.

  Args:
    raw_data: Mapping of round index to pd.DataFrame of all measurements in the
      round.
    parent_seq: The parent sequence.
    parent_df: pd.DataFrame of the parent sequence in long form.
    allowed_pos: IMGT positions where mutations are allowed.
    model_by_group: Mapping from source_group to ML model annotations.
    design_by_group: Mapping from source_group to design design annotations
    target_short_name_mapping: Mapping from target_name to target abbreviation.

  Returns:
    The `df` dataframe with a column indicating which sequences have invalid
    mutations.
  """

  def fix_singles(df):
    idx = df['source_num_mutations'] == 1
    df.loc[idx, 'source_group'] = 'singles'
    df.loc[idx, 'source_std_group'] = 'singles'
    return df

  raw_data = tree.map_structure(fix_singles, raw_data)

  group_annotations = get_source_annotations(raw_data, 'source_group')
  std_group_annotations = get_source_annotations(raw_data, 'source_std_group')

  pos_to_imgt = get_imgt_mapping(parent_df)

  def has_invalid_mutations(x):
    if pd.isna(x):
      return True
    return sequence_has_invalid_mutations(
        x, parent_seq, pos_to_imgt, allowed_pos=allowed_pos
    )

  def _update_series(series, mapping):
    return helper.map_values(series, mapping)

  def _annotate_df(df):
    return df.assign(
        has_invalid_mutation=df['source_seq'].apply(has_invalid_mutations),
        new_source_group=helper.map_values(df['source_seq'], group_annotations),
        new_source_std_group=helper.map_values(
            df['source_seq'], std_group_annotations
        ),
        source_model=(
            lambda df: _update_series(df['new_source_group'], model_by_group)
        ),
        source_design=(
            lambda df: _update_series(df['new_source_group'], design_by_group)
        ),
        target_short_name=helper.map_values(
            df['target_name'], target_short_name_mapping
        ),
    )

  return tree.map_structure(_annotate_df, raw_data)


def standardize_affinities(
    values, with_mean = True, with_std = True
):
  """Standardizes dataset affinities."""
  values = np.asarray(values)
  is_inf = np.isinf(values)
  if is_inf.all():
    return values
  values[~is_inf] = sklearn.preprocessing.scale(
      values[~is_inf], with_mean=with_mean, with_std=with_std
  )
  return values


def standardize_experimental_replicates(raw_df):
  """Normalization of binding values across experimental replicas.

  Binding values are normalized by the binding's mean and standard deviation
  computed per experimental replica (`replica`) and target.

  Args:
    raw_df: pd.DataFrame where each row corresponds to one binding measurement.
      Has columns `target_name`, `replica`, and `value`.

  Returns:
    pd.DataFrame with the same columns and length as `df`, where the `value`
    column has been normalized across replicas and targets.
  """
  groupby = ['target_name', 'replica']
  if 'round' in raw_df.columns:
    groupby.append('round')
  return raw_df.assign(
      value=raw_df.groupby(groupby)['value'].transform(
          standardize_affinities, with_mean=True, with_std=True
      )
  )


def standardize_by_parent(raw_df):
  """Robust normalization of binding by parent binding per target and replica.

  For each target, we compute the median binding and IQR for the parent, and
  normalize the bindings of all sequences by these values:
    new_binding = (binding - median[parent]) / iqr[parent].

  Args:
    raw_df: pd.DataFrame where each row corresponds to one binding measurement.
      Has columns `target_name`, `replica`, `source_num_mutations`, and `value`.

  Returns:
    pd.DataFrame with the same columns and length as `df`, where the `value`
    column has been normalized by the parent binding values.
  """
  parent_df = extract_parent_df(raw_df)
  groupby = ['target_name', 'replica']
  if 'round' in raw_df.columns:
    groupby.append('round')

  parent_stats = parent_df.groupby(groupby)['value'].agg(
      ['median', scipy.stats.iqr]
  )

  agg_df = helper.safe_merge(
      raw_df, parent_stats.reset_index(), how='left', on=groupby
  )
  agg_df.loc[agg_df['iqr'] == 0, 'iqr'] = 1.0
  agg_df['value'] = (agg_df['value'] - agg_df['median']) / agg_df['iqr']
  return agg_df.drop(columns=['median', 'iqr'])


def aggregate_affinities(raw_df):
  """Aggregates all measurements of a sequence-target pair into a single scalar.

  This proceeds in three steps:
  1. All bindings to a target are normalized (by mean and standard deviation)
     across experimental replicates, such that experimental replica `i` has mean
     0 and standard deviation 1.
  2. All bindings to a target are robustly normalized by the parent binding to
     the target (per replica), such that the parent binding has median binding 0
     and IQR 1.
  3. Finally, we compute the median binding between a sequence and its target.

  Args:
    raw_df: pd.DataFrame of binding measurements, where each row corresponds to
      an individual binding value. Must contain at least columns `source_seq`,
      `target_name`, `value`, `source_num_mutations`, `replica`.

  Returns:
    pd.DataFrame of aggregated binding values. Each row corresponds to a unique
    sequence-target pair, and the parent binding has been normalized to zero.
  """
  return (
      raw_df.pipe(standardize_experimental_replicates)
      .pipe(standardize_by_parent)
      .groupby(['source_seq', 'target_name'])['value']
      .median()
      .reset_index()
  )


def filter_sequences_with_min_replicas(
    raw_df, min_replicas
):
  """Filters out sequences with an insufficient number of binding measurements."""
  if raw_df['target_name'].nunique() != 1:
    raise ValueError('DataFrame contains more than one target!')
  measurements_by_seq = raw_df.groupby('source_seq')['value'].count()
  valid_seqs = measurements_by_seq[measurements_by_seq >= min_replicas].index
  return raw_df[raw_df['source_seq'].isin(valid_seqs)].copy()


def _compute_pvalues_by_target(
    raw_df,
    min_replicas,
    correction_method,
    alpha,
):
  """Mann-Whitney U-test and FDR correction for a single target."""
  if raw_df['target_name'].nunique() != 1:
    raise ValueError(
        'Cannot compute pvalues of binding measurements to multiple targets!'
    )
  parent_values = extract_parent_df(raw_df)['value'].values

  def _compute_pvalue(x):
    return scipy.stats.mannwhitneyu(x, parent_values, alternative='less').pvalue

  pvalues = (
      filter_sequences_with_min_replicas(raw_df, min_replicas)
      .groupby('source_seq')['value']
      .agg(_compute_pvalue)
      .rename('pvalue')
      .reset_index()
  )

  if not pvalues.empty:
    _, pvalues['pvalue_corrected'], _, _ = (
        statsmodels.stats.multitest.multipletests(
            pvalues['pvalue'],
            alpha=alpha,
            method=correction_method,
        )
    )
  return pvalues


def compute_pvalues(
    raw_df,
    min_replicas,
    correction_method = 'bonferroni',
    alpha = 0.05,
):
  """Mann-Whitney U-test and p-value correction on all targets.

  Args:
    raw_df: pd.DataFrame that has at least `source_seq`, `target_name`,
      `source_num_mutations`, and `value` columns. The `target_name` column must
      take a unique value.
    min_replicas: Number of measurements required for a (VHH, target) pair to
      compute a p-value.
    correction_method: Method used for p-value correction. Must be a valid
      `method` argument of statsmodels.stats.multitest.multipletests, including
      "fdr_bh", "fdr_by", and "bonferroni".
    alpha: Family-wise error rate used for the p-value correction.

  Returns:
    A pd.DataFrame with `source_seq`, `target_name`, `pvalue`, and
    `pvalue_corrected` columns.
  """
  raw_df = standardize_experimental_replicates(raw_df)
  return (
      raw_df.groupby('target_name')
      .apply(
          _compute_pvalues_by_target,
          min_replicas=min_replicas,
          correction_method=correction_method,
          alpha=alpha,
      )
      .reset_index()
      .drop(columns='level_1')
  )


def get_metadata(raw_df):
  """Returns metadata dataframe for sequence-target pairs."""
  counts = raw_df.groupby(['source_seq', 'target_name']).nunique()
  meta_cols = counts.columns[(counts <= 1).all()]
  metadata = (
      raw_df.groupby(['source_seq', 'target_name'])[meta_cols]
      .agg('first')
      .reset_index()
  )
  return metadata.assign(
      source_num_mutations=metadata['source_num_mutations'].astype('int')
  )


def join_aggregate_data_with_values(
    metadata_df,
    agg_df = None,
    pvalues_df = None,
):
  """Creates a unified dataframe of metadata, aggregated bindings, and pvalues.

  Args:
    metadata_df: pd.DataFrame of metadata for each source/target pair.
    agg_df: pd.DataFrame of aggregated bindings, one per source-target pair.
    pvalues_df: pd.DataFrame with pvalues and corrected_pvalues for each
      source-target pair.

  Returns:
    A pd.DataFrame containing metadata, aggregated bindings, and pvalues for
    each source-target pair (one per row).
  """
  groupby_cols = ['source_seq', 'target_name']
  df = metadata_df
  if agg_df is not None:
    df = helper.safe_merge(df, agg_df, on=groupby_cols)
  if pvalues_df is not None:
    df = df.merge(pvalues_df, on=groupby_cols, how='left')
  return df


def aggregate_over_rounds(raw_data):
  """Computes normalized log KDs and pvalues across all experimental rounds.

  Normalized log KDs and pvalues are computed similarly to the computations for
  single-round experiments. The difference is that measurements are median
  aggregates *across all rounds* (and experimental replicas) after standardizing
  them by a) experimental replicas, and 2) the parent sequence *per round*. All
  measurements across rounds are used when computing the pvalues.

  Args:
    raw_data: Mapping from round to raw alphaseq data (potentially after
      filtering out invalid measurements) for different rounds.

  Returns:
    pd.DataFrame containing a unique measurement per (sequence, target) pair,
    pvalues, corrected pvalues, and "round" columns.
  """
  dfs = [raw_data[round].assign(round=round) for round in raw_data]
  raw_df = pd.concat(dfs, ignore_index=True)

  metadata = get_metadata(raw_df)
  normalized_log_kds = aggregate_affinities(raw_df)
  round_by_seq = raw_df.groupby('source_seq')['round'].min().to_dict()
  all_rounds_df = join_aggregate_data_with_values(
      metadata_df=metadata,
      agg_df=normalized_log_kds,
      pvalues_df=None,  # P-values are round-specific.
  )
  return all_rounds_df.assign(
      round=all_rounds_df['source_seq'].map(round_by_seq)
  )


def compute_hit_rate(
    agg_df,
    groupby,
    thresholds,
    measurement_col,
    lesser_than = True,
):
  """Computes the hit rate, grouped by user-defined columns.

  A "hit" is defined as a sequence whose binding value improves over the
  parent's binding value by a fixed amount.

  Args:
    agg_df: pd.DataFrame of aggregated measurements (only one measurement per
      nanobody-target) pair. Must contain at least columns 'target_name',
      'value', and all elements fo `groupby`.
    groupby: Column, or list of columns, by which to group the hit rates.
    thresholds: Value(s) defining a "hit".
    measurement_col: Column reporting measurements that can be "hits".
    lesser_than: Whether a hit is defined by the measurement being lesser than
      the threshold(s). Defaults to True.

  Returns:
    pd.DataFrame whose columns are 'hit_rate' and the elements of `groupby`.
  """
  groupby = helper.to_list(groupby)
  thresholds = helper.to_list(thresholds)

  op = operator.le if lesser_than else operator.gt
  hit_dfs = []
  for threshold in thresholds:
    hit_dfs.append(
        agg_df.assign(hit=op(agg_df[measurement_col], threshold))
        .groupby(groupby)['hit']
        .agg(['mean', 'sum', 'count'])
        .reset_index()
        .assign(threshold=threshold)
    )

  return pd.concat(hit_dfs, ignore_index=True)


def extract_best_sequences_by_category(
    df,
    num_top_seqs,
    value_col,
    category_col,
):
  """Returns n best sequences for each value of `category_col`.

  Best sequences are sequences with the smallest values in the 'value' column.

  Args:
    df: pd.DataFrame with at least 'source_seq', category_col, and 'value'
      columns.
    num_top_seqs: Number of sequences to select per category.
    value_col: Column used to rank sequences.
    category_col: Column of `df`; `n` sequences are chosen for each unique value
      of `category_col`.

  Returns:
    np.ndarray of `num_top_seqs * df[category_col].nunique()` sequences.
  """
  df = df.set_index('source_seq')
  best_by_category = df.groupby(category_col)[value_col].nsmallest(
      n=num_top_seqs
  )
  return best_by_category.reset_index()['source_seq'].unique()


def find_multihits(agg_df, how):
  """Returns dataframe where each row describes bound targets per sequence.

  Computes a binary dataframe where rows correspond to sequences and columns to
  targets; `1`s indicate a positive binding event.

  Args:
    agg_df: DataFrame of binding values, one row per (VHH, target) pair.
    how: One of ['iqr', 'pvalue']. If 'iqr', a positive binding event
      corresponds to an improvement of at least 1 IQR over parent binding. If
      `pvalue`, a positive binding event corresponds to a (corrected) p-value
      smaller than 0.05.

  Returns:
    A pd.DataFrame of binary binding events.
  """
  if how == 'iqr':
    value_col = 'value'
    threshold = helper.get_unique_value(extract_parent_df(agg_df)['value']) - 1
  elif how == 'pvalue':
    value_col = 'pvalue_corrected'
    threshold = 0.05
  else:
    raise ValueError(f'"how" must be one of ["iqr", "pvalue"], but was {how}.')

  hits_df = agg_df.pivot(
      index='source_seq', columns='target_name', values=value_col
  )
  hits_df = (hits_df <= threshold).astype(int)

  return hits_df


def _as_list(x):
  return list(x) if isinstance(x, str) else x


def hamming_distance(
    x, y, normalize = False
):
  """The Hamming distance between two iterables of same length.

  Converts `x`, `y` to iterables if they are of type string, then computes the
  standard Hamming distance between sequences. If `normalize` is set to `True`,
  returns a number in [0,1] corresponding to the Hamming distance divided by the
  sequence length.

  Args:
    x: A single-depth iterable of characters or integers.
    y: A single-depth iterable of characters or integers.
    normalize: Whether to normalize the distance by sequence length.

  Returns:
    The Hamming distance between `x` and `y`.

  Raises:
    ValueError if `x` and `y` are of different lengths.
  """
  if len(x) != len(y):
    raise ValueError('Sequences of different lengths')
  x = _as_list(x)
  y = _as_list(y)
  normalized_distance = distance.hamming(x, y)
  if normalize:
    return normalized_distance
  else:
    return int(round(normalized_distance * len(x)))


def avg_distance_to_set(seq, other_seqs):
  """Computes average Hamming distance of a sequence to a set of sequences."""
  vocab = ProteinVocab()
  if not other_seqs.size:
    return 0

  return np.mean(
      [
          hamming_distance(
              vocab.encode(seq), vocab.encode(other_seq), normalize=False
          )
          for other_seq in other_seqs
      ]
  )


def compute_tsne_embedding(agg_df):
  """Returns a copy of the dataframe with t-sne sequence embeddings."""
  if agg_df['source_seq'].nunique() != len(agg_df):
    raise ValueError('Dataframe contains duplicate sequences!')
  vocab = ProteinVocab()
  structures = np.vstack(agg_df['source_seq'].apply(vocab.encode))
  latents = sklearn.manifold.TSNE(
      metric='hamming', random_state=0
  ).fit_transform(structures)
  return agg_df.assign(latents_x=latents[:, 0], latents_y=latents[:, 1])
