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

# coding=utf-8
# Copyright 2020 The Many Constraints Neurips 2020 Authors.
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
"""Cross-group ranking fairness experiments with per-query constraints."""

import math
import random
import sys

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_constrained_optimization as tfco

FLAGS = flags.FLAGS
flags.DEFINE_string('save_to_dir', 'tmp/', 'save the output to the path')
flags.DEFINE_string('input',
                    'https://www.microsoft.com/en-us/research/project/mslr/',
                    'path to unzipped MSLR-WEB10K	input file')
flags.DEFINE_string('prefix', 'sim', 'identicating prefix of saved files')
flags.DEFINE_float('learning_rate', 0.05, 'learning rate')
flags.DEFINE_integer('loops', 100, 'num of loops')
flags.DEFINE_integer('train_size', 1000, 'num of training queries')
flags.DEFINE_integer('microloops', 1000, 'num of microloops within a loop')
flags.DEFINE_string('optimizer', 'adagrad', 'optimizer')
flags.DEFINE_string('constraint_type', 'cross_group_equal_opportunity',
                    'constraint_type')
flags.DEFINE_integer('id', 0, 'variable for manual parallelism')
flags.DEFINE_string('type', 'unc', 'unc, tfco or new')


def pair_pos_neg_docs(data, max_num_pairs=10000, max_query_bandwidth=20):
  """Returns pairs of positive-negative docs from given DataFrame."""

  # Include a row number
  data.insert(0, 'tmp_row_id', list(range(data.shape[0])))

  # Separate pos and neg docs.
  pos_docs = data[data.label == 1]
  if pos_docs.empty:
    return
  neg_docs = data[data.label == 0]
  if neg_docs.empty:
    return

  # Include a merge key.
  pos_docs.insert(0, 'merge_key', 0)
  neg_docs.insert(0, 'merge_key', 0)

  # Merge docs and drop merge key column.
  pairs = pos_docs.merge(
      neg_docs, on='merge_key', how='outer', suffixes=('_pos', '_neg'))
  pairs = pairs[np.abs(pairs['tmp_row_id_pos'] -
                       pairs['tmp_row_id_neg']) <= max_query_bandwidth]
  pairs.drop(
      columns=['merge_key', 'tmp_row_id_pos', 'tmp_row_id_neg'], inplace=True)
  if pairs.shape[0] > max_num_pairs:
    pairs = pairs.sample(n=max_num_pairs, axis=0, random_state=543210)
  return pairs


def convert_labeled_to_paired_data(data_dict,
                                   index=None,
                                   max_num_pairs=200,
                                   max_query_bandwidth=200):
  """Convert data arrays to pandas DataFrame with required column names."""
  features = data_dict['features']
  labels = data_dict['labels']
  groups = data_dict['groups']
  queries = data_dict['queries']

  if index is not None:
    data_df = pd.DataFrame(features[queries == index, :])
    data_df = data_df.assign(label=pd.DataFrame(labels[queries == index]))
    data_df = data_df.assign(group=pd.DataFrame(groups[queries == index]))
    data_df = data_df.assign(query_id=pd.DataFrame(queries[queries == index]))
  else:
    data_df = pd.DataFrame(features)
    data_df = data_df.assign(label=pd.DataFrame(labels))
    data_df = data_df.assign(group=pd.DataFrame(groups))
    data_df = data_df.assign(query_id=pd.DataFrame(queries))

  def pair_pos_neg_docs_helper(x):
    return pair_pos_neg_docs(
        x, max_num_pairs=max_num_pairs, max_query_bandwidth=max_query_bandwidth)

  # Forms pairs of positive-negative docs for each query in given DataFrame
  # if the DataFrame has a query_id column. Otherise forms pairs from all rows
  # of the DataFrame.
  data_pairs = data_df.groupby('query_id').apply(pair_pos_neg_docs_helper)

  # Create groups ndarray.
  pos_groups = data_pairs['group_pos'].values.reshape(-1, 1)
  neg_groups = data_pairs['group_neg'].values.reshape(-1, 1)
  group_pairs = np.concatenate((pos_groups, neg_groups), axis=1)

  # Create queries ndarray.
  queries = data_pairs['query_id_pos'].values.reshape(-1,)

  # Create features ndarray.
  feature_names = data_df.columns
  feature_names = feature_names.drop(['query_id', 'label'])
  feature_names = feature_names.drop(['group'])

  pos_features = data_pairs[[str(s) + '_pos' for s in feature_names]].values
  pos_features = pos_features.reshape(-1, 1, len(feature_names))

  neg_features = data_pairs[[str(s) + '_neg' for s in feature_names]].values
  neg_features = neg_features.reshape(-1, 1, len(feature_names))

  features_pairs = np.concatenate((pos_features, neg_features), axis=1)

  # Paired data dict.
  paired_data = {
      'features': features_pairs,
      'groups': group_pairs,
      'queries': queries,
      'dimension': data_dict['dimension'],
      'num_queries': data_dict['num_queries']
  }

  return paired_data


def group_tensors(predictions,
                  groups,
                  pos_group,
                  neg_group=None,
                  queries=None,
                  query_index=None):
  """Select pairs based on given groups and queries."""

  # Returns predictions and labels for document-pairs belonging to query_index
  # (if specified) where the protected group for the positive document is
  # pos_group, and the protected group for the negative document (if specified)
  # is neg_group; and also the group mask.

  def group_mask():
    mask = np.reshape(get_mask(groups(), pos_group, neg_group), (-1))
    if (queries is not None) and (query_index is not None):
      mask = mask & (queries() == query_index)
    return mask

  def group_labels():
    return tf.constant(np.ones(np.sum(group_mask())), dtype=tf.float32)

  def group_predictions():
    return tf.boolean_mask(predictions(), group_mask())

  return group_predictions, group_labels


def add_query_mean(x, z=None, columns=None, queries=None):
  """Create query level features as the averages of its document features."""
  x = np.array(x)
  nrow = x.shape[0]
  if columns is not None:
    x_ = np.copy(x[:, columns])
  else:
    x_ = np.copy(x)
  if z is not None:
    # for concatenating grouping
    x = np.concatenate((x, z), axis=1)
  if queries is None:
    return np.concatenate((
        x,
        np.tile(np.mean(x_, axis=0), (nrow, 1)),
    ), axis=1)
  else:
    y_ = np.zeros(x_.shape)
    for query in np.unique(queries):
      query_mask = (queries == query)
      y_[query_mask, :] = np.mean(x_[query_mask, :], axis=0)
    return np.concatenate((x, y_), axis=1)


def add_query_median(x, columns=None, queries=None):
  """Create query level features as the medians of its document features."""
  x = np.array(x)
  nrow = x.shape[0]
  if columns is not None:
    x_ = x[:, columns]
  else:
    x_ = x
  if queries is None:
    return np.concatenate((
        x,
        np.tile(np.median(x_, axis=0), (nrow, 1)),
    ),
                          axis=1)
  else:
    y_ = np.zeros(x_.shape)
    for query in np.unique(queries):
      query_mask = (queries == query)
      y_[query_mask, :] = np.median(x_[query_mask, :], axis=0)
    return np.concatenate((x, y_), axis=1)


def get_mask(groups, pos_group, neg_group=None):
  """Returns a boolean mask selecting positive negative document pairs."""
  # Returns a boolean mask selecting positive-negative document pairs where
  # the protected group for  the positive document is pos_group and
  # the protected group for the negative document (if specified) is neg_group.
  # Repeat group membership positive docs as many times as negative docs.
  mask_pos = groups[:, 0] == pos_group

  if neg_group is None:
    return mask_pos
  else:
    mask_neg = groups[:, 1] == neg_group
    return mask_pos & mask_neg


def error_rate(model, dataset):
  """Returns error rate for Keras model on dataset."""
  d = dataset['dimension']
  scores = np.squeeze(model.predict(dataset['features'][:, :, 0:d]), axis=-1)
  diff = scores[:, 0] - scores[:, 1]
  return np.mean(diff.reshape((-1)) <= 0)


def group_error_rate(model, dataset, pos_group, neg_group=None):
  """Returns error rate for Keras model on data set."""
  # Returns error rate for Keras model on data set, considering only document
  # pairs where the protected group for the positive document is pos_group, and
  # the protected group for the negative document (if specified) is neg_group.
  d = dataset['dimension']
  scores = np.squeeze(model.predict(dataset['features'][:, :, 0:d]), axis=-1)
  mask = get_mask(dataset['groups'], pos_group, neg_group)
  diff = scores[:, 0] - scores[:, 1]
  diff = diff[mask > 0].reshape((-1))

  queries = dataset['queries']
  unique_qids = np.unique(queries)
  masked_queries = queries[mask > 0]
  query_group_errors = np.zeros(len(unique_qids))
  for qid in unique_qids:
    masked_queries_qid = (masked_queries == qid)
    if np.any(masked_queries_qid):
      query_group_errors[qid] = np.mean(
          diff[masked_queries_qid] < 0) + 0.5 * np.mean(
              diff[masked_queries_qid] == 0)
    else:
      query_group_errors[qid] = 0
  return np.mean(diff < 0) + 0.5 * np.mean(diff == 0), query_group_errors


# nDCG
def dcg(labels, at=None):
  """Compute DCG given labels in order."""
  result = 0.0
  position = 2
  for i in labels:
    if i != 0:
      result += 1 / math.log2(position)
    position += 1
    if at is not None and (position >= at + 2):
      break
  return result


def ndcg(labels, at=None):
  """Compute nDCG given labels in order."""
  return dcg(labels, at=at) / dcg(sorted(labels, reverse=True), at=at)


# A faster 'get_error_rate'
def error_rate_lambda(dataset):
  """Returns error rate for Keras model on data set."""
  pos_row_id = np.array(dataset['labels']) == 1
  neg_row_id = ~pos_row_id
  row_numbers = np.array(range(len(dataset['labels'])))
  pos_data = pd.DataFrame({
      'row_ids': row_numbers[pos_row_id],
      'labels': np.array(dataset['labels'])[pos_row_id],
      'queries': np.array(dataset['queries'])[pos_row_id],
      'groups': np.array(dataset['groups'])[pos_row_id]
  })
  neg_data = pd.DataFrame({
      'row_ids': row_numbers[neg_row_id],
      'labels': np.array(dataset['labels'])[neg_row_id],
      'queries': np.array(dataset['queries'])[neg_row_id],
      'groups': np.array(dataset['groups'])[neg_row_id]
  })
  pairs = pos_data.merge(
      neg_data, on='queries', how='outer', suffixes=('_pos', '_neg'))

  def error_rate_helper(model, groups=None, at=10):
    preds = np.squeeze(
        model.predict(dataset['features'][:, 0:dataset['dimension']]), axis=-1)
    error = (np.array(
        preds[pairs['row_ids_pos']] < preds[pairs['row_ids_neg']])) + 0.5 * (
            np.array(
                preds[pairs['row_ids_pos']] == preds[pairs['row_ids_neg']]))
    # error rate
    error_rate_ = np.nan_to_num(np.mean(error))
    group_error_rate_ = []
    index_ = []
    # group_error_rate
    for g0, g1 in groups:
      if g1 is None:
        index_ = (pairs['groups_pos'] == g0)
      else:
        index_ = (pairs['groups_pos'] == g0) & (pairs['groups_neg'] == g1)
      group_error_rate_.append(np.nan_to_num(np.mean(error[index_])))

    # query_error_rate and query ndcg
    query_error_rate = []
    query_ndcg = []
    for query_id in np.unique(dataset['queries']):
      # query error rate
      query_index_ = (pairs['queries'] == query_id)
      query_error_rate_ = []
      for g0, g1 in groups:
        if g1 is None:
          index_ = (pairs['groups_pos'] == g0) & query_index_
        else:
          index_ = ((pairs['groups_pos'] == g0) & (pairs['groups_neg'] == g1)
                    & query_index_)
        query_error_rate_.append(np.nan_to_num(np.mean(error[index_])))
      query_error_rate.append(query_error_rate_)

      # ndcg
      query_index_ = (dataset['queries'] == query_id)
      query_pred_ = preds[query_index_]
      query_labels = np.array(dataset['labels'])[query_index_]
      query_ndcg.append(ndcg(query_labels[np.argsort(-query_pred_)], at=at))

    return error_rate_, np.array(group_error_rate_), np.array(
        query_error_rate), np.array(query_ndcg)

  return error_rate_helper


def create_ranking_model(features, dimension):
  """Construct the ranking model."""
  layers = []
  layers.append(tf.keras.Input(shape=(dimension,)))
  layers.append(
      tf.keras.layers.Dense(
          128, use_bias=True, bias_initializer='ones', activation='relu'))
  layers.append(tf.keras.layers.Dense(1, use_bias=False))
  ranking_model = tf.keras.Sequential(layers)

  def predictions():
    predicted_scores = ranking_model(features()[:, :, 0:dimension])
    predicted_scores = tf.squeeze(predicted_scores, axis=-1)
    return predicted_scores[:, 0] - predicted_scores[:, 1]

  return ranking_model, predictions


def create_multipliers_model(features, dimension, num_constraints):
  """Construct the multiplier model."""
  layers = []
  layers.append(tf.keras.Input(shape=(dimension,)))
  layers.append(
      tf.keras.layers.Dense(
          64, use_bias=True, bias_initializer='ones', activation='tanh'))
  layers.append(tf.keras.layers.Dense(num_constraints, bias_initializer='ones'))
  multiplier_model = tf.keras.Sequential(layers)

  def multipliers():
    batch = features()[:, 0, (-dimension):].reshape(-1, dimension)
    batch = np.mean(batch, axis=0).reshape(-1, dimension)
    multiplier_scores = multiplier_model(batch)
    return multiplier_scores

  return multiplier_model, multipliers


def formulate_problem(features,
                      groups,
                      labels,
                      dimension,
                      constraint_groups,
                      constraint_slack=None):
  """Formulates a constrained problem."""
  #   Formulates a constrained problem that optimizes the error rate for a linear

  #   model on the specified dataset, subject to pairwise fairness constraints
  #   specified by the constraint_groups and the constraint_slack.

  #   Args:
  #     features: Nullary function returning features
  #     groups: Nullary function returning groups
  #     labels: Nullary function returning labels
  #     dimension: Input dimension for ranking model
  #     constraint_groups: List containing tuples of the form ((pos_group0,
  #       neg_group0), (pos_group1, neg_group1)), specifying the group memberships
  #       for the document pairs to compare in the constraints.
  #     constraint_slack: slackness '\epsilon' allowed in the constraints.

  #   Returns:
  #     A RateMinimizationProblem object, and a Keras ranking model.

  # Create linear ranking model: we get back a Keras model and a nullary
  # function returning predictions on the features.
  ranking_model, predictions = create_ranking_model(features, dimension)

  # Context for the optimization objective.
  context = tfco.rate_context(predictions, labels)

  # Constraint set.
  constraint_set = []

  # Context for the constraints.
  for ((pos_group0, neg_group0), (pos_group1, neg_group1)) in constraint_groups:
    # Context for group 0.
    group0_predictions, group0_labels = group_tensors(
        predictions, groups, pos_group0, neg_group=neg_group0)
    context_group0 = tfco.rate_context(group0_predictions, group0_labels)

    # Context for group 1.
    group1_predictions, group1_labels = group_tensors(
        predictions, groups, pos_group1, neg_group=neg_group1)
    context_group1 = tfco.rate_context(group1_predictions, group1_labels)

    # Add constraints to constraint set.
    constraint_set.append(
        tfco.false_negative_rate(context_group0) <= (
            tfco.false_negative_rate(context_group1) + constraint_slack))
    constraint_set.append(
        tfco.false_negative_rate(context_group1) <= (
            tfco.false_negative_rate(context_group0) + constraint_slack))

  # Formulate constrained minimization problem.
  problem = tfco.RateMinimizationProblem(
      tfco.error_rate(context, penalty_loss=tfco.SoftmaxCrossEntropyLoss()),
      constraint_set)

  return problem, ranking_model


def evaluate_results(model, test_set, params):
  """Returns error rates and violation metrics."""
  # Returns overall, group error rates, group-level constraint violations,
  # query-level constraint violations for model on test set.
  if params['constraint_type'] == 'marginal_equal_opportunity':
    g0_error, g0_query_error = group_error_rate(model, test_set, 0)
    g1_error, g1_query_error = group_error_rate(model, test_set, 1)
    group_violations = [g0_error - g1_error, g1_error - g0_error]
    query_violations = [np.max(np.abs(g0_query_error - g1_query_error))]
    query_violations_full = [np.abs(g0_query_error - g1_query_error)]
    return (error_rate(model, test_set), [g0_error, g1_error], group_violations,
            query_violations, query_violations_full)
  else:
    g00_error, g00_query_error = group_error_rate(model, test_set, 0, 0)
    g01_error, g01_query_error = group_error_rate(model, test_set, 0, 1)
    g10_error, g10_query_error = group_error_rate(model, test_set, 1, 1)
    g11_error, g11_query_error = group_error_rate(model, test_set, 1, 1)
    group_violations_offdiag = [g01_error - g10_error, g10_error - g01_error]
    group_violations_diag = [g00_error - g11_error, g11_error - g00_error]
    query_violations_offdiag = [
        np.max(np.abs(g01_query_error - g10_query_error))
    ]
    query_violations_diag = [np.max(np.abs(g00_query_error - g11_query_error))]
    query_violations_offdiag_full = np.abs(g01_query_error - g10_query_error)
    query_violations_diag_full = np.abs(g00_query_error - g11_query_error)

    if params['constraint_type'] == 'cross_group_equal_opportunity':
      return (error_rate(model,
                         test_set), [[g00_error, g01_error],
                                     [g10_error,
                                      g11_error]], group_violations_offdiag,
              query_violations_offdiag, [query_violations_offdiag_full])
    else:
      return (error_rate(model, test_set), [[g00_error, g01_error],
                                            [g10_error, g11_error]],
              group_violations_offdiag + group_violations_diag,
              query_violations_offdiag + query_violations_diag, [
                  np.concatenate((query_violations_offdiag_full,
                                  query_violations_diag_full))
              ])


def display_results(model,
                    objectives,
                    group_violations,
                    query_violations,
                    query_violations_full,
                    query_ndcgs,
                    test_set,
                    params,
                    method,
                    error_type,
                    show_header=False,
                    show_plots=False,
                    best_index=-1,
                    suffix='',
                    metric_fn=None,
                    output_file=None,
                    plot_ax=None):
  """Prints evaluation results and plots its decision boundary."""

  # Evaluate model on test set and print results.
  if metric_fn is None:
    error, group_error, _, _, viols = evaluate_results(model, test_set, params)
  else:
    if params['constraint_type'] == 'marginal_equal_opportunity':
      valid_groups = [(0, None), (1, None)]
    elif params['constraint_type'] == 'cross_group_equal_opportunity':
      valid_groups = [(0, 1), (1, 0)]
    error, group_error, query_error, _ = metric_fn(model, valid_groups)
    viols = [np.abs(query_error[:, 0] - query_error[:, 1])]

  result = []
  if params['constraint_type'] == 'marginal_equal_opportunity':
    if show_header:
      output_file.write(
          '{:>20}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}\n'.format(
              'Method', 'Error', 'Overall', 'Group 0', 'Group 1', 'Mean Query',
              'Median Query', '90p Query', 'Max Query'))
    output_file.write(('{:>20}{:>15}{:>15.3f}{:>15.3f}{:>15.3f}{:>15.3f}' +
                       '{:>15.3f}{:>15.3f}{:>15.3f}\n').format(
                           method,
                           error_type,
                           error,
                           group_error[0],
                           group_error[1],
                           np.mean(viols[0]),
                           np.median(viols[0]),
                           np.percentile(viols[0], 90),
                           np.max(viols[0]),
                       ))
    result = [
        error, group_error[0], group_error[1],
        np.mean(viols[0]),
        np.median(viols[0]),
        np.percentile(viols[0], 90),
        np.max(viols[0])
    ]
  elif params['constraint_type'] == 'cross_group_equal_opportunity':
    if show_header:
      output_file.write(
          '{:>20}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}\n'.format(
              'Method', 'Error', 'Overall', 'Group 0/1', 'Group 1/0',
              'Mean Query', 'Median Query', '90p Query', 'Max Query'))
    if metric_fn is None:
      output_file.write(('{:>20}{:>15}{:>15.3f}{:>15.3f}{:>15.3f}{:>15.3f}' +
                         '{:>15.3f}{:>15.3f}{:>15.3f}\n').format(
                             method,
                             error_type,
                             error,
                             group_error[0][1],
                             group_error[1][0],
                             np.mean(viols[0]),
                             np.median(viols[0]),
                             np.percentile(viols[0], 90),
                             np.max(viols[0]),
                         ))
      result = [
          error, group_error[0][1], group_error[1][0],
          np.mean(viols[0]),
          np.median(viols[0]),
          np.percentile(viols[0], 90),
          np.max(viols[0])
      ]
    else:
      output_file.write(('{:>20}{:>15}{:>15.3f}{:>15.3f}{:>15.3f}{:>15.3f}' +
                         '{:>15.3f}{:>15.3f}{:>15.3f}\n').format(
                             method,
                             error_type,
                             error,
                             group_error[0],
                             group_error[1],
                             np.mean(viols[0]),
                             np.median(viols[0]),
                             np.percentile(viols[0], 90),
                             np.max(viols[0]),
                         ))
      result = [
          error, group_error[0], group_error[1],
          np.mean(viols[0]),
          np.median(viols[0]),
          np.percentile(viols[0], 90),
          np.max(viols[0])
      ]

  # Plot decision boundary and progress of training objective/constraint viol.
  if show_plots:
    if plot_ax is None:
      ff, ax = plt.subplots(1, 6, figsize=(16.0, 3.5))
    else:
      ax = plot_ax

      ax[0].set_title('Overall Error')
      ax[0].set_xlabel('Number of epochs')
      ax[0].plot(range(params['loops']), objectives)

      ax[1].set_title('Group Constraint Violation')
      ax[1].set_xlabel('Number of epochs')
      ax[1].plot(range(params['loops']), np.max(group_violations, axis=1))

      ax[2].set_title('Max% Percentile Query \nConstraint Violation per Epoch')
      ax[2].set_xlabel('Number of epochs')
      ax[2].plot(
          range(params['loops']), np.percentile(query_violations, 90, axis=1))

      ax[3].set_title('Training Final Query\nConstraint Violation')
      ax[3].set_xlabel('Constraint violation')
      ax[3].set_ylim(bottom=0, top=20)
      ax[3].hist(
          np.array(query_violations_full)[best_index, :][0],
          range=(0, 1),
          bins=20,
          density=True)

      ax[4].set_title('Testing Query \nConstraint Violation')
      ax[4].set_xlabel('Constraint violation')
      ax[4].set_ylim(bottom=0, top=20)
      ax[4].hist(np.array(viols[0]), range=(0, 1), bins=20, density=True)

      ax[5].set_title('Mean Query nDCG')
      ax[5].set_xlabel('Number of Epochs')
      ax[5].plot(range(params['loops']), query_ndcgs)

    if plot_ax is None:
      ff.tight_layout()
      plt.savefig('{}/{}_plot_{}.png'.format(FLAGS.save_to_dir, FLAGS.prefix,
                                             suffix))

  return result


def train_model(train_set, params, metric_fn=None, valid_set=None):
  """Set up problem and model."""

  # include id = 0
  np.random.seed(121212 + FLAGS.id)
  random.seed(212121 + FLAGS.id)
  tf.compat.v1.set_random_seed(123456 + FLAGS.id)

  if params['multiplier_type'] == 'unconstrained':
    # Unconstrained optimization.
    constraint_groups = []
    if params['constraint_type'] == 'marginal_equal_opportunity':
      valid_groups = [(0, None), (1, None)]
    elif params['constraint_type'] == 'cross_group_equal_opportunity':
      valid_groups = [(0, 1), (1, 0)]
  else:
    # Constrained optimization.
    if params['constraint_type'] == 'marginal_equal_opportunity':
      constraint_groups = [((0, None), (1, None))]
      valid_groups = [(0, None), (1, None)]
    elif params['constraint_type'] == 'cross_group_equal_opportunity':
      constraint_groups = [((0, 1), (1, 0))]
      valid_groups = [(0, 1), (1, 0)]
    elif params['constraint_type'] == 'custom':
      constraint_groups = params['constraint_groups']
    else:
      constraint_groups = []

  if 'multiplier_dimension' not in params:
    multiplier_dimension = train_set['features'].shape[2] - train_set[
        'dimension']
  else:
    multiplier_dimension = params['multiplier_dimension']

  # Dictionary that will hold batch features pairs, group pairs and labels for
  # current batch. We include one query per-batch.
  paired_batch = {}
  batch_index = 0  # Index of current query.

  # Data functions.
  features = lambda: paired_batch['features']
  groups = lambda: paired_batch['groups']
  labels = lambda: np.ones(paired_batch['features'].shape[0])

  # Create ranking model and constrained optimization problem.
  problem, ranking_model = formulate_problem(features, groups, labels,
                                             train_set['dimension'],
                                             constraint_groups,
                                             params['constraint_slack'])

  if (params['multiplier_type'] == 'unconstrained') or (
      params['multiplier_type'] == 'common'):
    # Unconstrained optimization or constrained optimization with a common
    # set of Lagrange multipliers for all query.

    # Create Lagrangian loss for problem with standard TFCO.
    lagrangian_loss, update_ops, multipliers_variables = (
        tfco.create_lagrangian_loss(problem, dual_scale=params['dual_scale']))
    multipliers_variables_list = [multipliers_variables]

    # All paired queries are valid
    check_train_pair = lambda _: True
  else:
    # Constrained optimization with feature-dependent multiplier, or with
    # per-query multipliers, i.e. separate set of multipliers per each query.
    if params['multiplier_type'] == 'feature_dependent':
      # Create multipliers model.
      print('Creating multiplier model with {} features.'.format(
          multiplier_dimension))
      multiplier_model, multipliers = create_multipliers_model(
          features, multiplier_dimension, problem.num_constraints)
      multipliers_variables_list = multiplier_model.trainable_weights
      check_train_pair = lambda x: np.unique(x['groups'], axis=0).shape[0] >= 4
    elif params['multiplier_type'] == 'per-query':
      # Create separate set of multipliers per query.
      multipliers_variables = tf.Variable(
          np.ones((train_set['num_queries'], problem.num_constraints)),
          dtype=tf.float32)

      def multipliers():
        return tf.reshape(multipliers_variables[batch_index, :], (-1,))

      multipliers_variables_list = [multipliers_variables]
      check_train_pair = lambda _: True
    else:
      raise ValueError('Invalid multiplier type')

    # Create Lagrangian loss with multipliers defined above.
    def lagrangian_loss():
      # Separate out objective, constraints and proxy constraints.
      objective = problem.objective()
      constraints = problem.constraints()
      if constraints.shape[0] == 0:
        # If no constraints, just return objective.
        return objective

      # Set up custom Lagrangian loss.
      proxy_constraints = problem.proxy_constraints()
      multipliers_tensor = tf.abs(multipliers())  # Abs enforces non-negativity.

      primal = objective + tf.tensordot(
          tf.stop_gradient(multipliers_tensor), proxy_constraints, 1)
      dual = params['dual_scale'] * tf.tensordot(
          multipliers_tensor, tf.stop_gradient(constraints), 1)

      return primal - dual

    update_ops = problem.update_ops

  # Create optimizer
  if FLAGS.optimizer == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
  else:
    optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=params['learning_rate'])

  # List of trainable variables.
  if params['multiplier_type'] == 'unconstrained':
    var_list = ranking_model.trainable_weights + problem.trainable_variables
  else:
    var_list = (
        ranking_model.trainable_weights + problem.trainable_variables +
        multipliers_variables_list)

  # List of objectives, group constraint violations, per-query constraint
  # violations, and snapshot of models during course of training.
  objectives = []
  group_violations = []
  query_violations = []
  query_violations_full = []
  query_ndcgs = []
  models = []

  features = train_set['features']
  queries = train_set['queries']
  groups = train_set['groups']

  print()
  # Run loops * iterations_per_loop full batch iterations.
  for ii in range(params['loops']):
    for _ in range(params['iterations_per_loop']):

      # Populate paired_batch dict with all pairs for current query. The batch
      # index is the same as the current query index.
      paired_batch = {
          'features': features[queries == batch_index],
          'groups': groups[queries == batch_index]
      }

      # Optimize loss.
      if check_train_pair(paired_batch):
        update_ops()
        optimizer.minimize(lagrangian_loss, var_list=var_list)

      # Update batch_index, and cycle back once last query is reached.
      batch_index = (batch_index + 1) % train_set['num_queries']
      # print(var_list)

    # Snap shot current model.
    model_copy = tf.keras.models.clone_model(ranking_model)
    model_copy.set_weights(ranking_model.get_weights())
    models.append(model_copy)

    # Evaluate metrics for snapshotted model.
    # error, gerr, group_viol, query_viol, query_viols = evaluate_results(
    #     ranking_model, train_set, params)
    # sys.stdout.write('\r Evaluating')
    if metric_fn is None:
      error, _, group_viol, query_viol, query_viols = evaluate_results(
          ranking_model, valid_set, params)
      query_ndcgs.append(0)
    else:
      error, group_error, query_error, query_ndcg = metric_fn(
          ranking_model, valid_groups)
      group_viol = [
          group_error[0] - group_error[1], group_error[1] - group_error[0]
      ]
      query_viol = [np.max(np.abs(query_error[:, 0] - query_error[:, 1]))]
      query_viols = [np.abs(query_error[:, 0] - query_error[:, 1])]
      query_ndcgs.append(np.mean(query_ndcg))

    objectives.append(error)
    group_violations.append(
        [x - params['constraint_slack'] for x in group_viol])
    query_violations.append(
        [x - params['constraint_slack'] for x in query_viol])
    query_violations_full.append(
        [x - params['constraint_slack'] for x in query_viols])
    sys.stdout.write(
        '\r Epoch %d: error = %.3f, group violation = %.3f, query violation = %.3f'
        % (ii, objectives[-1], max(
            group_violations[-1]), max(query_violations[-1])))

  print()

  best_index_padding = params['loops'] // 2
  if params['multiplier_type'] == 'unconstrained':
    # Find model iterate that achieves lowest objective.
    best_index = np.argmin(objectives[best_index_padding:]) + best_index_padding
  elif params['multiplier_type'] == 'common':
    # Find model iterate that trades-off between objective and group violations.
    best_index = tfco.find_best_candidate_index(
        np.array(objectives[best_index_padding:]),
        np.array(group_violations[best_index_padding:]),
        rank_objectives=False) + best_index_padding
  else:
    # Find model iterate that trades-off between objective and per-query
    # violations.
    best_index = tfco.find_best_candidate_index(
        np.array(objectives[best_index_padding:]),
        np.array(query_violations[best_index_padding:]),
        rank_objectives=False) + best_index_padding

  return models[
      best_index], objectives, group_violations, query_violations, query_violations_full, query_ndcgs, best_index


###### EXPERIMENT ######
train_features_raw = None
valid_features_raw = None
test_features_raw = None


def create_dataset_msltr(seed=42,
                         train_size=None,
                         valid_size=None,
                         test_size=None):
  """Read and process MSLR-10K dataset."""
  global train_features_raw
  global valid_features_raw
  global test_features_raw
  train_file_path = FLAGS.input + '/train.txt'
  valid_file_path = FLAGS.input + '/vali.txt'
  test_file_path = FLAGS.input + '/test.txt'

  def _read_data(file_path):
    query_ids = []
    labels = []
    features = []
    n_examples = 0
    with open(file_path) as input_file:
      # The input file can be large. We manually process each line.
      for line in input_file:

        raw_features = line.strip().split(' ')
        labels.append(int(int(raw_features[0]) > 1))
        query_ids.append(int(raw_features[1].split(':')[1]))
        features.append([float(v.split(':')[1]) for v in raw_features[2:]])
        n_examples += 1
        if n_examples % 1000 == 0:
          print('\rFinished {} lines.'.format(str(n_examples)), end='')

      print('\rFinished {} lines.'.format(str(n_examples)), end='\n')
      return (query_ids, features, labels, n_examples)

  np.random.seed(seed=seed)
  selected_feature_indices_ = np.array(range(125))

  # TRAIN
  if train_features_raw is None:
    train_features_raw = _read_data(train_file_path)
  train_queries_, train_features_, train_labels_, _ = train_features_raw

  # Manipulate the training dataset
  # Grouping decided by the No.133 feature
  # We sample the docs in the query by a set of query level rules
  # to create heterogeneous queries and query level features.
  train_features_ = np.array(train_features_)

  # Group score is decided by the 40 percentile of QualityScore2
  group_threshold = np.percentile(train_features_[:, 132], 40)
  train_groups_ = np.array(train_features_[:, 132] > group_threshold, dtype=int)
  train_labels_ = np.array(train_labels_)
  train_queries_ = np.array(train_queries_)

  train_features = []
  train_queries = []
  train_groups = []
  train_labels = []
  train_feature_indices = selected_feature_indices_
  train_dimension = train_feature_indices.shape[0]

  train_num_queries = 0
  for query_id in np.unique(train_queries_):
    query_example_indices = np.where(train_queries_ == query_id)[0]
    # Same as in the paper, we exclude queries with less than 20 docs
    if query_example_indices.shape[0] < 20:
      continue
    # Sort by PageRank
    query_example_indices = query_example_indices[np.argsort(
        train_features_[query_example_indices, 129])]
    # For each, we generate two features and decide how to discard
    # negative docs accordingly
    query_features_ = np.random.uniform(low=-1, high=1, size=2)
    discard_probs = np.linspace(
        start=0.5 + 0.5 * query_features_[0],
        stop=0.5 - 0.5 * query_features_[0],
        num=query_example_indices.shape[0]) * (0.7 + 0.3 * query_features_[1])
    query_example_indices = query_example_indices[(
        train_labels_[query_example_indices] == 1) | (np.random.uniform(
            size=query_example_indices.shape[0]) > discard_probs)]

    # If there is less than 10 posdocs/neg we discard the query
    if (np.sum(train_labels_[query_example_indices]) <
        10) or (query_example_indices.shape[0] -
                np.sum(train_labels_[query_example_indices]) < 10):
      continue

    # Reconstruct the order
    query_example_indices.sort()

    # Only retain queries with minimum number of pos/neg candidates.
    query_groups = train_groups_[query_example_indices]
    query_labels = train_labels_[query_example_indices]
    if (np.sum(np.multiply(query_groups == 1, query_labels == 1)) <= 4) or (
        np.sum(np.multiply(query_groups == 0, query_labels == 1)) <= 4) or (
            np.sum(np.multiply(query_groups == 1, query_labels == 0)) <= 4) or (
                np.sum(np.multiply(query_groups == 0, query_labels == 0)) <= 4):
      continue

    train_groups.extend(train_groups_[query_example_indices])
    train_labels.extend(train_labels_[query_example_indices])

    train_features.extend(
        add_query_mean(
            train_features_[query_example_indices][:, train_feature_indices],
            z=np.reshape(query_groups, (-1, 1))).tolist())

    train_queries.extend([train_num_queries] * query_example_indices.shape[0])
    train_num_queries += 1
    if (train_size is not None) and (train_num_queries >= train_size):
      break

  # VALIDATION
  if valid_features_raw is None:
    valid_features_raw = _read_data(valid_file_path)
  valid_queries_, valid_features_, valid_labels_, _ = valid_features_raw

  # Manipulate the training dataset
  # Grouping decided by the No.133 feature
  # We sample the docs in the query by a set of query level rules
  # to create heterogeneous queries and query level features.
  valid_features_ = np.array(valid_features_)

  # Group score is decided by the 40 percentile of QualityScore2
  group_threshold = np.percentile(valid_features_[:, 132], 40)
  valid_groups_ = np.array(valid_features_[:, 132] > group_threshold, dtype=int)
  valid_labels_ = np.array(valid_labels_)
  valid_queries_ = np.array(valid_queries_)

  valid_features = []
  valid_queries = []
  valid_groups = []
  valid_labels = []
  valid_feature_indices = selected_feature_indices_

  valid_num_queries = 0
  for query_id in np.unique(valid_queries_):
    query_example_indices = np.where(valid_queries_ == query_id)[0]
    # Same as in the paper, we exclude queries with less than 20 docs
    if query_example_indices.shape[0] < 20:
      continue
    # Sort by PageRank
    query_example_indices = query_example_indices[np.argsort(
        valid_features_[query_example_indices, 129])]
    # For each, we generate two features and decide how to discard
    # negative docs accordingly
    query_features_ = np.random.uniform(low=-1, high=1, size=2)
    discard_probs = np.linspace(
        start=0.5 + 0.5 * query_features_[0],
        stop=0.5 - 0.5 * query_features_[0],
        num=query_example_indices.shape[0]) * (0.7 + 0.3 * query_features_[1])
    query_example_indices = query_example_indices[(
        valid_labels_[query_example_indices] == 1) | (np.random.uniform(
            size=query_example_indices.shape[0]) > discard_probs)]

    # If there is less than 10 posdocs/neg we discard the query
    if (np.sum(valid_labels_[query_example_indices]) <
        10) or (query_example_indices.shape[0] -
                np.sum(valid_labels_[query_example_indices]) < 10):
      continue
    if np.random.uniform() > 0.1:
      continue
    # Reconstruct the order
    query_example_indices.sort()

    # Only retain queries with minimum number of pos/neg candidates.
    query_groups = valid_groups_[query_example_indices]
    query_labels = valid_labels_[query_example_indices]
    if (np.sum(np.multiply(query_groups == 1, query_labels == 1)) <= 4) or (
        np.sum(np.multiply(query_groups == 0, query_labels == 1)) <= 4) or (
            np.sum(np.multiply(query_groups == 1, query_labels == 0)) <= 4) or (
                np.sum(np.multiply(query_groups == 0, query_labels == 0)) <= 4):
      continue

    valid_groups.extend(valid_groups_[query_example_indices])
    valid_labels.extend(valid_labels_[query_example_indices])

    valid_features.extend(
        add_query_mean(
            valid_features_[query_example_indices][:, valid_feature_indices],
            z=np.reshape(query_groups, (-1, 1))).tolist())

    valid_queries.extend([valid_num_queries] * query_example_indices.shape[0])
    valid_num_queries += 1
    if (valid_size is not None) and (valid_num_queries >= valid_size):
      break

  # TEST
  if test_features_raw is None:
    test_features_raw = _read_data(test_file_path)
  test_queries_, test_features_, test_labels_, _ = test_features_raw

  test_features_ = np.array(test_features_)
  # no need to calculate threshold again
  test_groups_ = np.array(test_features_[:, 132] > group_threshold, dtype=int)
  test_labels_ = np.array(test_labels_)
  test_queries_ = np.array(test_queries_)

  test_features = []
  test_queries = []
  test_groups = []
  test_labels = []
  test_feature_indices = selected_feature_indices_

  test_num_queries = 0
  for query_id in np.unique(test_queries_):
    query_example_indices = np.where(test_queries_ == query_id)[0]
    # Same as in the paper, we exclude queries with less than 20 docs
    if query_example_indices.shape[0] < 20:
      continue
    # Sort by PageRank
    query_example_indices = query_example_indices[np.argsort(
        test_features_[query_example_indices, 129])]
    # For each, we generate three features and decide how to discard
    # negative docs accordingly
    query_features_ = np.random.uniform(low=-1, high=1, size=2)
    discard_probs = np.linspace(
        start=0.5 + 0.5 * query_features_[0],
        stop=0.5 - 0.5 * query_features_[0],
        num=query_example_indices.shape[0]) * (0.7 + 0.3 * query_features_[1])
    query_example_indices = query_example_indices[(
        test_labels_[query_example_indices] == 1) | (np.random.uniform(
            size=query_example_indices.shape[0]) > discard_probs)]
    if (np.sum(test_labels_[query_example_indices]) <
        10) or (query_example_indices.shape[0] -
                np.sum(test_labels_[query_example_indices]) < 10):
      continue

    query_example_indices.sort()

    # Only retain queries with minimum number of pos/neg candidates.
    query_groups = test_groups_[query_example_indices]
    query_labels = test_labels_[query_example_indices]
    if (np.sum(np.multiply(query_groups == 1, query_labels == 1)) <= 4) or (
        np.sum(np.multiply(query_groups == 0, query_labels == 1)) <= 4) or (
            np.sum(np.multiply(query_groups == 1, query_labels == 0)) <= 4) or (
                np.sum(np.multiply(query_groups == 0, query_labels == 0)) <= 4):
      continue

    test_groups.extend(test_groups_[query_example_indices])
    test_labels.extend(test_labels_[query_example_indices])

    test_features.extend(
        add_query_mean(
            test_features_[query_example_indices][:, test_feature_indices],
            z=np.reshape(query_groups, (-1, 1))).tolist())

    test_queries.extend([test_num_queries] * query_example_indices.shape[0])
    test_num_queries += 1
    if (test_size is not None) and (test_num_queries >= test_size):
      break

  train_dataset = {
      'features': np.array(train_features),
      'queries': train_queries,
      'groups': train_groups,
      'labels': train_labels,
      'dimension': train_dimension,
      'multiplier_dimension': 1,
      'num_queries': train_num_queries,
  }
  # adjust dimensions accordingly
  multiplier_dimension = train_dataset['features'].shape[1] - train_dimension - 1
  print('multiplier_dimension: {}'.format(multiplier_dimension))
  train_dimension = train_dataset['features'].shape[1]

  # adjust dimensions accordingly
  train_dataset['dimension'] = train_dimension
  train_dataset['multiplier_dimension'] = multiplier_dimension

  valid_dataset = {
      'features': np.array(valid_features),
      'queries': valid_queries,
      'groups': valid_groups,
      'labels': valid_labels,
      'dimension': train_dimension,
      'num_queries': valid_num_queries,
  }

  test_dataset = {
      'features': np.array(test_features),
      'queries': test_queries,
      'groups': test_groups,
      'labels': test_labels,
      'dimension': train_dimension,
      'num_queries': test_num_queries,
  }

  return train_dataset, valid_dataset, test_dataset


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  global train_features_raw
  global valid_features_raw
  global test_features_raw
  # Generate train and test data.
  # Generate train and test data.
  train_set, _, test_set = create_dataset_msltr(
      train_size=FLAGS.train_size, valid_size=100, test_size=100)

  # Convert train/test set to paired data for later evaluation.
  paired_train_set = convert_labeled_to_paired_data(
      train_set, max_num_pairs=1000000, max_query_bandwidth=10000)
  paired_test_set = convert_labeled_to_paired_data(
      test_set, max_num_pairs=1000000, max_query_bandwidth=10000)

  # Model hyper-parameters.
  model_params = {
      'loops': FLAGS.loops,
      'iterations_per_loop': FLAGS.microloops,
      'learning_rate': FLAGS.learning_rate,
      'constraint_type': FLAGS.constraint_type,
      'constraint_slack': 0.25,
      'dual_scale': 2,
      'multiplier_dimension': train_set['multiplier_dimension'],
  }
  FLAGS.prefix = '{}_loops_{}_lr_{}_ct_{}_opt_{}'.format(
      FLAGS.prefix, FLAGS.loops * FLAGS.microloops, str(FLAGS.learning_rate),
      model_params['constraint_type'], FLAGS.optimizer)

  with open(
      '{}/{}_{}_ts_{}_id_{}_metric.txt'.format(FLAGS.save_to_dir, FLAGS.prefix,
                                               FLAGS.type, FLAGS.train_size,
                                               FLAGS.id), 'wt') as output_file:
    ff, ax = plt.subplots(1, 6, figsize=(16.0, 3.5))

    train_metric_fn = error_rate_lambda(train_set)
    final_metric_fn = error_rate_lambda(test_set)
    # Unconstrained optimization.
    if FLAGS.type == 'unc':
      model_params['multiplier_type'] = 'unconstrained'
      model_unc, objectives, group_violations, query_violations, query_violations_full, query_ndcgs, best_index = train_model(
          paired_train_set, model_params, metric_fn=train_metric_fn)
      display_results(
          model_unc,
          objectives,
          group_violations,
          query_violations,
          query_violations_full,
          query_ndcgs,
          paired_train_set,
          model_params,
          'Unconstrained',
          'Train',
          show_header=True,
          best_index=best_index,
          output_file=output_file)
      display_results(
          model_unc,
          objectives,
          group_violations,
          query_violations,
          query_violations_full,
          query_ndcgs,
          paired_test_set,
          model_params,
          'Unconstrained',
          'Test',
          show_plots=True,
          best_index=best_index,
          suffix='unconstrained_{}'.format(1),
          metric_fn=final_metric_fn,
          output_file=output_file,
          plot_ax=ax)
    elif FLAGS.type == 'tfco':
      # Constrained optimization with common multipliers (TFCO).
      model_params['multiplier_type'] = 'common'
      model_tfco, objectives, group_violations, query_violations, query_violations_full, query_ndcgs, best_index = train_model(
          paired_train_set, model_params, metric_fn=train_metric_fn)
      display_results(
          model_tfco,
          objectives,
          group_violations,
          query_violations,
          query_violations_full,
          query_ndcgs,
          paired_train_set,
          model_params,
          'Constrained (TFCO)',
          'Train',
          show_header=True,
          output_file=output_file)
      display_results(
          model_tfco,
          objectives,
          group_violations,
          query_violations,
          query_violations_full,
          query_ndcgs,
          paired_test_set,
          model_params,
          'Constrained (TFCO)',
          'Test',
          show_plots=True,
          best_index=best_index,
          suffix='tfco_{}'.format(1),
          metric_fn=final_metric_fn,
          output_file=output_file,
          plot_ax=ax)
    elif FLAGS.type == 'new':
      # Constrained optimization with feature dependent multipliers.
      model_params['multiplier_type'] = 'feature_dependent'
      model_new, objectives, group_violations, query_violations, query_violations_full, query_ndcgs, best_index = train_model(
          paired_train_set, model_params, metric_fn=train_metric_fn)
      display_results(
          model_new,
          objectives,
          group_violations,
          query_violations,
          query_violations_full,
          query_ndcgs,
          paired_train_set,
          model_params,
          'Constrained (New)',
          'Train',
          show_header=True,
          output_file=output_file)
      display_results(
          model_new,
          objectives,
          group_violations,
          query_violations,
          query_violations_full,
          query_ndcgs,
          paired_test_set,
          model_params,
          'Constrained (New)',
          'Test',
          show_plots=True,
          best_index=best_index,
          suffix='new_{}'.format(1),
          metric_fn=final_metric_fn,
          output_file=output_file,
          plot_ax=ax)

    ff.tight_layout()
    plt.savefig('{}/{}_{}_ts_{}_id_{}_plot.png'.format(FLAGS.save_to_dir,
                                                       FLAGS.prefix, FLAGS.type,
                                                       FLAGS.train_size,
                                                       FLAGS.id))


if __name__ == '__main__':
  app.run(main)
