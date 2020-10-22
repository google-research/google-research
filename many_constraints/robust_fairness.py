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
"""Fairness with noisy protected groups experiments."""

from collections import Counter
import random

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import tensorflow as tf
import tensorflow_constrained_optimization as tfco

flags.DEFINE_integer('data_seed', None, 'Seed for train/test/split.')
flags.DEFINE_boolean('feature_dependent_multiplier', True,
                     'If True, train lagrange multipliers based on '
                     'group features.')
flags.DEFINE_float('learning_rate', 0.1, 'Step size for model parameters.')
flags.DEFINE_integer('skip_iterations', 100,
                     'Number of training steps to skip before evaluation.')
flags.DEFINE_integer('num_steps', 10000, 'Number of gradient steps.')
flags.DEFINE_float('dual_scale', 5.0, 'Dual scale.')
flags.DEFINE_boolean('unconstrained', False,
                     'If True, train using TFCO with an empty constraint list.')
flags.DEFINE_boolean('standard_lagrangian', False,
                     'if True, use standard lagrangian of one multiplier per '
                     'constraint.')
flags.DEFINE_boolean('resample_proxy_groups', True,
                     'If True, resample proxy groups every epoch.')
flags.DEFINE_integer('n_resamples_per_candidate', 20,
                     'when using find_best_candidate_index, we take the '
                     'max constraint violations over n_resamples_per_candidate '
                     'resamples of the proxy groups.')
flags.DEFINE_string('group_features_type', 'full_group_vec', 'Type of group '
                    'features to compute. '
                    'full_group_vec: uses the full group membership vector of '
                    'size batch size. '
                    'size_alone: uses the proportional size of each group as a '
                    'single feature. '
                    'size_and_pr: uses the size of the group and the positive '
                    'rate for the group, resulting in 2 features. '
                    'avg_features: uses the average of the other features over '
                    'the group, as well as the group size.'
                    'kmeans: cluster all examples using kmeans. Group features '
                    'are the number of examples that fall in each cluster. ')
flags.DEFINE_integer('num_group_clusters', 100, 'number of clusters to use for '
                     'group_features_type=kmeans.')
flags.DEFINE_integer('num_multiplier_model_hidden_layers', 0,
                     'Number of hidden layers in the multiplier model.')
flags.DEFINE_float('noise_level', 0.3, 'Noise level of initial proxy groups.')
flags.DEFINE_boolean('uniform_groups', False, 'If True, ignore proxy groups '
                     'and sample groups uniformly.')
flags.DEFINE_float('min_group_frac', 0.01, 'smallest group size that we want '
                   'to constrain (as a fraction of the full dataset).')
flags.DEFINE_float('epsilon', 0.05, 'Slack to allow on constraints.')

FLAGS = flags.FLAGS


def load_dataset_adult(noise_level):
  """Loads Adult dataset."""
  df = preprocess_data_adult()
  df = add_proxy_columns_adult(df)
  label_name = 'label'
  feature_names = list(df.keys())
  feature_names.remove(label_name)
  protected_columns = ['race_White', 'race_Black', 'race_Other_combined']
  for column in protected_columns:
    feature_names.remove(column)
  proxy_columns = get_proxy_column_names(protected_columns, noise_level)
  feature_names = remove_saved_noise_levels(
      protected_columns, feature_names, keep_noise_level=noise_level)
  return df, feature_names, label_name, protected_columns, proxy_columns


def preprocess_data_adult():
  """Preprocess Adult dataset."""
  categorical_columns = [
      'workclass', 'education', 'marital_status', 'occupation', 'relationship',
      'race', 'gender', 'native_country'
  ]
  continuous_columns = [
      'age', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num'
  ]
  columns = [
      'age', 'workclass', 'fnlwgt', 'education', 'education_num',
      'marital_status', 'occupation', 'relationship', 'race', 'gender',
      'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
      'income_bracket'
  ]
  label_column = 'label'

  train_df_raw = pd.read_csv(
      'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
      names=columns,
      skipinitialspace=True)
  test_df_raw = pd.read_csv(
      'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
      names=columns,
      skipinitialspace=True,
      skiprows=1)

  train_df_raw[label_column] = (
      train_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
  test_df_raw[label_column] = (
      test_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
  # Preprocessing Features
  pd.options.mode.chained_assignment = None  # default='warn'

  # Functions for preprocessing categorical and continuous columns.
  def binarize_categorical_columns(input_train_df,
                                   input_test_df,
                                   categorical_columns=None):

    def fix_columns(input_train_df, input_test_df):
      test_df_missing_cols = set(input_train_df.columns) - set(
          input_test_df.columns)
      for c in test_df_missing_cols:
        input_test_df[c] = 0
        train_df_missing_cols = set(input_test_df.columns) - set(
            input_train_df.columns)
      for c in train_df_missing_cols:
        input_train_df[c] = 0
        input_train_df = input_train_df[input_test_df.columns]
      return input_train_df, input_test_df

    # Binarize categorical columns.
    binarized_train_df = pd.get_dummies(
        input_train_df, columns=categorical_columns)
    binarized_test_df = pd.get_dummies(
        input_test_df, columns=categorical_columns)
    # Make sure the train and test dataframes have the same binarized columns.
    fixed_train_df, fixed_test_df = fix_columns(binarized_train_df,
                                                binarized_test_df)
    return fixed_train_df, fixed_test_df

  def bucketize_continuous_column(input_train_df,
                                  input_test_df,
                                  continuous_column_name,
                                  num_quantiles=None,
                                  bins=None):
    assert (num_quantiles is None or bins is None)
    if num_quantiles is not None:
      _, bins_quantized = pd.qcut(
          input_train_df[continuous_column_name],
          num_quantiles,
          retbins=True,
          labels=False)
      input_train_df[continuous_column_name] = pd.cut(
          input_train_df[continuous_column_name], bins_quantized, labels=False)
      input_test_df[continuous_column_name] = pd.cut(
          input_test_df[continuous_column_name], bins_quantized, labels=False)
    elif bins is not None:
      input_train_df[continuous_column_name] = pd.cut(
          input_train_df[continuous_column_name], bins, labels=False)
      input_test_df[continuous_column_name] = pd.cut(
          input_test_df[continuous_column_name], bins, labels=False)

  # Filter out all columns except the ones specified.
  train_df = train_df_raw[categorical_columns + continuous_columns +
                          [label_column]]
  test_df = test_df_raw[categorical_columns + continuous_columns +
                        [label_column]]

  # Bucketize continuous columns.
  bucketize_continuous_column(train_df, test_df, 'age', num_quantiles=4)
  bucketize_continuous_column(
      train_df, test_df, 'capital_gain', bins=[-1, 1, 4000, 10000, 100000])
  bucketize_continuous_column(
      train_df, test_df, 'capital_loss', bins=[-1, 1, 1800, 1950, 4500])
  bucketize_continuous_column(
      train_df, test_df, 'hours_per_week', bins=[0, 39, 41, 50, 100])
  bucketize_continuous_column(
      train_df, test_df, 'education_num', bins=[0, 8, 9, 11, 16])
  train_df, test_df = binarize_categorical_columns(
      train_df,
      test_df,
      categorical_columns=categorical_columns + continuous_columns)
  full_df = train_df.append(test_df)
  full_df['race_Other_combined'] = full_df['race_Amer-Indian-Eskimo'] + full_df[
      'race_Asian-Pac-Islander'] + full_df['race_Other']
  return full_df


def add_proxy_columns_adult(df):
  """Adds noisy proxy columns to adult dataset."""
  proxy_noises = [0.1, 0.2, 0.3, 0.4, 0.5]
  protected_columns = ['race_White', 'race_Black', 'race_Other_combined']
  # Generate proxy groups.
  for noise in proxy_noises:
    df = generate_proxy_columns(df, protected_columns, noise_param=noise)
  return df


def generate_proxy_columns(df, protected_columns, noise_param=1):
  """Generates noisy proxy columns from binarized protected columns."""
  proxy_columns = get_proxy_column_names(protected_columns, noise_param)
  num_datapoints = len(df)
  num_groups = len(protected_columns)
  noise_idx = random.sample(
      range(num_datapoints), int(noise_param * num_datapoints))
  df_proxy = df.copy()
  for i in range(num_groups):
    df_proxy[proxy_columns[i]] = df_proxy[protected_columns[i]]
  for j in noise_idx:
    group_index = -1
    for i in range(num_groups):
      if df_proxy[proxy_columns[i]][j] == 1:
        df_proxy.at[j, proxy_columns[i]] = 0
        group_index = i
        allowed_new_groups = list(range(num_groups))
        allowed_new_groups.remove(group_index)
        new_group_index = random.choice(allowed_new_groups)
        df_proxy.at[j, proxy_columns[new_group_index]] = 1
        break
    if group_index == -1:
      print('missing group information for datapoint ', j)
  return df_proxy


# Split into train/val/test
def train_val_test_split(df, train_fraction, validate_fraction, seed=None):
  """Split the whole dataset into train/val/test."""
  if seed is not None:
    np.random.seed(seed=seed)
  perm = np.random.permutation(df.index)
  m = len(df.index)
  train_end = int(train_fraction * m)
  validate_end = int(validate_fraction * m) + train_end
  train = df.iloc[perm[:train_end]]
  validate = df.iloc[perm[train_end:validate_end]]
  test = df.iloc[perm[validate_end:]]
  return train, validate, test


def _print_metric(dataset_name, metric_name, metric_value):
  """Prints metrics."""
  print('[metric] %s.%s=%f' % (dataset_name, metric_name, metric_value))


def compute_quantiles(features,
                      num_keypoints=10,
                      clip_min=None,
                      clip_max=None,
                      missing_value=None):
  """Computes quantiles for feature columns."""
  # Clip min and max if desired.
  if clip_min is not None:
    features = np.maximum(features, clip_min)
    features = np.append(features, clip_min)
  if clip_max is not None:
    features = np.minimum(features, clip_max)
    features = np.append(features, clip_max)
  # Make features unique.
  unique_features = np.unique(features)
  # Remove missing values if specified.
  if missing_value is not None:
    unique_features = np.delete(unique_features,
                                np.where(unique_features == missing_value))
  # Compute and return quantiles over unique non-missing feature values.
  return np.quantile(
      unique_features,
      np.linspace(0., 1., num=num_keypoints),
      interpolation='nearest').astype(float)


def print_metrics_results_dict(results_dict, iterate='best'):
  """Prints metrics from results_dict."""
  index = -1
  if iterate == 'best':
    if FLAGS.unconstrained:
      index = np.argmin(np.array(results_dict['train.true_error_rates']))
    else:
      index = tfco.find_best_candidate_index(
          np.array(results_dict['train.true_error_rates']),
          np.array(results_dict['train.sampled_violations_max']).reshape(
              (-1, 1)),
          rank_objectives=True)
  for metric_name, values in results_dict.items():
    _print_metric(iterate, metric_name, values[index])


# Helper functions for evaluation.
def error_rate(labels, predictions):
  """Computes error rate."""
  # Recall that the labels are binary (0 or 1).
  signed_labels = (labels * 2) - 1
  return np.mean(signed_labels * predictions <= 0.0)


def group_error_rates(labels, predictions, groups):
  """Returns a list containing error rates for each protected group."""
  errors = []
  for jj in range(groups.shape[1]):
    if groups[:, jj].sum() == 0:  # Group is empty?
      errors.append(0.0)
    else:
      signed_labels_jj = 2 * labels[groups[:, jj] == 1] - 1
      predictions_jj = predictions[groups[:, jj] == 1]
      errors.append(np.mean(signed_labels_jj * predictions_jj <= 0))
  return errors


def tpr(labels, predictions):
  """Computes true positive rate."""
  # Recall that the labels are binary (0 or 1).
  signed_labels = (labels * 2) - 1
  predictions_pos = predictions[signed_labels > 0]
  return np.mean(predictions_pos > 0.0)


def group_tprs(labels, predictions, groups):
  """Returns a list containing tprs for each protected group."""
  tprs = []
  for jj in range(groups.shape[1]):
    if groups[:, jj].sum() == 0:  # Group is empty?
      tprs.append(0.0)
    else:
      signed_labels_jj = 2 * labels[groups[:, jj] == 1] - 1
      predictions_jj = predictions[groups[:, jj] == 1]
      predictions_jj_pos = predictions_jj[signed_labels_jj > 0]
      tprs.append(np.mean(predictions_jj_pos > 0))
  return tprs


# Get proxy columns.
def get_proxy_column_names(protected_columns, noise_param, noise_index=''):
  """Gets proxy column names."""
  return [
      'PROXY' + noise_index + '_' + '%0.2f_' % noise_param + column_name
      for column_name in protected_columns
  ]


def remove_saved_noise_levels(protected_columns, feature_names,
                              keep_noise_level):
  """Removes saved noise level columns from feature columns."""
  saved_noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
  saved_noise_levels.remove(keep_noise_level)
  for noise_level in saved_noise_levels:
    proxy_columns = get_proxy_column_names(protected_columns, noise_level)
    for column in proxy_columns:
      feature_names.remove(column)
  return feature_names


def generate_proxy_groups_single_noise(input_groups, noise_param=1):
  """Generate proxy groups within noise noise_param."""
  proxy_groups = np.copy(input_groups)
  num_groups = len(input_groups[0])
  num_datapoints = len(input_groups)
  noise_idx = random.sample(
      range(num_datapoints), int(noise_param * num_datapoints))
  for j in noise_idx:
    group_index = -1
    for i in range(num_groups):
      if proxy_groups[j][i] == 1:
        proxy_groups[j][i] = 0
        group_index = i
        allowed_new_groups = list(range(num_groups))
        allowed_new_groups.remove(group_index)
        new_group_index = random.choice(allowed_new_groups)
        proxy_groups[j][new_group_index] = 1
        break
    if group_index == -1:
      print('missing group information for datapoint ', j)
  return proxy_groups


def generate_proxy_groups_uniform(num_examples, min_group_frac=0.05):
  """Generate proxy groups within noise noise_param."""

  # Generate a random array of the same shape as input groups. Each column
  # in the array is a a random binary vector where the number of 1's is at least
  # min_group_size.
  group_frac = np.random.uniform(min_group_frac, 1)
  num_in_group = int(num_examples * group_frac)
  group_assignment = np.array([0] * (num_examples - num_in_group) +
                              [1] * num_in_group)
  np.random.shuffle(group_assignment)
  return group_assignment.reshape((-1, 1))


def generate_proxy_groups_noise_array(input_groups, noise_array=None):
  """Generate proxy groups within noise noise_param."""

  proxy_groups = np.copy(input_groups)
  num_groups = len(input_groups[0])

  for row in proxy_groups:
    new_j = -1
    for k in range(num_groups):
      if row[k] == 1:
        # draw from noise_params to decide which group to switch to.
        new_j = np.random.choice(num_groups, 1, p=noise_array[k])
        row[k] = 0
    assert new_j >= 0
    row[new_j] = 1

  return proxy_groups


def extract_group_features(input_groups,
                           input_features,
                           input_labels,
                           group_features_type,
                           num_group_clusters=None,
                           kmeans_model=None):
  """Extracts features from groups."""
  input_groups_t = input_groups.transpose().astype(int)
  all_group_features = []
  for group_indices in input_groups_t:
    group_fraction = np.mean(group_indices)
    if group_features_type == 'size_alone':
      all_group_features.append(np.array([group_fraction]))
    elif group_features_type == 'size_and_pr':
      mean_labels = np.mean(input_labels[group_indices == 1], axis=0)
      mean_features = np.append(mean_labels, group_fraction)
      all_group_features.append(mean_features)
    elif group_features_type == 'avg_features':
      mean_features = np.mean(input_features[group_indices == 1], axis=0)
      mean_features = np.append(mean_features, group_fraction)
      all_group_features.append(mean_features)
    elif group_features_type == 'full_group_vec':
      print('group_indices shape', group_indices.shape)
      all_group_features.append(group_indices)
    elif group_features_type == 'kmeans':
      group_xs = input_features[group_indices == 1]
      clusters = kmeans_model.predict(group_xs)
      # Counter doesn't include clusters with count 0.
      # Need to manually add 0 counts for clusters that aren't seen.
      count_dict = dict.fromkeys(range(num_group_clusters), 0)
      count_dict.update(Counter(clusters))
      compressed_clusters = np.fromiter(count_dict.values(), dtype='float32')
      all_group_features.append(compressed_clusters)
  return np.array(all_group_features)


# Calculate P(G=j | \hat{G}=k).
def get_noise_array(input_df,
                    protected_columns,
                    proxy_columns,
                    num_groups,
                    print_noises=False):
  """Returns an array where noise_params[k][j] = P(G=j | hatG=k)."""
  noise_array = np.zeros((num_groups, num_groups))
  for k in range(num_groups):
    for j in range(num_groups):
      frac = np.sum(
          input_df[protected_columns[j]] * input_df[proxy_columns[k]]) / np.sum(
              input_df[proxy_columns[k]])
      noise_array[k][j] = frac
      if print_noises:
        print('P(G=%d | hatG=%d) = %f' % (j, k, frac))
  return noise_array


# Custom function for extracting features from the dataframe.
def extract_features(dataframe,
                     feature_names,
                     label_name,
                     proxy_group_names,
                     true_group_names,
                     uniform_groups=False,
                     min_group_frac=0.05):
  """Extracts features from dataframe."""
  features = []
  for feature_name in feature_names:
    features.append(dataframe[feature_name].values.astype(float))
  labels = dataframe[label_name].values.astype(float)
  proxy_groups_array = None
  true_groups_array = None
  if uniform_groups:
    proxy_groups_array = generate_proxy_groups_uniform(
        len(dataframe), min_group_frac=min_group_frac)
    true_groups_array = generate_proxy_groups_uniform(
        len(dataframe), min_group_frac=min_group_frac)
  else:
    proxy_groups = []
    true_groups = []
    for group_name in proxy_group_names:
      proxy_groups.append(dataframe[group_name].values.astype(float))
    for group_name in true_group_names:
      true_groups.append(dataframe[group_name].values.astype(float))
    proxy_groups_array = np.transpose(np.array(proxy_groups))
    true_groups_array = np.transpose(np.array(true_groups))
  return np.transpose(np.array(features)), np.array(labels).reshape(
      (-1, 1)), proxy_groups_array, true_groups_array


def add_summary_viols_to_results_dict(input_df,
                                      model,
                                      results_dict,
                                      dataset_name,
                                      feature_names,
                                      label_name,
                                      proxy_columns,
                                      protected_columns,
                                      epsilon=0.03,
                                      n_resamples_per_candidate=10,
                                      use_noise_array=True,
                                      noise_array=None,
                                      uniform_groups=False,
                                      min_group_frac=0.05):
  """Adds metrics to results_dict."""
  features, labels, init_proxy_groups, true_groups = extract_features(
      input_df,
      feature_names=feature_names,
      label_name=label_name,
      proxy_group_names=proxy_columns,
      true_group_names=protected_columns,
      uniform_groups=uniform_groups,
      min_group_frac=min_group_frac)
  predictions = model.predict(features)
  overall_error = error_rate(labels, predictions)
  results_dict[dataset_name + '.true_error_rates'].append(overall_error)

  overall_tpr = tpr(labels, predictions)
  init_proxy_group_tprs = group_tprs(labels, predictions, init_proxy_groups)
  proxy_group_tpr_violations = [
      overall_tpr - group_tpr - epsilon for group_tpr in init_proxy_group_tprs
  ]
  results_dict[dataset_name + '.proxy_group_violations'].append(
      max(proxy_group_tpr_violations))

  true_group_tprs = group_tprs(labels, predictions, true_groups)
  true_group_tpr_violations = [
      overall_tpr - group_tpr - epsilon for group_tpr in true_group_tprs
  ]
  results_dict[dataset_name + '.true_group_violations'].append(
      max(true_group_tpr_violations))

  sampled_violations = []
  for _ in range(n_resamples_per_candidate):
    # Resample proxy groups.
    if uniform_groups:
      sampled_groups = generate_proxy_groups_uniform(
          len(input_df), min_group_frac=min_group_frac)
    elif use_noise_array:
      sampled_groups = generate_proxy_groups_noise_array(
          init_proxy_groups, noise_array=noise_array)
    else:
      sampled_groups = generate_proxy_groups_single_noise(
          init_proxy_groups, noise_param=FLAGS.noise_level)
    sampled_group_tprs = group_tprs(labels, predictions, sampled_groups)
    sampled_group_tpr_violations = [
        overall_tpr - group_tpr - epsilon for group_tpr in sampled_group_tprs
    ]
    sampled_violations.append(max(sampled_group_tpr_violations))
  results_dict[dataset_name + '.sampled_violations_max'].append(
      max(sampled_violations))
  results_dict[dataset_name + '.sampled_violations_90p'].append(
      np.percentile(np.array(sampled_violations), 90))
  return results_dict


def create_multiplier_model(feature_dependent_multiplier=True,
                            num_group_features=1,
                            hidden_layers=None):
  """Creates lagrange multiplier model."""
  if hidden_layers is None:
    hidden_layers = [100]
  if feature_dependent_multiplier:
    layers = []
    layers.append(tf.keras.Input(shape=(num_group_features,)))
    for num_nodes in hidden_layers:
      layers.append(tf.keras.layers.Dense(num_nodes, activation='relu'))

    # Add a final dense layer.
    layers.append(tf.keras.layers.Dense(1, bias_initializer='ones'))

    # Keras model.
    multiplier_model = tf.keras.Sequential(layers)
    multiplier_weights = multiplier_model.trainable_weights
  else:
    common_multiplier = tf.Variable(1.0, name='common_multiplier')
    # Ignore feature input, and return common multiplier.
    multiplier_model = lambda x: common_multiplier
    multiplier_weights = [common_multiplier]
  return multiplier_model, multiplier_weights


def new_epoch(batch_index, batch_size, num_examples):
  """Returns true if a new epoch occurs during batch number batch_index."""
  min_batch_index = batch_index * batch_size
  max_batch_index = (batch_index + 1) * batch_size - 1
  return ((min_batch_index %
           num_examples) == 0) or ((min_batch_index % num_examples) >
                                   (max_batch_index % num_examples))


def train_helper(train_df,
                 val_df,
                 test_df,
                 feature_names,
                 label_name,
                 proxy_columns,
                 protected_columns,
                 feature_dependent_multiplier=True,
                 learning_rate=0.1,
                 batch_size=None,
                 skip_iterations=100,
                 num_steps=1000,
                 dual_scale=1.0,
                 epsilon=0.03,
                 unconstrained=False,
                 standard_lagrangian=False,
                 use_noise_array=True,
                 resample_proxy_groups=True,
                 epochs_per_resample=1,
                 n_resamples_per_candidate=10,
                 group_features_type='full_group_vec',
                 num_group_clusters=100,
                 multiplier_model_hidden_layers=[100],
                 uniform_groups=False,
                 min_group_frac=0.05):
  """Helper function for training a model."""
  tf.keras.backend.clear_session()

  # init_proxy_groups_train is the initial noisy group assignments.
  features_train, labels_train, init_proxy_groups_train, _ = extract_features(
      train_df,
      feature_names=feature_names,
      label_name=label_name,
      proxy_group_names=proxy_columns,
      true_group_names=protected_columns,
      uniform_groups=uniform_groups,
      min_group_frac=min_group_frac)

  num_groups = init_proxy_groups_train.shape[1]
  noise_array = None
  if use_noise_array and not uniform_groups:
    noise_array = get_noise_array(
        train_df,
        protected_columns=protected_columns,
        proxy_columns=proxy_columns,
        num_groups=num_groups)

  num_examples = len(train_df)
  num_features = len(feature_names)

  if batch_size is None:
    batch_size = num_examples

  # Get number of group features.
  kmeans_model = None
  num_group_features = None
  if group_features_type == 'full_group_vec':
    num_group_features = num_examples
  elif group_features_type == 'size_alone':
    num_group_features = 1
  elif group_features_type == 'size_and_pr':
    num_group_features = 2
  elif group_features_type == 'avg_features':
    num_group_features = num_features + 1
  elif group_features_type == 'kmeans':
    kmeans_model = KMeans(
        n_clusters=num_group_clusters, random_state=0).fit(features_train)
    num_group_features = num_group_clusters

  # Features
  features_tensor = tf.Variable(
      np.zeros((batch_size, num_features), dtype='float32'), name='features')
  # Labels
  labels_tensor = tf.Variable(
      np.zeros((batch_size, 1), dtype='float32'), name='labels')
  # Protected groups
  # We will resample these groups every epoch during training.
  groups_tensor = tf.Variable(
      np.zeros((batch_size, num_groups), dtype='float32'), name='groups')
  # Protected group features.
  groups_features_tensor = tf.Variable(
      np.zeros((num_groups, num_group_features), dtype='float32'),
      name='group_features')

  # Linear model with no hidden layers.
  layers = []
  layers.append(tf.keras.Input(shape=(num_features,)))
  layers.append(tf.keras.layers.Dense(1))

  # Keras model.
  model = tf.keras.Sequential(layers)

  # Set up rate minimization problem.
  # We set up a constrained optimization problem, where we *minimize the overall
  # error rate subject to the TPR for individual groups being with an epsilon
  # of the overall TPR.
  def predictions():
    return model(features_tensor)

  context = tfco.rate_context(predictions, labels=lambda: labels_tensor)
  overall_error = tfco.error_rate(context)
  constraints = []
  if not unconstrained:
    # Add group rate constraints.
    pos_context = context.subset(lambda: labels_tensor > 0)
    overall_tpr = tfco.positive_prediction_rate(pos_context)
    for jj in range(num_groups):
      group_pos_context = pos_context.subset(
          lambda kk=jj: groups_tensor[:, kk] > 0)
      group_tpr = tfco.positive_prediction_rate(group_pos_context)
      constraints.append(group_tpr >= overall_tpr - epsilon)

  problem = tfco.RateMinimizationProblem(overall_error, constraints)

  # Set up multiplier model.
  if not unconstrained:
    if standard_lagrangian:
      common_multiplier = tf.Variable(
          np.ones((len(constraints), 1)),
          dtype='float32',
          name='common_multiplier')
      multiplier_weights = [common_multiplier]
    else:
      multiplier_model, multiplier_weights = create_multiplier_model(
          feature_dependent_multiplier=feature_dependent_multiplier,
          num_group_features=num_group_features,
          hidden_layers=multiplier_model_hidden_layers)

  # Set up lagrangian loss.
  def lagrangian_loss():
    # Separate out objective, constraints and proxy constraints.
    objective = problem.objective()
    constraints = problem.constraints()
    proxy_constraints = problem.proxy_constraints()

    # Set-up custom Lagrangian loss.
    multipliers = tf.abs(multiplier_model(groups_features_tensor))
    primal = objective + tf.stop_gradient(multipliers) * proxy_constraints
    dual = dual_scale * multipliers * tf.stop_gradient(constraints)

    return primal - dual

  # Standard lagrangian loss with a different multiplier for each constraint.
  def lagrangian_loss_standard():
    objective = problem.objective()
    constraints = problem.constraints()
    proxy_constraints = problem.proxy_constraints()

    # Set up standard lagrangian loss.
    multipliers = tf.abs(common_multiplier)
    primal = objective + tf.stop_gradient(multipliers) * proxy_constraints
    dual = dual_scale * multipliers * tf.stop_gradient(constraints)

    return primal - dual

  # Set up unconstrained loss.
  def unconstrained_loss():
    return problem.objective()

  # Create optimizer
  optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

  # List of variables to optimize (in this case, the model parameters).
  if unconstrained:
    var_list = model.trainable_weights
  else:
    var_list = model.trainable_weights + multiplier_weights

  # Set up counter for the minibatch stream
  batch_index = 0

  # Record objectives and constraint violations.
  results_dict = {
      'train.objectives': [],
      'train.batch_violations': [],
      'train.true_error_rates': [],
      'train.sampled_violations_max': [],
      'train.sampled_violations_90p': [],
      'train.proxy_group_violations': [],
      'train.true_group_violations': [],
      'val.true_error_rates': [],
      'val.sampled_violations_max': [],
      'val.sampled_violations_90p': [],
      'val.proxy_group_violations': [],
      'val.true_group_violations': [],
      'test.true_error_rates': [],
      'test.sampled_violations_max': [],
      'test.sampled_violations_90p': [],
      'test.proxy_group_violations': [],
      'test.true_group_violations': []
  }
  group_sample_epochs = 0

  # Loop over minibatches.
  groups_train = init_proxy_groups_train
  if not unconstrained:
    group_features = extract_group_features(
        groups_train,
        features_train,
        labels_train,
        group_features_type,
        num_group_clusters=num_group_features,
        kmeans_model=kmeans_model)
    groups_features_tensor.assign(group_features)
  for ii in range(num_steps):
    # Indices for current minibatch in the stream.
    batch_indices = np.arange(batch_index * batch_size,
                              (batch_index + 1) * batch_size)

    # Check for the beginning of a new epoch.
    if resample_proxy_groups and not unconstrained:
      if new_epoch(
          batch_index, batch_size=batch_size, num_examples=num_examples):
        # Only resample proxy groups every epochs_per_resample epochs.
        if group_sample_epochs % epochs_per_resample == 0:
          # Resample the group at the beginning of the epoch.
          # Get groups_train from a ball around init_proxy_groups_train.
          if uniform_groups:
            groups_train = generate_proxy_groups_uniform(
                num_examples, min_group_frac=min_group_frac)
          elif use_noise_array:
            groups_train = generate_proxy_groups_noise_array(
                init_proxy_groups_train, noise_array=noise_array)
          else:
            groups_train = generate_proxy_groups_single_noise(
                init_proxy_groups_train, noise_param=FLAGS.noise_level)
          # Recompute group features at the beginning of the epoch.
          group_features = extract_group_features(
              groups_train,
              features_train,
              labels_train,
              group_features_type,
              num_group_clusters=num_group_features,
              kmeans_model=kmeans_model)
          groups_features_tensor.assign(group_features)
        group_sample_epochs += 1

    # Cycle back to the beginning if we have reached the end of the stream.
    batch_indices = [ind % num_examples for ind in batch_indices]

    # Assign features, labels.
    features_tensor.assign(features_train[batch_indices, :])
    labels_tensor.assign(labels_train[batch_indices].reshape(-1, 1))
    groups_tensor.assign(groups_train[batch_indices])

    # Gradient update.
    with tf.control_dependencies(problem.update_ops()):
      if unconstrained:
        optimizer.minimize(unconstrained_loss, var_list=var_list)
      elif standard_lagrangian:
        optimizer.minimize(lagrangian_loss_standard, var_list=var_list)
      else:
        optimizer.minimize(lagrangian_loss, var_list=var_list)

    if (ii % skip_iterations == 0) or (ii == num_steps - 1):
      # Record metrics.
      results_dict['train.objectives'].append(problem.objective().numpy())
      if not unconstrained:
        results_dict['train.batch_violations'].append(
            np.max(problem.constraints().numpy()))
      else:
        results_dict['train.batch_violations'].append(0)
      add_summary_viols_to_results_dict(
          train_df,
          model,
          results_dict,
          'train',
          feature_names=feature_names,
          label_name=label_name,
          proxy_columns=proxy_columns,
          protected_columns=protected_columns,
          epsilon=epsilon,
          n_resamples_per_candidate=n_resamples_per_candidate,
          use_noise_array=use_noise_array,
          noise_array=noise_array,
          uniform_groups=uniform_groups,
          min_group_frac=min_group_frac)
      add_summary_viols_to_results_dict(
          val_df,
          model,
          results_dict,
          'val',
          feature_names=feature_names,
          label_name=label_name,
          proxy_columns=proxy_columns,
          protected_columns=protected_columns,
          epsilon=epsilon,
          n_resamples_per_candidate=n_resamples_per_candidate,
          use_noise_array=use_noise_array,
          noise_array=noise_array,
          uniform_groups=uniform_groups,
          min_group_frac=min_group_frac)
      add_summary_viols_to_results_dict(
          test_df,
          model,
          results_dict,
          'test',
          feature_names=feature_names,
          label_name=label_name,
          proxy_columns=proxy_columns,
          protected_columns=protected_columns,
          epsilon=epsilon,
          n_resamples_per_candidate=n_resamples_per_candidate,
          use_noise_array=use_noise_array,
          noise_array=noise_array,
          uniform_groups=uniform_groups,
          min_group_frac=min_group_frac)

      print(
          '%d: batch obj: %.3f | batch viol: %.3f | true error: %.3f | sampled viol: %.3f | true group viol: %.3f'
          % (ii, results_dict['train.objectives'][-1],
             results_dict['train.batch_violations'][-1],
             results_dict['train.true_error_rates'][-1],
             results_dict['train.sampled_violations_max'][-1],
             results_dict['train.true_group_violations'][-1]))

    batch_index += 1
  return model, results_dict


def train_model():
  """Runs full training."""
  # Load datasets.
  df, feature_names, label_name, protected_columns, proxy_columns = load_dataset_adult(
      FLAGS.noise_level)

  train_df, val_df, test_df = train_val_test_split(
      df, 0.2, 0.2, seed=FLAGS.data_seed)

  if FLAGS.uniform_groups:
    protected_columns = []
    proxy_columns = []

  _, results_dict = train_helper(
      train_df,
      val_df,
      test_df,
      feature_names=feature_names,
      label_name=label_name,
      proxy_columns=proxy_columns,
      protected_columns=protected_columns,
      feature_dependent_multiplier=FLAGS.feature_dependent_multiplier,
      learning_rate=FLAGS.learning_rate,
      skip_iterations=FLAGS.skip_iterations,
      num_steps=FLAGS.num_steps,
      dual_scale=FLAGS.dual_scale,
      epsilon=FLAGS.epsilon,
      unconstrained=FLAGS.unconstrained,
      standard_lagrangian=FLAGS.standard_lagrangian,
      resample_proxy_groups=FLAGS.resample_proxy_groups,
      n_resamples_per_candidate=FLAGS.n_resamples_per_candidate,
      group_features_type=FLAGS.group_features_type,
      num_group_clusters=FLAGS.num_group_clusters,
      multiplier_model_hidden_layers=[
          100 for _ in range(FLAGS.num_multiplier_model_hidden_layers)
      ],
      uniform_groups=FLAGS.uniform_groups,
      min_group_frac=FLAGS.min_group_frac,
  )
  print_metrics_results_dict(results_dict, iterate='last')
  print_metrics_results_dict(results_dict, iterate='best')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  train_model()
  return 0


if __name__ == '__main__':
  app.run(main)
