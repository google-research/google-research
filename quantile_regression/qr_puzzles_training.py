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

# pylint: skip-file
import math
import random

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_constrained_optimization as tfco
import tensorflow_lattice as tfl

from quantile_regression import qr_lib
from quantile_regression import qr_lib_gasthaus

# Dataset flags
flags.DEFINE_string('train_filename', '',
                    'Filename of the training dataset.')
flags.DEFINE_string('val_filename', '',
                    'Filename of the validation dataset.')
flags.DEFINE_string('test_filename', '',
                    'Filename of the test dataset.')

flags.DEFINE_string('model_type', 'linear',
                    'Model type to train. Supports linear, rtl, and dnn.')
flags.DEFINE_boolean('normalize_features', False,
                     'If True, normalize input features.')
flags.DEFINE_integer('num_hold_features', 5,
                     'Number of hold time features to use.')

# Optimization flags.
flags.DEFINE_float('model_step_size', 0.1, 'Step size for model parameters.')
flags.DEFINE_float('multiplier_step_size', 0.1,
                   'Step size for lagrange multipliers.')
flags.DEFINE_integer('batch_size', 2000, 'Batch size per iteration.')
flags.DEFINE_integer('epochs', 80, 'Number of epochs through the dataset.')
flags.DEFINE_integer(
    'save_iters', 100,
    'number of iterations between printing the objective and constraint'
    'violations. if 0, do not save objective and constraint violations.')

# q feature flags.
flags.DEFINE_integer(
    'q_monotonicity', 1,
    'Monotonicity parameter for the q feature. 1 indicates '
    'increasing monotonicity, 0 indicates no constraints. ')
flags.DEFINE_integer('num_q_keypoints', 10, 'Number of keypoints for q.')

# Flags for defining constraints.
flags.DEFINE_boolean('unconstrained_keras', False,
                     'If True, train using Keras with pinball loss.')
flags.DEFINE_boolean(
    'unconstrained', False,
    'If True, train using TFCO with an empty constraint list.')
flags.DEFINE_float('constraint_slack', 0.08, 'Slack for each constraint.')
flags.DEFINE_string(
    'qs_to_constrain', 'discrete_3_med', 'q values to constrain. '
    'discrete_3: constrain q in [0.1, 0.5, 0.9]. '
    'discrete_9: constrain q in [0.1,0.2,...,0.9]. '
    'discrete_3_med: constrain q in [0.5, 0.7, 0.9]. '
    'discrete_3_high: constrain q in [0.5, 0.9, 0.99]. '
    'We will create two rate constraints per '
    'group per q value in this list.')
flags.DEFINE_boolean('constrain_groups', True,
                     'If True, include rate constraints per group')
flags.DEFINE_boolean('constrain_overall', True,
                     'If True, include overall rate constraints.')

# Flags for q sampling.
flags.DEFINE_string(
    'q_sampling_method', 'batch', 'batch: resample the qs per batch. '
    'discrete_3: resample qs from [0.1, 0.5, 0.9] per batch. '
    'discrete_3_high: resample qs from [0.5, 0.9, 0.99] per batch. '
    'discrete_3_med: resample qs from [0.5, 0.7, 0.9] per batch. '
    'discrete_9: resample qs from [0.1,0.2,...,0.9] per batch. '
    'discrete_99: resample qs from [0.01,0.02,...,0.99] per batch. '
    'fixed: use fixed uniform sample of qs for all batches. '
    'exact_50: set q=0.5. '
    'exact_90: set q=0.9. '
    'exact_99: set q=0.99. '
    'beta: sample q from beta distribution. '
    'IMPORTANT: unconstrained_keras can only be run with '
    'q_sampling_method=fixed.')
flags.DEFINE_float('beta_mode', 0.5,
                   'mode for beta distribution if q_sampling_method = beta.')
flags.DEFINE_float(
    'beta_concentration', 100.0,
    'concentration for beta distribution if q_sampling_method = beta.')

# Dataset flags
flags.DEFINE_boolean(
    'time_ordered_splits', True,
    'If True, train using train/val/test splits where the validation and test '
    'sets come from the last and second-to-last 20% of the data in time order.')

# RTL model flags.
flags.DEFINE_integer('rtl_lattice_size', 2, 'Size of each rtl lattice.')
flags.DEFINE_integer('rtl_lattice_dim', 2, 'Number of features that go into '
                     'each rtl lattice.')
flags.DEFINE_integer('rtl_num_lattices', 8, 'Number of rtl lattices.')
flags.DEFINE_integer('q_lattice_size', 3, 'Lattice size for q feature.')
flags.DEFINE_integer('num_feature_keypoints', 10,
                     'Number of keypoints for numeric features.')
flags.DEFINE_integer('numeric_feature_monotonicity', 0,
                     'Monotonicity of numeric features.')
flags.DEFINE_boolean('numeric_feature_impute_missing', False,
                     'Impute missing in numeric features.')
flags.DEFINE_integer('f5_monotonicity', 0, 'Monotonicity of f5 feature.')
flags.DEFINE_boolean('f5_impute_missing', False,
                     'Impute missing in f5 feature.')

# DNN model flags.
flags.DEFINE_integer('dnn_num_layers', 4, 'Number of layers in the dnn.')
flags.DEFINE_integer('dnn_hidden_dim', 4, 'Dimension of each hidden layer.')

# Gasthaus flags
flags.DEFINE_boolean('unconstrained_gasthaus', False,
                     'If True, train using Gasthaus method.')
flags.DEFINE_integer('num_gasthaus_keypoints', 10, 'Number of keypoints used '
                     'by Gasthaus (L parameter).')
flags.DEFINE_boolean(
    'calculate_CRPS_loss', False,
    'If True, calculate average CRPS loss. Warning: this can '
    'be slow for large datasets.')

FLAGS = flags.FLAGS

# Define features.
CATEGORICAL_FEATURE_NAMES = ['group']
LABEL_NAME = 'label'
GROUPS = [-1.0, 1.0, 2.0, 3.0]


# Helper function to load data.
def load_dataset(filename):
  """Loads a dataframe from a tsv file."""
  df = pd.read_csv(filename, sep=',', error_bad_lines=False)
  return df


# Get violations for different values of q for different groups.
def get_group_rate_constraint_viols(input_df,
                                    model,
                                    feature_names,
                                    group_num=None,
                                    desired_rates=[]):
  if group_num is None:
    xs, ys = qr_lib.extract_features(
        input_df,
        feature_names=feature_names,
        label_name=LABEL_NAME,
        normalize_features=FLAGS.normalize_features)
  else:
    group_df = input_df[input_df['group'] == group_num]
    xs, ys = qr_lib.extract_features(
        group_df,
        feature_names=feature_names,
        label_name=LABEL_NAME,
        normalize_features=FLAGS.normalize_features)
  _, rate_constraint_viols = qr_lib.get_rate_constraint_viols(
      xs, ys, model, desired_rates=desired_rates)
  return np.array(rate_constraint_viols)


def add_summary_viols_to_results_dict(input_df, model, results_dict,
                                      dataset_name, feature_names):
  """Adds metrics to results_dict."""
  overall_q_loss = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      q_values=None,
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.pinball_loss_overall'].append(overall_q_loss)

  # Report the average pinball loss over specific q's.
  avg_99_q_loss = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      q_values=np.arange(0.01, 1, step=0.01),
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.pinball_loss_avg_99'].append(avg_99_q_loss)

  avg_99_calib_viol, avg_99_calib_viol_sq = qr_lib.get_avg_calibration_viols(
      input_df,
      model,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      q_values=np.arange(0.01, 1, 0.01),
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.calib_viol_avg_99'].append(avg_99_calib_viol)
  results_dict[dataset_name +
               '.calib_viol_sq_avg_99'].append(avg_99_calib_viol_sq)

  # Report the single pinball loss for specific q's.
  q_loss_50 = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      q_values=[0.5],
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.pinball_loss_50'].append(q_loss_50)

  q_loss_70 = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      q_values=[0.7],
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.pinball_loss_70'].append(q_loss_70)

  q_loss_90 = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      q_values=[0.9],
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.pinball_loss_90'].append(q_loss_90)

  avg_3_med_q_loss = np.mean(np.array([q_loss_50, q_loss_70, q_loss_90]))
  results_dict[dataset_name +
               '.pinball_loss_avg_3_med'].append(avg_3_med_q_loss)

  group_9_max_viols = []
  group_3_max_viols = []
  group_3_max_high_viols = []
  group_3_max_med_viols = []
  for group_num in GROUPS:
    group_rate_constraint_viols = get_group_rate_constraint_viols(
        input_df,
        model,
        feature_names=feature_names,
        group_num=group_num,
        desired_rates=np.concatenate(
            [np.arange(0.1, 1, step=0.1),
             np.array([0.99])]))
    group_abs_viols = np.abs(np.array(group_rate_constraint_viols))
    # max rate constraint violation over 9 q's (9-max)
    group_9_max_viols.append(max(group_abs_viols[:8]))

    # max rate constraint violation over 3 q's (3-max)
    group_3_max_viols.append(max(group_abs_viols[[0, 4, 8]]))

    # max rate constraint violation over q in [0.5, 0.9, 0.99]
    group_3_max_high_viols.append(max(group_abs_viols[[4, 8, 9]]))

    # max rate constraint violation over q in [0.5, 0.9, 0.99]
    group_3_max_med_viols.append(max(group_abs_viols[[4, 6, 8]]))

  # Max rate constraint violation over all groups and 9 q's
  results_dict[dataset_name + '.max_9-max_over_groups'].append(
      max(group_9_max_viols))

  # Max rate constraint violation over all groups and 3 q's
  results_dict[dataset_name + '.max_3-max_over_groups'].append(
      max(group_3_max_viols))

  # Max rate constraint violation over all groups and q in [0.5, 0.9, 0.99]
  results_dict[dataset_name + '.max_3-max_high_over_groups'].append(
      max(group_3_max_high_viols))

  # Max rate constraint violation over all groups and q in [0.5, 0.7, 0.9]
  results_dict[dataset_name + '.max_3-max_med_over_groups'].append(
      max(group_3_max_med_viols))
  return results_dict


def print_summary_viols_results_dict(results_dict, iterate='best'):
  index = -1
  if iterate == 'best':
    index = tfco.find_best_candidate_index(
        np.array(results_dict['train.objective']),
        np.array(results_dict['train.max_viols']).reshape((-1, 1)),
        rank_objectives=True)
  for metric_name, values in results_dict.items():
    qr_lib.print_metric(iterate, metric_name, values[index])


def print_summary_viols(input_df, model, dataset_name, feature_names):
  overall_q_loss = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      normalize_features=FLAGS.normalize_features)
  qr_lib.print_metric('last.' + dataset_name, 'pinball_loss_overall',
                      overall_q_loss)

  overall_rate_constraint_viols = get_group_rate_constraint_viols(
      input_df, model, feature_names=feature_names)
  overall_abs_viols = np.abs(np.array(overall_rate_constraint_viols))
  # max rate constraint violation over 9 q's (9-max)
  overall_9_max_viol = max(overall_abs_viols)
  qr_lib.print_metric('last.' + dataset_name, '9-max_overall',
                      overall_9_max_viol)

  # max rate constraint violation over 3 q's (3-max)
  overall_3_max_viol = max(overall_abs_viols[[0, 4, 8]])
  qr_lib.print_metric('last.' + dataset_name, '3-max_overall',
                      overall_3_max_viol)

  group_9_max_viols = []
  group_3_max_viols = []
  for group_num in range(12):
    group_rate_constraint_viols = get_group_rate_constraint_viols(
        input_df, model, feature_names=feature_names, group_num=group_num)
    group_abs_viols = np.abs(np.array(group_rate_constraint_viols))
    # max rate constraint violation over 9 q's (9-max)
    group_9_max_viols.append(max(group_abs_viols))

    # max rate constraint violation over 3 q's (3-max)
    group_3_max_viols.append(max(group_abs_viols[[0, 4, 8]]))

  # Max 9-mean rate constraint violation over all groups
  qr_lib.print_metric('last.' + dataset_name, 'max_9-max_over_groups',
                      max(group_9_max_viols))

  # Max 3-mean rate constraint violation over all groups
  qr_lib.print_metric('last.' + dataset_name, 'max_3-max_over_groups',
                      max(group_3_max_viols))


def print_summary_viols_gasthaus(input_df, model, dataset_name, feature_names):
  """Print summary metrics for Gasthaus."""
  overall_q_loss = qr_lib_gasthaus.calculate_q_loss_gasthaus(
      input_df,
      model,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      q_values=None,
      normalize_features=FLAGS.normalize_features)
  qr_lib.print_metric('last.' + dataset_name, 'pinball_loss_overall',
                      overall_q_loss)

  # Report the average pinball loss over specific q's.
  avg_99_q_loss = qr_lib_gasthaus.calculate_q_loss_gasthaus(
      input_df,
      model,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      q_values=np.arange(0.01, 1, step=0.01),
      normalize_features=FLAGS.normalize_features)
  qr_lib.print_metric('last.' + dataset_name, 'pinball_loss_avg_99',
                      avg_99_q_loss)

  avg_99_calib_viol, avg_99_calib_viol_sq = qr_lib_gasthaus.calculate_calib_viol_gasthaus(
      input_df,
      model,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      q_values=np.arange(0.01, 1, step=0.01),
      normalize_features=FLAGS.normalize_features)
  qr_lib.print_metric('last.' + dataset_name, 'calib_viol_avg_99',
                      avg_99_calib_viol)
  qr_lib.print_metric('last.' + dataset_name, 'calib_viol_sq_avg_99',
                      avg_99_calib_viol_sq)

  if FLAGS.calculate_CRPS_loss:
    # Report the CRPS loss.
    avg_CRPS_loss = qr_lib_gasthaus.calculate_avg_CRPS_loss(
        input_df,
        model,
        feature_names=feature_names,
        label_name=LABEL_NAME,
        normalize_features=FLAGS.normalize_features)
    qr_lib.print_metric('last.' + dataset_name, 'avg_CRPS_loss', avg_CRPS_loss)


def build_linear_model(train_xs,
                       train_ys,
                       numeric_feature_names,
                       feature_name_indices,
                       num_feature_keypoints=10):
  """Builds a calibrated linear model using tfl."""
  min_label = min(train_ys)
  max_label = max(train_ys)
  # Compute quantiles for each feature.
  feature_configs = []
  default_value = -1 if FLAGS.numeric_feature_impute_missing else None
  for numeric_feature in numeric_feature_names:
    monotonicity = FLAGS.numeric_feature_monotonicity
    if numeric_feature == 'f5':
      monotonicity = FLAGS.f5_monotonicity
      default_value = -1 if FLAGS.f5_impute_missing else None
    quantiles = qr_lib.compute_quantiles(
        train_xs[feature_name_indices[numeric_feature]],
        num_keypoints=num_feature_keypoints,
        missing_value=-1)
    if default_value is None:
      # Add specific keypoint for missing value.
      quantiles = [-1] + quantiles
    feature_config = tfl.configs.FeatureConfig(
        name=numeric_feature,
        pwl_calibration_num_keypoints=len(quantiles),
        pwl_calibration_input_keypoints=quantiles,
        monotonicity=monotonicity,
        default_value=default_value)
    feature_configs.append(feature_config)
  group_feature_config = tfl.configs.FeatureConfig(
      name='group',
      # There are 4 total groups.
      pwl_calibration_num_keypoints=4,
      pwl_calibration_input_keypoints=GROUPS,
  )
  feature_configs.append(group_feature_config)
  q_feature_config = tfl.configs.FeatureConfig(
      name='q',
      monotonicity=FLAGS.q_monotonicity,
      pwl_calibration_input_keypoints=np.linspace(0.0, 1.0,
                                                  FLAGS.num_q_keypoints),
  )
  feature_configs.append(q_feature_config)
  linear_model_config = tfl.configs.CalibratedLinearConfig(
      feature_configs=feature_configs,
      # Saw better performance without output bounds.
      output_initialization=[min_label, max_label],
  )
  model = tfl.premade.CalibratedLinear(linear_model_config)
  return model


def build_rtl_model(train_xs,
                    feature_names,
                    numeric_feature_names,
                    feature_name_indices,
                    lattice_size=2,
                    num_lattices=32,
                    lattice_dim=2,
                    num_feature_keypoints=10,
                    numeric_feature_monotonicity=0):
  """Builds an rtl model.

  Args:
    train_xs: numpy array of shape (number of examples, number of features).
    feature_names: list of feature names.
    numeric_feature_names: list of numeric features names.
    feature_name_indices: dictionary mapping feature name to index in train_xs.
    lattice_size: Lattice size for each tiny lattice.
    num_lattices: Number of lattices in the rtl model.
    lattice_dim: Number of features that go into each lattice.
    num_feature_keypoints: number of keypoints to use for each feature.
    numeric_feature_monotonicity: 1 or 0 to enforce monotonicity or not.

  Returns:
    a Keras RTL model.
  """
  # This says we will have inputs that are input_dim features.
  # input_dim includes all dataset features plus q.
  input_dim = len(feature_names) + 1
  input_layers = [tf.keras.layers.Input(shape=(1,)) for _ in range(input_dim)]
  merged_layer = tf.keras.layers.concatenate(input_layers)

  # Manually add all calibrators.
  all_calibrators = tfl.layers.ParallelCombination()

  # Parameters for numeric features
  for numeric_feature in numeric_feature_names:
    cur_impute_missing = FLAGS.numeric_feature_impute_missing
    cur_montonicity = numeric_feature_monotonicity
    # Parameters for f5 specifically
    if numeric_feature == 'f5':
      cur_montonicity = FLAGS.f5_monotonicity
      cur_impute_missing = FLAGS.f5_impute_missing

    # Handle missing values
    cur_missing_input_value = -1 if cur_impute_missing else None
    cur_input_keypoints = qr_lib.compute_quantiles(
        train_xs[feature_name_indices[numeric_feature]],
        num_keypoints=num_feature_keypoints,
        missing_value=-1)
    if not cur_impute_missing:
      # Add specific keypoint for missing value
      cur_input_keypoints = [-1] + cur_input_keypoints

    numeric_calibrator = tfl.layers.PWLCalibration(
        input_keypoints=cur_input_keypoints,
        output_min=0.0,
        output_max=lattice_size - 1.0,
        clamp_min=False,
        clamp_max=False,
        monotonicity=cur_montonicity,
        impute_missing=cur_impute_missing,
        missing_input_value=cur_missing_input_value)
    all_calibrators.append(numeric_calibrator)

  group_calibrator = tfl.layers.PWLCalibration(
      input_keypoints=GROUPS,
      output_min=0.0,
      output_max=lattice_size - 1.0,
      clamp_min=False,  # Clamping is not implemented for nonmonotonic.
      clamp_max=False,
      monotonicity=0,
  )
  all_calibrators.append(group_calibrator)

  clamp_q = bool(FLAGS.q_monotonicity)
  q_calibrator = tfl.layers.PWLCalibration(
      input_keypoints=np.linspace(0.0, 1.0, FLAGS.num_q_keypoints),
      output_min=0.0,
      output_max=lattice_size - 1.0,
      clamp_min=clamp_q,
      clamp_max=clamp_q,
      monotonicity=FLAGS.q_monotonicity,
  )
  all_calibrators.append(q_calibrator)
  calibration_output = all_calibrators(merged_layer)

  lattice_output_layers = []
  # Randomly draw inputs to go into each lattice.
  for _ in range(num_lattices - 1):
    indices = random.sample(range(input_dim), lattice_dim)
    lattice_input = tf.gather(calibration_output, indices, axis=1)
    lattice_output_layer = tfl.layers.Lattice(
        lattice_sizes=[lattice_size] * lattice_dim,
        kernel_initializer='random_uniform',
        monotonicities=[1 for _ in range(lattice_dim)
                       ],  # Fully monotonic lattices.
    )(
        lattice_input)
    lattice_output_layers.append(lattice_output_layer)

  # Include an additional lattice that has q and the country feature.
  q_index = input_dim - 1  # Elsewhere, we always pass q as the last feature.
  group_index = feature_name_indices['group']
  indices = [q_index, group_index]
  lattice_input = tf.gather(calibration_output, indices, axis=1)
  lattice_output_layer = tfl.layers.Lattice(
      lattice_sizes=[FLAGS.q_lattice_size] * 2,  # Explicitly 2 dimensional
      kernel_initializer='random_uniform',
      monotonicities=[1 for _ in range(2)],  # Fully monotonic lattices.
  )(
      lattice_input)
  lattice_output_layers.append(lattice_output_layer)

  final_lattice_output_layer = tf.concat(lattice_output_layers, axis=1)
  output_layer = tf.keras.layers.Dense(
      units=1, kernel_constraint=tf.keras.constraints.NonNeg())(
          final_lattice_output_layer)

  keras_model = tf.keras.models.Model(inputs=input_layers, outputs=output_layer)
  return keras_model


def train_rate_constraints(model,
                           train_xs,
                           train_ys,
                           q,
                           train_df,
                           val_df,
                           test_df,
                           feature_names,
                           feature_name_indices,
                           model_step_size=0.1,
                           multiplier_step_size=0.1,
                           constraint_slack=0.08,
                           batch_size=1000,
                           epochs=20,
                           qs_to_constrain='discrete_3_high',
                           unconstrained=False,
                           save_iters=0,
                           q_sampling_method='batch',
                           beta_mode=0.5,
                           beta_concentration=100,
                           constrain_groups=True,
                           constrain_overall=True):
  """Trains a linear model with pinball loss and rate constraints per group.

  Args:
    model: TFL model.
    train_xs: list of numpy arrays for each x feature.
    train_ys: numpy array for label.
    q: numpy array of quantiles.
    train_df: dataframe with train data.
    val_df: dataframe with val data.
    test_df: dataframe with test data.
    feature_names: list of feature names.
    numeric_feature_names: list of numeric feature names.
    feature_name_indices: dictionary mapping feature name to index in train_xs
    model_step_size: step size for model parameters.
    multiplier_step_size: step size for lagrange multipliers.
    constraint_slack: slack for each constraint.
    batch_size: batch size. Note that Keras's model.fit defaults to
      batch_size=32.
    epochs: number of epochs through the dataset.
    qs_to_constrain: q values to constrain. We will create two rate constraints
      per group per q value in this list.
    unconstrained: if True, train with the pinball loss an empty constraint
      list.
    save_iters: number of iterations between saving the objective and constraint
      violations. if 0, do not save objective and constraint violations.
    q_sampling_method: 'batch': resample the qs every batch. 'discrete_3':
      resample q's from [0.1, 0.5, 0.9] every batch. 'discrete_3_high': resample
      q's from [0.5, 0.9, 0.99] every batch. 'discrete_3_med': resample q's from
      [0.5, 0.7, 0.9] every batch. 'discrete_9': resample q's from
      [0.1,0.2,...,0.9] every batch. 'discrete_99': resample q's from
      [0.01,0.02,...,0.99] every batch. 'fixed': use a single fixed uniform
      sample of q's for all batches.
    beta_mode: mode when sampling from beta distribution.
    beta_concentration: concentration when sampling from beta distribution.
    constrain_groups: if True, add rate constraints per group.
    constrain_overall: if True, add overall rate constraints.

  Returns:
    model: a trained TFL model
    objectives: objective values for every 100 iterations
    constraints: constraint violations for every 100 iterations.
  """
  data_size = len(train_ys)
  # num_iterations = num batches.
  num_iterations = (math.ceil(data_size / batch_size)) * epochs
  num_features = len(feature_names)

  def input_function(batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(
        tuple(train_xs + [q] + [train_ys]))
    dataset = dataset.shuffle(
        data_size, reshuffle_each_iteration=True).repeat().batch(batch_size)
    return dataset

  # all features
  feature_tensors = [
      tf.Variable(
          np.zeros([batch_size, 1], dtype='float32'), name=feature_name)
      for feature_name in feature_names
  ]
  q_tensor = tf.Variable(np.zeros([batch_size, 1], dtype='float32'), name='q')
  y_tensor = tf.Variable(np.zeros([batch_size, 1], dtype='float32'), name='y')

  def predictions(feature_tensors, q):
    q_filled_tensor = tf.fill(feature_tensors[0].shape, q)
    return model(feature_tensors + [q_filled_tensor])

  # Constraints: Check if quantile property holds for x's above and below 0.5.
  # Set slack to +/- 0.03 for all constraints; could tune based on batch size.
  constraints = []
  if not unconstrained:
    if qs_to_constrain == 'discrete_3':
      qs_to_constrain_list = [0.1, 0.5, 0.9]
    elif qs_to_constrain == 'discrete_9':
      qs_to_constrain_list = np.arange(0.1, 1, step=0.1)
    elif qs_to_constrain == 'discrete_3_high':
      qs_to_constrain_list = [0.5, 0.9, 0.99]
    elif qs_to_constrain == 'discrete_3_med':
      qs_to_constrain_list = [0.5, 0.7, 0.9]
    for i in qs_to_constrain_list:
      context = tfco.rate_context(
          lambda k=i: predictions(feature_tensors, k) - y_tensor)
      # Add rate constraint for each group.
      if constrain_groups:
        for group_i in GROUPS:
          context_subset = context.subset(lambda group_k=group_i: tf.math.equal(
              feature_tensors[feature_name_indices['group']], group_k))
          constraints += [
              tfco.positive_prediction_rate(context_subset) <=
              i + constraint_slack,
              tfco.positive_prediction_rate(context_subset) >=
              i - constraint_slack
          ]
      # add overall rate constraints.
      if constrain_overall:
        constraints += [
            tfco.positive_prediction_rate(context) <= i + constraint_slack,
            tfco.positive_prediction_rate(context) >= i - constraint_slack
        ]

  def q_loss_fn(feature_tensors, q_tensor, y_tensor):
    diff = y_tensor - model(feature_tensors + [q_tensor])
    return tf.reduce_mean(
        tf.maximum(diff, 0.0) * q_tensor + tf.minimum(diff, 0.0) *
        (q_tensor - 1.0))

  objective = tfco.wrap_rate(
      lambda: q_loss_fn(feature_tensors, q_tensor, y_tensor))

  problem = tfco.RateMinimizationProblem(objective, constraints)

  optimizer = tfco.ProxyLagrangianOptimizerV2(
      optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=model_step_size),
      constraint_optimizer=tf.keras.optimizers.legacy.Adam(
          learning_rate=multiplier_step_size
      ),
      num_constraints=problem.num_constraints,
  )

  var_list = (
      model.trainable_weights + list(problem.trainable_variables) +
      optimizer.trainable_variables())

  results_dict = {
      'train.objective': [],
      'train.max_viols': [],
      'train.pinball_loss_overall': [],
      'train.pinball_loss_avg_99': [],
      'train.calib_viol_avg_99': [],
      'train.calib_viol_sq_avg_99': [],
      'train.pinball_loss_avg_3_med': [],
      'train.pinball_loss_50': [],
      'train.pinball_loss_70': [],
      'train.pinball_loss_90': [],
      'train.max_9-max_over_groups': [],
      'train.max_3-max_over_groups': [],
      'train.max_3-max_high_over_groups': [],
      'train.max_3-max_med_over_groups': [],
      'val.pinball_loss_overall': [],
      'val.pinball_loss_avg_99': [],
      'val.calib_viol_avg_99': [],
      'val.calib_viol_sq_avg_99': [],
      'val.pinball_loss_avg_3_med': [],
      'val.pinball_loss_50': [],
      'val.pinball_loss_70': [],
      'val.pinball_loss_90': [],
      'val.max_9-max_over_groups': [],
      'val.max_3-max_over_groups': [],
      'val.max_3-max_high_over_groups': [],
      'val.max_3-max_med_over_groups': [],
      'test.pinball_loss_overall': [],
      'test.pinball_loss_avg_99': [],
      'test.calib_viol_avg_99': [],
      'test.calib_viol_sq_avg_99': [],
      'test.pinball_loss_avg_3_med': [],
      'test.pinball_loss_50': [],
      'test.pinball_loss_70': [],
      'test.pinball_loss_90': [],
      'test.max_9-max_over_groups': [],
      'test.max_3-max_over_groups': [],
      'test.max_3-max_high_over_groups': [],
      'test.max_3-max_med_over_groups': [],
  }

  iteration = 0
  for batches in input_function(batch_size):
    for i in range(num_features):
      feature_tensors[i].assign(np.reshape(batches[i], (-1, 1)))
    if q_sampling_method == 'batch':
      q_batch = tf.random.uniform(q_tensor.shape)
      q_tensor.assign(q_batch)
    elif q_sampling_method == 'discrete_3':
      q_batch = np.random.choice([0.1, 0.5, 0.9], size=q_tensor.shape)
      q_tensor.assign(q_batch)
    elif q_sampling_method == 'discrete_3_high':
      q_batch = np.random.choice([0.5, 0.9, 0.99], size=q_tensor.shape)
      q_tensor.assign(q_batch)
    elif q_sampling_method == 'discrete_3_med':
      q_batch = np.random.choice([0.5, 0.7, 0.9], size=q_tensor.shape)
      q_tensor.assign(q_batch)
    elif q_sampling_method == 'discrete_9':
      q_batch = np.random.choice(
          np.arange(0.1, 1, step=0.1), size=q_tensor.shape)
      q_tensor.assign(q_batch)
    elif q_sampling_method == 'discrete_99':
      q_batch = np.random.choice(
          np.arange(0.01, 1, step=0.01), size=q_tensor.shape)
      q_tensor.assign(q_batch)
    elif q_sampling_method == 'beta':
      q_batch = np.random.beta(
          beta_mode * (beta_concentration - 2) + 1,
          (1 - beta_mode) * (beta_concentration - 2) + 1,
          size=q_tensor.shape)
      q_tensor.assign(q_batch)
    else:
      # 'fixed', 'exact_50', 'exact_90', 'exact_99'
      q_tensor.assign(np.reshape(batches[-2], (-1, 1)))
    y_tensor.assign(np.reshape(batches[-1], (-1, 1)))
    optimizer.minimize(problem, var_list=var_list)
    objective = problem.objective()
    violations = problem.constraints()
    if save_iters and (iteration % save_iters == 0):
      max_viol = 0
      if len(violations) > 0:
        max_viol = max(violations)
      print('Iteration: %d, Objective: %.3f, Max Constraint Violation: %.3f' %
            (iteration, objective, max_viol))
      results_dict['train.objective'].append(objective)
      results_dict['train.max_viols'].append(max_viol)
      add_summary_viols_to_results_dict(
          train_df, model, results_dict, 'train', feature_names=feature_names)
      add_summary_viols_to_results_dict(
          val_df, model, results_dict, 'val', feature_names=feature_names)
      add_summary_viols_to_results_dict(
          test_df, model, results_dict, 'test', feature_names=feature_names)
    iteration += 1
    if iteration == num_iterations:
      # Add final metrics to results_dict
      max_viol = 0
      if len(violations) > 0:
        max_viol = max(violations)
      print('Iteration: %d, Objective: %.3f, Max Constraint Violation: %.3f' %
            (iteration, objective, max_viol))
      results_dict['train.objective'].append(objective)
      results_dict['train.max_viols'].append(max_viol)
      add_summary_viols_to_results_dict(
          train_df, model, results_dict, 'train', feature_names=feature_names)
      add_summary_viols_to_results_dict(
          val_df, model, results_dict, 'val', feature_names=feature_names)
      add_summary_viols_to_results_dict(
          test_df, model, results_dict, 'test', feature_names=feature_names)
      break

  return model, results_dict


def train_model(feature_names, numeric_feature_names, feature_name_indices):
  print('RUNNING WITH TF VERSION', tf.__version__)
  print('EAGER EXECUTION:', tf.executing_eagerly())

  # Load datasets.
  train_df = load_dataset(FLAGS.train_filename)
  val_df = load_dataset(FLAGS.val_filename)
  test_df = load_dataset(FLAGS.test_filename)

  # Extract the features from the dataframe.
  train_xs, train_ys = qr_lib.extract_features(
      train_df,
      feature_names=feature_names,
      label_name=LABEL_NAME,
      normalize_features=FLAGS.normalize_features)

  # Create a feature column for q with fixed values.
  q = np.zeros(train_xs[-1].shape)
  if FLAGS.q_sampling_method == 'fixed':
    q = np.random.uniform(0.0, 1.0, size=len(train_df))
  elif FLAGS.q_sampling_method == 'exact_50':
    q = np.full(train_xs[-1].shape, 0.5)
  elif FLAGS.q_sampling_method == 'exact_70':
    q = np.full(train_xs[-1].shape, 0.7)
  elif FLAGS.q_sampling_method == 'exact_90':
    q = np.full(train_xs[-1].shape, 0.9)

  if FLAGS.unconstrained_gasthaus:
    model = qr_lib_gasthaus.build_gasthaus_dnn_model(
        num_features=len(feature_names),
        num_hidden_layers=FLAGS.dnn_num_layers,
        hidden_dim=FLAGS.dnn_hidden_dim,
        keypoints_L=FLAGS.num_gasthaus_keypoints)

    trained_model = qr_lib_gasthaus.train_gasthaus_CRPS(
        model,
        train_xs,
        train_ys,
        model_step_size=FLAGS.model_step_size,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size)

    print_summary_viols_gasthaus(
        train_df,
        trained_model,
        dataset_name='train',
        feature_names=feature_names)
    print_summary_viols_gasthaus(
        val_df, trained_model, dataset_name='val', feature_names=feature_names)
    print_summary_viols_gasthaus(
        test_df,
        trained_model,
        dataset_name='test',
        feature_names=feature_names)
    return

  if FLAGS.model_type == 'linear':
    model = build_linear_model(
        train_xs,
        train_ys,
        feature_names=feature_names,
        numeric_feature_names=numeric_feature_names,
        feature_name_indices=feature_name_indices,
        num_feature_keypoints=FLAGS.num_feature_keypoints)
  elif FLAGS.model_type == 'rtl':
    model = build_rtl_model(
        train_xs,
        feature_names=feature_names,
        numeric_feature_names=numeric_feature_names,
        feature_name_indices=feature_name_indices,
        lattice_size=FLAGS.rtl_lattice_size,
        num_lattices=FLAGS.rtl_num_lattices,
        lattice_dim=FLAGS.rtl_lattice_dim,
        num_feature_keypoints=FLAGS.num_feature_keypoints,
        numeric_feature_monotonicity=FLAGS.numeric_feature_monotonicity)
  elif FLAGS.model_type == 'dnn':
    model = qr_lib.build_dnn_model(
        num_features=len(feature_names),
        num_layers=FLAGS.dnn_num_layers,
        hidden_dim=FLAGS.dnn_hidden_dim)

  if FLAGS.unconstrained_keras:
    trained_model = qr_lib.train_pinball_keras(
        model,
        train_xs,
        train_ys,
        q,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size)

    print_summary_viols(
        train_df,
        trained_model,
        dataset_name='train',
        feature_names=feature_names)
    print_summary_viols(
        val_df, trained_model, dataset_name='val', feature_names=feature_names)
    print_summary_viols(
        test_df,
        trained_model,
        dataset_name='test',
        feature_names=feature_names)
    return

  else:
    _, results_dict = train_rate_constraints(
        model,
        train_xs,
        train_ys,
        q,
        train_df,
        val_df,
        test_df,
        feature_names=feature_names,
        numeric_feature_names=numeric_feature_names,
        feature_name_indices=feature_name_indices,
        model_step_size=FLAGS.model_step_size,
        multiplier_step_size=FLAGS.multiplier_step_size,
        constraint_slack=FLAGS.constraint_slack,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        qs_to_constrain=FLAGS.qs_to_constrain,
        unconstrained=FLAGS.unconstrained,
        save_iters=FLAGS.save_iters,
        q_sampling_method=FLAGS.q_sampling_method,
        beta_mode=FLAGS.beta_mode,
        beta_concentration=FLAGS.beta_concentration,
        constrain_groups=FLAGS.constrain_groups,
        constrain_overall=FLAGS.constrain_overall)

    print_summary_viols_results_dict(results_dict, iterate='last')
    print_summary_viols_results_dict(results_dict, iterate='best')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # Parse feature names.
  numeric_feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
  numeric_feature_names = numeric_feature_names[-FLAGS.num_hold_features:]
  feature_names = numeric_feature_names + CATEGORICAL_FEATURE_NAMES
  feature_name_indices = {
      name: index for index, name in enumerate(feature_names)
  }
  train_model(
      feature_names=feature_names,
      numeric_feature_names=numeric_feature_names,
      feature_name_indices=feature_name_indices)
  return 0


if __name__ == '__main__':
  app.run(main)
