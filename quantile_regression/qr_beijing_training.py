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
import sys

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

# Model flags
flags.DEFINE_string('model_type', 'linear',
                    'Model type to train. Supports linear, rtl, and dnn.')
flags.DEFINE_boolean('log_label', False, 'If true, take the log of the label.')
flags.DEFINE_boolean('normalize_features', True,
                     'If True, normalize input features.')

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
    'qs_to_constrain', 'discrete_3_high', 'q values to constrain. '
    'discrete_3: constrain q in [0.1, 0.5, 0.9]. '
    'discrete_9: constrain q in [0.1,0.2,...,0.9]. '
    'discrete_3_high: constrain q in [0.5, 0.9, 0.99]. '
    'We will create two rate constraints per '
    'region per q value in this list.')
flags.DEFINE_boolean('constrain_regions', True,
                     'If True, include rate constraints per region')
flags.DEFINE_boolean('constrain_overall', True,
                     'If True, include overall rate constraints.')

# Flags for q sampling.
flags.DEFINE_string(
    'q_sampling_method', 'batch', 'batch: resample the qs per batch. '
    'discrete_3: resample qs from [0.1, 0.5, 0.9] per batch. '
    'discrete_3_high: resample qs from [0.5, 0.9, 0.99] per batch. '
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
NUMERIC_FEATURE_NAMES = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
CATEGORICAL_FEATURE_NAMES = ['region', 'wd_cat']
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
LABEL_NAME = 'PM2.5'
FEATURE_NAME_INDICES = {name: index for index, name in enumerate(FEATURE_NAMES)}


# Helper function to load data.
def load_dataset(filename):
  df = pd.read_csv(filename, sep=',', error_bad_lines=False)
  df = df.fillna(0.0)
  # Convert wind direction into numeric variable
  df['wd'] = df['wd'].astype('category')
  df['wd_cat'] = df['wd'].cat.codes
  if FLAGS.log_label:
    df[LABEL_NAME] = np.log(df[LABEL_NAME])
  return df


def get_region_rate_constraint_viols(input_df,
                                     model,
                                     region_num=None,
                                     desired_rates=[]):
  """Get violations for different values of q for different regions."""
  if region_num is None:
    xs, ys = qr_lib.extract_features(
        input_df,
        feature_names=FEATURE_NAMES,
        label_name=LABEL_NAME,
        normalize_features=FLAGS.normalize_features)
  else:
    region_df = input_df[input_df['region'] == region_num]
    xs, ys = qr_lib.extract_features(
        region_df,
        feature_names=FEATURE_NAMES,
        label_name=LABEL_NAME,
        normalize_features=FLAGS.normalize_features)
  _, rate_constraint_viols = qr_lib.get_rate_constraint_viols(
      xs, ys, model, desired_rates=desired_rates)
  return np.array(rate_constraint_viols)


def add_summary_viols_to_results_dict(input_df, model, results_dict,
                                      dataset_name):
  """Adds metrics to results_dict."""
  overall_q_loss = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=FEATURE_NAMES,
      label_name=LABEL_NAME,
      q_values=None,
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.pinball_loss_overall'].append(overall_q_loss)

  # Report the average pinball loss over specific q's.
  avg_99_q_loss = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=FEATURE_NAMES,
      label_name=LABEL_NAME,
      q_values=np.arange(0.01, 1, step=0.01),
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.pinball_loss_avg_99'].append(avg_99_q_loss)

  avg_99_calib_viol, avg_99_calib_viol_sq = qr_lib.get_avg_calibration_viols(
      input_df,
      model,
      feature_names=FEATURE_NAMES,
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
      feature_names=FEATURE_NAMES,
      label_name=LABEL_NAME,
      q_values=[0.5],
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.pinball_loss_50'].append(q_loss_50)

  q_loss_90 = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=FEATURE_NAMES,
      label_name=LABEL_NAME,
      q_values=[0.9],
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.pinball_loss_90'].append(q_loss_90)

  q_loss_99 = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=FEATURE_NAMES,
      label_name=LABEL_NAME,
      q_values=[0.99],
      normalize_features=FLAGS.normalize_features)
  results_dict[dataset_name + '.pinball_loss_99'].append(q_loss_99)

  avg_3_high_q_loss = np.mean(np.array([q_loss_50, q_loss_90, q_loss_99]))
  results_dict[dataset_name +
               '.pinball_loss_avg_3_high'].append(avg_3_high_q_loss)

  region_9_max_viols = []
  region_3_max_viols = []
  region_3_max_high_viols = []
  for region_num in range(12):
    region_rate_constraint_viols = get_region_rate_constraint_viols(
        input_df,
        model,
        region_num=region_num,
        desired_rates=np.concatenate(
            [np.arange(0.1, 1, step=0.1),
             np.array([0.99])]))
    region_abs_viols = np.abs(np.array(region_rate_constraint_viols))
    # max rate constraint violation over 9 q's (9-max)
    region_9_max_viols.append(max(region_abs_viols[:8]))

    # max rate constraint violation over 3 q's (3-max)
    region_3_max_viols.append(max(region_abs_viols[[0, 4, 8]]))

    # max rate constraint violation over q in [0.5, 0.9, 0.99]
    region_3_max_high_viols.append(max(region_abs_viols[[4, 8, 9]]))

  # Max rate constraint violation over all regions and 9 q's
  results_dict[dataset_name + '.max_9-max_over_regions'].append(
      max(region_9_max_viols))

  # Max rate constraint violation over all regions and 3 q's
  results_dict[dataset_name + '.max_3-max_over_regions'].append(
      max(region_3_max_viols))

  # Max rate constraint violation over all regions and q in [0.5, 0.9, 0.99]
  results_dict[dataset_name + '.max_3-max_high_over_regions'].append(
      max(region_3_max_high_viols))
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


def print_summary_viols(input_df, model, dataset_name):
  overall_q_loss = qr_lib.calculate_q_loss(
      input_df,
      model,
      feature_names=FEATURE_NAMES,
      label_name=LABEL_NAME,
      normalize_features=FLAGS.normalize_features)
  qr_lib.print_metric('last.' + dataset_name, 'pinball_loss_overall',
                      overall_q_loss)

  overall_rate_constraint_viols = get_region_rate_constraint_viols(
      input_df, model)
  overall_abs_viols = np.abs(np.array(overall_rate_constraint_viols))
  # max rate constraint violation over 9 q's (9-max)
  overall_9_max_viol = max(overall_abs_viols)
  qr_lib.print_metric('last.' + dataset_name, '9-max_overall',
                      overall_9_max_viol)

  # max rate constraint violation over 3 q's (3-max)
  overall_3_max_viol = max(overall_abs_viols[[0, 4, 8]])
  qr_lib.print_metric('last.' + dataset_name, '3-max_overall',
                      overall_3_max_viol)

  region_9_max_viols = []
  region_3_max_viols = []
  for region_num in range(12):
    region_rate_constraint_viols = get_region_rate_constraint_viols(
        input_df, model, region_num=region_num)
    region_abs_viols = np.abs(np.array(region_rate_constraint_viols))
    # max rate constraint violation over 9 q's (9-max)
    region_9_max_viols.append(max(region_abs_viols))

    # max rate constraint violation over 3 q's (3-max)
    region_3_max_viols.append(max(region_abs_viols[[0, 4, 8]]))

  # Max 9-mean rate constraint violation over all regions
  qr_lib.print_metric('last.' + dataset_name, 'max_9-max_over_regions',
                      max(region_9_max_viols))

  # Max 3-mean rate constraint violation over all regions
  qr_lib.print_metric('last.' + dataset_name, 'max_3-max_over_regions',
                      max(region_3_max_viols))


def print_summary_viols_gasthaus(input_df, model, dataset_name):
  """Print summary metrics for Gasthaus."""
  overall_q_loss = qr_lib_gasthaus.calculate_q_loss_gasthaus(
      input_df,
      model,
      feature_names=FEATURE_NAMES,
      label_name=LABEL_NAME,
      q_values=None,
      normalize_features=FLAGS.normalize_features)
  qr_lib.print_metric('last.' + dataset_name, 'pinball_loss_overall',
                      overall_q_loss)

  # Report the average pinball loss over specific q's.
  avg_99_q_loss = qr_lib_gasthaus.calculate_q_loss_gasthaus(
      input_df,
      model,
      feature_names=FEATURE_NAMES,
      label_name=LABEL_NAME,
      q_values=np.arange(0.01, 1, step=0.01),
      normalize_features=FLAGS.normalize_features)
  qr_lib.print_metric('last.' + dataset_name, 'pinball_loss_avg_99',
                      avg_99_q_loss)

  avg_99_calib_viol, avg_99_calib_viol_sq = qr_lib_gasthaus.calculate_calib_viol_gasthaus(
      input_df,
      model,
      feature_names=FEATURE_NAMES,
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
        feature_names=FEATURE_NAMES,
        label_name=LABEL_NAME,
        normalize_features=FLAGS.normalize_features)
    qr_lib.print_metric('last.' + dataset_name, 'avg_CRPS_loss', avg_CRPS_loss)


def build_linear_model(train_xs, train_ys):
  """Builds a calibrated linear model using tfl."""
  min_label = min(train_ys)
  max_label = max(train_ys)
  linear_model_config = tfl.configs.CalibratedLinearConfig(
      feature_configs=[
          tfl.configs.FeatureConfig(
              name='TEMP',
              pwl_calibration_num_keypoints=10,
              pwl_calibration_input_keypoints=qr_lib.compute_quantiles(
                  train_xs[FEATURE_NAME_INDICES['TEMP']], num_keypoints=10),
          ),
          tfl.configs.FeatureConfig(
              name='PRES',
              pwl_calibration_num_keypoints=10,
              pwl_calibration_input_keypoints=qr_lib.compute_quantiles(
                  train_xs[FEATURE_NAME_INDICES['PRES']], num_keypoints=10),
          ),
          tfl.configs.FeatureConfig(
              name='DEWP',
              pwl_calibration_num_keypoints=10,
              pwl_calibration_input_keypoints=qr_lib.compute_quantiles(
                  train_xs[FEATURE_NAME_INDICES['DEWP']], num_keypoints=10),
          ),
          tfl.configs.FeatureConfig(
              name='RAIN',
              pwl_calibration_num_keypoints=10,
              pwl_calibration_input_keypoints=qr_lib.compute_quantiles(
                  train_xs[FEATURE_NAME_INDICES['RAIN']], num_keypoints=10),
          ),
          tfl.configs.FeatureConfig(
              name='WSPM',
              pwl_calibration_num_keypoints=10,
              pwl_calibration_input_keypoints=qr_lib.compute_quantiles(
                  train_xs[FEATURE_NAME_INDICES['WSPM']], num_keypoints=10),
          ),
          tfl.configs.FeatureConfig(
              name='region',
              # There are 12 total regions.
              pwl_calibration_num_keypoints=12,
              # pwl_calibration_input_keypoints=np.linspace(0.0, 1.0, 12),
              pwl_calibration_input_keypoints=np.linspace(
                  np.min(train_xs[FEATURE_NAME_INDICES['region']]),
                  np.max(train_xs[FEATURE_NAME_INDICES['region']]),
                  num=12),
          ),
          tfl.configs.FeatureConfig(
              name='wd_cat',
              # There are 17 total wind directions.
              pwl_calibration_num_keypoints=17,
              # pwl_calibration_input_keypoints=np.linspace(0.0, 1.0, 17),
              pwl_calibration_input_keypoints=np.linspace(
                  np.min(train_xs[FEATURE_NAME_INDICES['wd_cat']]),
                  np.max(train_xs[FEATURE_NAME_INDICES['wd_cat']]),
                  num=17),
          ),
          tfl.configs.FeatureConfig(
              name='q',
              monotonicity=FLAGS.q_monotonicity,
              pwl_calibration_input_keypoints=np.linspace(
                  0.0, 1.0, FLAGS.num_q_keypoints),
          ),
      ],

      # Saw better performance without output bounds.
      output_initialization=[min_label, max_label],
  )
  model = tfl.premade.CalibratedLinear(linear_model_config)
  return model


def build_rtl_model(train_xs, lattice_size=2, num_lattices=32, lattice_dim=2):
  """Builds an rtl model.

  Args:
    train_xs: numpy array of shape (number of examples, number of features).
    lattice_size: Lattice size for each tiny lattice.
    num_lattices: Number of lattices in the rtl model.
    lattice_dim: Number of features that go into each lattice.

  Returns:
    a Keras RTL model.
  """
  # This says we will have inputs that are input_dim features.
  # input_dim includes all dataset features plus q.
  input_dim = len(FEATURE_NAMES) + 1
  input_layers = [tf.keras.layers.Input(shape=(1,)) for _ in range(input_dim)]
  merged_layer = tf.keras.layers.concatenate(input_layers)

  # Manually add all calibrators.
  all_calibrators = tfl.layers.ParallelCombination()
  for numeric_feature in NUMERIC_FEATURE_NAMES:
    numeric_calibrator = tfl.layers.PWLCalibration(
        input_keypoints=qr_lib.compute_quantiles(
            train_xs[FEATURE_NAME_INDICES[numeric_feature]], num_keypoints=10),
        output_min=0.0,
        output_max=lattice_size - 1.0,
        clamp_min=False,
        clamp_max=False,
        monotonicity=0,
    )
    all_calibrators.append(numeric_calibrator)

  region_calibrator = tfl.layers.PWLCalibration(
      input_keypoints=np.linspace(
          np.min(train_xs[FEATURE_NAME_INDICES['region']]),
          np.max(train_xs[FEATURE_NAME_INDICES['region']]),
          num=12),
      output_min=0.0,
      output_max=lattice_size - 1.0,
      clamp_min=False,  # Clamping is not implemented for nonmonotonic.
      clamp_max=False,
      monotonicity=0,
  )
  all_calibrators.append(region_calibrator)

  wd_cat_calibrator = tfl.layers.PWLCalibration(
      input_keypoints=np.linspace(
          np.min(train_xs[FEATURE_NAME_INDICES['wd_cat']]),
          np.max(train_xs[FEATURE_NAME_INDICES['wd_cat']]),
          num=17),
      output_min=0.0,
      output_max=lattice_size - 1.0,
      clamp_min=False,  # Clamping is not implemented for nonmonotonic.
      clamp_max=False,
      monotonicity=0,
  )
  all_calibrators.append(wd_cat_calibrator)

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
  region_index = FEATURE_NAME_INDICES['region']
  indices = [q_index, region_index]
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
                           constrain_regions=True,
                           constrain_overall=True):
  """Trains a linear model with pinball loss and rate constraints per region.

  Args:
    model: TFL model.
    train_xs: list of numpy arrays for each feature.
    train_ys: numpy array of labels.
    q: numpy array of quantile feature.
    train_df: dataframe with train data.
    val_df: dataframe with validation data.
    test_df: dataframe with test data.
    model_step_size: step size for model parameters.
    multiplier_step_size: step size for lagrange multipliers.
    constraint_slack: slack for each constraint.
    batch_size: batch size. Note that Keras's model.fit defaults to
      batch_size=32.
    epochs: number of epochs through the dataset.
    qs_to_constrain: q values to constrain. We will create two rate constraints
      per region per q value in this list.
    unconstrained: if True, train with the pinball loss an empty constraint
      list.
    save_iters: number of iterations between saving the objective and constraint
      violations. if 0, do not save objective and constraint violations.
    q_sampling_method: 'batch': resample the qs every batch. 'discrete_3':
      resample q's from [0.1, 0.5, 0.9] every batch. 'discrete_3_high': resample
      q's from [0.5, 0.9, 0.99] every batch. 'discrete_9': resample q's from
      [0.1,0.2,...,0.9] every batch. 'discrete_99': resample q's from
      [0.01,0.02,...,0.99] every batch. 'fixed': use a single fixed uniform
      sample of q's for all batches.
    beta_mode: mode if sampling from beta distribution.
    beta_concentration: concentration if sampling from beta distribution.
    constrain_regions: if True, add rate constraints per region.
    constrain_overall: if True, add overall rate constraints.

  Returns:
    model: a trained TFL model
    objectives: objective values for every 100 iterations
    constraints: constraint violations for every 100 iterations.
  """
  data_size = len(train_ys)
  # num_iterations = num batches.
  num_iterations = (math.ceil(data_size / batch_size)) * epochs
  num_features = len(FEATURE_NAMES)
  num_regions = 12

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
      for feature_name in FEATURE_NAMES
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
    for i in qs_to_constrain_list:
      context = tfco.rate_context(
          lambda k=i: predictions(feature_tensors, k) - y_tensor)
      # Add rate constraint for each region.
      if constrain_regions:
        for region_i in range(num_regions):
          context_subset = context.subset(
              lambda region_k=region_i: tf.math.equal(
                  feature_tensors[FEATURE_NAME_INDICES['region']], region_k))
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
      'train.pinball_loss_avg_3_high': [],
      'train.pinball_loss_50': [],
      'train.pinball_loss_90': [],
      'train.pinball_loss_99': [],
      'train.max_9-max_over_regions': [],
      'train.max_3-max_over_regions': [],
      'train.max_3-max_high_over_regions': [],
      'val.pinball_loss_overall': [],
      'val.pinball_loss_avg_99': [],
      'val.calib_viol_avg_99': [],
      'val.calib_viol_sq_avg_99': [],
      'val.pinball_loss_avg_3_high': [],
      'val.pinball_loss_50': [],
      'val.pinball_loss_90': [],
      'val.pinball_loss_99': [],
      'val.max_9-max_over_regions': [],
      'val.max_3-max_over_regions': [],
      'val.max_3-max_high_over_regions': [],
      'test.pinball_loss_overall': [],
      'test.pinball_loss_avg_99': [],
      'test.calib_viol_avg_99': [],
      'test.calib_viol_sq_avg_99': [],
      'test.pinball_loss_avg_3_high': [],
      'test.pinball_loss_50': [],
      'test.pinball_loss_90': [],
      'test.pinball_loss_99': [],
      'test.max_9-max_over_regions': [],
      'test.max_3-max_over_regions': [],
      'test.max_3-max_high_over_regions': [],
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
      add_summary_viols_to_results_dict(train_df, model, results_dict, 'train')
      add_summary_viols_to_results_dict(val_df, model, results_dict, 'val')
      add_summary_viols_to_results_dict(test_df, model, results_dict, 'test')
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
      add_summary_viols_to_results_dict(train_df, model, results_dict, 'train')
      add_summary_viols_to_results_dict(val_df, model, results_dict, 'val')
      add_summary_viols_to_results_dict(test_df, model, results_dict, 'test')
      break

  return model, results_dict


def train_model():
  print('RUNNING WITH TF VERSION', tf.__version__)
  print('EAGER EXECUTION:', tf.executing_eagerly())

  # Load datasets.
  train_df = load_dataset(FLAGS.train_filename)
  val_df = load_dataset(FLAGS.val_filename)
  test_df = load_dataset(FLAGS.test_filename)

  # Extract the features from the dataframe.
  train_xs, train_ys = qr_lib.extract_features(
      train_df,
      feature_names=FEATURE_NAMES,
      label_name=LABEL_NAME,
      normalize_features=FLAGS.normalize_features)

  # Create a feature column for q with fixed values.
  q = np.zeros(train_xs[-1].shape)
  if FLAGS.q_sampling_method == 'fixed':
    q = np.random.uniform(0.0, 1.0, size=len(train_df))
  elif FLAGS.q_sampling_method == 'exact_50':
    q = np.full(train_xs[-1].shape, 0.5)
  elif FLAGS.q_sampling_method == 'exact_90':
    q = np.full(train_xs[-1].shape, 0.9)
  elif FLAGS.q_sampling_method == 'exact_99':
    q = np.full(train_xs[-1].shape, 0.99)

  if FLAGS.unconstrained_gasthaus:
    model = qr_lib_gasthaus.build_gasthaus_dnn_model(
        num_features=len(FEATURE_NAMES),
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

    print_summary_viols_gasthaus(train_df, trained_model, dataset_name='train')
    print_summary_viols_gasthaus(val_df, trained_model, dataset_name='val')
    print_summary_viols_gasthaus(test_df, trained_model, dataset_name='test')
    return

  if FLAGS.model_type == 'linear':
    model = build_linear_model(train_xs, train_ys)
  elif FLAGS.model_type == 'rtl':
    model = build_rtl_model(
        train_xs,
        lattice_size=FLAGS.rtl_lattice_size,
        num_lattices=FLAGS.rtl_num_lattices,
        lattice_dim=FLAGS.rtl_lattice_dim)
  elif FLAGS.model_type == 'dnn':
    model = qr_lib.build_dnn_model(
        num_features=len(FEATURE_NAMES),
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

    print_summary_viols(train_df, trained_model, dataset_name='train')
    print_summary_viols(val_df, trained_model, dataset_name='val')
    print_summary_viols(test_df, trained_model, dataset_name='test')
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
        constrain_regions=FLAGS.constrain_regions,
        constrain_overall=FLAGS.constrain_overall)

    print_summary_viols_results_dict(results_dict, iterate='last')
    print_summary_viols_results_dict(results_dict, iterate='best')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  train_model()
  return 0


if __name__ == '__main__':
  app.run(main)
