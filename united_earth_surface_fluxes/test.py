# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

# pylint: disable=logging-fstring-interpolation
# pylint: disable=g-importing-member

"""Main testing loop."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import json
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from united_earth_surface_fluxes import config
from united_earth_surface_fluxes import dataset
from united_earth_surface_fluxes import trainer
from united_earth_surface_fluxes import utils

# --- Flags ---
_MODEL_DIR = flags.DEFINE_string(
    'model_dir', None, 'Directory containing the saved model.', required=True
)
_TEST_OUTPUT_DIR = flags.DEFINE_string(
    'test_output_dir', './output/test', 'Directory to save test results.'
)
_DATASET_DIR = flags.DEFINE_string(
    'dataset_dir', './data', 'Dataset directory.'
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', None, 'Batch size.', required=True
)
_SEED = flags.DEFINE_integer(
    'seed', None, 'Seed for reproducibility.', required=True
)

# Model architecture flags, passed from training json for compatibility
_TRUNK_NUM_LAYERS = flags.DEFINE_integer(
    'trunk_num_layers', None, 'Number of layers in model trunk.', required=True
)
_EXP_NUM_LAYERS = flags.DEFINE_integer(
    'exp_num_layers', None, 'Number of layers in model head.', required=True
)
_HID_DIM = flags.DEFINE_integer(
    'hid_dim', None, 'Hidden dimension.', required=True
)
_TRANSFORMER_EMBED_DIM = flags.DEFINE_integer(
    'transformer_embed_dim',
    None,
    'Transformer embed dimension.',
    required=True,
)
_TRANSFORMER_FF_DIM = flags.DEFINE_integer(
    'transformer_ff_dim',
    None,
    'Transformer feed forward dimension.',
    required=True,
)
_TRANSFORMER_NUM_HEADS = flags.DEFINE_integer(
    'transformer_num_heads',
    None,
    'Transformer number of heads.',
    required=True,
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name', None, 'Model name.', required=True
)
_TEST_PERIOD_LIST = flags.DEFINE_string(
    'test_period_list',
    (
        '2024-01-22,2024-02-22,2024-03-22,2024-04-22,2024-05-22,2024-06-22,'
        '2024-07-22,2024-08-22,2024-09-22,2024-10-22,2024-11-22,2024-12-22'
    ),
    'Testing period list (supports comma-separated or start:end colon range).',
)


def _validate_var_lists():
  """Ensures no overlap between feature variable lists."""
  feature_lists = {
      'DYNAMIC_VARS': config.DYNAMIC_VARS,
      'STATIC_VARS': config.STATIC_VARS,
      'VEG_AND_SOIL_TYPES': config.VEG_AND_SOIL_TYPES,
      'TARGET_VARS': config.TARGET_VARS,
  }
  for name1, list1 in feature_lists.items():
    for name2, list2 in feature_lists.items():
      if name1 == name2:
        continue
      intersection = set(list1) & set(list2)
      if intersection:
        raise ValueError(f'Overlap between {name1} and {name2}: {intersection}')


def _validate_var_scale_dict():
  """Ensures all variables have an entry in dataset.var_scale_dict."""
  all_vars = (
      config.DYNAMIC_VARS
      + config.STATIC_VARS
      + config.VEG_AND_SOIL_TYPES
      + config.TARGET_VARS
  )
  missing_vars = [var for var in all_vars if var not in dataset.var_scale_dict]
  if missing_vars:
    raise ValueError(f'Missing variables in var_scale_dict: {missing_vars}')


def _get_distribution_strategy():
  """Returns the appropriate tf.distribute strategy."""
  gpus = tf.config.list_physical_devices('GPU')
  if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    logging.info(f'Detected {len(gpus)} GPUs. Using MirroredStrategy.')
  else:
    strategy = tf.distribute.get_strategy()
    if gpus:
      logging.info('Detected 1 GPU. Using Default Strategy.')
    else:
      logging.info('No GPUs detected. Using Default Strategy (CPU).')
  logging.info(f'Replicas in sync: {strategy.num_replicas_in_sync}')
  return strategy


def save_predictions_and_weights(
    predictions_list,
    weights_list,
    labels_list,
    residuals_list,
    test_file,
    file_name,
):
  """Saves predictions, weights, and labels to files."""
  if not predictions_list:
    logging.info(f'No predictions for file {test_file}, skipping save.')
    return

  all_preds = tf.concat(predictions_list, axis=0).numpy()
  preds_path = f'{_TEST_OUTPUT_DIR.value}/{file_name}_predictions.npy'
  with io.BytesIO() as buffer:
    np.save(buffer, all_preds)
    with tf.io.gfile.GFile(preds_path, 'wb') as f:
      f.write(buffer.getvalue())
  logging.info(f'Predictions for {file_name} saved to {preds_path}')

  if weights_list and weights_list[0] is not None:
    all_weights = tf.concat(weights_list, axis=0).numpy()
    weights_path = f'{_TEST_OUTPUT_DIR.value}/{file_name}_weights.npy'
    with io.BytesIO() as buffer:
      np.save(buffer, all_weights)
      with tf.io.gfile.GFile(weights_path, 'wb') as f:
        f.write(buffer.getvalue())
    logging.info(f'Weights for {file_name} saved to {weights_path}')

  if labels_list:
    all_labels = tf.concat(labels_list, axis=0).numpy()
    labels_path = f'{_TEST_OUTPUT_DIR.value}/{file_name}_labels.npy'
    with io.BytesIO() as buffer:
      np.save(buffer, all_labels)
      with tf.io.gfile.GFile(labels_path, 'wb') as f:
        f.write(buffer.getvalue())
    logging.info(f'Labels for {file_name} saved to {labels_path}')

  if residuals_list:
    all_residuals = tf.concat(residuals_list, axis=0).numpy()
    residuals_path = f'{_TEST_OUTPUT_DIR.value}/{file_name}_residuals.npy'
    with io.BytesIO() as buffer:
      np.save(buffer, all_residuals)
      with tf.io.gfile.GFile(residuals_path, 'wb') as f:
        f.write(buffer.getvalue())
    logging.info(f'Residuals for {file_name} saved to {residuals_path}')


def main(_):
  """Main testing function."""
  utils.set_seed(_SEED.value)
  _validate_var_lists()
  _validate_var_scale_dict()

  strategy = _get_distribution_strategy()
  per_replica_batch_size = _BATCH_SIZE.value
  global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
  logging.info(f'{per_replica_batch_size=}, {global_batch_size=}')

  tf.io.gfile.makedirs(_TEST_OUTPUT_DIR.value)
  logging.info(f'Test output dir: {_TEST_OUTPUT_DIR.value}')
  # --- Dataset Setup ---
  test_dates = utils.parse_period_list(_TEST_PERIOD_LIST.value)
  test_period_list = [f'{_DATASET_DIR.value}/{d}.tfrecord' for d in test_dates]
  logging.info(f'test_period_list: {len(test_period_list)} dates')

  logging.info(
      f'Model dir: {_MODEL_DIR.value}\n'
      f'Test output dir: {_TEST_OUTPUT_DIR.value}\n'
      f'Testing period list: {test_period_list}\n'
      f'Per replica batch size: {per_replica_batch_size}\n'
      f'Global batch size: {global_batch_size}\n'
      f'Num replicas: {strategy.num_replicas_in_sync}\n'
  )

  (
      veg_params_data,
      soil_params_data,
      ground_soil_params_data,
      veg_type_names,
      soil_type_names,
      ground_soil_type_names,
  ) = dataset.load_veg_soil_params_data()

  (
      dynamic_var_indices,
      dynamic_var_scales,
      static_var_indices,
      static_var_scales,
      veg_soil_type_var_indices,
      target_var_indices,
      target_var_scales,
      all_dyn_var_names,
      all_stat_var_names,
  ) = dataset.load_var_indices_and_scales(
      _DATASET_DIR.value,
      config.DYNAMIC_VARS,
      config.STATIC_VARS,
      config.TARGET_VARS,
      config.VEG_AND_SOIL_TYPES,
  )

  snow_cover_idx = np.where(all_dyn_var_names == 'snow_cover')[0][0]
  high_veg_cover_idx = np.where(all_dyn_var_names == 'high_vegetation_cover')[
      0
  ][0]
  low_veg_cover_idx = np.where(all_dyn_var_names == 'low_vegetation_cover')[0][
      0
  ]
  type_high_veg_idx = np.where(all_stat_var_names == 'type_of_high_vegetation')[
      0
  ][0]
  type_low_veg_idx = np.where(all_stat_var_names == 'type_of_low_vegetation')[
      0
  ][0]
  lai_hv_idx = np.where(all_dyn_var_names == 'leaf_area_index_high_vegetation')[
      0
  ][0]
  lai_lv_idx = np.where(all_dyn_var_names == 'leaf_area_index_low_vegetation')[
      0
  ][0]
  src_idx = np.where(all_dyn_var_names == 'skin_reservoir_content')[0][0]
  solar_sin_idx = np.where(all_dyn_var_names == 'solar_time_sin')[0][0]
  solar_cos_idx = np.where(all_dyn_var_names == 'solar_time_cos')[0][0]

  x_names, y_names = dataset.get_feature_names(
      all_dyn_var_names,
      all_stat_var_names,
      dynamic_var_indices,
      static_var_indices,
      veg_type_names,
      soil_type_names,
      ground_soil_type_names,
      target_var_indices,
  )
  logging.info(f'Input feature names: {x_names}')
  logging.info(f'Target feature names: {y_names}')

  var_in_budget_eq = [
      'instantaneous_surface_net_solar_radiation',
      'instantaneous_surface_thermal_radiation_downwards',
  ]
  var_idx_in_budget_eq = np.array(
      [x_names.index(var) for var in var_in_budget_eq]
  ).astype(np.int32)
  var_scales_in_budget_eq = np.array(
      [dataset.var_scale_dict[var]['scale'] for var in var_in_budget_eq]
  ).astype(np.float32)

  var_idx_in_budget_eq_tf = tf.constant(var_idx_in_budget_eq)
  var_scales_in_budget_eq_tf = tf.constant(var_scales_in_budget_eq)

  logging.info('Using target scales')

  logging.info(f'Target scales: {target_var_scales}')

  transform_fn = partial(
      dataset.data_post_processing,
      dynamic_scales=tf.constant(dynamic_var_scales),
      dynamic_feat_indices=tf.constant(dynamic_var_indices),
      static_scales=tf.constant(static_var_scales),
      static_feat_indices=tf.constant(static_var_indices),
      high_low_veg_soil_type_idx=tf.constant(veg_soil_type_var_indices),
      veg_params=tf.constant(veg_params_data),
      soil_params=tf.constant(soil_params_data),
      soil_params_ground_heat=tf.constant(ground_soil_params_data),
      sample_weights=tf.constant(config.SAMPLE_WEIGHTS),
      target_feat_indices=tf.constant(target_var_indices),
  )

  # --- Model Setup ---
  with strategy.scope():
    num_targets = len(config.TARGET_VARS)
    test_metrics_dict = {
        'rmse': trainer.PerFeatureMetric(
            num_targets,
            tf.keras.metrics.RootMeanSquaredError,
            name='test_rmse',
        ),
    }
    target_var_scales_tf = tf.constant(target_var_scales, dtype=tf.float32)

    model_class = utils.get_model_class(_MODEL_NAME.value)

    model = model_class(
        hid_dim=_HID_DIM.value,
        trunk_num_layers=_TRUNK_NUM_LAYERS.value,
        num_experts=config.NUM_EXPERTS,
        num_outputs=len(config.TARGET_VARS),
        expert_num_layers=_EXP_NUM_LAYERS.value,
        transformer_embed_dim=_TRANSFORMER_EMBED_DIM.value,
        transformer_ff_dim=_TRANSFORMER_FF_DIM.value,
        transformer_num_heads=_TRANSFORMER_NUM_HEADS.value,
    )
    model_path = f'{_MODEL_DIR.value}/best_model'
    logging.info(f'Loading weights from {model_path}')
    model.load_weights(f'{model_path}/variables/variables')
    logging.info('Weights loaded successfully.')

  @tf.function
  def test_step(features, labels, land_fractions, sample_weights):
    """Perform a single test step."""
    del sample_weights
    labels_shape = tf.shape(labels)
    features = tf.reshape(features, [-1, tf.shape(features)[-1]])
    labels = tf.reshape(labels, [-1, tf.shape(labels)[-1]])
    lf = tf.reshape(land_fractions, [-1, tf.shape(land_fractions)[-1]])

    model_output = model((features, lf), training=False)
    if isinstance(model_output, tuple):
      predictions, expert_weights = model_output
    else:
      predictions = model_output
      expert_weights = lf
    if len(predictions.shape) == 2:
      predictions = tf.expand_dims(predictions, axis=-1)
      expert_weights = tf.ones_like(expert_weights[:, :1])

    predictions_unscaled = (
        predictions * target_var_scales_tf[tf.newaxis, :, tf.newaxis]
    )
    weights_expanded = tf.expand_dims(expert_weights, axis=1)
    predictions_agg_unscaled = tf.reduce_sum(
        predictions_unscaled * weights_expanded, axis=-1
    )
    test_metrics_dict['rmse'].update_state(labels, predictions_agg_unscaled)
    num_experts_preds = tf.shape(predictions_unscaled)[2]
    preds_reshaped = tf.reshape(
        predictions_unscaled,
        [labels_shape[0], labels_shape[1], labels_shape[2], num_experts_preds],
    )

    b_eq_radiation_vars = tf.gather(features, var_idx_in_budget_eq_tf, axis=-1)
    b_eq_radiation_vars = b_eq_radiation_vars * var_scales_in_budget_eq_tf
    b_eq_energy_vars = predictions_agg_unscaled
    balance_residue = tf.reduce_sum(
        b_eq_radiation_vars, axis=-1
    ) + tf.reduce_sum(b_eq_energy_vars, axis=-1)

    residue_reshaped = tf.reshape(
        balance_residue, [labels_shape[0], labels_shape[1]]
    )

    return preds_reshaped, expert_weights, residue_reshaped

  # --- Testing Loop ---
  logging.info('Starting testing...')
  futures = []
  with ThreadPoolExecutor(
      max_workers=min(len(test_period_list), 10)
  ) as executor:
    for test_file in test_period_list:
      file_name = os.path.splitext(os.path.basename(test_file))[0]
      logging.info(f'Testing file: {test_file}')
      test_ds_single = dataset.create_tf_dataset(
          file_paths=[test_file],
          batch_size=global_batch_size,
          snow_cover_idx=snow_cover_idx,
          high_veg_cover_idx=high_veg_cover_idx,
          low_veg_cover_idx=low_veg_cover_idx,
          type_high_veg_idx=type_high_veg_idx,
          type_low_veg_idx=type_low_veg_idx,
          lai_hv_idx=lai_hv_idx,
          lai_lv_idx=lai_lv_idx,
          src_idx=src_idx,
          veg_params_data=veg_params_data,
          solar_sin_idx=solar_sin_idx,
          solar_cos_idx=solar_cos_idx,
          fraction_filter_fn=None,
          transform_fn=transform_fn,
          shuffle=False,
          drop_remainder=False,
      )
      test_ds_single = strategy.experimental_distribute_dataset(test_ds_single)

      predictions_list = []
      weights_list = []
      labels_list = []
      residuals_list = []
      print_once = True
      for features, labels, land_fractions, sample_weights in test_ds_single:
        if print_once:
          logging.info(f'features: {features.shape}')
          logging.info(f'labels: {labels.shape}')
          logging.info(f'land_fractions: {land_fractions.shape}')
          logging.info(f'sample_weights: {sample_weights.shape}')
          print_once = False
        preds_per_replica, weights_per_replica, residuals_per_replica = (
            strategy.run(
                test_step,
                args=(features, labels, land_fractions, sample_weights),
            )
        )
        predictions_list.extend(strategy.unwrap(preds_per_replica))
        weights_list.extend(strategy.unwrap(weights_per_replica))
        labels_list.extend(strategy.unwrap(labels))
        residuals_list.extend(strategy.unwrap(residuals_per_replica))

      futures.append(
          executor.submit(
              save_predictions_and_weights,
              predictions_list,
              weights_list,
              labels_list,
              residuals_list,
              test_file,
              file_name,
          )
      )

    for future in futures:
      future.result()

  results = {k: v.result().numpy() for k, v in test_metrics_dict.items()}
  logging.info(f'Test results: {results}')

  results_path = f'{_TEST_OUTPUT_DIR.value}/test_results.json'
  with tf.io.gfile.GFile(results_path, 'w') as f:
    json_results = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in results.items()
    }
    json.dump(json_results, f, indent=4)
  logging.info(f'Test results: {json_results}')
  logging.info(f'Test results saved to {results_path}')


if __name__ == '__main__':
  app.run(main)
