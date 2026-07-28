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

"""Training module."""

from collections import defaultdict
from functools import partial
import itertools
import json
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from united_earth_surface_fluxes import config
from united_earth_surface_fluxes import dataset
from united_earth_surface_fluxes import trainer
from united_earth_surface_fluxes import utils

# --- Flags ---
_OUTPUT_DIR = flags.DEFINE_string('output_dir', './output', 'Output directory.')
_SEED = flags.DEFINE_integer('seed', 1000, 'Seed for reproducibility.')
_DATASET_DIR = flags.DEFINE_string(
    'dataset_dir', './data', 'Dataset directory.'
)
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', 500, 'Number of epochs.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 80, 'Batch size.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
_TRUNK_NUM_LAYERS = flags.DEFINE_integer(
    'trunk_num_layers', 1, 'Number of layers in model trunk.'
)
_EXP_NUM_LAYERS = flags.DEFINE_integer(
    'exp_num_layers', 2, 'Number of layers in model head.'
)
_HID_DIM = flags.DEFINE_integer('hid_dim', 256, 'Hidden dimension.')
_BUDGET_EQ_WEIGHT = flags.DEFINE_float(
    'budget_eq_weight', 1e-3, 'Budget equation weight.'
)
_TRANSFORMER_EMBED_DIM = flags.DEFINE_integer(
    'transformer_embed_dim',
    128,
    'Transformer embed dimension.',
)
_TRANSFORMER_FF_DIM = flags.DEFINE_integer(
    'transformer_ff_dim',
    512,
    'Transformer feed forward dimension.',
)
_TRANSFORMER_NUM_HEADS = flags.DEFINE_integer(
    'transformer_num_heads',
    4,
    'Transformer number of heads.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name', None, 'Model to train.', required=True
)
_FRACTION_FILTER_TYPE = flags.DEFINE_string(
    'fraction_filter_type',
    'none',
    'Type of fraction filter to apply to the dataset. Options: "none", "pure",'
    ' "mixed".',
)
_TRAIN_PERIOD_LIST = flags.DEFINE_string(
    'train_period_list',
    '2024-02-07,2024-04-07,2024-06-07,2024-08-07,2024-10-07,2024-12-07',
    'Training period list (supports comma-separated or start:end colon range).',
)
_VAL_PERIOD_LIST = flags.DEFINE_string(
    'val_period_list',
    '2024-02-15,2024-04-15,2024-06-15,2024-08-15,2024-10-15,2024-12-15',
    (
        'Validation period list (supports comma-separated or start:end colon'
        ' range).'
    ),
)


def _validate_var_lists():
  """Ensures no overlap between feature variable lists."""
  feature_lists = {
      'DYNAMIC_VARS': config.DYNAMIC_VARS,
      'STATIC_VARS': config.STATIC_VARS,
      'VEG_AND_SOIL_TYPES': config.VEG_AND_SOIL_TYPES,
      'TARGET_VARS': config.TARGET_VARS,
  }
  for (name1, list1), (name2, list2) in itertools.combinations(
      feature_lists.items(), 2
  ):
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


def main(_):
  """Main training function."""
  utils.set_seed(_SEED.value)
  _validate_var_lists()
  _validate_var_scale_dict()
  balance_dataset = _FRACTION_FILTER_TYPE.value == 'pure'

  strategy = _get_distribution_strategy()
  per_replica_batch_size = _BATCH_SIZE.value
  global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
  logging.info(f'{per_replica_batch_size=}, {global_batch_size=}')

  tf.io.gfile.makedirs(_OUTPUT_DIR.value)
  logging.info(f'Output dir: {_OUTPUT_DIR.value}')
  # --- Dataset Setup ---
  train_dates = utils.parse_period_list(_TRAIN_PERIOD_LIST.value)
  train_period_list = [
      f'{_DATASET_DIR.value}/{d}.tfrecord' for d in train_dates
  ]
  logging.info(f'train_period_list: {len(train_period_list)} dates')

  val_dates = utils.parse_period_list(_VAL_PERIOD_LIST.value)
  val_period_list = [f'{_DATASET_DIR.value}/{d}.tfrecord' for d in val_dates]
  logging.info(f'val_period_list: {len(val_period_list)} dates')

  logging.info(
      f'Output dir: {_OUTPUT_DIR.value}\n'
      f'Seed: {_SEED.value}\n'
      f'Training period list: {train_period_list}\n'
      f'Validation period list: {val_period_list}\n'
      f'Num epochs: {_NUM_EPOCHS.value}\n'
      f'Per replica batch size: {per_replica_batch_size}\n'
      f'Global batch size: {global_batch_size}\n'
      f'Num replicas: {strategy.num_replicas_in_sync}\n'
      f'Learning rate: {_LEARNING_RATE.value}\n'
      f'Num trunk layers: {_TRUNK_NUM_LAYERS.value}\n'
      f'Expert number of layers: {_EXP_NUM_LAYERS.value}\n'
      f'Hidden dimension: {_HID_DIM.value}\n'
      f'Budget eq weight: {_BUDGET_EQ_WEIGHT.value}\n'
      f'Model name: {_MODEL_NAME.value}\n'
      f'Balance dataset: {balance_dataset}\n'
  )

  if _FRACTION_FILTER_TYPE.value == 'none':
    ff_fn = None
    ff_fn_val = None
  elif _FRACTION_FILTER_TYPE.value == 'pure':
    ff_fn = dataset.filter_pure_fractions
    ff_fn_val = dataset.filter_pure_fractions
  elif _FRACTION_FILTER_TYPE.value == 'mixed':
    ff_fn = dataset.validation_fraction_filter
    ff_fn_val = dataset.validation_fraction_filter
  else:
    raise ValueError(
        f'Unknown fraction filter type: {_FRACTION_FILTER_TYPE.value}'
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
  logging.info(len(all_dyn_var_names))
  logging.info(f'Target scales: {target_var_scales}')

  var_in_budget_eq = [
      'instantaneous_surface_net_solar_radiation',
      'instantaneous_surface_thermal_radiation_downwards',
  ]
  var_idx_in_budget_eq = np.array(
      [x_names.index(var) for var in var_in_budget_eq]
  ).astype(np.int64)
  var_scales_in_budget_eq = np.array(
      [dataset.var_scale_dict[var]['scale'] for var in var_in_budget_eq]
  ).astype(np.float32)

  feature_names_path = f'{_OUTPUT_DIR.value}/feature_names.json'
  with tf.io.gfile.GFile(feature_names_path, 'w') as f:
    json.dump({'x_names': x_names, 'y_names': y_names}, f, indent=2)
  logging.info(f'Feature names saved to {feature_names_path}')

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
      sample_weights=tf.constant(
          np.ones(6) if balance_dataset else config.SAMPLE_WEIGHTS,
          dtype=tf.float32,
      ),
      target_feat_indices=tf.constant(target_var_indices),
  )

  train_ds = dataset.create_tf_dataset(
      file_paths=train_period_list,
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
      fraction_filter_fn=ff_fn,
      transform_fn=transform_fn,
      shuffle=True,
      balance_dataset=balance_dataset,
  )
  val_ds = dataset.create_tf_dataset(
      file_paths=val_period_list,
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
      fraction_filter_fn=ff_fn_val,
      transform_fn=transform_fn,
      shuffle=False,
      balance_dataset=False,
  )

  train_ds = strategy.experimental_distribute_dataset(train_ds)
  val_ds = strategy.experimental_distribute_dataset(val_ds)

  # --- Model Setup ---
  with strategy.scope():
    loss_object = tf.math.squared_difference
    optimizer = tf.keras.optimizers.AdamW(learning_rate=_LEARNING_RATE.value)

    num_targets = len(config.TARGET_VARS)

    model_class = utils.get_model_class(_MODEL_NAME.value)
    num_experts = config.NUM_EXPERTS
    model_kwargs = {
        'hid_dim': _HID_DIM.value,
        'trunk_num_layers': _TRUNK_NUM_LAYERS.value,
        'num_experts': num_experts,
        'num_outputs': len(config.TARGET_VARS),
        'expert_num_layers': _EXP_NUM_LAYERS.value,
        'transformer_embed_dim': _TRANSFORMER_EMBED_DIM.value,
        'transformer_ff_dim': _TRANSFORMER_FF_DIM.value,
        'transformer_num_heads': _TRANSFORMER_NUM_HEADS.value,
    }
    model = model_class(**model_kwargs)
    # Build model weights by calling it on dummy input
    model((tf.ones((1, len(x_names))), tf.ones((1, num_experts))))

    train_metrics_dict = {
        'rmse': trainer.PerFeatureMetric(
            num_targets,
            tf.keras.metrics.RootMeanSquaredError,
            name='train_rmse',
        ),
        'loss': trainer.PerFeatureMetric(
            num_targets, tf.keras.metrics.Mean, name='train_loss'
        ),
        'budget_loss': tf.keras.metrics.Mean(name='train_budget_loss'),
        'total_loss': tf.keras.metrics.Mean(name='train_total_loss'),
    }
    val_metrics_dict = {
        'rmse': trainer.PerFeatureMetric(
            num_targets, tf.keras.metrics.RootMeanSquaredError, name='val_rmse'
        ),
        'loss': trainer.PerFeatureMetric(
            num_targets, tf.keras.metrics.Mean, name='val_loss'
        ),
        'budget_loss': tf.keras.metrics.Mean(name='val_budget_loss'),
        'total_loss': tf.keras.metrics.Mean(name='val_total_loss'),
    }

  lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
      monitor='val_total_loss',
      factor=0.5,
      patience=10,
      min_lr=1e-6,
      verbose=1,
  )
  trainer_obj = trainer.Trainer(
      model=model,
      optimizer=optimizer,
      lr_reducer=lr_reducer,
      loss_fn=loss_object,
      outdir_path=_OUTPUT_DIR.value,
      target_var_scales=target_var_scales,
      train_metrics_dict=train_metrics_dict,
      val_metrics_dict=val_metrics_dict,
      global_batch_size=global_batch_size,
      budget_eq_weight=_BUDGET_EQ_WEIGHT.value,
      var_idx_in_budget_eq=var_idx_in_budget_eq,
      var_scales_in_budget_eq=var_scales_in_budget_eq,
  )

  # --- Checkpoint Setup ---
  checkpoint_dir = f'{_OUTPUT_DIR.value}/tf_ckps'
  checkpoint = tf.train.Checkpoint(trainer=trainer_obj)
  manager = tf.train.CheckpointManager(
      checkpoint, checkpoint_dir, max_to_keep=3
  )
  checkpoint.restore(manager.latest_checkpoint)
  experiment_metric_dict = defaultdict(list)

  if manager.latest_checkpoint:
    logging.info(
        f'Restored from {manager.latest_checkpoint}\n'
        f'{trainer_obj.best_val_metric.numpy()=:}\n'
        f'{trainer_obj.best_val_metric_epoch.numpy()=:}'
    )
    csv_results_path = f'{_OUTPUT_DIR.value}/results.csv'
    if tf.io.gfile.exists(csv_results_path):
      experiment_metric_dict = pd.read_csv(csv_results_path).to_dict(
          orient='list'
      )
  else:
    logging.info('Initializing from scratch.')

  # --- Training Loop ---
  total_start_time = time.perf_counter()
  logging.info(f'Starting training from epoch {trainer_obj.epoch.numpy()}')

  for epoch in range(trainer_obj.epoch.numpy(), _NUM_EPOCHS.value):
    start_time = time.perf_counter()
    epoch_results = trainer_obj.train_and_validate_epoch(
        train_ds, val_ds, strategy
    )
    epoch_time = time.perf_counter() - start_time

    metrics_str_parts = []
    for k, v in epoch_results.items():
      if 'lr' in k or 'budget' in k:
        metrics_str_parts.append(f'{k}={v}')
      elif isinstance(v, np.ndarray):
        metrics_str_parts.append(f'{k}=[{", ".join(f"{x:.4f}" for x in v)}]')
      else:
        metrics_str_parts.append(f'{k}={v:.4f}')
    metrics_str = ' '.join(metrics_str_parts)

    logging.info(
        f'Epoch {epoch} {metrics_str} '
        f'best_metric={trainer_obj.best_val_metric.numpy():.4f} '
        f'epoch={trainer_obj.best_val_metric_epoch.numpy()} '
        f'({epoch_time/60:.2f}m)'
    )

    experiment_metric_dict['epoch'].append(epoch)
    for k, v in epoch_results.items():
      if isinstance(v, np.ndarray) and v.size > 1:
        for i, val in enumerate(v):
          experiment_metric_dict[f'{k}_{i}'].append(val)
      else:
        experiment_metric_dict[k].append(v)
    pd.DataFrame(experiment_metric_dict).to_csv(
        f'{_OUTPUT_DIR.value}/results.csv', index=False
    )
    manager.save()

    if trainer_obj.model_stop_training():
      logging.info('Early stopping triggered.')
      break

  logging.info(
      f' {(time.perf_counter() - total_start_time)/60:.4f}min',
  )


if __name__ == '__main__':
  app.run(main)
