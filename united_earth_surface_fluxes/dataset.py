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

"""Dataset creation, feature transformation, and balancing pipeline."""

# pylint: disable=logging-fstring-interpolation
# pylint: disable=g-importing-member

from functools import partial

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from united_earth_surface_fluxes import config

var_scale_dict = {
    '10m_u_component_of_wind': {'scale': 10},
    '10m_v_component_of_wind': {'scale': 10},
    'skin_temperature': {'scale': 350},
    '2m_temperature': {'scale': 350},
    '2m_dewpoint_temperature': {'scale': 350},
    'soil_temperature_level_1': {'scale': 350},
    'soil_temperature_level_2': {'scale': 350},
    'soil_temperature_level_3': {'scale': 350},
    'soil_temperature_level_4': {'scale': 350},
    'boundary_layer_height': {'scale': 3_000},
    'skin_reservoir_content': {'scale': 1e-3},
    'leaf_area_index_high_vegetation': {'scale': 10},
    'leaf_area_index_low_vegetation': {'scale': 10},
    'type_of_high_vegetation': {'scale': 1},
    'type_of_low_vegetation': {'scale': 1},
    'soil_type': {'scale': 1},
    'surface_pressure': {'scale': 1e5},
    'volumetric_soil_water_layer_1': {'scale': 1},
    'volumetric_soil_water_layer_2': {'scale': 1},
    'volumetric_soil_water_layer_3': {'scale': 1},
    'volumetric_soil_water_layer_4': {'scale': 1},
    'instantaneous_ground_heat_flux': {
        'scale': 1e6 / 3600,
    },
    'instantaneous_surface_latent_heat_flux': {
        'scale': 1e6 / 3600,
    },
    'instantaneous_surface_sensible_heat_flux': {
        'scale': 1e6 / 3600,
    },
    'instantaneous_surface_net_solar_radiation': {
        'scale': 2 * 1e6 / 3600,
    },
    'instantaneous_surface_net_thermal_radiation': {
        'scale': 1e6 / 3600,
    },
    'instantaneous_surface_thermal_radiation_downwards': {
        'scale': 1e6 / 3600,
    },
    'instantaneous_surface_thermal_radiation_upwards': {
        'scale': 1e6 / 3600,
    },
    '2m_specific_humidity': {'scale': 1e-2},
    'vapor_pressure': {'scale': 611.2 * 6},
    'vapor_pressure_deficit': {'scale': 610.8 * 10},
    'solar_time_sin': {'scale': 1},
    'solar_time_cos': {'scale': 1},
}


def load_veg_soil_params_data():
  """Loads vegetation and soil physical parameters from CSV files."""
  with tf.io.gfile.GFile(config.VEG_PARAMS_PATH, 'r') as f:
    veg_params_df = pd.read_csv(f)
    veg_params_data = veg_params_df.values
    veg_type_names = veg_params_df.columns.values

  with tf.io.gfile.GFile(config.SOIL_PARAMS_PATH, 'r') as f:
    soil_params_df = pd.read_csv(f)
    soil_params_data = soil_params_df.values
    soil_type_names = soil_params_df.columns.values

  with tf.io.gfile.GFile(config.GROUND_HEAT_PARAMS_PATH, 'r') as f:
    ground_heat_df = pd.read_csv(f)
    ground_soil_params_data = ground_heat_df.values
    ground_soil_type_names = ground_heat_df.columns.values

  return (
      veg_params_data.astype(np.float32),
      soil_params_data.astype(np.float32),
      ground_soil_params_data.astype(np.float32),
      veg_type_names,
      soil_type_names,
      ground_soil_type_names,
  )


def get_var_idx_and_scale(all_var_names, var_names):
  var_indices = np.where(np.isin(all_var_names, var_names))[0]
  assert len(var_indices) == len(var_names)
  var_in_all_names = all_var_names[var_indices]
  var_scales = np.asarray(
      [var_scale_dict[var].get('scale', 1) for var in var_in_all_names]
  ).astype(np.float32)
  return var_indices, var_scales


def load_var_indices_and_scales(
    dir_path,
    dyn_var_names,
    stat_var_names,
    target_var_names,
    veg_soil_type_var_names,
):
  """Loads variable indices and scales from a directory."""
  with tf.io.gfile.GFile(
      f'{dir_path}/var_names_and_time_period.npz', 'rb'
  ) as f:
    var_names_and_time = np.load(f)
    all_dyn_var_names = var_names_and_time['dynamic_var_names']
    all_stat_var_names = var_names_and_time['static_var_names']

  dynamic_var_indices, dynamic_var_scales = get_var_idx_and_scale(
      all_dyn_var_names, dyn_var_names
  )
  static_var_indices, static_var_scales = get_var_idx_and_scale(
      all_stat_var_names, stat_var_names
  )
  veg_soil_type_var_indices, _ = get_var_idx_and_scale(
      all_stat_var_names, veg_soil_type_var_names
  )
  target_var_indices, target_var_scales = get_var_idx_and_scale(
      all_dyn_var_names, target_var_names
  )
  return (
      dynamic_var_indices,
      dynamic_var_scales,
      static_var_indices,
      static_var_scales,
      veg_soil_type_var_indices,
      target_var_indices,
      target_var_scales,
      all_dyn_var_names,
      all_stat_var_names,
  )


@tf.function
def data_post_processing(
    data_dict,
    dynamic_scales,
    dynamic_feat_indices,
    static_scales,
    static_feat_indices,
    high_low_veg_soil_type_idx,
    veg_params,
    soil_params,
    soil_params_ground_heat,
    sample_weights,
    target_feat_indices,
):
  """Performs feature engineering and shaping on a batch of data.

  Args:
    data_dict: Dictionary of data tensors.
    dynamic_scales: Scaling factors for dynamic data.
    dynamic_feat_indices: Feature indices for dynamic data.
    static_scales: Scaling factors for static data.
    static_feat_indices: Feature indices for static data.
    high_low_veg_soil_type_idx: Indices representing vegetation and soil.
    veg_params: Vegetation parameters array.
    soil_params: Soil parameters array.
    soil_params_ground_heat: Soil parameters for ground heat.
    sample_weights: Array of sample weights.
    target_feat_indices: Feature indices for target variables.

  Returns:
    Transformed X, targeted Y, land fractions, and sample weights.
  """
  slice_start_idx = 0
  # --- Feature Engineering for Dynamic Data ---
  dynamic_data = tf.gather(data_dict['dynamic'], dynamic_feat_indices, axis=-1)
  dynamic_data = dynamic_data / dynamic_scales  # (batch, time, dyn_vars)
  dynamic_data = dynamic_data[:, slice_start_idx:]

  # --- Feature Engineering for Static Data ---
  static_data = tf.gather(data_dict['static'], static_feat_indices, axis=-1)
  static_data = static_data / static_scales  # (batch, stat_vars)

  h_v_l_v_soil_type = tf.gather(
      data_dict['static'], high_low_veg_soil_type_idx, axis=-1
  )
  h_v_l_v_soil_type = tf.cast(tf.round(h_v_l_v_soil_type), tf.int32)

  h_v_data = tf.gather(veg_params, h_v_l_v_soil_type[:, 0])
  l_v_data = tf.gather(veg_params, h_v_l_v_soil_type[:, 1])
  soil_data = tf.gather(soil_params, h_v_l_v_soil_type[:, 2])
  soil_data_ground = tf.gather(soil_params_ground_heat, h_v_l_v_soil_type[:, 2])

  soil_and_veg_params = tf.concat(
      [h_v_data, l_v_data, soil_data, soil_data_ground], axis=-1
  )
  static_data = tf.concat([static_data, soil_and_veg_params], axis=-1)

  # --- Land Fraction Data ---
  batch_land_fraction_tiled = data_dict['land_fraction'][:, slice_start_idx:]

  # --- Sample Weights ---
  batch_sample_weights_tiled = tf.reduce_sum(
      batch_land_fraction_tiled * sample_weights, axis=-1, keepdims=True
  )

  # --- Final Assembly ---
  time_steps = tf.shape(dynamic_data)[1]
  static_data = tf.tile(tf.expand_dims(static_data, axis=1), [1, time_steps, 1])

  x_data = tf.concat([dynamic_data, static_data], axis=-1)

  target_data = tf.gather(data_dict['dynamic'], target_feat_indices, axis=-1)
  target_data = target_data[:, slice_start_idx:]

  return (
      tf.cast(x_data, tf.float32),
      tf.cast(target_data, tf.float32),
      tf.cast(batch_land_fraction_tiled, tf.float32),
      tf.cast(batch_sample_weights_tiled, tf.float32),
  )


def get_feature_names(
    all_dyn_var_names,
    all_stat_var_names,
    dynamic_var_indices,
    static_var_indices,
    veg_type_names,
    soil_type_names,
    ground_soil_type_names,
    target_var_indices,
):
  """Returns feature names for inputs (x) and targets (y)."""
  dyn_names = all_dyn_var_names[dynamic_var_indices]
  stat_names = all_stat_var_names[static_var_indices]
  target_names = all_dyn_var_names[target_var_indices]

  x_feature_names = []
  x_feature_names.extend([str(name) for name in dyn_names])
  x_feature_names.extend(stat_names)
  x_feature_names.extend([f'high_veg_{name}' for name in veg_type_names])
  x_feature_names.extend([f'low_veg_{name}' for name in veg_type_names])
  x_feature_names.extend([f'soil_{name}' for name in soil_type_names])
  x_feature_names.extend(
      [f'ground_soil_{name}' for name in ground_soil_type_names]
  )

  return x_feature_names, target_names.tolist()


def parse_tfrecord_fn(example_proto):
  """Parses a TFRecord containing feature and label data."""
  feature_description = {
      'dynamic': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
      'static': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
      'lat_lon': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
      'idx': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  }
  parsed_example = tf.io.parse_single_example(
      example_proto, feature_description
  )
  return {
      'dynamic': tf.io.parse_tensor(parsed_example['dynamic'], tf.float32),
      'static': tf.io.parse_tensor(parsed_example['static'], tf.float32),
      'idx': tf.io.parse_tensor(parsed_example['idx'], tf.int64),
      'lat_lon': tf.io.parse_tensor(parsed_example['lat_lon'], tf.float32),
  }


def compute_fractions_era5_data(
    data_dict,
    snow_cover_idx,
    high_veg_cover_idx,
    low_veg_cover_idx,
    type_high_veg_idx,
    type_low_veg_idx,
    lai_hv_idx,
    lai_lv_idx,
    src_idx,
    veg_params_data,
):
  """Computes fraction arrays of land covers based on ERA5 conventions."""
  snow_cover_data = tf.gather(data_dict['dynamic'], snow_cover_idx, axis=-1)
  high_veg_cover_data = tf.gather(
      data_dict['dynamic'], high_veg_cover_idx, axis=-1
  )
  low_veg_cover_data = tf.gather(
      data_dict['dynamic'], low_veg_cover_idx, axis=-1
  )
  type_h_veg_data = tf.math.round(
      tf.gather(data_dict['static'], type_high_veg_idx, axis=-1)
  )
  type_h_veg_data = tf.cast(type_h_veg_data, tf.int64)
  type_low_veg_data = tf.math.round(
      tf.gather(data_dict['static'], type_low_veg_idx, axis=-1)
  )
  type_low_veg_data = tf.cast(type_low_veg_data, tf.int64)
  lai_hv_data = tf.gather(data_dict['dynamic'], lai_hv_idx, axis=-1)
  lai_lv_data = tf.gather(data_dict['dynamic'], lai_lv_idx, axis=-1)
  src_data = tf.gather(data_dict['dynamic'], src_idx, axis=-1)

  # Compute fractions
  # ['rs,min (sm−1)'; 'cveg'; 'ar'; 'br']
  h_cveg = tf.gather(veg_params_data, type_h_veg_data, axis=0)[1]
  l_cveg = tf.gather(veg_params_data, type_low_veg_data, axis=0)[1]

  c_h = high_veg_cover_data * h_cveg
  c_l = low_veg_cover_data * l_cveg
  c_b = 1 - c_h - c_l

  c_sn = snow_cover_data
  w_l = src_data
  w_lmax = 0.0002
  w_lm = w_lmax * (c_b + c_h * lai_hv_data + c_l * lai_lv_data)
  c_1 = tf.minimum(1.0, tf.math.divide_no_nan(w_l, w_lm))

  no_snow = 1 - c_sn
  no_inter_w = 1 - c_1
  c_3 = no_snow * c_1  # Intercepted Water
  c_4 = no_snow * no_inter_w * c_l  # Dry Low Vegetation
  c_5 = c_sn * (1 - c_h)  # Exposed snow
  c_6 = no_snow * no_inter_w * c_h  # Dry High Vegetation
  c_7 = c_sn * c_h  # Shaded snow
  c_8 = no_snow * no_inter_w * c_b  # Dry bare ground

  fraction_data = tf.stack([c_3, c_4, c_5, c_6, c_7, c_8], axis=-1)
  data_dict['land_fraction'] = fraction_data

  return data_dict


def filter_pure_fractions(data_dict):
  fractions = data_dict['land_fraction']
  max_fraction_per_time_step = tf.reduce_any(
      fractions
      == tf.constant(config.PURE_FRACTION_THRESHOLDS, dtype=fractions.dtype),
      axis=-1,
  )
  return tf.reduce_all(max_fraction_per_time_step)


def validation_fraction_filter(data_dict):
  """Keeps samples where all land fractions are <= 0.5."""
  fractions = data_dict['land_fraction']
  return tf.reduce_all(fractions <= 0.5)


@tf.function
def get_lst_from_sin_cos_tf(
    solar_time_sin, solar_time_cos
):
  """Retrieves Local Solar Time (LST) in hours [0, 24) from sin/cos tensors.

  Assumes sin/cos were generated using angle = 2*pi*(LST+0.5)/24.
  This function is suitable for use in tf.data.Dataset.map().

  Args:
      solar_time_sin: Tensor containing sin(angle).
      solar_time_cos: Tensor containing cos(angle).

  Returns:
      Tensor containing LST in hours [0, 24).
  """
  pi = tf.constant(np.pi, dtype=tf.float32)

  solar_time_sin = tf.cast(solar_time_sin, tf.float32)
  solar_time_cos = tf.cast(solar_time_cos, tf.float32)

  angle_rad = tf.math.atan2(solar_time_sin, solar_time_cos)
  angle_rad_0_2pi = tf.where(angle_rad < 0.0, angle_rad + 2.0 * pi, angle_rad)

  # Invert the formula: solar_time = angle * 12/pi - 0.5
  # This yields LST in range [-0.5, 23.5)
  solar_hour_raw = angle_rad_0_2pi * (12.0 / pi) - 0.5
  solar_hour = tf.math.floormod(solar_hour_raw, 24.0)

  return solar_hour


def solar_time_filter_map_fn(data_dict, solar_sin_idx, solar_cos_idx):
  """Filters data to keep only hours 10-15."""
  solar_sin = data_dict['dynamic'][:, solar_sin_idx]
  solar_cos = data_dict['dynamic'][:, solar_cos_idx]
  lst = get_lst_from_sin_cos_tf(solar_sin, solar_cos)
  time_mask = tf.logical_and(
      lst >= config.SOLAR_START_HOUR, lst <= config.SOLAR_END_HOUR
  )
  indices = tf.reshape(tf.where(time_mask), [-1])
  data_dict['dynamic'] = tf.gather(data_dict['dynamic'], indices)
  return data_dict


def create_tf_dataset(
    file_paths,
    batch_size,
    snow_cover_idx,
    high_veg_cover_idx,
    low_veg_cover_idx,
    type_high_veg_idx,
    type_low_veg_idx,
    lai_hv_idx,
    lai_lv_idx,
    src_idx,
    veg_params_data,
    solar_sin_idx,
    solar_cos_idx,
    fraction_filter_fn=None,
    transform_fn=None,
    shuffle=False,
    drop_remainder=True,
    balance_dataset=False,
):
  """Creates and configures a tf.data.Dataset built from TFRecord paths."""
  if shuffle:
    np.random.shuffle(file_paths)
  ds = tf.data.TFRecordDataset(file_paths, num_parallel_reads=tf.data.AUTOTUNE)
  ds = ds.map(
      parse_tfrecord_fn,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,
  )
  ds = ds.map(
      partial(
          solar_time_filter_map_fn,
          solar_sin_idx=solar_sin_idx,
          solar_cos_idx=solar_cos_idx,
      ),
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,
  )
  ds = ds.map(
      partial(
          compute_fractions_era5_data,
          snow_cover_idx=snow_cover_idx,
          high_veg_cover_idx=high_veg_cover_idx,
          low_veg_cover_idx=low_veg_cover_idx,
          type_high_veg_idx=type_high_veg_idx,
          type_low_veg_idx=type_low_veg_idx,
          lai_hv_idx=lai_hv_idx,
          lai_lv_idx=lai_lv_idx,
          src_idx=src_idx,
          veg_params_data=veg_params_data,
      ),
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,
  )
  if fraction_filter_fn is not None:
    logging.info('Applying fraction filter: %s', fraction_filter_fn.__name__)
    ds = ds.filter(fraction_filter_fn)

  if transform_fn is not None:
    # Batch before transform_fn for vectorization
    ds = ds.batch(batch_size)
    ds = ds.map(
        transform_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )  # (batch_size, time-1, all_vars)
    # Unbatch because transform_fn flattens the batch and time dimensions
    ds = ds.unbatch()  # (time-1, all_vars)

  if balance_dataset:
    # --- Dataset Balancing Logic ---
    # When enabled, this loops through the dataset, evaluates the dominant
    # land cover type fraction per sample, and splits the data into 6 discrete
    # buckets (one for each land cover type).
    #
    # These buckets are then zipped together, and evenly interleaved into
    # balanced batches. This guarantees the model trains on an equal proportion
    # of each terrain type simultaneously, completely bypassing original
    # geographic imbalances in the dataset.
    if transform_fn is None:
      raise ValueError(
          'balance_dataset=True requires transform_fn to be provided.'
      )
    logging.info('Loading dataset to memory for balancing...')
    samples_per_class = [[] for _ in range(config.NUM_EXPERTS)]
    for sample in ds:
      fractions = sample[2]
      mean_fractions = tf.reduce_mean(fractions, axis=0)
      class_idx = tf.argmax(mean_fractions).numpy()
      samples_per_class[class_idx].append(sample)
    logging.info('Finished loading dataset to memory.')

    class_lengths = [len(samples) for samples in samples_per_class]
    sorted_class_indices = np.argsort(class_lengths)[::-1]

    balanced_batch_size = max(1, batch_size // config.NUM_EXPERTS)
    datasets = []
    for i, class_idx in enumerate(sorted_class_indices):
      logging.info(
          'Class %s has %s samples.',
          class_idx,
          class_lengths[class_idx],
      )
      if samples_per_class[class_idx]:
        class_ds = tf.data.Dataset.from_generator(
            lambda class_idx=class_idx: samples_per_class[class_idx],
            output_signature=ds.element_spec,
        )
      else:
        class_ds = tf.data.Dataset.from_tensor_slices(
            tuple(
                tf.zeros(shape=(0,) + s.shape, dtype=s.dtype)
                for s in ds.element_spec
            )
        )

      logging.info('Caching class %s dataset to memory', class_idx)
      class_ds = class_ds.cache()

      if shuffle:
        buffer_size = int(600_000)
        class_ds = class_ds.shuffle(buffer_size=buffer_size)

      if i > 0:
        class_ds = class_ds.repeat()

      class_ds = class_ds.batch(
          batch_size=balanced_batch_size, drop_remainder=drop_remainder
      )
      datasets.append(class_ds.prefetch(tf.data.AUTOTUNE))
    return tf.data.Dataset.zip(tuple(datasets))

  logging.info('Caching dataset to memory')
  ds = ds.cache()

  if shuffle:
    buffer_size = int(600_000)  # Adjust based on dataset size and memory
    ds = ds.shuffle(buffer_size=buffer_size)

  ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)

  return ds
