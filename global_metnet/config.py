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

"""Config for Global MetNet model."""

from ml_collections import config_dict
import numpy as np

from global_metnet import geosat_mosaic
from global_metnet import hres_global
from global_metnet import normalizers
from global_metnet import preprocessors


def get_config():
  """Returns the config."""
  config = config_dict.ConfigDict()

  # --- Training parameters ---
  config.dtype = 'bfloat16'
  config.target_tds = list(range(0, 13 * 60 + 1, 15))
  config.eval_tds = list(range(0, 13 * 60 + 1, 60))
  config.train_steps = 100000
  config.checkpoint_frequency = 1000

  # Note: this uses yearly splits.
  config.directory = ''  # Path to the dataset.
  config.compressed_tf_examples = True
  config.filename = {
      'train': 'training.-*',
  }

  config.optimizer = 'adam'
  config.lr_schedule = 'step'
  config.optimizer_beta = 0.9
  config.learning_rate = 3e-4
  config.optimizer_decay_steps = '40000,80000'
  config.optimizer_decay_rate = 0.5

  config.polyak_decay = 0.9999
  config.weight_decay = 1e-1

  config.context_size_km = (18000, 36000)
  config.target_size_km = (18000, 36000)
  config.input_resolution_km = 10
  config.batch_size = 32
  config.target_features = 512
  config.fix_cond_init = True

  config.xy_splits = (4, 4)

  # --- Model parameters ---
  # ResNet with multiple cropping stages
  config.encoder_channels = [256, 384, 384, 384]
  config.encoder_num_blocks = [8, 8, 8, 8]
  config.encoder_crops = [45, 45, 45, 45]
  config.space_to_depth_factor = 2
  config.remat = True

  # --- Inputs ---
  config.inputs = {
      'imerg_early_source': preprocessors.PrecipitationRatePreprocessor(
          dataset_keys=['imerg_early_source'],
          replace_not_finite=True,
          rate_channel='precipitationCal',
      ),
      'elevation_source': preprocessors.StandardPreprocessor(
          dataset_keys=['elevation_5km_source'],
          replace_not_finite=True,
          output_normalizer_fn=lambda _: normalizers.Normalizer(  # pylint: disable=g-long-lambda
              center=[200.34], scale=[589.78]
          ),
          space_to_depth=2,
      ),
      'geosat_mosaic_source': preprocessors.StandardPreprocessor(
          dataset_keys=['geosat_mosaic_source'],
          replace_not_finite=True,
          output_normalizer_fn=lambda _: geosat_mosaic.create_mosaic_normalizer_no_clipping(
              geosat_mosaic.mosaic_channels
          ),
          filter_channels=geosat_mosaic.mosaic_channels,
          space_to_depth=2,
      ),
      'hres_fs_source': preprocessors.StandardPreprocessor(
          dataset_keys=['hres_f00_source', 'hres_fs_source'],
          replace_not_finite=True,
          output_normalizer_fn=hres_global.create_normalizer,
      ),
      'hres_fs_precip_source': preprocessors.StandardPreprocessor(
          dataset_keys=['hres_fs_precip_source'],
          replace_not_finite=True,
          output_normalizer_fn=hres_global.create_normalizer,
      ),
      # Loads the lon lat keys along with timestamp_day,
      # timestamp_hour and timestamp_month keys.
      'rest_lon_lat_with_timestamp': preprocessors.RestPreprocessor(
          lon_lat_key='lon_lat_5km_source',
          include_elevation=False,
          replace_not_finite=True,
          repeat_inputs=False,
          space_to_depth=2,
      ),
  }
  config.lon_lat_input_key = 'rest_lon_lat_with_timestamp'

  # Targets preprocessors.
  gpm_2b_15_min_res_target_preprocessor = (
      preprocessors.PrecipitationTargetPreprocessor(
          dataset_keys=['GPM_2B_15_MIN_RES_target'],
          filter_channels=['KuKaGMI_nearSurfPrecipTotRate'],
      )
  )
  imerg_final_target_preprocessor = (
      preprocessors.PrecipitationTargetPreprocessor(
          dataset_keys=['imerg_final_target'],
      )
  )
  radar_target_preprocessor = preprocessors.PrecipitationTargetPreprocessor(
      dataset_keys=['mrms_target', 'opera_target', 'jma_radar_target'],
      dataset_keys_op='sum',
  )

  rate_bins = np.concatenate(
      [np.arange(0, 4.1, 0.2), np.arange(5, 11), np.arange(15, 31, 5)]
  )
  eval_rates = [0.2, 0.4, 1, 1.6, 2.4, 4, 7, 15, 25]
  config.heads = {
      # Actual heads
      'GPM_2B_15_MIN_RES_target': preprocessors.Head(
          bins=rate_bins,
          eval_bins=eval_rates,
          resolution_km=5,
          preprocessor=gpm_2b_15_min_res_target_preprocessor,
          sample_non_nan_target_lead_times=True,
          extra_logs=True,
      ),
      'imerg_final_target': preprocessors.Head(
          bins=rate_bins,
          eval_bins=eval_rates,
          resolution_km=10,
          preprocessor=imerg_final_target_preprocessor,
          extra_logs=True,
      ),
      'radar_target': preprocessors.Head(
          bins=rate_bins,
          eval_bins=eval_rates,
          resolution_km=5,
          preprocessor=radar_target_preprocessor,
          extra_logs=True,
          timedeltas_key='opera_target_timedeltas',
      ),
  }

  config.targets = sorted(config.heads.keys())
  return config
