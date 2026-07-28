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

"""Main entry point for data processing."""

from concurrent import futures
import datetime
import io
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from united_earth_surface_fluxes.data_processing import data_processing

_OUTPUT_DIR = flags.DEFINE_string('output_dir', '../data', 'Output directory.')
_PERIOD_LIST = flags.DEFINE_string(
    'period_list', '2024-01-01', 'Times to process.'
)
_PAD_SIZE = flags.DEFINE_integer(
    'pad_size',
    4,
    'Number of padding hours to buffer out the daily period boundaries.',
)
_NUM_WORKERS = flags.DEFINE_integer('num_workers', 30, 'Number of workers.')
_SPATIAL_COORDINATES_PATH = flags.DEFINE_string(
    'spatial_coordinates_path',
    '../data/spatial_coordinates.json',
    'Path to spatial coordinates JSON file.',
)
_ERA5_ZARR_PATH = flags.DEFINE_string(
    'era5_zarr_path',
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    'Path to the ERA5 Zarr store.',
)


def main(_):
  loc_data_dict = data_processing.load_location_lat_lons(
      _SPATIAL_COORDINATES_PATH.value
  )
  all_times = _PERIOD_LIST.value.split(',')
  num_workers = _NUM_WORKERS.value
  logging.info(
      f'{_OUTPUT_DIR.value=}\n'
      f'{_PERIOD_LIST.value=}\n'
      f'{_PAD_SIZE.value=}\n'
      f'{_NUM_WORKERS.value=}\n'
      f'{_SPATIAL_COORDINATES_PATH.value=}'
  )
  logging.info(f'Number of locations: {len(loc_data_dict["all_lat_arr"]):,}')

  future_list = []
  with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    for time_t in all_times:
      thread_f = executor.submit(
          process_and_save_to_tfrecord,
          time_t,
          _OUTPUT_DIR.value,
          loc_data_dict,
          _PAD_SIZE.value,
          _ERA5_ZARR_PATH.value,
      )
      future_list.append(thread_f)

    done, not_done = futures.wait(
        future_list, return_when=futures.FIRST_EXCEPTION
    )

    for future in done:
      try:
        (
            dynamic_var_names,
            static_var_names,
            time_kept,
        ) = future.result()
      except Exception as e:
        logging.exception('Failing fast! An exception was caught.')
        for f in not_done:
          f.cancel()
        raise e
      else:
        io_buffer = io.BytesIO()
        np.savez_compressed(
            io_buffer,
            dynamic_var_names=dynamic_var_names,
            static_var_names=static_var_names,
            period=time_kept,
        )
        with tf.io.gfile.GFile(
            f'{_OUTPUT_DIR.value}/var_names_and_time_period.npz', 'wb'
        ) as f:
          f.write(io_buffer.getvalue())


def process_and_save_to_tfrecord(
    time_str,
    output_dir,
    land_fraction_dict,
    pad_size,
    era5_zarr_path,
):
  """Performs transformation on a given day and dumps out a TFRecord."""
  time_t = datetime.datetime.strptime(time_str, '%Y-%m-%d')
  period_start = time_t - datetime.timedelta(hours=pad_size)
  period_end = time_t + datetime.timedelta(hours=23 + pad_size)
  era5_data = data_processing.load_era5_for_training(era5_zarr_path)
  era5_data_time_t = era5_data.sel(time=slice(period_start, period_end))
  (
      dynamic_vars,
      dynamic_var_names,
      static_vars,
      static_var_names,
      lat_lon_arr,
      indices,
      time_kept,
  ) = data_processing.get_features(
      era5_data_time_t,
      pad_size,
      land_fraction_dict,
      verbose=False,
  )

  start_time = time.perf_counter()
  record_file = f'{output_dir}/{time_str}.tfrecord'
  indices = indices.reshape(-1, 1)
  with tf.io.TFRecordWriter(record_file) as tf_writer:
    for i, d_v in enumerate(dynamic_vars):
      s_v = static_vars[i]
      lat_lon_v = lat_lon_arr[i]
      idx_v = indices[i]
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'dynamic': tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[tf.io.serialize_tensor(d_v).numpy()]
                      )
                  ),
                  'static': tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[tf.io.serialize_tensor(s_v).numpy()]
                      )
                  ),
                  'lat_lon': tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[tf.io.serialize_tensor(lat_lon_v).numpy()]
                      )
                  ),
                  'idx': tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[tf.io.serialize_tensor(idx_v).numpy()]
                      )
                  ),
              }
          )
      )
      tf_writer.write(example.SerializeToString())
      log_step = max(1, len(indices) // 10)
      if i % log_step == 0:
        logging.info(
            f'Completed {i:,} of {time_str} in'
            f' {(time.perf_counter() - start_time)/60} minutes'
            f' {lat_lon_arr.shape=:} {dynamic_vars.shape=} {static_vars.shape=}'
        )

  logging.info(
      f'Completed {time_str} in '
      f'{(time.perf_counter() - start_time)/60:.2f} minutes'
  )

  return (
      dynamic_var_names,
      static_var_names,
      time_kept,
  )


if __name__ == '__main__':
  app.run(main)
