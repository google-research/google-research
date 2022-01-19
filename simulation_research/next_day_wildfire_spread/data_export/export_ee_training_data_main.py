# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

r"""Main program to export training data for the wildfire ML project."""

import json
import os

from absl import app
from absl import flags
from absl import logging
from absl.testing import flagsaver
import ee

from simulation_research.next_day_wildfire_spread.data_export import export_ee_data


FLAGS = flags.FLAGS
flags.DEFINE_string('bucket', '', 'Output file bucket on GCP.')
flags.DEFINE_string('folder', 'tmp',
                    'Output folder path under the file bucket.')
flags.DEFINE_string('start_date', '2020-01-01',
                    'Start date to export in YYYY-MM-DD.')
flags.DEFINE_string('end_date', '2021-01-01',
                    'End date to export in YYYY-MM-DD (inclusive).')
flags.DEFINE_string('prefix', '', 'File prefix for output files.')
flags.DEFINE_string(
    'config_dir', '',
    'If non-empty, a JSON file of flag values is written to this directory.')
flags.DEFINE_integer('kernel_size', 64,
                     'Size of the exported tiles in pixels (square).')
flags.DEFINE_integer('sampling_scale', 1000,
                     'Resolution at which to export the data (in meters).')
flags.DEFINE_float('eval_split_ratio', 0.1,
                   'Evaluation dataset as a proportion of the total dataset.')
flags.DEFINE_integer('num_samples_per_file', 2000,
                     'Approximate number of samples per tf.Examples file.')

f_open = open



def main(argv):
  if len(argv) > 6:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Starting wildfire ee export job...')
  logging.info('bucket=%s', FLAGS.bucket)
  logging.info('folder=%s', FLAGS.folder)
  logging.info('start_date=%s', FLAGS.start_date)
  logging.info('end_date=%s', FLAGS.end_date)
  logging.info('prefix=%s', FLAGS.prefix)
  logging.info('eval_split_ratio=%f', FLAGS.eval_split_ratio)

  ee.Initialize()
  logging.info('ee authenticated!')

  start_date = ee.Date(FLAGS.start_date)
  end_date = ee.Date(FLAGS.end_date)
  export_ee_data.export_ml_datasets(
      bucket=FLAGS.bucket,
      folder=FLAGS.folder,
      start_date=start_date,
      end_date=end_date,
      prefix=FLAGS.prefix,
      kernel_size=FLAGS.kernel_size,
      sampling_scale=FLAGS.sampling_scale,
      eval_split_ratio=FLAGS.eval_split_ratio,
      num_samples_per_file=FLAGS.num_samples_per_file,
  )

  saved_flag_values = flagsaver.save_flag_values()
  # Save the names and values of the flags as a json file in a local folder.
  # Note that this includes more flags than just those defined in this file,
  # since FLAGS includes many other flags, including default flags.
  saved_flag_values = {
      key: flag_dict['_value'] for key, flag_dict in saved_flag_values.items()
  }
  saved_flag_path = os.path.join(FLAGS.config_dir, 'export_flags.json')
  json_str = json.dumps(saved_flag_values, indent=2) + '\n'
  with f_open(saved_flag_path, 'w') as f:
    f.write(json_str)

  logging.info('Ending wildfire ee export job!'
               'Note that the export job may continue in the background by EE.')


if __name__ == '__main__':
  logging.use_python_logging()
  app.run(main)
