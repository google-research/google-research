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

"""Regenerate validation report and plots from existing results.csv."""

from absl import app
from absl import flags
import pandas as pd
from gem_15 import utils

_CSV_PATH = flags.DEFINE_string(
    'csv_path', None, 'Path to results.csv', required=True
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Output directory for report and plots', required=True
)
_REF_NAME = flags.DEFINE_string('ref_name', 'GEDI', 'Reference dataset name')
_IS_LIDAR = flags.DEFINE_bool('is_lidar', False, 'Whether reference is LiDAR')


def main(argv):
  del argv
  df = pd.read_csv(_CSV_PATH.value)
  utils.generate_plots_and_report(
      df, _OUTPUT_DIR.value, ref_name=_REF_NAME.value, is_lidar=_IS_LIDAR.value
  )


if __name__ == '__main__':
  app.run(main)
