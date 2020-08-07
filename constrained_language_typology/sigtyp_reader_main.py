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

r"""Reader for the format provided by SIGTYP 2020 Shared Task.

More information on the format is available here:
  https://sigtyp.github.io/st2020.html

Example:
--------
 Clone the GitHub data to ST2020_DIR. Then run:

 > ST2020_DIR=...
 > python3 sigtyp_reader_main.py --sigtyp_dir ${ST2020_DIR}/data \
    --output_dir ${OUTPUT_DIR}

 The above will create "train.csv", "dev.csv" and "test_blinded.csv" files
 converted from the format provided by SIGTYP. Our models should be able to
 injest these csv files. Along each of the above files, an accompanying
 "data_train_*.json.gz" file is generated that contains metainformation on
 various features and their values.

TODO:
-----
Following needs to be done:
  - Latitude and longitude need to be on a point on a unit sphere? Keep as is
    and add three further columns for (x,y,z)?
  - Country codes are *several*.
  - Other types of SOMs.
  - Use BaseMap for visualizations?
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import tempfile

from absl import app
from absl import flags
from absl import logging

import constants as const
import data_info as data_lib
import sigtyp_reader as sigtyp

flags.DEFINE_string(
    "sigtyp_dir", "",
    "Directory containing SIGTYP original training and development.")

flags.DEFINE_string(
    "output_dir", "",
    "Output directory for preprocessed files.")

flags.DEFINE_bool(
    "categorical_as_ints", False,
    "Encode all the categorical features as ints.")

FLAGS = flags.FLAGS


def _write_dict(data, file_type, output_filename):
  """Writes dictionary of a specified type to a file in output directory."""
  output_filename = os.path.join(
      FLAGS.output_dir,
      output_filename + "_" + file_type + data_lib.FILE_EXTENSION)
  data_lib.write_data_info(output_filename, data)


def _process_file(filename, base_dir=None):
  """Preprocesses supplied data file."""
  if not base_dir:
    base_dir = FLAGS.sigtyp_dir
  full_path = os.path.join(base_dir, filename + ".csv")
  _, df, data_info = sigtyp.read(
      full_path, categorical_as_ints=FLAGS.categorical_as_ints)
  _write_dict(data_info, filename, const.DATA_INFO_FILENAME)

  # Save preprocessed data frames to a csv.
  output_file = os.path.join(FLAGS.output_dir, filename + ".csv")
  logging.info("Saving preprocessed data to \"%s\" ...", output_file)
  df.to_csv(output_file, sep="|", index=False, float_format="%g")
  return data_info


def _write_combined_data(file_types, output_file_name):
  """Combine data from multiple files."""
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_file = os.path.join(temp_dir, output_file_name + ".csv")
    with open(temp_file, "w", encoding=const.ENCODING) as out_f:
      header = None
      all_lines = []
      for file_type in file_types:
        input_path = os.path.join(FLAGS.sigtyp_dir, file_type + ".csv")
        with open(input_path, "r", encoding=const.ENCODING) as in_f:
          lines = in_f.readlines()
          if not header:
            header = lines[0]
          lines.pop(0)  # Remove header.
          all_lines.extend(lines)

      # Sort the lines by the WALS code and dump them.
      all_lines = sorted(all_lines, key=lambda x: x.split("|")[0])
      all_lines.insert(0, header)
      out_f.write("".join(all_lines))
    _process_file(output_file_name, base_dir=temp_dir)


def _process_files():
  """Processes input files."""
  # Process training and development files individually.
  _process_file(const.TRAIN_FILENAME)
  _process_file(const.DEV_FILENAME)
  test_data_info = _process_file(const.TEST_BLIND_FILENAME)

  # Save features requested for prediction in the test set.
  features_to_predict = test_data_info[const.DATA_KEY_FEATURES_TO_PREDICT]
  if not features_to_predict:
    raise ValueError("No features requested for prediction!")
  predict_dict_path = os.path.join(FLAGS.output_dir,
                                   const.FEATURES_TO_PREDICT_FILENAME + ".json")
  logging.info("Saving features for prediction in \"%s\" ...",
               predict_dict_path)
  with open(predict_dict_path, "w", encoding=const.ENCODING) as f:
    json.dump(features_to_predict, f)

  # Process the combine the datasets.
  _write_combined_data([const.TRAIN_FILENAME, const.DEV_FILENAME],
                       const.TRAIN_DEV_FILENAME)
  _write_combined_data([const.TRAIN_FILENAME, const.DEV_FILENAME,
                        const.TEST_BLIND_FILENAME],
                       const.TRAIN_DEV_TEST_FILENAME)


def main(unused_argv):
  # Check flags.
  if not FLAGS.sigtyp_dir:
    raise ValueError("Specify --sigtyp_dir for input data!")
  if not FLAGS.output_dir:
    raise ValueError("Specify --output_dir for preprocessed data!")
  _process_files()


if __name__ == "__main__":
  app.run(main)
