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

"""Convert xls to csv."""

from absl import app
from absl import flags
import pandas as pd

flags.DEFINE_string('xls_file', '', 'Input xls file to convert')
flags.DEFINE_string('csv_file', '', 'Output csv file')


def main(argv):
  del argv
  data = pd.read_excel(flags.FLAGS.xls_file)
  data.to_csv(flags.FLAGS.csv_file, index=None, header=True)


if __name__ == '__main__':
  app.run(main)
